#!/usr/bin/env node
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { inflateSync } from "node:zlib";
import { chromium } from "@playwright/test";

const ROOT = process.cwd();
const OUT_DIR = join(ROOT, "assets/readme/studio");
const HOST = "127.0.0.1";
const PORT = Number(process.env.STUDIO_CAPTURE_PORT || "4173");
const BASE_URL = process.env.STUDIO_CAPTURE_URL || `http://${HOST}:${PORT}`;
const SKIP_SERVER = process.env.STUDIO_CAPTURE_SKIP_SERVER === "1";
const SERVER_MODE = process.env.STUDIO_CAPTURE_SERVER || "studio";
const READY_TIMEOUT_MS = Number(
	process.env.STUDIO_CAPTURE_READY_TIMEOUT_MS || "120000",
);
const HI_DPI_SCALE = Number(process.env.STUDIO_CAPTURE_SCALE || "2");
const GIF_WIDTH = 960;
const GIF_HEIGHT = 720;
const GIF_CAPTURE_WIDTH = 1600;
const GIF_CAPTURE_HEIGHT = 1000;

const CAPTURES = [
	{
		name: "command-center",
		testId: null,
		waitTestId: "command-center",
		width: 1600,
		height: 1000,
	},
	{
		name: "live-radar",
		testId: "tab-radar",
		waitTestId: "radar-console",
		width: 1600,
		height: 1000,
		delayMs: 1200,
	},
	{
		name: "monte-carlo-builder",
		testId: "tab-builder",
		waitTestId: "monte-carlo-builder",
		width: 1600,
		height: 1000,
	},
];

const STORYBOARD = [
	{
		name: "command-center",
		testId: null,
		waitTestId: "command-center",
		delayMs: 500,
	},
	{
		name: "live-radar",
		testId: "tab-radar",
		waitTestId: "radar-console",
		delayMs: 1200,
	},
	{
		name: "monte-carlo-builder",
		testId: "tab-builder",
		waitTestId: "monte-carlo-builder",
		delayMs: 500,
	},
];

function formatManifestJson(manifest) {
	const pretty = JSON.stringify(manifest, null, "\t");
	const frameLines = manifest.capture.gif_frames
		.map((frame) => `\t\t\t${JSON.stringify(frame)}`)
		.join(",\n");
	const multilineFrames = `\n\t\t"gif_frames": [\n${frameLines}\n\t\t]`;
	const inlineFrames = `\n\t\t"gif_frames": [${manifest.capture.gif_frames
		.map((frame) => JSON.stringify(frame))
		.join(", ")}]`;
	return `${pretty.replace(multilineFrames, inlineFrames)}\n`;
}

function runChecked(command, args) {
	return new Promise((resolve, reject) => {
		const child = spawn(command, args, { cwd: ROOT, stdio: "inherit" });
		child.on("exit", (code) => {
			if (code === 0) resolve();
			else reject(new Error(`${command} ${args.join(" ")} exited ${code}`));
		});
	});
}

function spawnServer(command, args, env = {}) {
	return spawn(command, args, {
		cwd: ROOT,
		stdio: "inherit",
		detached: true,
		env: { ...process.env, ...env },
	});
}

function startVitePreview() {
	return spawnServer("./node_modules/.bin/vite", [
		"preview",
		"--config",
		"apps/web/vite.config.ts",
		"--host",
		HOST,
		"--port",
		String(PORT),
	]);
}

function startStudioService() {
	return spawnServer("cargo", ["run", "-p", "echoforge-studio", "--locked"], {
		HOST,
		PORT: String(PORT),
		ECHOFORGE_WEB_DIST: "apps/web/dist",
		ECHOFORGE_SIM_AUTOSTART: "false",
	});
}

function startServer() {
	if (SERVER_MODE === "vite") return startVitePreview();
	if (SERVER_MODE === "studio") return startStudioService();
	throw new Error(`Unsupported STUDIO_CAPTURE_SERVER=${SERVER_MODE}`);
}

function stopServer(child) {
	if (!child?.pid) return;
	try {
		process.kill(-child.pid, "SIGTERM");
	} catch {
		child.kill("SIGTERM");
	}
}

async function waitForServer(url, timeoutMs = READY_TIMEOUT_MS) {
	const started = Date.now();
	while (Date.now() - started < timeoutMs) {
		try {
			const response = await fetch(url);
			if (response.ok) return;
		} catch {}
		await new Promise((resolve) => setTimeout(resolve, 250));
	}
	throw new Error(`Timed out waiting for ${url}`);
}

async function stabilizePage(page, delayMs = 350) {
	await page
		.evaluate(() => document.fonts?.ready ?? Promise.resolve())
		.catch(() => undefined);
	await page.waitForTimeout(delayMs);
}

async function openStudioState(page, captureSpec) {
	await page.setViewportSize({
		width: captureSpec.width,
		height: captureSpec.height,
	});
	await page.goto(BASE_URL, { waitUntil: "domcontentloaded" });
	await page
		.waitForLoadState("networkidle", { timeout: 5000 })
		.catch(() => undefined);
	if (captureSpec.testId) {
		await page.getByTestId(captureSpec.testId).click();
	}
	if (captureSpec.waitTestId) {
		await page
			.getByTestId(captureSpec.waitTestId)
			.waitFor({ state: "visible", timeout: 15000 });
	}
	await stabilizePage(page, captureSpec.delayMs ?? 500);
}

function pushWord(bytes, value) {
	bytes.push(value & 0xff, (value >> 8) & 0xff);
}

function pushAscii(bytes, text) {
	for (const char of text) bytes.push(char.charCodeAt(0));
}

function writeSubBlocks(bytes, data) {
	for (let offset = 0; offset < data.length; offset += 255) {
		const block = data.slice(offset, offset + 255);
		bytes.push(block.length, ...block);
	}
	bytes.push(0);
}

function lzwEncode(indices, minCodeSize) {
	const clear = 1 << minCodeSize;
	const end = clear + 1;
	let codeSize = minCodeSize + 1;
	const packed = [];
	let bitBuffer = 0;
	let bitCount = 0;

	function emit(code) {
		bitBuffer |= code << bitCount;
		bitCount += codeSize;
		while (bitCount >= 8) {
			packed.push(bitBuffer & 0xff);
			bitBuffer >>= 8;
			bitCount -= 8;
		}
	}

	// Literal runs trade compression for decoder compatibility. Clearing before
	// the dictionary reaches 10-bit codes keeps every emitted code at 9 bits.
	const maxLiteralRun = 250;
	emit(clear);
	let literalRun = 0;
	for (const index of indices) {
		if (literalRun >= maxLiteralRun) {
			emit(clear);
			codeSize = minCodeSize + 1;
			literalRun = 0;
		}
		emit(index);
		literalRun += 1;
	}
	emit(end);
	if (bitCount > 0) packed.push(bitBuffer & 0xff);
	return packed;
}

function paethPredictor(left, up, upLeft) {
	const p = left + up - upLeft;
	const pa = Math.abs(p - left);
	const pb = Math.abs(p - up);
	const pc = Math.abs(p - upLeft);
	if (pa <= pb && pa <= pc) return left;
	if (pb <= pc) return up;
	return upLeft;
}

function decodePng(input) {
	const buffer = Buffer.isBuffer(input) ? input : Buffer.from(input);
	if (
		buffer.length < 8 ||
		buffer.toString("hex", 0, 8) !== "89504e470d0a1a0a"
	) {
		throw new Error("screenshot is not a PNG");
	}

	let width = 0;
	let height = 0;
	let bitDepth = 0;
	let colorType = 0;
	let interlace = 0;
	let palette = [];
	let transparency = [];
	const idat = [];

	for (let offset = 8; offset < buffer.length; ) {
		const length = buffer.readUInt32BE(offset);
		const type = buffer.toString("ascii", offset + 4, offset + 8);
		const data = buffer.subarray(offset + 8, offset + 8 + length);
		offset += 12 + length;

		if (type === "IHDR") {
			width = data.readUInt32BE(0);
			height = data.readUInt32BE(4);
			bitDepth = data[8];
			colorType = data[9];
			interlace = data[12];
		} else if (type === "PLTE") {
			palette = [];
			for (let i = 0; i < data.length; i += 3) {
				palette.push([data[i], data[i + 1], data[i + 2]]);
			}
		} else if (type === "tRNS") {
			transparency = Array.from(data);
		} else if (type === "IDAT") {
			idat.push(data);
		} else if (type === "IEND") {
			break;
		}
	}

	if (!width || !height) throw new Error("PNG is missing IHDR dimensions");
	if (bitDepth !== 8) throw new Error(`unsupported PNG bit depth ${bitDepth}`);
	if (interlace !== 0)
		throw new Error("interlaced PNG screenshots are not supported");

	const channels = {
		0: 1,
		2: 3,
		3: 1,
		4: 2,
		6: 4,
	}[colorType];
	if (!channels) throw new Error(`unsupported PNG color type ${colorType}`);

	const raw = inflateSync(Buffer.concat(idat));
	const stride = width * channels;
	const rows = new Uint8Array(height * stride);
	let sourceOffset = 0;
	let previous = new Uint8Array(stride);

	for (let y = 0; y < height; y += 1) {
		const filter = raw[sourceOffset];
		sourceOffset += 1;
		const current = new Uint8Array(stride);
		for (let x = 0; x < stride; x += 1) {
			const value = raw[sourceOffset];
			sourceOffset += 1;
			const left = x >= channels ? current[x - channels] : 0;
			const up = previous[x] ?? 0;
			const upLeft = x >= channels ? previous[x - channels] : 0;
			let predictor = 0;
			if (filter === 1) predictor = left;
			else if (filter === 2) predictor = up;
			else if (filter === 3) predictor = Math.floor((left + up) / 2);
			else if (filter === 4) predictor = paethPredictor(left, up, upLeft);
			else if (filter !== 0)
				throw new Error(`unsupported PNG filter ${filter}`);
			current[x] = (value + predictor) & 0xff;
		}
		rows.set(current, y * stride);
		previous = current;
	}

	const rgba = new Uint8ClampedArray(width * height * 4);
	for (let i = 0, p = 0; i < rows.length; p += 4) {
		if (colorType === 0) {
			const gray = rows[i];
			i += 1;
			rgba[p] = gray;
			rgba[p + 1] = gray;
			rgba[p + 2] = gray;
			rgba[p + 3] = 255;
		} else if (colorType === 2) {
			rgba[p] = rows[i];
			rgba[p + 1] = rows[i + 1];
			rgba[p + 2] = rows[i + 2];
			rgba[p + 3] = 255;
			i += 3;
		} else if (colorType === 3) {
			const index = rows[i];
			i += 1;
			const [r, g, b] = palette[index] || [0, 0, 0];
			rgba[p] = r;
			rgba[p + 1] = g;
			rgba[p + 2] = b;
			rgba[p + 3] = transparency[index] ?? 255;
		} else if (colorType === 4) {
			const gray = rows[i];
			rgba[p] = gray;
			rgba[p + 1] = gray;
			rgba[p + 2] = gray;
			rgba[p + 3] = rows[i + 1];
			i += 2;
		} else {
			rgba[p] = rows[i];
			rgba[p + 1] = rows[i + 1];
			rgba[p + 2] = rows[i + 2];
			rgba[p + 3] = rows[i + 3];
			i += 4;
		}
	}

	return { width, height, rgba };
}

function rgb332Palette() {
	const palette = [];
	for (let index = 0; index < 256; index += 1) {
		const r = (index >> 5) & 0x07;
		const g = (index >> 2) & 0x07;
		const b = index & 0x03;
		palette.push([
			Math.round((r * 255) / 7),
			Math.round((g * 255) / 7),
			Math.round((b * 255) / 3),
		]);
	}
	return palette;
}

function blendChannel(value, alpha, background) {
	return Math.round((value * alpha + background * (255 - alpha)) / 255);
}

function quantizeRgb332(r, g, b) {
	const ri = Math.max(0, Math.min(7, Math.round((r * 7) / 255)));
	const gi = Math.max(0, Math.min(7, Math.round((g * 7) / 255)));
	const bi = Math.max(0, Math.min(3, Math.round((b * 3) / 255)));
	return (ri << 5) | (gi << 2) | bi;
}

function fitAndQuantize(image, outputWidth, outputHeight) {
	const background = [5, 6, 7];
	const backgroundIndex = quantizeRgb332(...background);
	const scale = Math.min(
		outputWidth / image.width,
		outputHeight / image.height,
	);
	const fittedWidth = Math.max(1, Math.round(image.width * scale));
	const fittedHeight = Math.max(1, Math.round(image.height * scale));
	const offsetX = Math.floor((outputWidth - fittedWidth) / 2);
	const offsetY = Math.floor((outputHeight - fittedHeight) / 2);
	const indices = new Uint8Array(outputWidth * outputHeight);
	indices.fill(backgroundIndex);
	for (let y = 0; y < outputHeight; y += 1) {
		const targetY = y - offsetY;
		if (targetY < 0 || targetY >= fittedHeight) continue;
		const sy = Math.min(
			image.height - 1,
			Math.floor((targetY * image.height) / fittedHeight),
		);
		for (let x = 0; x < outputWidth; x += 1) {
			const targetX = x - offsetX;
			if (targetX < 0 || targetX >= fittedWidth) continue;
			const sx = Math.min(
				image.width - 1,
				Math.floor((targetX * image.width) / fittedWidth),
			);
			const source = (sy * image.width + sx) * 4;
			const alpha = image.rgba[source + 3];
			const r = blendChannel(image.rgba[source], alpha, background[0]);
			const g = blendChannel(image.rgba[source + 1], alpha, background[1]);
			const b = blendChannel(image.rgba[source + 2], alpha, background[2]);
			indices[y * outputWidth + x] = quantizeRgb332(r, g, b);
		}
	}
	return indices;
}

function writeStudioGif(path, frames) {
	const palette = rgb332Palette();
	const bytes = [];
	pushAscii(bytes, "GIF89a");
	pushWord(bytes, GIF_WIDTH);
	pushWord(bytes, GIF_HEIGHT);
	bytes.push(0b11110111, 0, 0);
	for (const [r, g, b] of palette) bytes.push(r, g, b);
	bytes.push(0x21, 0xff, 0x0b);
	pushAscii(bytes, "NETSCAPE2.0");
	bytes.push(0x03, 0x01);
	pushWord(bytes, 0);
	bytes.push(0);

	for (const frame of frames) {
		bytes.push(0x21, 0xf9, 0x04, 0x00);
		pushWord(bytes, 120);
		bytes.push(0, 0);
		bytes.push(0x2c);
		pushWord(bytes, 0);
		pushWord(bytes, 0);
		pushWord(bytes, GIF_WIDTH);
		pushWord(bytes, GIF_HEIGHT);
		bytes.push(0);
		bytes.push(8);
		writeSubBlocks(bytes, lzwEncode(frame, 8));
	}
	bytes.push(0x3b);
	writeFileSync(path, Buffer.from(bytes));
}

async function assertGifRenders(browser, gifPath) {
	const gifData = readFileSync(gifPath).toString("base64");
	const context = await browser.newContext({
		deviceScaleFactor: 1,
		viewport: { width: GIF_WIDTH, height: GIF_HEIGHT },
	});
	const page = await context.newPage();
	try {
		await page.setContent(
			`<!doctype html><style>body{margin:0;background:#050607}img{display:block;width:${GIF_WIDTH}px;height:${GIF_HEIGHT}px}</style><img alt="Studio GIF check" src="data:image/gif;base64,${gifData}">`,
		);
		const result = await page.evaluate(async () => {
			const img = document.querySelector("img");
			if (!img) return { ok: false, reason: "missing image" };
			await img.decode().catch(() => undefined);
			const width = img.naturalWidth;
			const height = img.naturalHeight;
			if (width === 0 || height === 0) {
				return { ok: false, reason: "image did not decode", width, height };
			}
			const canvas = document.createElement("canvas");
			canvas.width = width;
			canvas.height = height;
			const ctx = canvas.getContext("2d");
			if (!ctx)
				return { ok: false, reason: "canvas unavailable", width, height };
			ctx.drawImage(img, 0, 0);
			const pixels = ctx.getImageData(0, 0, width, height).data;
			let visibleSamples = 0;
			for (let y = 0; y < height; y += 8) {
				for (let x = 0; x < width; x += 8) {
					const offset = (y * width + x) * 4;
					const r = pixels[offset];
					const g = pixels[offset + 1];
					const b = pixels[offset + 2];
					const a = pixels[offset + 3];
					if (a > 0 && Math.max(r, g, b) > 24) visibleSamples += 1;
				}
			}
			return { ok: true, width, height, visibleSamples };
		});
		if (
			!result.ok ||
			result.width !== GIF_WIDTH ||
			result.height !== GIF_HEIGHT ||
			result.visibleSamples < 500
		) {
			throw new Error(
				`Studio GIF render check failed: ${JSON.stringify(result)}`,
			);
		}
	} finally {
		await context.close();
	}
}

async function captureScreenshots(browser) {
	const context = await browser.newContext({
		deviceScaleFactor: HI_DPI_SCALE,
		viewport: { width: CAPTURES[0].width, height: CAPTURES[0].height },
	});
	const page = await context.newPage();
	const assets = [];
	for (const captureSpec of CAPTURES) {
		await openStudioState(page, captureSpec);
		const path = join(OUT_DIR, `${captureSpec.name}.png`);
		await page.screenshot({
			path,
			fullPage: true,
			animations: "disabled",
			caret: "hide",
		});
		assets.push({
			kind: "screenshot",
			path: `assets/readme/studio/${captureSpec.name}.png`,
			viewport: `${captureSpec.width}x${captureSpec.height}@${HI_DPI_SCALE}x`,
			route: "/",
		});
	}
	await context.close();
	return assets;
}

async function captureGif(browser) {
	const context = await browser.newContext({
		deviceScaleFactor: 1,
		viewport: { width: GIF_CAPTURE_WIDTH, height: GIF_CAPTURE_HEIGHT },
	});
	const page = await context.newPage();
	const frames = [];
	for (const step of STORYBOARD) {
		await openStudioState(page, {
			...step,
			width: GIF_CAPTURE_WIDTH,
			height: GIF_CAPTURE_HEIGHT,
		});
		const png = await page.screenshot({
			fullPage: true,
			type: "png",
			animations: "disabled",
			caret: "hide",
		});
		frames.push(fitAndQuantize(decodePng(png), GIF_WIDTH, GIF_HEIGHT));
	}
	await context.close();

	const gifPath = join(OUT_DIR, "studio-demo.gif");
	writeStudioGif(gifPath, frames);
	await assertGifRenders(browser, gifPath);
}

async function capture() {
	mkdirSync(OUT_DIR, { recursive: true });
	if (!SKIP_SERVER) {
		await runChecked("npm", ["run", "web:build"]);
	}

	const server = SKIP_SERVER ? null : startServer();
	try {
		await waitForServer(BASE_URL);
		const browser = await chromium.launch();
		const assets = await captureScreenshots(browser);
		await captureGif(browser);
		await browser.close();

		const badgePath = join(OUT_DIR, "jankurai-score.svg");
		assets.unshift({
			kind: "gif",
			path: "assets/readme/studio/studio-demo.gif",
			viewport: `${GIF_CAPTURE_WIDTH}x${GIF_CAPTURE_HEIGHT} full-page fit to ${GIF_WIDTH}x${GIF_HEIGHT}`,
			route: "/",
			source: "playwright-storyboard",
		});
		if (existsSync(badgePath)) {
			assets.unshift({
				kind: "badge",
				path: "assets/readme/studio/jankurai-score.svg",
				viewport: "n/a",
				route: "/",
			});
		}

		const manifest = {
			generated_utc: new Date(0).toISOString(),
			tool: "tools/capture_studio_media.mjs",
			route: "/",
			capture: {
				server: SKIP_SERVER ? "external" : SERVER_MODE,
				sim_autostart: SKIP_SERVER ? "external" : false,
				screenshot_scale: HI_DPI_SCALE,
				gif_frames: STORYBOARD.map((frame) => frame.name),
			},
			strict_open_note:
				"README media is deterministic documentation capture from Playwright-rendered Studio states; it is not measured truth, proprietary-equivalent behavior, or field-performance evidence.",
			assets,
		};
		writeFileSync(join(OUT_DIR, "manifest.json"), formatManifestJson(manifest));
	} finally {
		stopServer(server);
	}
}

capture().catch((err) => {
	console.error(err);
	process.exit(1);
});
