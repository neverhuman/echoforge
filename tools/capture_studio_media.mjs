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
const GIF_WIDTH = 1280;
const GIF_HEIGHT = 720;
const GIF_FPS = Number(process.env.STUDIO_CAPTURE_GIF_FPS || "3");
const GIF_SECONDS = Number(process.env.STUDIO_CAPTURE_GIF_SECONDS || "6");
const GIF_FRAME_COUNT = Math.max(12, Math.round(GIF_FPS * GIF_SECONDS));
const GIF_FRAME_INTERVAL_MS = Math.max(100, Math.round(1000 / GIF_FPS));
const GIF_FRAME_DELAY_CS = Math.max(4, Math.round(100 / GIF_FPS));
const GIF_BACKGROUND = [5, 6, 7];

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

function clampByte(value) {
	return Math.max(0, Math.min(255, Math.round(value)));
}

function lerpChannel(a, b, t) {
	return clampByte(a + (b - a) * t);
}

function interpolateRgb(stops, t) {
	if (stops.length === 1) return stops[0];
	const scaled = t * (stops.length - 1);
	const index = Math.min(stops.length - 2, Math.max(0, Math.floor(scaled)));
	const localT = scaled - index;
	const start = stops[index];
	const end = stops[index + 1];
	return [
		lerpChannel(start[0], end[0], localT),
		lerpChannel(start[1], end[1], localT),
		lerpChannel(start[2], end[2], localT),
	];
}

function makeRamp(stops, count) {
	const ramp = [];
	for (let i = 0; i < count; i += 1) {
		const t = count === 1 ? 0 : i / (count - 1);
		ramp.push(interpolateRgb(stops, t));
	}
	return ramp;
}

function appendRamp(palette, stops, count) {
	const base = palette.length;
	palette.push(...makeRamp(stops, count));
	return base;
}

function buildStudioGifPalette() {
	const palette = [];
	const bands = {};

	bands.gray = appendRamp(
		palette,
		[
			[5, 6, 7],
			[18, 20, 22],
			[49, 58, 61],
			[95, 106, 112],
			[156, 169, 175],
			[219, 228, 232],
			[237, 244, 247],
		],
		64,
	);
	bands.cyan = appendRamp(
		palette,
		[
			[6, 9, 16],
			[18, 46, 54],
			[31, 127, 158],
			[53, 214, 197],
			[159, 233, 255],
		],
		32,
	);
	bands.amber = appendRamp(
		palette,
		[
			[16, 12, 8],
			[43, 33, 15],
			[88, 68, 28],
			[184, 146, 74],
			[232, 184, 107],
			[255, 210, 122],
			[255, 248, 222],
		],
		32,
	);
	bands.green = appendRamp(
		palette,
		[
			[7, 19, 13],
			[18, 49, 31],
			[33, 87, 53],
			[79, 149, 96],
			[127, 207, 143],
			[178, 242, 196],
		],
		32,
	);
	bands.red = appendRamp(
		palette,
		[
			[22, 7, 7],
			[58, 18, 18],
			[104, 34, 34],
			[165, 70, 70],
			[239, 111, 108],
			[255, 184, 182],
		],
		32,
	);
	bands.blue = appendRamp(
		palette,
		[
			[8, 16, 29],
			[17, 33, 56],
			[28, 74, 117],
			[60, 119, 178],
			[142, 196, 255],
			[207, 231, 255],
		],
		24,
	);
	bands.magenta = appendRamp(
		palette,
		[
			[18, 8, 23],
			[46, 15, 72],
			[62, 18, 96],
			[126, 44, 120],
			[190, 54, 96],
			[248, 162, 58],
		],
		24,
	);
	bands.warm = appendRamp(
		palette,
		[
			[26, 19, 9],
			[66, 49, 20],
			[120, 90, 37],
			[195, 156, 82],
			[246, 210, 130],
			[255, 248, 222],
		],
		16,
	);

	if (palette.length !== 256) {
		throw new Error(
			`studio GIF palette must contain 256 colors, got ${palette.length}`,
		);
	}

	return { palette, bands };
}

const STUDIO_GIF = buildStudioGifPalette();

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

function blendChannel(value, alpha, background) {
	return Math.round((value * alpha + background * (255 - alpha)) / 255);
}

function rgbToHsv(r, g, b) {
	const max = Math.max(r, g, b);
	const min = Math.min(r, g, b);
	const delta = max - min;
	let hue = 0;
	if (delta > 0) {
		if (max === r) hue = ((g - b) / delta) % 6;
		else if (max === g) hue = (b - r) / delta + 2;
		else hue = (r - g) / delta + 4;
		hue *= 60;
		if (hue < 0) hue += 360;
	}
	const saturation = max === 0 ? 0 : delta / max;
	return { hue, saturation, value: max };
}

function rampIndex(base, count, value) {
	return (
		base +
		Math.max(0, Math.min(count - 1, Math.round((value * (count - 1)) / 255)))
	);
}

function quantizeStudioGif(r, g, b) {
	const { hue, saturation, value } = rgbToHsv(r, g, b);
	const chroma = Math.max(r, g, b) - Math.min(r, g, b);
	if (chroma < 18 || saturation < 0.16 || value < 40) {
		return rampIndex(STUDIO_GIF.bands.gray, 64, Math.round((r + g + b) / 3));
	}
	if (hue >= 155 && hue < 220) {
		return rampIndex(STUDIO_GIF.bands.cyan, 32, value);
	}
	if (hue >= 220 && hue < 265) {
		return rampIndex(STUDIO_GIF.bands.blue, 24, value);
	}
	if (hue >= 265 && hue < 335) {
		return rampIndex(STUDIO_GIF.bands.magenta, 24, value);
	}
	if (hue >= 335 || hue < 20) {
		return rampIndex(STUDIO_GIF.bands.red, 32, value);
	}
	if (hue >= 20 && hue < 65) {
		return rampIndex(STUDIO_GIF.bands.amber, 32, value);
	}
	if (hue >= 65 && hue < 155) {
		return rampIndex(STUDIO_GIF.bands.green, 32, value);
	}
	return rampIndex(STUDIO_GIF.bands.warm, 16, value);
}

function quantizeFrame(image) {
	const indices = new Uint8Array(image.width * image.height);
	for (let i = 0, p = 0; i < indices.length; i += 1, p += 4) {
		const alpha = image.rgba[p + 3];
		const r = blendChannel(image.rgba[p], alpha, GIF_BACKGROUND[0]);
		const g = blendChannel(image.rgba[p + 1], alpha, GIF_BACKGROUND[1]);
		const b = blendChannel(image.rgba[p + 2], alpha, GIF_BACKGROUND[2]);
		indices[i] = quantizeStudioGif(r, g, b);
	}
	return indices;
}

function writeStudioGif(path, frames) {
	const bytes = [];
	pushAscii(bytes, "GIF89a");
	pushWord(bytes, GIF_WIDTH);
	pushWord(bytes, GIF_HEIGHT);
	bytes.push(0b11110111, 0, 0);
	for (const [r, g, b] of STUDIO_GIF.palette) bytes.push(r, g, b);
	bytes.push(0x21, 0xff, 0x0b);
	pushAscii(bytes, "NETSCAPE2.0");
	bytes.push(0x03, 0x01);
	pushWord(bytes, 0);
	bytes.push(0);

	for (const frame of frames) {
		bytes.push(0x21, 0xf9, 0x04, 0x00);
		pushWord(bytes, GIF_FRAME_DELAY_CS);
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

async function openLiveRadarRun(page) {
	await openStudioState(page, {
		name: "live-radar-run",
		testId: "tab-radar",
		waitTestId: "radar-console",
		width: GIF_WIDTH,
		height: GIF_HEIGHT,
		delayMs: 500,
	});
	const startButton = page.getByTestId("sim-start");
	if (await startButton.isVisible().catch(() => false)) {
		await startButton.click();
	}
	await page.waitForFunction(
		() => /scans\s+[1-9][0-9]*/.test(document.body.innerText),
		null,
		{ timeout: 15000 },
	);
	await page.waitForTimeout(750);
}

async function captureGif(browser) {
	const context = await browser.newContext({
		deviceScaleFactor: 1,
		viewport: { width: GIF_WIDTH, height: GIF_HEIGHT },
	});
	const page = await context.newPage();
	const frames = [];
	await openLiveRadarRun(page);
	for (let frame = 0; frame < GIF_FRAME_COUNT; frame += 1) {
		const png = await page.screenshot({
			animations: "allow",
			caret: "hide",
		});
		frames.push(quantizeFrame(decodePng(png)));
		await page.waitForTimeout(GIF_FRAME_INTERVAL_MS);
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
			viewport: `${GIF_WIDTH}x${GIF_HEIGHT}@1x full-page`,
			route: "/",
			source: `${GIF_SECONDS}s-playwright-live-radar-run`,
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
				gif_frames: [
					`live-radar-run ${GIF_FRAME_COUNT} frames at ${GIF_FPS} fps`,
				],
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
