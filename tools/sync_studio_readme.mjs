#!/usr/bin/env node
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

function fatal(code, message) {
	console.error(`sync_studio_readme: ${message}`);
	process.exit(code);
}

function readJson(path, label) {
	try {
		return JSON.parse(readFileSync(path, "utf8"));
	} catch (err) {
		fatal(1, `cannot read ${label} ${path}: ${err.message}`);
	}
}

const args = process.argv.slice(2);
const options = {};
for (let i = 0; i < args.length; i += 1) {
	const arg = args[i];
	const next = args[i + 1];
	if (arg === "--readme" || arg === "--manifest" || arg === "--badge") {
		if (!next || next.startsWith("--"))
			fatal(3, `${arg} requires a path argument`);
		options[arg.slice(2)] = resolve(process.cwd(), next);
		i += 1;
	} else {
		fatal(3, `unknown argument: ${arg}`);
	}
}

if (!options.readme || !options.manifest || !options.badge) {
	fatal(
		3,
		"usage: node tools/sync_studio_readme.mjs --readme README.md --manifest assets/readme/studio/manifest.json --badge assets/readme/studio/jankurai-score.svg",
	);
}

const heroLine =
	'<img src="./assets/ecoforge.png" alt="EchoForge hero" width="100%" />';
const readme = readFileSync(options.readme, "utf8");
const lines = readme.split(/\r?\n/);
if (lines[0] !== heroLine)
	fatal(1, "README hero image must remain the first line");
if (lines[2] !== "# EchoForge")
	fatal(1, "README title must remain directly below the hero image");

const valueIndex = lines.indexOf("## Value");
if (valueIndex === -1) fatal(1, 'README is missing the "## Value" anchor');

const manifest = readJson(options.manifest, "manifest");
try {
	readFileSync(options.badge, "utf8");
} catch (err) {
	fatal(1, `cannot read badge ${options.badge}: ${err.message}`);
}
const assets = Array.isArray(manifest.assets) ? manifest.assets : [];
const gif = assets.find((asset) => asset?.kind === "gif");
if (!gif) fatal(1, "manifest is missing the studio demo gif");

const screenshotOrder = [
	"command-center.png",
	"live-radar.png",
	"monte-carlo-builder.png",
];
const screenshotsByName = new Map();
for (const asset of assets) {
	if (asset?.kind !== "screenshot") continue;
	const name = String(asset.path || "")
		.split("/")
		.pop();
	screenshotsByName.set(name, asset.path);
}
for (const name of screenshotOrder) {
	if (!screenshotsByName.has(name))
		fatal(1, `manifest is missing screenshot ${name}`);
}

const replacement = [
	'<img src="./assets/readme/studio/jankurai-score.svg" alt="Jankurai score" />',
	"",
	"EchoForge is a strict-open, radar-first, GPU-native synthetic sensing foundry. It publishes public-proxy object signatures, uncertainty-scored radar artifacts, and reproducible validation evidence so downstream work can be inspected, rerun, and compared without claiming measured truth or proprietary-equivalent sensor behavior.",
	"",
	"The claim boundary stays narrow: public-source priors only, explicit uncertainty, hard negatives treated as robustness work, and no classified, vendor-private, or exact field-performance claims.",
	"",
	"## Studio Preview",
	"",
	"![EchoForge Studio demo](./assets/readme/studio/studio-demo.gif)",
	"",
	"EchoForge Studio is a same-origin Rust + Vite interface for live radar playback, Monte Carlo campaign setup, run archive/restore/duplicate workflows, artifact review, and headless API usage. The UI keeps source cards, validation tier, uncertainty language, seed, and scenario hash visible before export.",
	"",
	"| Command Center | Live Radar | Monte Carlo Builder |",
	"| --- | --- | --- |",
	`| ![Command Center](./${screenshotsByName.get("command-center.png")}) | ![Live Radar](./${screenshotsByName.get("live-radar.png")}) | ![Monte Carlo Builder](./${screenshotsByName.get("monte-carlo-builder.png")}) |`,
	"",
	"README media is tracked under [assets/readme/studio](./assets/readme/studio/) with a deterministic capture manifest at [assets/readme/studio/manifest.json](./assets/readme/studio/manifest.json). Regenerate after a Studio UI change with the same lane used by the post-merge sync:",
	"",
	"```bash",
	"rtk bash ops/run-lane.sh studio-sync",
	"```",
].join("\n");

const next = [
	heroLine,
	"",
	"# EchoForge",
	"",
	replacement,
	"",
	...lines.slice(valueIndex),
].join("\n");

if (next !== readme) {
	writeFileSync(options.readme, next.endsWith("\n") ? next : `${next}\n`);
}
