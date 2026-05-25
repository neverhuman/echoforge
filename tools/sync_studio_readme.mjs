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

function asInteger(value, label) {
	if (!Number.isInteger(value)) fatal(1, `score report is missing ${label}`);
	return value;
}

function asString(value, label) {
	if (typeof value !== "string" || value.length === 0)
		fatal(1, `score report is missing ${label}`);
	if (!/^[A-Za-z0-9_.-]+$/.test(value))
		fatal(1, `score report has unexpected ${label}: ${value}`);
	return value;
}

function asBoolean(value, label) {
	if (typeof value !== "boolean") fatal(1, `score report is missing ${label}`);
	return value;
}

function scoreBadgeColor(score, minimumScore, decisionPassed) {
	if (decisionPassed && score >= 90) return "brightgreen";
	if (decisionPassed && score >= minimumScore) return "green";
	if (score >= 70) return "yellow";
	return "red";
}

const args = process.argv.slice(2);
const options = {};
for (let i = 0; i < args.length; i += 1) {
	const arg = args[i];
	const next = args[i + 1];
	if (
		arg === "--readme" ||
		arg === "--manifest" ||
		arg === "--badge" ||
		arg === "--score-json"
	) {
		if (!next || next.startsWith("--"))
			fatal(3, `${arg} requires a path argument`);
		options[arg.slice(2).replace("-", "")] = resolve(process.cwd(), next);
		i += 1;
	} else {
		fatal(3, `unknown argument: ${arg}`);
	}
}

if (!options.readme || !options.manifest || !options.badge) {
	fatal(
		3,
		"usage: node tools/sync_studio_readme.mjs --readme README.md --manifest assets/readme/studio/manifest.json --badge assets/readme/studio/jankurai-score.svg [--score-json target/jankurai/repo-score.json]",
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
const scoreJson =
	options.scorejson ||
	resolve(process.cwd(), "target/jankurai/repo-score.json");
const scoreReport = readJson(scoreJson, "score report");
try {
	readFileSync(options.badge, "utf8");
} catch (err) {
	fatal(1, `cannot read badge ${options.badge}: ${err.message}`);
}

const score = asInteger(Number(scoreReport?.score), "numeric score");
const minimumScore = asInteger(
	Number(scoreReport?.decision?.minimum_score ?? scoreReport?.minimum_score),
	"numeric minimum_score",
);
const decisionStatus = asString(
	scoreReport?.decision?.status,
	"decision.status",
);
const decisionPassed = asBoolean(
	scoreReport?.decision?.passed,
	"decision.passed",
);
const observedLevel = asString(
	scoreReport?.observed_conformance_level,
	"observed_conformance_level",
);
const auditorVersion = asString(
	scoreReport?.auditor_version,
	"auditor_version",
);
if (score < 0 || score > 100) fatal(1, `score out of range: ${score}`);

const scoreHref = "./docs/testing.md#studio-sync-lane-studio-sync";
const badgeColor = scoreBadgeColor(score, minimumScore, decisionPassed);
const jankuraiBadge = [
	`<a href="${scoreHref}">`,
	`  <img alt="jankurai score ${score}" src="https://img.shields.io/badge/jankurai-${score}-${badgeColor}" />`,
	"</a>",
].join("\n");
const jankuraiScoreBlock = [
	"<!-- jankurai-score:start -->",
	`<p><strong>Jankurai score:</strong> <a href="${scoreHref}"><code>${score}/100</code></a> (${decisionStatus}, minimum ${minimumScore}, ${observedLevel}, auditor ${auditorVersion})</p>`,
	"<!-- jankurai-score:end -->",
].join("\n");

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
	jankuraiBadge,
	"",
	jankuraiScoreBlock,
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
