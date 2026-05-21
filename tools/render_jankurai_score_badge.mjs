#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

function fatal(code, message) {
	console.error(`render_jankurai_score_badge: ${message}`);
	process.exit(code);
}

function escapeXml(value) {
	return String(value)
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");
}

const [inputArg, outputArg] = process.argv.slice(2);
if (!inputArg || !outputArg) {
	fatal(
		3,
		"usage: node tools/render_jankurai_score_badge.mjs <score.json> <out.svg>",
	);
}

const inputPath = resolve(process.cwd(), inputArg);
const outputPath = resolve(process.cwd(), outputArg);

let report;
try {
	report = JSON.parse(readFileSync(inputPath, "utf8"));
} catch (err) {
	fatal(1, `cannot read score report ${inputPath}: ${err.message}`);
}

const score = Number(report?.score);
const minimum = Number(
	report?.decision?.minimum_score ?? report?.minimum_score ?? 85,
);
const hardFindings = Number(
	report?.decision?.hard_findings ?? report?.hard_findings ?? 0,
);

if (!Number.isFinite(score))
	fatal(1, `score report ${inputPath} is missing a numeric score`);
if (!Number.isFinite(minimum))
	fatal(1, `score report ${inputPath} is missing a numeric minimum_score`);
if (!Number.isFinite(hardFindings))
	fatal(
		1,
		`score report ${inputPath} is missing a numeric hard_findings count`,
	);

const status =
	hardFindings > 0
		? score >= minimum
			? "advisory"
			: "blocked"
		: score >= minimum
			? "pass"
			: "blocked";
const fill = {
	pass: "#0f766e",
	advisory: "#b45309",
	blocked: "#b91c1c",
}[status];

const labelWidth = 112;
const valueText = `score ${score}/${minimum}`;
const valueWidth = Math.max(94, Math.round(valueText.length * 7.6) + 20);
const width = labelWidth + valueWidth;

const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="28" viewBox="0 0 ${width} 28" role="img" aria-labelledby="title desc">
  <title id="title">${escapeXml(`Jankurai score ${score} of ${minimum}`)}</title>
  <desc id="desc">${escapeXml(`${hardFindings} hard finding${hardFindings === 1 ? "" : "s"}; status ${status}`)}</desc>
  <defs>
    <clipPath id="clip">
      <rect x="0" y="0" width="${width}" height="28" rx="14" ry="14" />
    </clipPath>
  </defs>
  <g clip-path="url(#clip)">
    <rect x="0" y="0" width="${width}" height="28" fill="#0f172a" />
    <rect x="${labelWidth}" y="0" width="${valueWidth}" height="28" fill="${fill}" />
  </g>
  <g fill="#f8fafc" font-family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" font-size="11" font-weight="700" text-anchor="middle">
    <text x="56" y="18">Jankurai</text>
    <text x="${labelWidth + Math.round(valueWidth / 2)}" y="18">${escapeXml(valueText)}</text>
  </g>
</svg>
`;

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, svg);
