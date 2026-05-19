#!/usr/bin/env node
// Validation-dossier HTML renderer for EchoForge.
//
// Reads a `validation_report.json` document (per
// `schemas/validation_report.schema.json`) and emits a single static HTML5
// page summarising the report: subject identity, overall status, validation
// tier (and optional fidelity_class), uncertainty score, provenance and
// license envelopes, and the per-check table.
//
// The output is intentionally a single self-contained file: inline CSS in a
// `<style>` block, no JavaScript, no external assets. Drop it on a file
// share, open it in any browser, and the page renders.
//
// The renderer is faithful — it shows exactly what the JSON says. It does
// not synthesize claims, smooth status values, or hide checks. If a field
// is missing/empty, the output reflects that (em-dash placeholders, empty
// counts, etc.) instead of inventing content.
//
// Usage:
//   node tools/render_validation_dossier.mjs <input.json> [--out <output.html>]
//
// Default output path: `<input>.html` next to the input file.
//
// Exit codes:
//   0 = HTML written successfully
//   1 = input failed required-field validation, or IO failure on read/write
//   3 = usage error (bad/missing args)
//
// Determinism: For a fixed input JSON the rendered HTML is byte-identical
// across reruns (no timestamps, no random ids, no Map iteration on
// unspecified key order — every loop iterates either an input array in its
// declared order or an explicit fixed key list). This is what makes the
// dossier safe to include in the repro-bytes lane.

import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const FOOTER_TEXT =
  "public-proxy validation dossier — strict-open EchoForge";

// Required top-level fields per the prompt + schema. The schema also
// requires `kind: "validation_report"`, which we check as a literal.
const REQUIRED_TOP_LEVEL = [
  "id",
  "kind",
  "subject_kind",
  "subject_id",
  "overall_status",
  "checks",
  "provenance",
  "license",
];

// The prompt also calls out `tier`. Per the schema, `tier` lives inside the
// `validation` envelope (`validation.tier`), so we accept either shape:
// `validation.tier` (canonical), or a top-level `tier` (older fixtures).
function extractTier(report) {
  if (report && typeof report === "object") {
    if (report.validation && typeof report.validation === "object" &&
        typeof report.validation.tier === "string") {
      return report.validation.tier;
    }
    if (typeof report.tier === "string") return report.tier;
  }
  return null;
}

function extractFidelity(report) {
  if (report && typeof report === "object") {
    if (report.validation && typeof report.validation === "object" &&
        typeof report.validation.fidelity_class === "string") {
      return report.validation.fidelity_class;
    }
    if (typeof report.fidelity_class === "string") return report.fidelity_class;
  }
  return null;
}

function extractUncertainty(report) {
  if (report && typeof report === "object" &&
      report.validation && typeof report.validation === "object" &&
      typeof report.validation.uncertainty_score === "number") {
    return report.validation.uncertainty_score;
  }
  return null;
}

function fatal(code, msg) {
  process.stderr.write(`render_validation_dossier: ${msg}\n`);
  process.exit(code);
}

function parseArgs(argv) {
  const positional = [];
  let out = null;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--out") {
      const next = argv[i + 1];
      if (!next || next.startsWith("--")) {
        fatal(3, "--out requires a path argument");
      }
      out = next;
      i++;
      continue;
    }
    if (arg.startsWith("--")) {
      fatal(3, `unknown flag ${arg}`);
    }
    positional.push(arg);
  }
  if (positional.length !== 1) {
    fatal(3, "expected exactly one positional argument <input.json>");
  }
  return { input: positional[0], out };
}

function loadReport(path) {
  let raw;
  try {
    raw = readFileSync(path, "utf8");
  } catch (e) {
    fatal(1, `cannot read input ${path}: ${e.message}`);
  }
  let report;
  try {
    report = JSON.parse(raw);
  } catch (e) {
    fatal(1, `input ${path} is not valid JSON: ${e.message}`);
  }
  if (!report || typeof report !== "object" || Array.isArray(report)) {
    fatal(1, `input ${path} must be a JSON object`);
  }
  return report;
}

function validateReport(report, path) {
  const missing = [];
  for (const key of REQUIRED_TOP_LEVEL) {
    if (!(key in report)) missing.push(key);
  }
  if (report.kind !== undefined && report.kind !== "validation_report") {
    fatal(1, `input ${path}: kind must be "validation_report" (got ${JSON.stringify(report.kind)})`);
  }
  if (extractTier(report) === null) missing.push("tier (validation.tier or top-level tier)");
  if (missing.length > 0) {
    fatal(1, `input ${path} is missing required field(s): ${missing.join(", ")}`);
  }
  if (!Array.isArray(report.checks) || report.checks.length === 0) {
    fatal(1, `input ${path}: \`checks\` must be a non-empty array`);
  }
  const valid = new Set(["pass", "warn", "fail"]);
  if (!valid.has(report.overall_status)) {
    fatal(1, `input ${path}: overall_status must be one of pass|warn|fail (got ${JSON.stringify(report.overall_status)})`);
  }
}

// HTML-escape; sufficient for text node and attribute contexts since we
// always quote attributes with double-quotes.
function esc(value) {
  if (value === null || value === undefined) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderBadge(kind, text) {
  return `<span class="badge badge-${esc(kind)}">${esc(text)}</span>`;
}

function statusBadge(status) {
  const t = typeof status === "string" ? status : "";
  const cls = t === "pass" || t === "warn" || t === "fail" ? t : "unknown";
  const label = t || "unknown";
  return `<span class="status status-${cls}">${esc(label)}</span>`;
}

// Color-coded uncertainty bar. Higher uncertainty = more saturated. We do
// not interpret what the value means — we just render whatever the JSON
// says, clamped into [0, 1] for the bar width.
function renderUncertaintyGauge(score) {
  if (score === null) {
    return `<div class="gauge-empty">no uncertainty_score in validation envelope</div>`;
  }
  const clamped = Math.max(0, Math.min(1, score));
  const pct = (clamped * 100).toFixed(1);
  let band = "low";
  if (clamped >= 0.66) band = "high";
  else if (clamped >= 0.33) band = "mid";
  return [
    `<div class="gauge" aria-label="uncertainty score">`,
    `  <div class="gauge-label">uncertainty_score = <code>${esc(score)}</code></div>`,
    `  <div class="gauge-bar"><div class="gauge-fill gauge-${band}" style="width: ${pct}%"></div></div>`,
    `  <div class="gauge-scale"><span>0.0 (low)</span><span>1.0 (high)</span></div>`,
    `</div>`,
  ].join("\n");
}

function renderChecksTable(checks) {
  const rows = checks.map((c) => {
    const metricCell = c && c.metric !== undefined
      ? `<code>${esc(JSON.stringify(c.metric))}</code>`
      : "&mdash;";
    return [
      `<tr>`,
      `  <td><code>${esc(c && c.name)}</code></td>`,
      `  <td>${statusBadge(c && c.status)}</td>`,
      `  <td>${esc(c && c.message ? c.message : "")}</td>`,
      `  <td>${metricCell}</td>`,
      `</tr>`,
    ].join("\n");
  });
  return [
    `<table class="checks">`,
    `  <thead>`,
    `    <tr><th>name</th><th>status</th><th>message</th><th>metric</th></tr>`,
    `  </thead>`,
    `  <tbody>`,
    rows.join("\n"),
    `  </tbody>`,
    `</table>`,
  ].join("\n");
}

function renderSourceRefs(refs) {
  if (!Array.isArray(refs) || refs.length === 0) {
    return `<em>(no source_refs)</em>`;
  }
  const items = refs.map((r) => `<li><code>${esc(r)}</code></li>`).join("\n");
  return `<ul class="refs">\n${items}\n</ul>`;
}

function renderProvenance(prov) {
  if (!prov || typeof prov !== "object") {
    return `<p><em>(provenance envelope absent)</em></p>`;
  }
  return [
    `<dl class="provenance">`,
    `  <dt>source_kind</dt><dd>${esc(prov.source_kind || "")}</dd>`,
    `  <dt>source_refs</dt><dd>${renderSourceRefs(prov.source_refs)}</dd>`,
    `  <dt>generated_by</dt><dd>${esc(prov.generated_by || "")}</dd>`,
    `  <dt>generated_at</dt><dd><time>${esc(prov.generated_at || "")}</time></dd>`,
    `  <dt>fingerprint_sha256</dt><dd><code class="hash">${esc(prov.fingerprint_sha256 || "")}</code></dd>`,
    `</dl>`,
  ].join("\n");
}

function renderLicense(lic) {
  if (!lic || typeof lic !== "object") {
    return `<p><em>(license envelope absent)</em></p>`;
  }
  const noticeBlock = lic.notice
    ? `  <dt>notice</dt><dd><pre class="notice">${esc(lic.notice)}</pre></dd>\n`
    : "";
  return [
    `<dl class="license">`,
    `  <dt>spdx_id</dt><dd><code>${esc(lic.spdx_id || "")}</code></dd>`,
    noticeBlock,
    `</dl>`,
  ].join("");
}

// CSS kept inline so the dossier opens as a single file. No web fonts; we
// rely on the system stack so the page renders identically offline.
const STYLE = `
  :root {
    color-scheme: light dark;
    --bg: #ffffff;
    --fg: #1a1a1a;
    --muted: #555;
    --border: #d0d0d0;
    --card: #fafafa;
    --pass: #2e7d32;
    --warn: #b58100;
    --fail: #b3261e;
    --pass-bg: #e6f4ea;
    --warn-bg: #fff5d1;
    --fail-bg: #fbe9e7;
    --unknown-bg: #ececec;
    --unknown-fg: #444;
    --tier-bg: #1f4e8c;
    --fidelity-bg: #54327a;
    --gauge-low: #2e7d32;
    --gauge-mid: #b58100;
    --gauge-high: #b3261e;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #15171a;
      --fg: #e8e8e8;
      --muted: #b8b8b8;
      --border: #383b40;
      --card: #1d2024;
      --pass-bg: #1c3a23;
      --warn-bg: #3a3015;
      --fail-bg: #3a1d1a;
      --unknown-bg: #2a2c30;
      --unknown-fg: #ddd;
    }
  }
  html, body { background: var(--bg); color: var(--fg); }
  body {
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI",
      Roboto, sans-serif;
    line-height: 1.5;
    margin: 0;
    padding: 2rem 1.5rem 4rem;
    max-width: 64rem;
    margin-left: auto;
    margin-right: auto;
  }
  h1, h2 { font-weight: 600; letter-spacing: -0.01em; }
  h1 { font-size: 1.6rem; margin: 0 0 0.5rem; }
  h2 { font-size: 1.15rem; margin: 2rem 0 0.75rem; border-bottom: 1px solid var(--border); padding-bottom: 0.25rem; }
  code, pre { font-family: ui-monospace, SFMono-Regular, "JetBrains Mono", Menlo, Consolas, monospace; }
  code { background: var(--card); padding: 0.05rem 0.3rem; border-radius: 3px; font-size: 0.92em; }
  pre.notice { background: var(--card); border: 1px solid var(--border); border-radius: 4px; padding: 0.75rem; overflow-x: auto; white-space: pre-wrap; }
  .muted { color: var(--muted); }
  header.dossier-head { border-bottom: 2px solid var(--border); padding-bottom: 1rem; margin-bottom: 1rem; }
  header.dossier-head .subject-id { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.95rem; color: var(--muted); word-break: break-all; }
  .badges { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.75rem 0; }
  .status, .badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.85rem; font-weight: 600; line-height: 1.4; }
  .status-pass { background: var(--pass-bg); color: var(--pass); }
  .status-warn { background: var(--warn-bg); color: var(--warn); }
  .status-fail { background: var(--fail-bg); color: var(--fail); }
  .status-unknown { background: var(--unknown-bg); color: var(--unknown-fg); }
  .badge-tier { background: var(--tier-bg); color: #fff; }
  .badge-fidelity { background: var(--fidelity-bg); color: #fff; }
  .gauge { margin: 0.75rem 0; }
  .gauge-label { font-size: 0.95rem; margin-bottom: 0.35rem; }
  .gauge-bar { background: var(--card); border: 1px solid var(--border); border-radius: 4px; height: 0.75rem; overflow: hidden; }
  .gauge-fill { height: 100%; transition: none; }
  .gauge-low { background: var(--gauge-low); }
  .gauge-mid { background: var(--gauge-mid); }
  .gauge-high { background: var(--gauge-high); }
  .gauge-scale { display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem; }
  .gauge-empty { color: var(--muted); font-style: italic; }
  dl.provenance, dl.license { display: grid; grid-template-columns: max-content 1fr; gap: 0.4rem 1rem; margin: 0; }
  dl.provenance dt, dl.license dt { color: var(--muted); font-size: 0.9rem; }
  dl.provenance dd, dl.license dd { margin: 0; word-break: break-word; }
  ul.refs { margin: 0; padding-left: 1.2rem; }
  code.hash { word-break: break-all; }
  table.checks { width: 100%; border-collapse: collapse; margin-top: 0.5rem; }
  table.checks th, table.checks td { text-align: left; padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--border); vertical-align: top; }
  table.checks th { font-size: 0.85rem; color: var(--muted); font-weight: 600; }
  footer.dossier-foot { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.85rem; color: var(--muted); text-align: center; }
`;

function renderHtml(report) {
  const tier = extractTier(report);
  const fidelity = extractFidelity(report);
  const uncertainty = extractUncertainty(report);
  const prov = report.provenance || {};
  const generatedAt = prov && typeof prov === "object" ? prov.generated_at : "";
  const fingerprint = prov && typeof prov === "object" ? prov.fingerprint_sha256 : "";
  const generatedBy = prov && typeof prov === "object" ? prov.generated_by : "";

  const badges = [];
  badges.push(renderBadge("tier", `tier: ${tier}`));
  if (fidelity) {
    badges.push(renderBadge("fidelity", `fidelity: ${fidelity}`));
  }

  const title = `Validation Dossier — ${esc(report.subject_id || report.id || "")}`;

  const out = [];
  out.push(`<!doctype html>`);
  out.push(`<html lang="en">`);
  out.push(`<head>`);
  out.push(`<meta charset="utf-8">`);
  out.push(`<meta name="viewport" content="width=device-width, initial-scale=1">`);
  out.push(`<meta name="generator" content="echoforge.render_validation_dossier">`);
  out.push(`<title>${title}</title>`);
  out.push(`<style>${STYLE}</style>`);
  out.push(`</head>`);
  out.push(`<body>`);
  out.push(`<header class="dossier-head">`);
  out.push(`  <h1>Validation Dossier</h1>`);
  out.push(`  <div><strong>subject_kind:</strong> <code>${esc(report.subject_kind || "")}</code></div>`);
  out.push(`  <div class="subject-id"><strong>subject_id:</strong> ${esc(report.subject_id || "")}</div>`);
  out.push(`  <div><strong>report_id:</strong> <code>${esc(report.id || "")}</code></div>`);
  if (report.report_name) {
    out.push(`  <div><strong>report_name:</strong> ${esc(report.report_name)}</div>`);
  }
  if (report.public_proxy_id) {
    out.push(`  <div><strong>public_proxy_id:</strong> <code>${esc(report.public_proxy_id)}</code></div>`);
  }
  out.push(`  <div><strong>generated_at:</strong> <time>${esc(generatedAt || "")}</time> <span class="muted">by <code>${esc(generatedBy || "")}</code></span></div>`);
  out.push(`  <div><strong>fingerprint_sha256:</strong> <code class="hash">${esc(fingerprint || "")}</code></div>`);
  out.push(`  <div class="badges">`);
  out.push(`    <span class="muted">overall_status:</span> ${statusBadge(report.overall_status)}`);
  out.push(`    ${badges.join("\n    ")}`);
  out.push(`  </div>`);
  out.push(`</header>`);

  out.push(`<section>`);
  out.push(`<h2>Uncertainty</h2>`);
  out.push(renderUncertaintyGauge(uncertainty));
  out.push(`</section>`);

  out.push(`<section>`);
  out.push(`<h2>Provenance</h2>`);
  out.push(renderProvenance(report.provenance));
  out.push(`</section>`);

  out.push(`<section>`);
  out.push(`<h2>License</h2>`);
  out.push(renderLicense(report.license));
  out.push(`</section>`);

  out.push(`<section>`);
  out.push(`<h2>Checks (${report.checks.length})</h2>`);
  out.push(renderChecksTable(report.checks));
  out.push(`</section>`);

  out.push(`<footer class="dossier-foot">${esc(FOOTER_TEXT)}</footer>`);
  out.push(`</body>`);
  out.push(`</html>`);
  out.push(``); // trailing newline

  return out.join("\n");
}

function defaultOutPath(input) {
  return `${input}.html`;
}

function main() {
  const { input, out } = parseArgs(process.argv.slice(2));
  const report = loadReport(input);
  validateReport(report, input);
  const html = renderHtml(report);
  const outPath = out || defaultOutPath(input);
  try {
    writeFileSync(resolve(outPath), html);
  } catch (e) {
    fatal(1, `cannot write output ${outPath}: ${e.message}`);
  }
  process.stdout.write(`${outPath}\n`);
  process.exit(0);
}

main();
