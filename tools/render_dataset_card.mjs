#!/usr/bin/env node
// Dataset-card HTML renderer for EchoForge.
//
// Converts a `dataset_card.json` (matching `schemas/dataset_card.schema.json`)
// into a single static HTML page summarizing the dataset's class balance,
// leakage status, validation tier, and source campaigns.
//
// Usage:
//   node tools/render_dataset_card.mjs <dataset_card.json>
//       [--out <output.html>]
//       [--class-balance <class_balance.json>]
//       [--leakage <leakage_report.json>]
//       [--benchmark <benchmark_report.json>]
//
// Default <output.html>: <input>.html (next to the input).
//
// Sections rendered:
//   - Header: dataset_name + public_proxy_id + validation tier badge
//   - Splits: train/validation/test counts with a proportion bar
//   - Class balance (optional): per-class counts as inline horizontal bars
//   - Source campaigns: linked list of campaign ids
//   - Leakage (optional): pass/warn/fail badge + summary
//   - Benchmark (optional): top-line metrics (TPR, FPR, mean time-to-detect)
//   - Provenance + license footer
//
// Output is deterministic for the same input set (no timestamps, no rand,
// stable object iteration). Inline CSS only; no JS, no external assets.
//
// Exit codes:
//   0 = HTML rendered successfully
//   1 = validation or IO error (bad JSON, missing required field, write fail)
//   3 = usage error (bad flags, missing positional)

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, basename, resolve, join, relative } from "node:path";

function fatal(code, msg) {
  process.stderr.write(`render_dataset_card: ${msg}\n`);
  process.exit(code);
}

function parseArgs(argv) {
  const out = {
    input: null,
    output: null,
    classBalance: null,
    leakage: null,
    benchmark: null,
  };
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--out") {
      out.output = argv[++i];
      if (!out.output) fatal(3, "--out requires a path argument");
    } else if (arg === "--class-balance") {
      out.classBalance = argv[++i];
      if (!out.classBalance) fatal(3, "--class-balance requires a path argument");
    } else if (arg === "--leakage") {
      out.leakage = argv[++i];
      if (!out.leakage) fatal(3, "--leakage requires a path argument");
    } else if (arg === "--benchmark") {
      out.benchmark = argv[++i];
      if (!out.benchmark) fatal(3, "--benchmark requires a path argument");
    } else if (arg === "--help" || arg === "-h") {
      process.stdout.write(
        "Usage: node tools/render_dataset_card.mjs <dataset_card.json> " +
          "[--out <output.html>] [--class-balance <p>] [--leakage <p>] [--benchmark <p>]\n",
      );
      process.exit(0);
    } else if (arg.startsWith("--")) {
      fatal(3, `unknown flag ${arg}`);
    } else {
      positional.push(arg);
    }
  }
  if (positional.length !== 1) {
    fatal(3, `expected exactly one positional <dataset_card.json>; got ${positional.length}`);
  }
  out.input = positional[0];
  return out;
}

function readJson(path, label) {
  let raw;
  try {
    raw = readFileSync(path, "utf8");
  } catch (e) {
    fatal(1, `cannot read ${label} (${path}): ${e.message}`);
  }
  try {
    return JSON.parse(raw);
  } catch (e) {
    fatal(1, `invalid JSON in ${label} (${path}): ${e.message}`);
  }
}

function validateDatasetCard(card, path) {
  const required = [
    "id",
    "kind",
    "public_proxy_id",
    "dataset_name",
    "source_campaign_ids",
    "provenance",
    "license",
    "validation",
    "splits",
  ];
  for (const field of required) {
    if (card[field] === undefined || card[field] === null) {
      fatal(1, `dataset_card missing required field "${field}" (${path})`);
    }
  }
  if (card.kind !== "dataset_card") {
    fatal(1, `dataset_card has wrong kind "${card.kind}" (expected "dataset_card") (${path})`);
  }
  for (const splitKey of ["train", "validation", "test"]) {
    if (!card.splits || typeof card.splits[splitKey] !== "number") {
      fatal(1, `dataset_card splits.${splitKey} missing or not a number (${path})`);
    }
  }
  if (!Array.isArray(card.source_campaign_ids) || card.source_campaign_ids.length === 0) {
    fatal(1, `dataset_card source_campaign_ids must be a non-empty array (${path})`);
  }
  if (!card.provenance || typeof card.provenance !== "object") {
    fatal(1, `dataset_card provenance must be an object (${path})`);
  }
  if (!card.license || typeof card.license !== "object" || !card.license.spdx_id) {
    fatal(1, `dataset_card license.spdx_id missing (${path})`);
  }
  if (!card.validation || typeof card.validation !== "object" || !card.validation.tier) {
    fatal(1, `dataset_card validation.tier missing (${path})`);
  }
}

// Minimal HTML escaping. Stable iteration of input characters; deterministic.
function esc(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function statusBadgeClass(status) {
  const s = String(status || "").toLowerCase();
  if (s === "pass") return "badge badge-pass";
  if (s === "warn") return "badge badge-warn";
  if (s === "fail") return "badge badge-fail";
  return "badge badge-neutral";
}

function tierBadgeClass(tier) {
  const t = String(tier || "").toLowerCase();
  if (t === "measured" || t === "benchmarked") return "badge badge-pass";
  if (t === "cross_checked") return "badge badge-info";
  if (t === "basic") return "badge badge-warn";
  return "badge badge-neutral";
}

// Sort object entries by key for deterministic rendering. Returns array of
// [key, value] pairs.
function sortedEntries(obj) {
  if (!obj || typeof obj !== "object") return [];
  return Object.entries(obj).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
}

// Pre-baked palette. Cycled deterministically by index, so the same input
// always gets the same colors.
const BAR_PALETTE = [
  "#1f78b4",
  "#33a02c",
  "#e31a1c",
  "#ff7f00",
  "#6a3d9a",
  "#b15928",
  "#a6cee3",
  "#b2df8a",
  "#fb9a99",
  "#fdbf6f",
  "#cab2d6",
  "#ffff99",
];

function renderSplitsBar(splits) {
  const total = (splits.train || 0) + (splits.validation || 0) + (splits.test || 0);
  if (total === 0) {
    return `<p class="empty">No samples recorded across splits.</p>`;
  }
  const pcts = {
    train: ((splits.train || 0) / total) * 100,
    validation: ((splits.validation || 0) / total) * 100,
    test: ((splits.test || 0) / total) * 100,
  };
  // Round to 2 decimal places deterministically.
  function pct(p) {
    return (Math.round(p * 100) / 100).toFixed(2);
  }
  return `
      <div class="splits-bar" role="img" aria-label="Dataset split proportions">
        <div class="split-seg split-train" style="width: ${pct(pcts.train)}%" title="train: ${splits.train}">${splits.train > 0 ? `train ${splits.train}` : ""}</div>
        <div class="split-seg split-validation" style="width: ${pct(pcts.validation)}%" title="validation: ${splits.validation}">${splits.validation > 0 ? `val ${splits.validation}` : ""}</div>
        <div class="split-seg split-test" style="width: ${pct(pcts.test)}%" title="test: ${splits.test}">${splits.test > 0 ? `test ${splits.test}` : ""}</div>
      </div>
      <table class="splits-table">
        <thead><tr><th>Split</th><th>Count</th><th>Share</th></tr></thead>
        <tbody>
          <tr><td>train</td><td>${splits.train}</td><td>${pct(pcts.train)}%</td></tr>
          <tr><td>validation</td><td>${splits.validation}</td><td>${pct(pcts.validation)}%</td></tr>
          <tr><td>test</td><td>${splits.test}</td><td>${pct(pcts.test)}%</td></tr>
          <tr class="splits-total"><td>total</td><td>${total}</td><td>100.00%</td></tr>
        </tbody>
      </table>`;
}

function renderClassBalanceGroup(title, counts) {
  const entries = sortedEntries(counts).filter(([, v]) => typeof v === "number");
  if (entries.length === 0) {
    return `<div class="cb-group"><h3>${esc(title)}</h3><p class="empty">No entries.</p></div>`;
  }
  const max = entries.reduce((m, [, v]) => (v > m ? v : m), 0);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  const rows = entries.map(([k, v], i) => {
    const widthPct = max > 0 ? (v / max) * 100 : 0;
    const sharePct = total > 0 ? (v / total) * 100 : 0;
    const color = BAR_PALETTE[i % BAR_PALETTE.length];
    return `
        <div class="cb-row">
          <div class="cb-label" title="${esc(k)}">${esc(k)}</div>
          <div class="cb-track"><div class="cb-fill" style="width: ${(Math.round(widthPct * 100) / 100).toFixed(2)}%; background-color: ${color};"></div></div>
          <div class="cb-count">${v}</div>
          <div class="cb-share">${(Math.round(sharePct * 100) / 100).toFixed(2)}%</div>
        </div>`;
  }).join("");
  return `
      <div class="cb-group">
        <h3>${esc(title)} <span class="cb-total">(total: ${total})</span></h3>
        ${rows}
      </div>`;
}

function renderClassBalance(cb) {
  if (!cb) return "";
  const sections = [];
  // Headline counts if present.
  const headline = [];
  if (typeof cb.total_records === "number") {
    headline.push(`<dt>total records</dt><dd>${cb.total_records}</dd>`);
  }
  if (typeof cb.shahed_positive_records === "number") {
    headline.push(`<dt>positive (target) records</dt><dd>${cb.shahed_positive_records}</dd>`);
  }
  if (typeof cb.shahed_min_required === "number") {
    headline.push(`<dt>positive-class minimum required</dt><dd>${cb.shahed_min_required}</dd>`);
  }
  if (typeof cb.meets_shahed_min === "boolean") {
    headline.push(
      `<dt>meets minimum?</dt><dd><span class="${cb.meets_shahed_min ? "badge badge-pass" : "badge badge-fail"}">${cb.meets_shahed_min ? "yes" : "no"}</span></dd>`,
    );
  }
  const headlineHtml = headline.length
    ? `<dl class="cb-headline">${headline.join("")}</dl>`
    : "";

  // Per-bucket / per-family groups, only the ones present.
  if (cb.bucket_counts && typeof cb.bucket_counts === "object") {
    sections.push(renderClassBalanceGroup("Bucket counts", cb.bucket_counts));
  }
  if (cb.target_family_counts && typeof cb.target_family_counts === "object") {
    sections.push(renderClassBalanceGroup("Target family counts", cb.target_family_counts));
  }
  if (cb.hard_negative_family_counts && typeof cb.hard_negative_family_counts === "object") {
    sections.push(
      renderClassBalanceGroup("Hard-negative family counts", cb.hard_negative_family_counts),
    );
  }

  if (!headlineHtml && sections.length === 0) {
    return "";
  }
  return `
    <section class="card">
      <h2>Class balance</h2>
      ${headlineHtml}
      ${sections.join("")}
    </section>`;
}

function renderSourceCampaigns(ids) {
  const items = ids.map((id) => `<li><code>${esc(id)}</code></li>`).join("");
  return `
    <section class="card">
      <h2>Source campaigns</h2>
      <ul class="campaign-list">${items}</ul>
    </section>`;
}

function renderLeakage(leakage) {
  if (!leakage) return "";
  const findings = Array.isArray(leakage.findings) ? leakage.findings : [];
  const status = findings.length === 0 ? "pass" : "warn";
  const checkedKeys = Array.isArray(leakage.checked_keys) ? leakage.checked_keys : [];
  const findingRows = findings.length
    ? findings.map((f, i) => {
        const keys = sortedEntries(f).map(([k, v]) => {
          let val;
          if (typeof v === "object" && v !== null) {
            val = JSON.stringify(v);
          } else {
            val = String(v);
          }
          return `<dt>${esc(k)}</dt><dd>${esc(val)}</dd>`;
        }).join("");
        return `<li><strong>Finding ${i + 1}</strong><dl>${keys}</dl></li>`;
      }).join("")
    : "";
  return `
    <section class="card">
      <h2>Leakage report <span class="${statusBadgeClass(status)}">${status}</span></h2>
      <dl class="kv">
        <dt>policy id</dt><dd><code>${esc(leakage.policy_id || "")}</code></dd>
        <dt>schema</dt><dd><code>${esc(leakage.schema_ref || "")}</code></dd>
        <dt>checked keys</dt><dd>${checkedKeys.length ? checkedKeys.map((k) => `<code>${esc(k)}</code>`).join(", ") : "<em>none</em>"}</dd>
        <dt>findings</dt><dd>${findings.length}</dd>
      </dl>
      ${findings.length ? `<ol class="leakage-findings">${findingRows}</ol>` : `<p class="empty">No leakage findings reported.</p>`}
    </section>`;
}

function renderBenchmark(bench) {
  if (!bench) return "";
  const limits = Array.isArray(bench.limitations) ? bench.limitations : [];
  const models = Array.isArray(bench.model_reports) ? bench.model_reports : [];
  const modelRows = models.map((m) => {
    const pd = typeof m.pd === "number" ? (Math.round(m.pd * 10000) / 10000).toFixed(4) : "n/a";
    const pfa = typeof m.pfa === "number" ? (Math.round(m.pfa * 10000) / 10000).toFixed(4) : "n/a";
    const mttd =
      typeof m.mean_first_detection_latency_frames === "number"
        ? (Math.round(m.mean_first_detection_latency_frames * 100) / 100).toFixed(2)
        : "n/a";
    return `<tr>
        <td><code>${esc(m.model_id || "")}</code></td>
        <td>${pd}</td>
        <td>${pfa}</td>
        <td>${mttd}</td>
        <td>${m.positive_records ?? "n/a"}</td>
        <td>${m.negative_records ?? "n/a"}</td>
      </tr>`;
  }).join("");
  const status = bench.status ? `<span class="${statusBadgeClass(bench.status)}">${esc(bench.status)}</span>` : "";
  return `
    <section class="card">
      <h2>Benchmark report ${status}</h2>
      <dl class="kv">
        <dt>benchmark id</dt><dd><code>${esc(bench.benchmark_id || "")}</code></dd>
        <dt>dataset id</dt><dd><code>${esc(bench.dataset_id || "")}</code></dd>
        ${bench.neutral_campaign_id ? `<dt>neutral campaign id</dt><dd><code>${esc(bench.neutral_campaign_id)}</code></dd>` : ""}
        ${bench.campaign_request_id ? `<dt>campaign request id</dt><dd><code>${esc(bench.campaign_request_id)}</code></dd>` : ""}
      </dl>
      ${models.length ? `
      <table class="bench-table">
        <thead><tr>
          <th>model</th><th>TPR (pd)</th><th>FPR (pfa)</th>
          <th>mean time-to-detect (frames)</th><th>positives</th><th>negatives</th>
        </tr></thead>
        <tbody>${modelRows}</tbody>
      </table>` : ""}
      ${limits.length ? `<h3>Known limitations</h3><ul>${limits.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>` : ""}
    </section>`;
}

function renderValidationChecks(validation) {
  const checks = Array.isArray(validation.checks) ? validation.checks : [];
  if (checks.length === 0) return "";
  const rows = checks.map((c) => `
        <tr>
          <td>${esc(c.name || "")}</td>
          <td><span class="${statusBadgeClass(c.status)}">${esc(c.status || "")}</span></td>
          <td>${esc(c.message || "")}</td>
        </tr>`).join("");
  return `
      <table class="checks-table">
        <thead><tr><th>check</th><th>status</th><th>message</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
}

function renderProvenanceLicenseFooter(card) {
  const p = card.provenance || {};
  const refs = Array.isArray(p.source_refs) ? p.source_refs : [];
  const refsHtml = refs.length
    ? `<ul class="ref-list">${refs.map((r) => `<li><code>${esc(r)}</code></li>`).join("")}</ul>`
    : "<em>none</em>";
  return `
    <footer class="card">
      <h2>Provenance &amp; license</h2>
      <dl class="kv">
        <dt>source kind</dt><dd><code>${esc(p.source_kind || "")}</code></dd>
        <dt>generated by</dt><dd><code>${esc(p.generated_by || "")}</code></dd>
        <dt>generated at</dt><dd><code>${esc(p.generated_at || "")}</code></dd>
        <dt>fingerprint (sha256)</dt><dd><code>${esc(p.fingerprint_sha256 || "")}</code></dd>
        <dt>source refs</dt><dd>${refsHtml}</dd>
        <dt>license (SPDX)</dt><dd><code>${esc(card.license.spdx_id)}</code></dd>
        ${card.license.notice ? `<dt>license notice</dt><dd>${esc(card.license.notice)}</dd>` : ""}
        <dt>schema version</dt><dd><code>${esc(card.schema_version || "")}</code></dd>
      </dl>
      <p class="disclaimer">
        This page renders the dataset_card JSON as-is. It does not assert anything
        beyond what the source document declares. Validation tier and check status
        are reproduced verbatim from the dataset_card validation envelope.
      </p>
    </footer>`;
}

// All styles are inline. Keep CSS deterministic (no rand, no timestamps).
const CSS = `
:root {
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #555555;
  --border: #d8d8d8;
  --card-bg: #fafafa;
  --pass: #2e7d32;
  --pass-bg: #e8f5e9;
  --warn: #ef6c00;
  --warn-bg: #fff3e0;
  --fail: #c62828;
  --fail-bg: #ffebee;
  --info: #1565c0;
  --info-bg: #e3f2fd;
  --neutral: #424242;
  --neutral-bg: #eeeeee;
  --train: #1f78b4;
  --val: #33a02c;
  --test: #ff7f00;
  --track-bg: #ececec;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 24px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--fg);
  background: var(--bg);
  line-height: 1.45;
}
main { max-width: 1080px; margin: 0 auto; }
header.dataset-header {
  border-bottom: 1px solid var(--border);
  padding-bottom: 12px;
  margin-bottom: 24px;
}
header.dataset-header h1 { margin: 0 0 6px 0; font-size: 1.7em; }
header.dataset-header .proxy { color: var(--muted); font-family: ui-monospace, monospace; }
header.dataset-header .tier-row { margin-top: 8px; }
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 16px 18px;
  margin: 16px 0;
}
.card h2 { margin: 0 0 12px 0; font-size: 1.2em; }
.card h3 { margin: 16px 0 8px 0; font-size: 1.0em; color: var(--muted); }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 0.85em;
  font-weight: 600;
  vertical-align: middle;
}
.badge-pass { color: var(--pass); background: var(--pass-bg); }
.badge-warn { color: var(--warn); background: var(--warn-bg); }
.badge-fail { color: var(--fail); background: var(--fail-bg); }
.badge-info { color: var(--info); background: var(--info-bg); }
.badge-neutral { color: var(--neutral); background: var(--neutral-bg); }
.splits-bar {
  display: flex;
  height: 28px;
  border-radius: 4px;
  overflow: hidden;
  border: 1px solid var(--border);
  margin-bottom: 12px;
  background: var(--track-bg);
}
.split-seg {
  color: white;
  font-size: 0.78em;
  font-weight: 600;
  padding: 4px 6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.split-train { background: var(--train); }
.split-validation { background: var(--val); }
.split-test { background: var(--test); }
table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.92em; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }
thead th { background: #f0f0f0; }
.splits-total td { font-weight: 600; background: #f6f6f6; }
.cb-group { margin: 12px 0 18px 0; }
.cb-total { font-weight: normal; color: var(--muted); font-size: 0.9em; }
.cb-row {
  display: grid;
  grid-template-columns: 240px 1fr 72px 72px;
  align-items: center;
  gap: 8px;
  padding: 3px 0;
  font-size: 0.88em;
}
.cb-label {
  font-family: ui-monospace, monospace;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.cb-track {
  background: var(--track-bg);
  border-radius: 3px;
  height: 14px;
  overflow: hidden;
}
.cb-fill { height: 100%; }
.cb-count { text-align: right; font-variant-numeric: tabular-nums; }
.cb-share { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
.kv {
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 4px 12px;
  margin: 0;
}
.kv dt { color: var(--muted); }
.kv dd { margin: 0; word-break: break-all; }
.cb-headline { grid-template-columns: 280px 1fr; margin-bottom: 12px; }
.campaign-list, .ref-list { padding-left: 18px; margin: 0; }
.campaign-list li, .ref-list li { margin: 4px 0; word-break: break-all; }
.empty { color: var(--muted); font-style: italic; margin: 4px 0; }
.checks-table th, .checks-table td { vertical-align: top; }
.bench-table { font-size: 0.9em; }
.bench-table th, .bench-table td { font-variant-numeric: tabular-nums; }
.leakage-findings { padding-left: 20px; }
.leakage-findings dl { display: grid; grid-template-columns: 160px 1fr; gap: 2px 8px; margin: 4px 0 0 0; }
.disclaimer {
  font-size: 0.85em;
  color: var(--muted);
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px dashed var(--border);
}
code { font-family: ui-monospace, monospace; font-size: 0.95em; }
`;

function render(card, opts) {
  const cb = opts.classBalance;
  const leakage = opts.leakage;
  const bench = opts.benchmark;
  const title = `${card.dataset_name} | EchoForge dataset card`;
  const tier = card.validation.tier;
  const overallStatus = card.validation.status;
  const uncertainty =
    typeof card.validation.uncertainty_score === "number"
      ? (Math.round(card.validation.uncertainty_score * 1000) / 1000).toFixed(3)
      : "n/a";
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<style>${CSS}</style>
</head>
<body>
<main>
  <header class="dataset-header">
    <h1>${esc(card.dataset_name)}</h1>
    <div class="proxy">public proxy: <code>${esc(card.public_proxy_id)}</code></div>
    <div class="proxy">id: <code>${esc(card.id)}</code></div>
    <div class="tier-row">
      <span class="${tierBadgeClass(tier)}" title="validation tier">tier: ${esc(tier)}</span>
      <span class="${statusBadgeClass(overallStatus)}" title="overall validation status">status: ${esc(overallStatus)}</span>
      <span class="badge badge-neutral" title="uncertainty score">uncertainty: ${uncertainty}</span>
      ${card.validation.fidelity_class ? `<span class="badge badge-info" title="method-ceiling fidelity">fidelity: ${esc(card.validation.fidelity_class)}</span>` : ""}
    </div>
  </header>

  <section class="card">
    <h2>Splits</h2>
    ${renderSplitsBar(card.splits)}
  </section>

  <section class="card">
    <h2>Validation checks</h2>
    ${renderValidationChecks(card.validation) || `<p class="empty">No validation checks recorded.</p>`}
  </section>

  ${renderClassBalance(cb)}

  ${renderSourceCampaigns(card.source_campaign_ids)}

  ${renderLeakage(leakage)}

  ${renderBenchmark(bench)}

  ${renderProvenanceLicenseFooter(card)}
</main>
</body>
</html>
`;
}

function main() {
  const opts = parseArgs(process.argv.slice(2));
  const card = readJson(opts.input, "dataset_card");
  validateDatasetCard(card, opts.input);

  const companionOpts = {
    classBalance: opts.classBalance ? readJson(opts.classBalance, "class_balance") : null,
    leakage: opts.leakage ? readJson(opts.leakage, "leakage_report") : null,
    benchmark: opts.benchmark ? readJson(opts.benchmark, "benchmark_report") : null,
  };

  const html = render(card, companionOpts);

  const outPath = opts.output
    ? opts.output
    : join(dirname(opts.input), basename(opts.input).replace(/\.json$/i, "") + ".html");

  try {
    writeFileSync(outPath, html);
  } catch (e) {
    fatal(1, `cannot write ${outPath}: ${e.message}`);
  }

  // Brief status to stdout. Path printed relative to cwd when shorter, else
  // absolute — purely for human readability; not part of the file.
  const cwd = process.cwd();
  const absOut = resolve(outPath);
  const rel = relative(cwd, absOut);
  const display = rel && !rel.startsWith("..") ? rel : absOut;
  process.stdout.write(`render_dataset_card: wrote ${display} (${html.length} bytes)\n`);
  process.exit(0);
}

main();
