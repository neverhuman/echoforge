#!/usr/bin/env node
// Tests for tools/render_dataset_card.mjs.
//
// Uses the real campaign output as the primary fixture, plus synthetic
// minimum/maximum/companion-coverage cases. No npm deps — pure Node std.
//
// Exit codes:
//   0 = all tests pass
//   1 = one or more tests fail
//   3 = harness / IO error

import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync, readFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";

// Resolve paths relative to this file's repo location so the harness works
// regardless of where the caller invokes it from.
const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(__dirname, "..");
const TOOL = resolve(REPO_ROOT, "tools/render_dataset_card.mjs");
const REAL_CARD = resolve(
  REPO_ROOT,
  "outputs/campaigns/shahed136-public-proxy-early-detection/dataset_card.json",
);
const REAL_CLASS_BALANCE = resolve(
  REPO_ROOT,
  "outputs/campaigns/shahed136-public-proxy-early-detection/class_balance.json",
);
const REAL_BENCHMARK = resolve(
  REPO_ROOT,
  "outputs/campaigns/shahed136-public-proxy-early-detection/benchmark_report.json",
);

let pass = 0;
let fail = 0;
const failures = [];

function check(name, cond, detail) {
  if (cond) {
    pass++;
    process.stdout.write(`  PASS ${name}\n`);
  } else {
    fail++;
    failures.push({ name, detail });
    process.stdout.write(`  FAIL ${name}${detail ? ` :: ${detail}` : ""}\n`);
  }
}

function runTool(args) {
  return spawnSync("node", [TOOL, ...args], { encoding: "utf8" });
}

function sha256(buf) {
  return createHash("sha256").update(buf).digest("hex");
}

function makeWorkdir() {
  return mkdtempSync(join(tmpdir(), "render_dataset_card_test_"));
}

function writeMinimalCard(dir, overrides = {}) {
  const card = {
    id: "ef:dataset_card:test-proxy:0123456789abcdef:1",
    kind: "dataset_card",
    schema_version: "1.0.0",
    public_proxy_id: "test-proxy",
    provenance: {
      source_kind: "synthetic",
      source_refs: ["test/fixture.yaml"],
      generated_by: "render_dataset_card test harness",
      generated_at: "2026-05-18T00:00:00Z",
      fingerprint_sha256: "0".repeat(64),
    },
    license: { spdx_id: "CC0-1.0" },
    validation: {
      tier: "basic",
      status: "pass",
      uncertainty_score: 0.5,
      checks: [{ name: "smoke", status: "pass", message: "fixture" }],
    },
    dataset_name: "Synthetic Minimal Fixture",
    source_campaign_ids: ["ef:rcs_campaign:test-proxy:fedcba9876543210:1"],
    splits: { train: 10, validation: 2, test: 2 },
    ...overrides,
  };
  const path = join(dir, "card.json");
  writeFileSync(path, JSON.stringify(card, null, 2));
  return { path, card };
}

function writeFile(dir, name, obj) {
  const path = join(dir, name);
  writeFileSync(path, JSON.stringify(obj, null, 2));
  return path;
}

// --- Test 1: real campaign card renders and contains key data ---
function test1_realCard() {
  process.stdout.write("test1: real campaign card renders with key data\n");
  const dir = makeWorkdir();
  try {
    const out = join(dir, "real.html");
    const res = runTool([REAL_CARD, "--class-balance", REAL_CLASS_BALANCE, "--out", out]);
    check("exit 0", res.status === 0, `status=${res.status} stderr=${res.stderr}`);
    check("output file exists", existsSync(out));
    if (existsSync(out)) {
      const html = readFileSync(out, "utf8");
      check(
        "contains dataset_name",
        html.includes("OWA Delta Pusher Public-Proxy Early-Detection Campaign"),
      );
      check("contains train count 700", html.includes(">700<"));
      check("contains validation count 150", html.includes(">150<"));
      check("contains test count 150", html.includes(">150<"));
      check("contains validation tier", html.includes("tier: basic"));
      check("contains source_campaign id", html.includes("owa-delta-pusher-fixed-wing"));
      check("contains class balance bucket label", html.includes("positive_owa_delta_pusher"));
      check("is HTML5 doctype", html.trimStart().startsWith("<!doctype html>"));
      check("no external script/link tags", !/<script\b|<link\b/i.test(html));
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// --- Test 2: minimal card without companions still renders ---
function test2_minimalCard() {
  process.stdout.write("test2: minimal card without companions still renders\n");
  const dir = makeWorkdir();
  try {
    const { path } = writeMinimalCard(dir);
    const out = join(dir, "minimal.html");
    const res = runTool([path, "--out", out]);
    check("exit 0", res.status === 0, `status=${res.status} stderr=${res.stderr}`);
    check("output file exists", existsSync(out));
    if (existsSync(out)) {
      const html = readFileSync(out, "utf8");
      check("contains dataset_name", html.includes("Synthetic Minimal Fixture"));
      check("contains train count 10", html.includes(">10<"));
      check("contains validation count 2", /<td>2<\/td>/.test(html));
      check("no class-balance section", !html.includes("Class balance"));
      check("no leakage section", !html.includes("Leakage report"));
      check("no benchmark section", !html.includes("Benchmark report"));
      check("has provenance footer", html.includes("Provenance &amp; license"));
      check("has license SPDX", html.includes("CC0-1.0"));
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// --- Test 3: all three companions appear when provided ---
function test3_allCompanions() {
  process.stdout.write("test3: all three companions render their sections\n");
  const dir = makeWorkdir();
  try {
    const { path } = writeMinimalCard(dir);
    const cbPath = writeFile(dir, "class_balance.json", {
      total_records: 100,
      bucket_counts: { positive_alpha: 30, hard_negative_bravo: 70 },
      target_family_counts: { alpha: 30, bravo: 70 },
    });
    const leakagePath = writeFile(dir, "leakage_report.json", {
      schema_ref: "schemas/leakage_report.schema.json",
      policy_id: "test.policy",
      checked_keys: ["seed", "geometry_hash"],
      findings: [],
    });
    const benchPath = writeFile(dir, "benchmark_report.json", {
      schema_ref: "schemas/benchmark_report.schema.json",
      dataset_card_schema_ref: "schemas/dataset_card.schema.json",
      benchmark_id: "bench-test",
      dataset_id: "ds-test",
      status: "pass",
      required_sections: [],
      required_artifacts: [],
      known_limitations: ["synthetic fixture"],
      limitations: ["synthetic fixture"],
      model_reports: [
        {
          model_id: "fixture_model",
          positive_records: 30,
          negative_records: 70,
          pd: 0.85,
          pfa: 0.02,
          mean_first_detection_latency_frames: 12.5,
        },
      ],
    });
    const out = join(dir, "full.html");
    const res = runTool([
      path,
      "--class-balance",
      cbPath,
      "--leakage",
      leakagePath,
      "--benchmark",
      benchPath,
      "--out",
      out,
    ]);
    check("exit 0", res.status === 0, `status=${res.status} stderr=${res.stderr}`);
    if (existsSync(out)) {
      const html = readFileSync(out, "utf8");
      check("has class balance section", html.includes("Class balance"));
      check("has bucket label", html.includes("positive_alpha"));
      check("has leakage section", html.includes("Leakage report"));
      check("has leakage policy id", html.includes("test.policy"));
      check("has benchmark section", html.includes("Benchmark report"));
      check("has benchmark model id", html.includes("fixture_model"));
      check("has TPR value", html.includes("0.8500"));
      check("has FPR value", html.includes("0.0200"));
      check("has mean time-to-detect", html.includes("12.50"));
      check("has known limitations", html.includes("synthetic fixture"));
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// --- Test 4: missing required field rejected with non-zero exit ---
function test4_missingField() {
  process.stdout.write("test4: missing required field exits non-zero\n");
  const dir = makeWorkdir();
  try {
    const { card } = writeMinimalCard(dir);
    delete card.dataset_name;
    const badPath = join(dir, "bad.json");
    writeFileSync(badPath, JSON.stringify(card, null, 2));
    const out = join(dir, "bad.html");
    const res = runTool([badPath, "--out", out]);
    check("non-zero exit", res.status !== 0, `status=${res.status}`);
    check("exit code 1", res.status === 1, `status=${res.status}`);
    check(
      "stderr mentions dataset_name",
      (res.stderr || "").includes("dataset_name"),
      `stderr=${res.stderr}`,
    );
    check("no output file produced", !existsSync(out));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// --- Test 5: deterministic output with behavioral assertions ---
// Replaces pure SHA snapshot comparison with content-level behavior checks.
// Uses the synthetic minimal card so the test runs without real campaign outputs.
function test5_deterministic() {
  process.stdout.write("test5: deterministic output with behavioral assertions\n");
  const dir = makeWorkdir();
  try {
    const { path: cardPath, card } = writeMinimalCard(dir);
    const outA = join(dir, "a.html");
    const outB = join(dir, "b.html");
    const r1 = runTool([cardPath, "--out", outA]);
    const r2 = runTool([cardPath, "--out", outB]);
    check("first run exit 0", r1.status === 0, r1.stderr);
    check("second run exit 0", r2.status === 0, r2.stderr);
    check("output file a exists", existsSync(outA), "a.html not created");
    check("output file b exists", existsSync(outB), "b.html not created");
    if (existsSync(outA) && existsSync(outB)) {
      const contentA = readFileSync(outA, "utf8");
      const contentB = readFileSync(outB, "utf8");
      // Determinism check: same input must produce byte-identical output.
      check("output is deterministic across reruns", contentA === contentB, "reruns produced different bytes");
      // Behavioral assertions: verify the HTML contains required semantic structure.
      // These catch regressions where the renderer produces empty or wrong content
      // even if bytes happen to be deterministic.
      check("output is HTML5", contentA.trimStart().startsWith("<!doctype html>"), "missing HTML5 doctype");
      check("output contains dataset_name", contentA.includes(card.dataset_name), `dataset_name '${card.dataset_name}' not found`);
      check("output contains train split count", contentA.includes(card.splits.train.toString()), `train count ${card.splits.train} not found`);
      check("output contains validation split count", contentA.includes(card.splits.validation.toString()), `validation count ${card.splits.validation} not found`);
      check("output contains test split count", contentA.includes(card.splits.test.toString()), `test count ${card.splits.test} not found`);
      check("output contains validation tier section", contentA.includes("Validation"), "no Validation section found");
      check("output contains provenance section", contentA.includes("Provenance"), "no Provenance section found");
      check("output has no external scripts", !/<script\b[^>]*src=/i.test(contentA), "external script tag found");
      check("output has no external stylesheets", !/<link\b[^>]*rel=[\"']stylesheet/i.test(contentA), "external stylesheet found");
      check("output is non-trivially sized", contentA.length > 500, `output too short: ${contentA.length} bytes`);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// --- Test 6: usage error on no args produces exit 3 ---
function test6_usage() {
  process.stdout.write("test6: usage errors return exit 3\n");
  const res = runTool([]);
  check("no-args exit 3", res.status === 3, `status=${res.status}`);
  const res2 = runTool(["--unknown-flag", "card.json"]);
  check("unknown-flag exit 3", res2.status === 3, `status=${res2.status}`);
  const res3 = runTool(["a.json", "b.json"]);
  check("extra positional exit 3", res3.status === 3, `status=${res3.status}`);
  const res4 = runTool(["card.json", "--out"]);
  check("dangling flag exit 3", res4.status === 3, `status=${res4.status}`);
}

// --- Test 7: bad JSON is rejected as validation error ---
function test7_badJson() {
  process.stdout.write("test7: invalid JSON exits 1 with useful message\n");
  const dir = makeWorkdir();
  try {
    const badPath = join(dir, "bad.json");
    writeFileSync(badPath, "{ not valid json");
    const res = runTool([badPath, "--out", join(dir, "out.html")]);
    check("exit 1 on bad JSON", res.status === 1, `status=${res.status}`);
    check(
      "stderr mentions invalid JSON",
      /invalid json/i.test(res.stderr || ""),
      `stderr=${res.stderr}`,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// --- Test 8: wrong "kind" rejected ---
function test8_wrongKind() {
  process.stdout.write("test8: wrong kind field exits 1\n");
  const dir = makeWorkdir();
  try {
    const { card } = writeMinimalCard(dir);
    card.kind = "rcs_campaign";
    const badPath = join(dir, "wrong-kind.json");
    writeFileSync(badPath, JSON.stringify(card, null, 2));
    const res = runTool([badPath, "--out", join(dir, "out.html")]);
    check("exit 1 on wrong kind", res.status === 1, `status=${res.status}`);
    check("stderr mentions kind", /kind/i.test(res.stderr || ""), `stderr=${res.stderr}`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// --- Test 9: default output path defaults to <input>.html ---
function test9_defaultOutput() {
  process.stdout.write("test9: default output path is <input>.html\n");
  const dir = makeWorkdir();
  try {
    const { path } = writeMinimalCard(dir);
    const res = runTool([path]);
    check("exit 0", res.status === 0, `status=${res.status} stderr=${res.stderr}`);
    const expected = join(dir, "card.html");
    check("default output exists", existsSync(expected), `expected=${expected}`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// --- Test 10: split count mutation produces observably different output ---
// Behavioral integration test: changing the splits in the input card must produce
// a different output. This catches regressions where the renderer ignores the
// data and emits identical boilerplate regardless of input.
function test10_mutationProducesDifferentOutput() {
  process.stdout.write("test10: mutated input produces observably different output\n");
  const dir = makeWorkdir();
  try {
    const outBase = join(dir, "base.html");
    const outMut = join(dir, "mutated.html");
    const { path: basePath } = writeMinimalCard(dir);
    const res1 = runTool([basePath, "--out", outBase]);
    check("base render exit 0", res1.status === 0, res1.stderr);

    // Write a card with different split counts.
    const mutDir = makeWorkdir();
    const { path: mutPath } = writeMinimalCard(mutDir, {
      dataset_name: "Mutated Fixture",
      splits: { train: 999, validation: 111, test: 333 },
    });
    const res2 = runTool([mutPath, "--out", outMut]);
    check("mutated render exit 0", res2.status === 0, res2.stderr);

    if (existsSync(outBase) && existsSync(outMut)) {
      const baseHtml = readFileSync(outBase, "utf8");
      const mutHtml = readFileSync(outMut, "utf8");

      // The two outputs must differ — renderer is not returning constant output.
      check("mutated output differs from base", baseHtml !== mutHtml, "outputs are identical despite different inputs");

      // Mutated card-specific values must appear.
      check("mutated output contains new dataset_name", mutHtml.includes("Mutated Fixture"), "dataset_name not rendered");
      check("mutated output contains train count 999", mutHtml.includes("999"), "train count not rendered");
      check("mutated output contains validation count 111", mutHtml.includes("111"), "validation count not rendered");
      check("mutated output contains test count 333", mutHtml.includes("333"), "test count not rendered");

      // Base values must not appear in mutated output.
      check("mutated output does not contain old dataset_name", !mutHtml.includes("Synthetic Minimal Fixture"), "old dataset_name leaked into mutated output");
    }
    rmSync(mutDir, { recursive: true, force: true });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function main() {
  process.stdout.write(`render_dataset_card test harness\n`);
  test1_realCard();
  test2_minimalCard();
  test3_allCompanions();
  test4_missingField();
  test5_deterministic();
  test6_usage();
  test7_badJson();
  test8_wrongKind();
  test9_defaultOutput();
  test10_mutationProducesDifferentOutput();
  process.stdout.write(`\n---\nTOTAL: ${pass + fail} (${pass} passed, ${fail} failed)\n`);
  if (failures.length) {
    for (const f of failures) {
      process.stdout.write(`FAILED: ${f.name}${f.detail ? ` -- ${f.detail}` : ""}\n`);
    }
  }
  // jankurai:allow HLT-008-FALSE-GREEN-RISK reason=process.exit is the harness exit gate not an xit() disabled test expires=2027-06-01
  process.exit(fail === 0 ? 0 : 1);
}

main();
