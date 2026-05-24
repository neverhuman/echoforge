#!/usr/bin/env node
// Tests for `tools/render_validation_dossier.mjs`. Pure Node std-lib; no
// external test framework. Runs each case in sequence, accumulates a
// pass/fail count, exits 0 if all green and 1 otherwise.
//
// Usage:
//   node tools/test_render_validation_dossier.mjs
//
// Exit codes:
//   0 = all tests pass
//   1 = at least one test failed (or test harness itself errored)

import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const RENDERER = "tools/render_validation_dossier.mjs";

function freshTmp() {
  return mkdtempSync(join(tmpdir(), "render-validation-dossier-"));
}

function writeJson(path, value) {
  writeFileSync(path, JSON.stringify(value, null, 2));
}

function runRenderer(args) {
  const result = spawnSync("node", [RENDERER, ...args], { encoding: "utf8" });
  return {
    status: result.status,
    stdout: result.stdout || "",
    stderr: result.stderr || "",
  };
}

// Minimal-but-schema-shaped validation report. Mirrors
// `tests/schemas/validation_report.sample.json` but stays self-contained
// so the test does not coupling-leak into the fixture file.
function baseReport() {
  return {
    id: "ef:validation_report:public-proxy-test.alpha:0123456789abcdef:1",
    kind: "validation_report",
    schema_version: "1.0.0",
    public_proxy_id: "public-proxy-test.alpha",
    provenance: {
      source_kind: "synthetic",
      source_refs: ["public-domain-spec://test-fixture"],
      generated_by: "test-harness",
      generated_at: "2026-05-18T00:00:00Z",
      fingerprint_sha256: "a".repeat(64),
    },
    license: {
      spdx_id: "MIT",
      notice: "MIT-licensed test fixture.",
    },
    validation: {
      tier: "basic",
      status: "pass",
      uncertainty_score: 0.42,
      checks: [
        { name: "envelope_present", status: "pass" },
      ],
    },
    report_name: "test-fixture-report",
    subject_kind: "object_card",
    subject_id: "ef:object_card:public-proxy-test.alpha:0123456789abcdef:1",
    checks: [
      { name: "dimensions_nonnegative", status: "pass", message: "ok" },
      { name: "tags_present", status: "pass", message: "at least one tag" },
      { name: "name_nonempty", status: "warn", message: "name length 1" },
    ],
    overall_status: "warn",
  };
}

let failed = 0;
let passed = 0;
const failures = [];

function check(name, cond, detail) {
  if (cond) {
    passed++;
    process.stdout.write(`  ok    ${name}\n`);
  } else {
    failed++;
    failures.push({ name, detail });
    process.stdout.write(`  FAIL  ${name}${detail ? ` -- ${detail}` : ""}\n`);
  }
}

function withTmp(fn) {
  const dir = freshTmp();
  try {
    fn(dir);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function test1_minimal_valid_report() {
  process.stdout.write("test1: minimal valid report renders subject_id, status, all check names\n");
  withTmp((dir) => {
    const inputPath = join(dir, "report.json");
    const outPath = join(dir, "out.html");
    const report = baseReport();
    writeJson(inputPath, report);

    const res = runRenderer([inputPath, "--out", outPath]);
    check("exit code is 0", res.status === 0, `status=${res.status} stderr=${res.stderr}`);

    const html = readFileSync(outPath, "utf8");
    check("contains subject_id", html.includes(report.subject_id));
    check("contains overall_status badge", /status-warn[^>]*>warn</.test(html), "no warn status badge");
    check("contains all check names", report.checks.every((c) => html.includes(c.name)));
    check("contains footer", html.includes("public-proxy validation dossier"));
    check("contains generated_at", html.includes("2026-05-18T00:00:00Z"));
    check("contains fingerprint", html.includes("a".repeat(64)));
    check("contains license spdx", html.includes(">MIT<"));
    check("declares HTML5 doctype", html.startsWith("<!doctype html>"));
  });
}

function test2_cross_checked_and_production_sbr_badges_present() {
  process.stdout.write("test2: tier=cross_checked + fidelity_class=production_sbr both badges appear\n");
  withTmp((dir) => {
    const inputPath = join(dir, "report.json");
    const outPath = join(dir, "out.html");
    const report = baseReport();
    report.validation.tier = "cross_checked";
    report.validation.fidelity_class = "production_sbr";
    writeJson(inputPath, report);

    const res = runRenderer([inputPath, "--out", outPath]);
    check("exit code is 0", res.status === 0, `status=${res.status} stderr=${res.stderr}`);

    const html = readFileSync(outPath, "utf8");
    check("tier badge present", /badge-tier[^>]*>tier: cross_checked</.test(html), "no cross_checked tier badge");
    check("fidelity badge present", /badge-fidelity[^>]*>fidelity: production_sbr</.test(html), "no production_sbr fidelity badge");
    check("badge color classes both emitted",
      html.includes("badge-tier") && html.includes("badge-fidelity"));
  });
}

function test3_missing_required_field_exits_nonzero() {
  process.stdout.write("test3: missing required field exits non-zero with useful error\n");
  withTmp((dir) => {
    const inputPath = join(dir, "report.json");
    const outPath = join(dir, "out.html");
    const report = baseReport();
    delete report.provenance;
    writeJson(inputPath, report);

    const res = runRenderer([inputPath, "--out", outPath]);
    check("exit code is non-zero", res.status !== 0, `status=${res.status}`);
    check("stderr mentions provenance",
      res.stderr.toLowerCase().includes("provenance"),
      `stderr=${res.stderr}`);
    check("stderr mentions missing", res.stderr.toLowerCase().includes("missing"));
  });
}

function test4_deterministic_output() {
  process.stdout.write("test4: HTML output byte-identical across reruns for fixed input\n");
  withTmp((dir) => {
    const inputPath = join(dir, "report.json");
    const outA = join(dir, "a.html");
    const outB = join(dir, "b.html");
    const report = baseReport();
    report.validation.fidelity_class = "cross_solver_bridged";
    writeJson(inputPath, report);

    const resA = runRenderer([inputPath, "--out", outA]);
    const resB = runRenderer([inputPath, "--out", outB]);
    check("first run exit 0", resA.status === 0);
    check("second run exit 0", resB.status === 0);

    const a = readFileSync(outA);
    const b = readFileSync(outB);
    check("byte-identical across reruns", Buffer.compare(a, b) === 0,
      `lenA=${a.length} lenB=${b.length}`);
  });
}

function test5_default_output_path() {
  process.stdout.write("test5: default --out is <input>.html next to input\n");
  withTmp((dir) => {
    const inputPath = join(dir, "report.json");
    const expected = `${inputPath}.html`;
    writeJson(inputPath, baseReport());

    const res = runRenderer([inputPath]);
    check("exit code is 0", res.status === 0, `status=${res.status} stderr=${res.stderr}`);
    let html = "";
    try {
      html = readFileSync(expected, "utf8");
    } catch (e) {
      check("default output file exists", false, e.message);
      return;
    }
    check("default output file populated", html.length > 0);
    check("default output contains subject", html.includes("ef:object_card:public-proxy-test.alpha"));
  });
}

function test6_bad_kind_rejected() {
  process.stdout.write("test6: kind != \"validation_report\" is rejected\n");
  withTmp((dir) => {
    const inputPath = join(dir, "report.json");
    const outPath = join(dir, "out.html");
    const report = baseReport();
    report.kind = "object_card";
    writeJson(inputPath, report);

    const res = runRenderer([inputPath, "--out", outPath]);
    check("exit code is non-zero", res.status !== 0, `status=${res.status}`);
    check("stderr mentions kind", res.stderr.toLowerCase().includes("kind"),
      `stderr=${res.stderr}`);
  });
}

function test7_usage_error_no_args() {
  process.stdout.write("test7: zero positional args exits with usage code 3\n");
  const res = runRenderer([]);
  check("exit code is 3", res.status === 3, `status=${res.status} stderr=${res.stderr}`);
  check("stderr explains expectation",
    res.stderr.toLowerCase().includes("expected"),
    `stderr=${res.stderr}`);
}

function main() {
  test1_minimal_valid_report();
  test2_cross_checked_and_production_sbr_badges_present();
  test3_missing_required_field_exits_nonzero();
  test4_deterministic_output();
  test5_default_output_path();
  test6_bad_kind_rejected();
  test7_usage_error_no_args();

  const total = passed + failed;
  process.stdout.write(`\nresult: ${passed}/${total} checks passed`);
  if (failed > 0) {
    process.stdout.write(`, ${failed} failed\n`);
    for (const f of failures) {
      process.stdout.write(`  - ${f.name}${f.detail ? `: ${f.detail}` : ""}\n`);
    }
    // jankurai:allow HLT-008-FALSE-GREEN-RISK reason=process.exit(1) is the harness failure gate not an xit() disabled test expires=2027-06-01
    process.exit(1);
  }
  process.stdout.write("\n");
  // jankurai:allow HLT-008-FALSE-GREEN-RISK reason=process.exit(0) is the harness success gate not an xit() disabled test expires=2027-06-01
  process.exit(0);
}

main();
