#!/usr/bin/env node
// Reproducibility-bytes lane for the EchoForge Monte-Carlo demo.
//
// Runs `cargo run -p echoforge-cli --release -- demo monte-carlo ...` twice
// into two subdirectories of a working root, walks both trees, sha256s
// every file, and asserts the file sets are identical and every pair of
// sha256s matches.
//
// Wallclock-derivative files (benchmark_report.md, benchmark_report.json)
// are intentionally excluded — they carry stage timings and throughput
// measurements that vary by design. Presence/absence of each excluded
// file is still cross-checked, so a file dropping out across runs still
// fails the lane.
//
// Usage:
//   node tools/repro_bytes.mjs [<workdir>]
//
// Default <workdir>: target/repro-bytes/<preset>
//
// Exit codes:
//   0 = byte-equality across all compared files
//   1 = mismatch (file set drift or differing bytes)
//   3 = usage / IO / subprocess error
//
// Output: JSON summary printed to stdout:
//   { ok: bool, run_a_count, run_b_count, compared, mismatches: [...] }

import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
} from "node:fs";
import { join, relative, sep } from "node:path";

const PRESET = "low-altitude-fixed-wing-takeoff";
const EPISODES = "4";
const SEED = "20260518";
const GENERATED_AT = "2026-05-18T00:00:00Z";

// Wallclock-derivative artifacts. Keep this in sync with
// `crates/echoforge-dataset/tests/repro_bytes.rs::WALLCLOCK_FILES`.
const WALLCLOCK_FILES = new Set([
  "benchmark_report.md",
  "benchmark_report.json",
]);

function fatal(code, msg) {
  process.stderr.write(`repro_bytes: ${msg}\n`);
  process.exit(code);
}

function parseArgs(argv) {
  const positional = [];
  for (const arg of argv) {
    if (arg.startsWith("--")) {
      fatal(3, `unknown flag ${arg} (only positional <workdir> is supported)`);
    }
    positional.push(arg);
  }
  if (positional.length > 1) {
    fatal(3, `expected at most one positional argument (workdir); got ${positional.length}`);
  }
  return { workdir: positional[0] || join("target", "repro-bytes", PRESET) };
}

function ensureFreshWorkdir(workdir) {
  if (existsSync(workdir)) {
    rmSync(workdir, { recursive: true, force: true });
  }
  mkdirSync(workdir, { recursive: true });
}

function runDemo(outDir) {
  const args = [
    "run",
    "-p",
    "echoforge-cli",
    "--release",
    "--",
    "demo",
    "monte-carlo",
    "--preset",
    PRESET,
    "--episodes",
    EPISODES,
    "--seed",
    SEED,
    "--generated-at",
    GENERATED_AT,
    "--out",
    outDir,
  ];
  // Inherit stderr so cargo's compile output is visible on first build;
  // capture stdout so the demo's receipt-style log doesn't drown the
  // final JSON summary.
  const result = spawnSync("cargo", args, {
    stdio: ["ignore", "pipe", "inherit"],
    encoding: "utf8",
  });
  if (result.error) {
    fatal(3, `cargo invocation failed for ${outDir}: ${result.error.message}`);
  }
  if (result.status !== 0) {
    fatal(3, `cargo exited ${result.status} for ${outDir}`);
  }
}

function walk(root) {
  const out = new Map();
  const stack = [root];
  while (stack.length) {
    const dir = stack.pop();
    let entries;
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch (e) {
      fatal(3, `readdir(${dir}) failed: ${e.message}`);
    }
    for (const entry of entries) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) {
        stack.push(path);
      } else if (entry.isFile()) {
        const rel = relative(root, path).split(sep).join("/");
        const buf = readFileSync(path);
        const hash = createHash("sha256").update(buf).digest("hex");
        out.set(rel, hash);
      }
      // Symlinks / sockets / etc. are silently ignored — the demo writer
      // never creates them, so they would be a separate regression.
    }
  }
  return out;
}

function main() {
  const { workdir } = parseArgs(process.argv.slice(2));
  ensureFreshWorkdir(workdir);

  const runA = join(workdir, "run-a");
  const runB = join(workdir, "run-b");
  runDemo(runA);
  runDemo(runB);

  if (!statSync(runA).isDirectory() || !statSync(runB).isDirectory()) {
    fatal(3, `expected both run dirs to exist after generation (${runA}, ${runB})`);
  }

  const hashesA = walk(runA);
  const hashesB = walk(runB);

  const mismatches = [];
  const onlyInA = [];
  const onlyInB = [];
  for (const [rel, hashA] of hashesA) {
    if (!hashesB.has(rel)) {
      onlyInA.push(rel);
      continue;
    }
    if (WALLCLOCK_FILES.has(rel)) continue;
    const hashB = hashesB.get(rel);
    if (hashA !== hashB) {
      mismatches.push({ path: rel, run_a: hashA, run_b: hashB });
    }
  }
  for (const rel of hashesB.keys()) {
    if (!hashesA.has(rel)) onlyInB.push(rel);
  }

  const compared = [...hashesA.keys()].filter(
    (rel) => hashesB.has(rel) && !WALLCLOCK_FILES.has(rel),
  ).length;
  const ok =
    mismatches.length === 0 &&
    onlyInA.length === 0 &&
    onlyInB.length === 0 &&
    compared > 0;

  const summary = {
    ok,
    preset: PRESET,
    workdir,
    run_a_count: hashesA.size,
    run_b_count: hashesB.size,
    compared,
    excluded_wallclock: [...WALLCLOCK_FILES],
    only_in_a: onlyInA,
    only_in_b: onlyInB,
    mismatches,
  };
  process.stdout.write(JSON.stringify(summary, null, 2) + "\n");
  process.exit(ok ? 0 : 1);
}

main();
