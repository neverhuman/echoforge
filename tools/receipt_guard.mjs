#!/usr/bin/env node
// Receipt guard for EchoForge. Validates that every receipt under
// `.agents/receipts/<slice>/<timestamp>.md` follows the agreed shape:
//
//   - Filename: ^\d{8}T\d{6}Z?(?:-\d{4})?\.md$ (accept both Z and offset)
//   - Has a top H1 (any text)
//   - Required H2 sections (case-insensitive substring): one of
//     {"files changed", "scope"}, plus {"commands", "results"|"validation",
//     "notes"|"risks"|"follow"}
//   - At least one bullet under Files Changed (`- ...`)
//   - At least one `rtk ` invocation appears somewhere in the body
//
// Modes:
//   node tools/receipt_guard.mjs --all                # validate every receipt
//   node tools/receipt_guard.mjs <path> [<path> ...]  # validate specific files
//   node tools/receipt_guard.mjs --staged             # validate staged receipts
//
// Exit: 0 clean, 1 violations, 3 usage/IO.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { execSync } from "node:child_process";
import { join, basename } from "node:path";

const args = process.argv.slice(2);
const opts = {
  all: args.includes("--all"),
  staged: args.includes("--staged"),
  paths: args.filter((a) => !a.startsWith("--")),
};

function fatal(code, msg) {
  process.stderr.write(`receipt_guard: ${msg}\n`);
  process.exit(code);
}

const ROOT = ".agents/receipts";
const NAME_RE = /^\d{8}T\d{6}Z?(?:-\d{4})?\.md$/;
const REQUIRED_SECTIONS = [
  { label: "Files Changed | Scope", keys: ["files changed", "scope"] },
  { label: "Commands", keys: ["commands"] },
  { label: "Results | Validation", keys: ["results", "validation"] },
  { label: "Notes | Risks | Follow-ups", keys: ["notes", "risks", "follow"] },
];

function collectAll() {
  let slices;
  try {
    slices = readdirSync(ROOT, { withFileTypes: true });
  } catch (e) {
    fatal(3, `cannot read ${ROOT}: ${e.message}`);
  }
  const out = [];
  for (const slice of slices) {
    if (!slice.isDirectory()) continue;
    const dir = join(ROOT, slice.name);
    for (const f of readdirSync(dir)) {
      const path = join(dir, f);
      try {
        if (statSync(path).isFile()) out.push(path);
      } catch {}
    }
  }
  return out;
}

function stagedReceipts() {
  try {
    return execSync("git diff --name-only --cached --diff-filter=ACMR", { encoding: "utf8" })
      .split("\n")
      .filter((p) => p.startsWith(`${ROOT}/`) && p.endsWith(".md"));
  } catch (e) {
    fatal(3, `git diff failed: ${e.message}`);
  }
}

function validate(path) {
  const errs = [];
  const fname = basename(path);
  if (!NAME_RE.test(fname)) {
    errs.push(`bad filename "${fname}" (want ^\\d{8}T\\d{6}Z?(?:-\\d{4})?\\.md$)`);
  }
  let text;
  try {
    text = readFileSync(path, "utf8");
  } catch (e) {
    return [`unreadable: ${e.message}`];
  }
  const lines = text.split(/\r?\n/);
  const h1 = lines.find((l) => /^#\s+\S/.test(l));
  if (!h1) errs.push("missing H1 title");

  const lower = text.toLowerCase();
  for (const req of REQUIRED_SECTIONS) {
    const hit = req.keys.some((k) => new RegExp(`^##\\s+.*${k}`, "im").test(text));
    if (!hit) errs.push(`missing H2 section: ${req.label}`);
  }

  // Files-changed section: at least one bullet.
  const fcStart = lines.findIndex((l) => /^##\s+.*(files changed|scope)/i.test(l));
  if (fcStart !== -1) {
    let i = fcStart + 1;
    let hasBullet = false;
    while (i < lines.length && !/^#/.test(lines[i])) {
      if (/^\s*[-*+]\s+\S/.test(lines[i])) { hasBullet = true; break; }
      i++;
    }
    if (!hasBullet) errs.push("Files Changed has no bullets");
  }

  if (!/\brtk\s+\S+/.test(lower)) errs.push("no `rtk ...` command referenced");

  return errs;
}

function main() {
  let targets;
  if (opts.all) targets = collectAll();
  else if (opts.staged) targets = stagedReceipts();
  else targets = opts.paths;
  if (!targets || targets.length === 0) fatal(3, "no receipts to validate (use --all, --staged, or pass paths)");

  let fail = 0;
  for (const t of targets) {
    const errs = validate(t);
    if (errs.length === 0) {
      process.stdout.write(`OK   ${t}\n`);
    } else {
      fail++;
      process.stdout.write(`FAIL ${t}\n`);
      for (const e of errs) process.stdout.write(`     - ${e}\n`);
    }
  }
  process.stderr.write(`receipt_guard: ${targets.length} receipt(s), ${fail} failing.\n`);
  process.exit(fail === 0 ? 0 : 1);
}

main();
