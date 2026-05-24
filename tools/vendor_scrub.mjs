#!/usr/bin/env node
// Vendor-name scrub for EchoForge. Walks `git ls-files`, scans each tracked
// text file for banned terms declared in `agent/banned-terms.toml`, and
// reports matches as JSONL + Markdown.
//
// Exit codes:
//   0 = clean
//   1 = matches found
//   2 = banlist empty/malformed (when fail_when_banlist_empty=true)
//   3 = IO/usage error
//
// Flags:
//   --staged       scan only staged files (for pre-commit)
//   --report-only  always exit 0, dump inventory
//   --json <path>  write JSONL report (default: target/jankurai/vendor-scrub.jsonl)
//   --md   <path>  write Markdown summary (default: target/jankurai/vendor-scrub.md)

import { readFileSync, writeFileSync, mkdirSync, statSync } from "node:fs";
import { execSync } from "node:child_process";
import { dirname, resolve, sep } from "node:path";

const args = process.argv.slice(2);
const opts = {
  staged: args.includes("--staged"),
  reportOnly: args.includes("--report-only"),
  json: argValue("--json", "target/jankurai/vendor-scrub.jsonl"),
  md: argValue("--md", "target/jankurai/vendor-scrub.md"),
};

function argValue(name, def) {
  const i = args.indexOf(name);
  if (i === -1) return def;
  return args[i + 1] || def;
}

function fatal(code, msg) {
  process.stderr.write(`vendor_scrub: ${msg}\n`);
  process.exit(code);
}

function loadBanlist() {
  const path = "agent/banned-terms.toml";
  let raw;
  try {
    raw = readFileSync(path, "utf8");
  } catch (e) {
    fatal(3, `cannot read ${path}: ${e.message}`);
  }
  return parseTomlBanlist(raw);
}

// Minimal TOML parser sufficient for banned-terms.toml shape.
function parseTomlBanlist(text) {
  const out = {
    version: 1,
    compliance: {
      fail_on_match: true,
      fail_when_banlist_empty: true,
      placeholder_token_pattern: "^__PLACEHOLDER_",
      exclude: { paths: [], file_globs: [] },
    },
    terms: [],
  };
  let section = null;
  let currentTerm = null;
  // Multi-line `key = [...]` collector. When non-null, lines accumulate
  // into `pendingArray.buffer` until the closing `]` is seen, then the
  // joined string is parsed by parseValue and routed to the right
  // section. This is the minimum we need so that `paths = [\n "a",\n ...]`
  // works without pulling in a real TOML library.
  let pendingArray = null;
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.replace(/#.*$/, "").trim();
    if (!line) continue;
    if (pendingArray) {
      pendingArray.buffer += " " + line;
      if (line.includes("]")) {
        const val = parseValue(pendingArray.buffer.trim());
        applyKey(out, pendingArray.section, currentTerm, pendingArray.key, val);
        pendingArray = null;
      }
      continue;
    }
    if (line.startsWith("[[term]]")) {
      if (currentTerm) out.terms.push(currentTerm);
      currentTerm = { case_sensitive: false };
      section = "term";
      continue;
    }
    if (line.startsWith("[compliance.exclude]")) { section = "exclude"; continue; }
    if (line.startsWith("[compliance]")) { section = "compliance"; continue; }
    if (line.startsWith("[")) { section = null; continue; }
    const eq = line.indexOf("=");
    if (eq === -1) continue;
    const key = line.slice(0, eq).trim();
    const valRaw = line.slice(eq + 1).trim();
    if (valRaw.startsWith("[") && !valRaw.endsWith("]")) {
      pendingArray = { section, key, buffer: valRaw };
      continue;
    }
    const val = parseValue(valRaw);
    applyKey(out, section, currentTerm, key, val);
  }
  if (currentTerm) out.terms.push(currentTerm);
  return out;
}

function applyKey(out, section, currentTerm, key, val) {
  if (section === "term") currentTerm[key] = val;
  else if (section === "compliance") out.compliance[key] = val;
  else if (section === "exclude") out.compliance.exclude[key] = val;
  else if (section === null && key === "version") out.version = val;
}

function parseValue(raw) {
  if (raw.startsWith("[")) {
    // Tolerate multi-line buffers: strip newlines that the array collector
    // joined in, then split on commas. Trailing commas are allowed.
    const inner = raw.replace(/^\[|\]$/g, "").trim();
    if (!inner) return [];
    return inner
      .split(",")
      .map((s) => s.trim().replace(/^"|"$/g, ""))
      .filter((s) => s.length > 0);
  }
  if (raw === "true") return true;
  if (raw === "false") return false;
  if (/^-?\d+$/.test(raw)) return Number(raw);
  return raw.replace(/^"|"$/g, "");
}

function gitListFiles() {
  const cmd = opts.staged
    ? "git diff --name-only --cached --diff-filter=ACMR"
    : "git ls-files";
  try {
    return execSync(cmd, { encoding: "utf8" })
      .split("\n")
      .filter(Boolean);
  } catch (e) {
    fatal(3, `git invocation failed: ${e.message}`);
  }
}

function isExcluded(path, excludes) {
  const norm = path.split(sep).join("/");
  for (const prefix of excludes.paths) {
    if (norm.startsWith(prefix)) return true;
  }
  for (const glob of excludes.file_globs) {
    if (globMatch(glob, norm)) return true;
  }
  return false;
}

function globMatch(glob, str) {
  const re = new RegExp(
    "^" +
      glob
        .replace(/[.+^${}()|[\]\\]/g, "\\$&")
        .replace(/\*/g, ".*") +
      "$",
  );
  return re.test(str);
}

function isBinary(path) {
  try {
    const buf = readFileSync(path);
    const slice = buf.subarray(0, Math.min(buf.length, 4096));
    for (const byte of slice) if (byte === 0) return true;
    return false;
  } catch {
    return true;
  }
}

function scan(file, terms) {
  let text;
  try {
    text = readFileSync(file, "utf8");
  } catch {
    return [];
  }
  const lines = text.split(/\r?\n/);
  const hits = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    for (const term of terms) {
      const needle = term.case_sensitive ? line : line.toLowerCase();
      const probe = term.case_sensitive ? term.pattern : term.pattern.toLowerCase();
      const idx = needle.indexOf(probe);
      if (idx !== -1) {
        hits.push({
          path: file,
          line: i + 1,
          col: idx + 1,
          term: term.pattern,
          reason: term.reason || "",
          snippet: line.slice(Math.max(0, idx - 16), idx + probe.length + 16),
        });
      }
    }
  }
  return hits;
}

function ensureDir(path) {
  mkdirSync(dirname(resolve(path)), { recursive: true });
}

function writeReports(hits) {
  ensureDir(opts.json);
  writeFileSync(opts.json, hits.map((h) => JSON.stringify(h)).join("\n") + (hits.length ? "\n" : ""));
  const md = [
    "# Vendor scrub report",
    "",
    `Scanned ${hits.length === 0 ? "clean" : `${hits.length} match${hits.length === 1 ? "" : "es"}`}.`,
    "",
    "| path | line | term | snippet |",
    "|---|---|---|---|",
    ...hits.map((h) => `| ${h.path} | ${h.line} | \`${h.term}\` | \`${h.snippet.replace(/\|/g, "\\|")}\` |`),
  ].join("\n");
  ensureDir(opts.md);
  writeFileSync(opts.md, md + "\n");
}

function main() {
  const banlist = loadBanlist();
  const placeholderRe = new RegExp(banlist.compliance.placeholder_token_pattern);
  const realTerms = banlist.terms.filter((t) => !placeholderRe.test(t.pattern));
  const placeholderTerms = banlist.terms.filter((t) => placeholderRe.test(t.pattern));

  if (banlist.compliance.fail_when_banlist_empty && realTerms.length === 0) {
    writeReports([]);
    process.stderr.write(
      `vendor_scrub: banlist contains only ${placeholderTerms.length} placeholder entr${placeholderTerms.length === 1 ? "y" : "ies"} ` +
        `(matching ${banlist.compliance.placeholder_token_pattern}). Populate agent/banned-terms.toml before merging.\n`,
    );
    if (opts.reportOnly) process.exit(0);
    process.exit(2);
  }

  const files = gitListFiles().filter((f) => {
    try {
      return statSync(f).isFile();
    } catch {
      return false;
    }
  });

  const scanned = [];
  for (const file of files) {
    if (isExcluded(file, banlist.compliance.exclude)) continue;
    if (isBinary(file)) continue;
    scanned.push(file);
  }

  const hits = [];
  for (const file of scanned) {
    hits.push(...scan(file, realTerms));
  }

  writeReports(hits);

  for (const h of hits) {
    process.stdout.write(`${h.path}:${h.line}:${h.col}: ${h.term}\n`);
  }
  process.stderr.write(
    `vendor_scrub: scanned ${scanned.length} file(s), ${realTerms.length} real banned term(s), ${hits.length} match(es). Reports: ${opts.json}, ${opts.md}.\n`,
  );

  if (opts.reportOnly) process.exit(0);
  if (hits.length > 0 && banlist.compliance.fail_on_match) process.exit(1);
  process.exit(0);
}

main();
