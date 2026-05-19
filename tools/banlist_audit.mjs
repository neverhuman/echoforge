#!/usr/bin/env node
// Banlist sanity audit. Asserts that `agent/banned-terms.toml`:
//   1. Parses cleanly with the same minimal TOML reader vendor-scrub uses.
//   2. Contains zero rows matching `[compliance].placeholder_token_pattern`.
//   3. Contains at least MIN_REAL_TERMS real entries.
//   4. Every real entry has both `pattern` and `reason` populated.
//   5. The exclude paths list parsed by vendor-scrub is non-empty.
//
// Exit 0 on success, 1 on audit failure, 2 on IO/parse error.
//
// This is the test that locks in packet `banlist-activation` so the file
// cannot silently regress to the all-placeholder state.

import { readFileSync } from "node:fs";

const MIN_REAL_TERMS = 10;

function fail(msg) {
  process.stderr.write(`banlist_audit: ${msg}\n`);
  process.exit(1);
}

function fatal(msg) {
  process.stderr.write(`banlist_audit: ${msg}\n`);
  process.exit(2);
}

// Inline a tiny TOML parser equivalent to the one in vendor_scrub.mjs so
// this audit is self-contained and does not depend on importing from a
// sibling .mjs (Node import semantics around .mjs sibling imports stay
// brittle across versions; the file is small enough to duplicate).
function parseBanlist(raw) {
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
  let pendingArray = null;
  for (const rawLine of raw.split(/\r?\n/)) {
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

function main() {
  let raw;
  try {
    raw = readFileSync("agent/banned-terms.toml", "utf8");
  } catch (err) {
    fatal(`cannot read agent/banned-terms.toml: ${err.message}`);
  }
  const banlist = parseBanlist(raw);
  const placeholderRe = new RegExp(banlist.compliance.placeholder_token_pattern);

  const placeholderTerms = banlist.terms.filter((t) => placeholderRe.test(t.pattern || ""));
  if (placeholderTerms.length > 0) {
    fail(
      `${placeholderTerms.length} placeholder term(s) remain (matching ${banlist.compliance.placeholder_token_pattern}). ` +
        `Replace them with real banned terms.`,
    );
  }

  const realTerms = banlist.terms.filter((t) => !placeholderRe.test(t.pattern || ""));
  if (realTerms.length < MIN_REAL_TERMS) {
    fail(
      `${realTerms.length} real banned term(s) found, expected at least ${MIN_REAL_TERMS}. ` +
        `The wave-1 claim-guardrail set must remain populated.`,
    );
  }

  for (const term of realTerms) {
    if (!term.pattern || typeof term.pattern !== "string") {
      fail(`term missing pattern: ${JSON.stringify(term)}`);
    }
    if (!term.reason || typeof term.reason !== "string" || term.reason.length < 8) {
      fail(`term has missing or too-short reason: ${JSON.stringify(term)}`);
    }
  }

  const excludePaths = banlist.compliance.exclude.paths || [];
  if (excludePaths.length === 0) {
    fail(
      `[compliance.exclude].paths parsed as empty. ` +
        `vendor-scrub would scan the banlist file itself and self-match. ` +
        `Check the multi-line array parser in tools/vendor_scrub.mjs.`,
    );
  }
  if (!excludePaths.includes("agent/banned-terms.toml")) {
    fail(`exclude paths must include "agent/banned-terms.toml" (was: ${JSON.stringify(excludePaths)})`);
  }
  if (!excludePaths.some((p) => p === "tips/")) {
    fail(`exclude paths must include "tips/" (was: ${JSON.stringify(excludePaths)})`);
  }

  process.stdout.write(
    `banlist_audit: OK (${realTerms.length} real term(s), ${excludePaths.length} exclude path(s), 0 placeholders).\n`,
  );
}

main();
