#!/usr/bin/env node
/**
 * Manifest-based source LOC self-classification audit.
 *
 * This reports an auditable self-classification of tracked source-style files
 * as either generative/AI-inspired or human-standard. It is not evidence of
 * authorship history.
 */

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

function fail(message) {
	process.stderr.write(`generative_origin_audit: ${message}\n`);
	process.exit(1);
}

function readJson(filePath, label) {
	try {
		return JSON.parse(readFileSync(filePath, "utf8"));
	} catch (error) {
		fail(`could not read ${label} at ${filePath}: ${error.message}`);
	}
}

function repoRoot() {
	try {
		return execFileSync("git", ["rev-parse", "--show-toplevel"], {
			encoding: "utf8",
			stdio: ["ignore", "pipe", "pipe"],
		}).trim();
	} catch (error) {
		fail(
			`could not resolve repo root via git: ${error.stderr?.toString() || error.message}`,
		);
	}
}

function trackedFiles(root) {
	try {
		const stdout = execFileSync(
			"git",
			["-C", root, "ls-files", "--cached", "--others", "--exclude-standard"],
			{ encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] },
		);
		return stdout
			.split(/\r?\n/)
			.map((line) => line.trim())
			.filter((line) => line.length > 0);
	} catch (error) {
		fail(`git ls-files failed: ${error.stderr?.toString() || error.message}`);
	}
}

function escapeRegexChar(ch) {
	return "\\^$+?.()|{}[]-".includes(ch) ? `\\${ch}` : ch;
}

function globToRegExp(glob) {
	let pattern = "^";
	for (let index = 0; index < glob.length; index += 1) {
		const char = glob[index];
		const next = glob[index + 1];
		if (char === "*") {
			if (next === "*") {
				const afterNext = glob[index + 2];
				if (afterNext === "/") {
					pattern += "(?:.*/)?";
					index += 2;
				} else {
					pattern += ".*";
					index += 1;
				}
			} else {
				pattern += "[^/]*";
			}
			continue;
		}
		if (char === "?") {
			pattern += "[^/]";
			continue;
		}
		pattern += escapeRegexChar(char);
	}
	pattern += "$";
	return new RegExp(pattern);
}

function countLoc(filePath) {
	const text = readFileSync(filePath, "utf8");
	return text.split(/\r?\n/).filter((line) => line.trim().length > 0).length;
}

function csvEscape(value) {
	const text = String(value);
	if (/[",\r\n]/.test(text)) {
		return `"${text.replace(/"/g, '""')}"`;
	}
	return text;
}

function main() {
	const args = process.argv.slice(2);
	let manifestPath = null;
	let outRoot = null;
	for (let index = 0; index < args.length; index += 1) {
		const arg = args[index];
		if (arg === "--manifest") {
			manifestPath = args[++index];
			continue;
		}
		if (arg === "--out-root") {
			outRoot = args[++index];
			continue;
		}
		fail(`unknown argument: ${arg}`);
	}

	if (!manifestPath || !outRoot) {
		fail(
			"usage: generative_origin_audit.mjs --manifest <manifest.json> --out-root <dir>",
		);
	}

	const resolvedManifestPath = path.resolve(manifestPath);
	const resolvedOutRoot = path.resolve(outRoot);
	const root = repoRoot();
	const manifest = readJson(resolvedManifestPath, "manifest");
	if (!manifest || typeof manifest !== "object") {
		fail("manifest must be a JSON object");
	}
	const categories = Array.isArray(manifest.categories)
		? manifest.categories
		: null;
	if (!categories || categories.length === 0) {
		fail("manifest.categories must be a non-empty array");
	}
	const compiledRules = categories.map((category) => {
		if (!category || typeof category !== "object") {
			fail("each manifest category must be an object");
		}
		const globs = Array.isArray(category.globs) ? category.globs : null;
		if (!globs || globs.length === 0) {
			fail(`category ${category.id || "<unknown>"} must define globs`);
		}
		return {
			id: String(category.id || ""),
			label: String(category.label || category.id || ""),
			reason: String(category.reason || ""),
			regexes: globs.map((glob) => globToRegExp(String(glob))),
		};
	});

	const allFiles = trackedFiles(root);
	const relevantFiles = [];
	for (const relPath of allFiles) {
		const match = compiledRules.find((rule) =>
			rule.regexes.some((regex) => regex.test(relPath)),
		);
		if (!match) {
			continue;
		}
		relevantFiles.push({
			path: relPath,
			rule: match,
		});
	}

	const rows = relevantFiles.map((entry) => {
		const absPath = path.join(root, entry.path);
		const loc = countLoc(absPath);
		return {
			path: entry.path,
			category_id: entry.rule.id,
			category_label: entry.rule.label,
			loc,
			reason: entry.rule.reason,
		};
	});

	const totals = rows.reduce(
		(acc, row) => {
			acc.total_loc += row.loc;
			acc.category_totals[row.category_id] =
				(acc.category_totals[row.category_id] || 0) + row.loc;
			return acc;
		},
		{ total_loc: 0, category_totals: {} },
	);
	const generativeLoc = totals.category_totals.generative_ai_inspired || 0;
	const humanLoc = totals.category_totals.human_standard || 0;
	const classifiedLoc = generativeLoc + humanLoc;
	const percentageGenerative =
		classifiedLoc > 0 ? (generativeLoc / classifiedLoc) * 100.0 : 0.0;
	const percentageHuman =
		classifiedLoc > 0 ? (humanLoc / classifiedLoc) * 100.0 : 0.0;

	mkdirSync(resolvedOutRoot, { recursive: true });
	const jsonSummary = {
		status: rows.length > 0 ? "pass" : "fail",
		manifest_path: resolvedManifestPath,
		scope: manifest.scope || "",
		notes: Array.isArray(manifest.notes) ? manifest.notes : [],
		file_count: rows.length,
		total_loc: totals.total_loc,
		classified_loc: classifiedLoc,
		generative_loc: generativeLoc,
		human_standard_loc: humanLoc,
		percentage_generative_loc: Number(percentageGenerative.toFixed(6)),
		percentage_human_standard_loc: Number(percentageHuman.toFixed(6)),
		category_totals: totals.category_totals,
	};
	const csvRows = [
		"path,category_id,category_label,loc,reason",
		...rows
			.sort((a, b) => a.path.localeCompare(b.path))
			.map((row) =>
				[
					csvEscape(row.path),
					csvEscape(row.category_id),
					csvEscape(row.category_label),
					csvEscape(row.loc),
					csvEscape(row.reason),
				].join(","),
			),
	];

	writeFileSync(
		path.join(resolvedOutRoot, "generative_origin_audit.json"),
		`${JSON.stringify(jsonSummary, null, 2)}\n`,
	);
	writeFileSync(
		path.join(resolvedOutRoot, "generative_origin_audit.csv"),
		`${csvRows.join("\n")}\n`,
	);
	process.stdout.write(`${JSON.stringify(jsonSummary)}\n`);
}

main();
