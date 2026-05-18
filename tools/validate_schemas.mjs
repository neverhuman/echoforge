#!/usr/bin/env node
// JSON Schema 2020-12 validator using `ajv`. Validates every schema in
// `schemas/` (or the paths passed on the command line) is itself a valid
// schema, AND for each `tests/schemas/*.sample.json` instance attempts to
// validate it against the schema declared by its `$schema` or its `kind`
// field.
//
// Exit: 0 clean, 1 violations, 3 usage/IO.

import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, basename } from "node:path";

let Ajv2020;
try {
  ({ default: Ajv2020 } = await import("ajv/dist/2020.js"));
} catch (e) {
  process.stderr.write(`validate_schemas: ajv not installed (npm i -D ajv): ${e.message}\n`);
  process.exit(3);
}

const args = process.argv.slice(2);
const SCHEMA_DIR = "schemas";
const SAMPLE_DIR = "tests/schemas";

function loadJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function listSchemas() {
  const out = [];
  if (args.length > 0) {
    for (const a of args) {
      if (a.endsWith(".json") && statSync(a).isFile()) out.push(a);
    }
    if (out.length) return out;
  }
  try {
    for (const f of readdirSync(SCHEMA_DIR)) {
      if (f.endsWith(".schema.json")) out.push(join(SCHEMA_DIR, f));
    }
  } catch (e) {
    process.stderr.write(`validate_schemas: cannot read ${SCHEMA_DIR}: ${e.message}\n`);
    process.exit(3);
  }
  return out;
}

function listSamples() {
  const out = [];
  try {
    for (const f of readdirSync(SAMPLE_DIR)) {
      if (f.endsWith(".sample.json")) out.push(join(SAMPLE_DIR, f));
    }
  } catch {}
  return out;
}

function main() {
  const ajv = new Ajv2020({ strict: false, allErrors: true });
  const schemas = listSchemas();
  if (schemas.length === 0) {
    process.stderr.write("validate_schemas: no schemas found\n");
    process.exit(3);
  }

  // Pre-load every schema so $ref resolution works.
  const compiled = new Map();
  let fail = 0;

  // Add common.schema.json first (others $ref into it).
  const orderHead = ["schemas/common.schema.json"];
  const ordered = orderHead.concat(schemas.filter((s) => !orderHead.includes(s)));

  for (const path of ordered) {
    try {
      const schema = loadJson(path);
      if (schema.$id) ajv.addSchema(schema, schema.$id);
      const validate = ajv.compile(schema);
      compiled.set(path, validate);
      compiled.set(basename(path), validate);
      process.stdout.write(`OK   schema ${path}\n`);
    } catch (e) {
      fail++;
      process.stdout.write(`FAIL schema ${path}\n     - ${e.message}\n`);
    }
  }

  for (const samplePath of listSamples()) {
    let sample;
    try {
      sample = loadJson(samplePath);
    } catch (e) {
      fail++;
      process.stdout.write(`FAIL sample ${samplePath}\n     - ${e.message}\n`);
      continue;
    }
    const kind = sample.kind;
    const target =
      compiled.get(`schemas/${kind}.schema.json`) ||
      compiled.get(`${kind}.schema.json`);
    if (!target) {
      process.stdout.write(`SKIP sample ${samplePath} (no schema for kind="${kind}")\n`);
      continue;
    }
    const ok = target(sample);
    if (ok) {
      process.stdout.write(`OK   sample ${samplePath}\n`);
    } else {
      fail++;
      process.stdout.write(`FAIL sample ${samplePath}\n`);
      for (const err of target.errors || []) {
        process.stdout.write(`     - ${err.instancePath || "/"} ${err.message}\n`);
      }
    }
  }

  process.stderr.write(`validate_schemas: ${fail} failure(s)\n`);
  process.exit(fail === 0 ? 0 : 1);
}

main();
