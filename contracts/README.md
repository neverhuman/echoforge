# Contracts

`contracts/schema_catalog.json` is the canonical 12-schema catalog for the
studio slice. The Rust studio service reads the file directly and serves it to
the React app on the same origin, so the browser no longer owns a second
hard-coded list.

The live demo path is:

- `contracts/schema_catalog.json` for the catalog
- `tests/science/fixtures/bundles/v1_pass` for the canonical validation report
- `crates/echoforge-studio` for the runtime API and static asset server

The contract smoke package under `tests/contracts/` keeps the schema catalog,
schema validation, and canonical payload drift checks in Rust.
