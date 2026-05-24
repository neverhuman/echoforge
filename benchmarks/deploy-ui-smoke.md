# Deploy / UI Smoke Benchmark

This directory holds the smallest possible smoke definition for the deployment
and product-surface slice.

Scope:

- The web UI builds with Vite and renders through React/TypeScript
- The React surface fetches live health, catalog, and validation state from the
  Rust studio service on the same origin
- The generated client remains available for contract helpers and tests
- GPU doctor emits a JSON receipt through the Rust smoke package without
  claiming hardware validation
- `rtk just demo` launches the same-origin Rust service + Vite build path

This is a scaffold, not a performance or correctness benchmark.
