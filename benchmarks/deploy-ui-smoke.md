# Deploy / UI Smoke Benchmark

This directory holds the smallest possible smoke definition for the deployment
and product-surface slice.

Scope:

- API health responds on `/healthz`
- API returns a placeholder schema catalog on `/api/catalog`
- Web UI loads the generated client stub and renders the schema list
- GPU doctor stub emits a JSON receipt without claiming validation

This is a scaffold, not a performance or correctness benchmark.

