set shell := ["sh", "-eu", "-c"]

default: fast

bootstrap:
    rtk jankurai version
    rtk jankurai adapters verify .

fast:
    rtk cargo test --workspace --locked
    rtk jankurai adapters verify .

contracts:
    rtk cargo test -p echoforge-core --locked

science-smoke:
    rtk cargo test -p echoforge-sig --locked
    rtk cargo test -p echoforge-radar --locked
    rtk cargo test -p echoforge-dataset --locked

gpu-smoke:
    node --test tests/gpu/doctor.test.mjs

dataset-smoke:
    rtk cargo test -p echoforge-dataset --locked

doctor:
    mkdir -p target/jankurai
    rtk jankurai doctor . --fail-on critical --json target/jankurai/doctor.json --md target/jankurai/doctor.md

score:
    mkdir -p target/jankurai
    rtk jankurai audit . --mode advisory --json target/jankurai/repo-score.json --md target/jankurai/repo-score.md
