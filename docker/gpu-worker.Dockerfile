FROM rust:1.85-bookworm AS rust-build

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY tests/contracts ./tests/contracts
COPY tests/gpu ./tests/gpu
COPY contracts ./contracts
COPY crates ./crates
COPY schemas ./schemas
COPY tests/radar_chain ./tests/radar_chain
COPY tests/science ./tests/science

RUN cargo build --locked -p echoforge-gpu-doctor --release

FROM nvidia/cuda:12.6.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
  ECHOFORGE_DOCTOR_OUTPUT=/tmp/gpu-doctor.json

WORKDIR /workspace

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=rust-build /workspace/target/release/gpu-doctor /usr/local/bin/gpu-doctor

CMD ["gpu-doctor"]
