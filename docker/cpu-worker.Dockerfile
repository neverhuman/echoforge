FROM node:26-bookworm-slim AS web-build

ENV DEBIAN_FRONTEND=noninteractive \
  NODE_ENV=development

WORKDIR /workspace

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
COPY apps ./apps
RUN npm ci --no-fund --no-audit
RUN npm run web:build

FROM rust:1.85-bookworm AS rust-build

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY contracts ./contracts
COPY crates ./crates
COPY schemas ./schemas
COPY tests/contracts ./tests/contracts
COPY tests/gpu ./tests/gpu
COPY tests/radar_chain ./tests/radar_chain
COPY tests/science ./tests/science

RUN cargo build --locked -p echoforge-studio --release

FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
  HOST=0.0.0.0 \
  PORT=8080 \
  ECHOFORGE_PUBLIC_BASE_URL=/ \
  ECHOFORGE_CATALOG_PATH=/app/contracts/schema_catalog.json \
  ECHOFORGE_BUNDLE_PATH=/app/tests/science/fixtures/bundles/v1_pass \
  ECHOFORGE_WEB_DIST=/app/apps/web/dist

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/* \
  && groupadd --system nonroot \
  && useradd --system --gid nonroot --create-home nonroot

COPY --from=web-build /workspace/apps/web/dist ./apps/web/dist
COPY --from=rust-build /workspace/target/release/echoforge-studio /usr/local/bin/echoforge-studio
COPY contracts ./contracts
COPY tests/science/fixtures/bundles/v1_pass ./tests/science/fixtures/bundles/v1_pass

EXPOSE 8080

USER nonroot
CMD ["echoforge-studio"]
