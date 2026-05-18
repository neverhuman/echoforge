FROM node:20-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
  NODE_ENV=development

WORKDIR /workspace

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY apps ./apps
COPY benchmarks ./benchmarks
COPY examples ./examples
COPY tests ./tests

EXPOSE 3000 8080

CMD ["node", "apps/api/server.mjs"]

