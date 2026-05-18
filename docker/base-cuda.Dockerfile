FROM nvidia/cuda:12.6.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates nodejs npm \
  && rm -rf /var/lib/apt/lists/*

