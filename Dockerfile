# Multi-stage build: compile in builder, run in minimal image
FROM rust:1.85-slim AS builder

WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/

RUN cargo build --release && strip target/release/engram

# Runtime: distroless-like minimal image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/engram /usr/local/bin/engram

ENV ENGRAM_PORT=3917
ENV ENGRAM_DB=/data/engram.db
EXPOSE 3917
VOLUME /data

ENTRYPOINT ["engram"]
