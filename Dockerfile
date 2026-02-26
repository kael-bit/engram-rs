FROM rust:1.84-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/engram /usr/local/bin/engram
EXPOSE 3917
ENV ENGRAM_PORT=3917
ENV ENGRAM_DB=/data/engram.db
VOLUME /data
ENTRYPOINT ["engram"]
