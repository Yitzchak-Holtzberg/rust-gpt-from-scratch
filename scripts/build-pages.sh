#!/usr/bin/env bash
# Rebuild the static Cloudflare Pages bundle from the current trained weights.
# Run after training a new model with `cargo run --release --bin train`.
#
# Usage: bash scripts/build-pages.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f weights.safetensors || ! -f vocab.json ]]; then
  echo "Missing weights.safetensors or vocab.json."
  echo "Train a model first: cargo run --release --bin train -- data/input.txt 500 16"
  exit 1
fi

echo "==> Building WASM module..."
(cd wasm-inference && wasm-pack build --target web --release)

echo "==> Staging public/..."
mkdir -p public/pkg
cp wasm-inference/pkg/* public/pkg/
cp weights.safetensors public/
cp vocab.json public/

echo
echo "public/ is ready. Deploy with:"
echo "  npx wrangler pages deploy public/ --project-name=nano-gpt-rs"
