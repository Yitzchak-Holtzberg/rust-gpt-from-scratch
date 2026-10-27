# CLAUDE.md

Guidance for coding agents working in this repository.

## Build and test

```bash
cargo build
cargo build --release
cargo test
cargo run --release --bin train -- data/input.txt 5000 64
cargo run --release --bin generate -- "ROMEO:" 200
cargo run --release --bin serve
cargo run --release --features cuda --bin serve
```

## What this repository is

This is an educational GPT project in Rust. It began with a hand-written tensor engine and model implementation; that implementation remains in git history through commit `1ebcd6d`. Commit `ff9148c` migrated tensor/autograd work to Candle so end-to-end training could be completed.

Do not describe the current code as “no ML framework” or as a 2M-parameter model.

## Current architecture

`Config::nano()` in `src/gpt.rs` is roughly 11M parameters with BPE(1024):

- 6 transformer blocks
- 6 attention heads
- embedding width 384
- context 256
- pre-norm RMSNorm
- RoPE position encoding
- SwiGLU MLP
- tied token embedding / LM-head weights
- dropout 0.1
- Candle tensors + autograd

## Two tokenizer/training paths

Keep these distinct:

- **Server:** `BpeTokenizer`, 1024-token byte-level BPE, live dashboard, checkpoint support, optional CUDA.
- **CLI/WASM:** `CharTokenizer`, character-level artifacts (`vocab.json`).

Do not assume server BPE checkpoints are interchangeable with the character-tokenizer WASM path.

## Server layout

- `src/server.rs` — runtime, router, device selection
- `src/server/state.rs` — shared state/status
- `src/server/handlers.rs` — endpoints, model/tokenizer loading, checkpoints
- `src/server/training.rs` — optimizer loop, LR schedule, gradient clipping
- `src/server/monitoring.rs` — generation probes, token probabilities, GPU stats

The current split is intentional; avoid collapsing these concerns back into the binary.

## Training caveats

- A `/api/train` request currently creates a fresh AdamW optimizer. Model weights persist, optimizer moments do not.
- Checkpoints do not persist optimizer state.
- There is currently no validation split.
- The original short runs were badly undertrained relative to Karpathy's Shakespeare baseline. Do not diagnose sample quality from a few hundred steps.
- When debugging training, change one variable at a time and prefer a fixed-seed character-level baseline before BPE or architectural experiments.

Karpathy's reference Shakespeare-char recipe uses 6 layers, 6 heads, width 384, context 256, batch 64, 5000 iterations, LR 1e-3, beta2 0.99, warmup 100, and dropout 0.2.

## Historical hand-rolled implementation

The custom tensor engine was substantive learning work, not abandoned because it was broken. It implemented tensor operations, forward model components, loss, generation, and primitive backward derivatives. The project switched to Candle before implementing the complete layer-level backward graph.

Preserve that history in documentation and avoid deleting historical notes simply because the current implementation uses Candle.
