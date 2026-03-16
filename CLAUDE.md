# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build                    # Debug build
cargo build --release          # Optimized build (opt-level 3)
cargo test                     # Run all tests
cargo test tensor              # Run tensor tests only
cargo test tokenizer           # Run tokenizer tests only
cargo test <test_name>         # Run a single test by name
cargo run --release --bin train -- data/input.txt   # Train on text file
cargo run --release --bin generate -- "prompt"       # Generate text
```

## Architecture

This is a 2M-parameter GPT built from scratch in pure Rust with no ML framework. The codebase follows a 6-phase incremental build (see README.md for full roadmap):

1. **Tensor engine** (`src/tensor/mod.rs`) — Flat `Vec<f32>` with `Vec<usize>` shape. Row-major layout. All ops return new tensors (immutable style). No automatic broadcasting. Panics on shape mismatches.

2. **Tokenizer & Dataset** (`src/tokenizer/`) — `CharTokenizer` maps characters to integer IDs. `Dataset` produces sliding-window `(input, target)` pairs of length `context_len`.

3. **Model** (`src/model/`) — `Config` holds hyperparameters (2 layers, 2 heads, embed_dim=128, context_len=256). Each component (`attention`, `mlp`, `block`, `gpt`) has a `*Weights` struct and a free-function `*_forward` (not a method). Shape flow for attention: `[T,D] → [T,3D] → split Q/K/V → [H,T,Dh] → scores → softmax → [T,D]`.

4. **Training** (`src/training/mod.rs`) — Cross-entropy loss, AdamW optimizer, backpropagation.

5. **Inference** (`src/inference/mod.rs`) — Temperature + top-k sampling, autoregressive generation loop.

Entry points: `src/bin/train.rs` and `src/bin/generate.rs`.

## Key Conventions

- **TDD approach**: Tests are written first (in `tests/`), implementations fill in `todo!()` stubs
- Forward pass functions are module-level free functions, not methods on weight structs
- Weight structs (`AttentionWeights`, `MlpWeights`, etc.) have a `::zeros(config)` constructor
- Dependencies are minimal: `rand`, `serde`, `serde_json` only
- Data files (`.txt`, `.bin`, `.safetensors`) are gitignored; download Tiny Shakespeare per `data/README.md`

## Collaboration Style

The user is learning Rust and LLMs from scratch through this project.
- Break each step into a lesson: explain the concept and *why* it works
- Provide relevant Rust syntax patterns as hints
- Do NOT give the exact solution — guide the user to figure it out themselves

## Signature Conventions (deviations from original stubs)

- `mlp_forward` does not take `cfg` — the weights already encode the dimensions, so `cfg` was redundant. Removed to keep signatures clean.
