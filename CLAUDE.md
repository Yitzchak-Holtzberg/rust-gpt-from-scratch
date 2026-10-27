# CLAUDE.md

This branch is a learning project: a GPT and the minimum autograd machinery needed to train it, written in ordinary Rust without an ML framework.

## Build and test

```bash
cargo build
cargo test
cargo test tensor
cargo test tokenizer
cargo test training
cargo run --release --bin train -- data/input.txt
cargo run --release --bin generate -- "prompt"
```

## Current architecture

- `src/tensor.rs` — hand-written row-major `Tensor` backed by `Vec<f32>`
- `src/tokenizer/` — character tokenizer + sliding-window dataset
- `src/model/` — 2-layer, 2-head, 128-wide GPT forward implementation
- `src/training.rs` — cross-entropy, several local backward helpers, hand-written Adam
- `src/inference.rs` — temperature/top-k sampling and autoregressive generation

The current missing capability is end-to-end reverse-mode automatic differentiation. `Tensor::matmul_backward` is still unfinished and is the intended next step.

## Collaboration rules

The primary goal is understanding, not velocity.

- Do **not** introduce Candle, PyTorch, burn, tch, or another tensor/autograd framework.
- Do **not** modernize the architecture, tokenizer, optimizer, or infrastructure unless explicitly requested.
- Do **not** add CUDA, BPE, a web UI, a server, distributed training, or model scaling as incidental improvements.
- Keep changes narrowly scoped to the current learning milestone in README.md.
- Prefer one concept and one small commit at a time.
- Preserve the hand-written tensor implementation and simple free-function model style.
- Avoid unrelated cleanup while a mathematical concept is being learned.

## Teaching style

The user wants to implement and understand the important pieces personally.

By default:

1. Explain the invariant and shape flow first.
2. Give a tiny numerical example.
3. Suggest the test that would prove correctness.
4. Give Rust syntax hints if useful.
5. Do **not** provide the complete implementation unless the user explicitly asks for it.

If the user explicitly asks to implement a step, implement only that step and explain what changed.

For gradient code, prefer finite-difference checks over trusting algebra or generated code.

## Immediate milestone

Finish and verify `Tensor::matmul_backward` only. Do not jump directly to `attention_backward`, `gpt_backward`, or a full training loop. Once primitive derivatives are verified, the next design problem is a small reverse-mode computation graph so layer-level backward passes emerge from composition rather than bespoke giant backward functions.
