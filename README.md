# rust-gpt-from-scratch — hand-rolled autograd

This branch is the learning version of the project: a GPT built from ordinary Rust data structures and math, with no ML framework doing tensors or automatic differentiation.

It was split from commit `1ebcd6d`, the last pre-Candle state of the original project. The goal here is not to get impressive Shakespeare output as quickly as possible. The goal is to understand, piece by piece, how a Transformer actually trains.

## Current checkpoint

Already working:

- A hand-written row-major `Tensor` backed by `Vec<f32>`
- Core forward operations: reshape, transpose, slicing, concatenation, add/bias, multiply, scale, matmul, softmax, layer norm, GELU, causal masks
- Character tokenizer and sliding-window dataset
- Multi-head causal self-attention
- MLP, Transformer block, and full GPT forward pass
- Cross-entropy loss and a hand-written Adam optimizer
- Temperature/top-k sampling and autoregressive generation
- Several local backward derivatives for tensor operations
- Tests for tensors, model pieces, tokenizer, and training helpers

Not working yet:

- End-to-end training
- Automatic differentiation / computation graph
- Gradient propagation through the whole Transformer

The most concrete unfinished primitive is currently `Tensor::matmul_backward`, which is still a `todo!()`.

## Ground rules

This branch exists to learn the machinery, not to race to a result.

1. **No Candle, PyTorch, burn, tch, or other tensor/autograd framework.** `rand`, `serde`, and ordinary utility crates are fine.
2. **One conceptual step at a time.** Do not combine autograd, architecture changes, tokenization experiments, CUDA, dashboards, or model scaling.
3. **Correctness before speed.** Tiny matrices and slow loops are fine.
4. **Tests before integration.** Every derivative should be checked independently before it is trusted inside the model.
5. **Use numerical gradient checks.** Finite differences are the referee for hand-derived gradients.
6. **Do not use Shakespeare as the first success criterion.** First make tiny functions, then tiny networks, then a tiny Transformer deliberately overfit microscopic data.
7. **No architecture modernization until training works.** Learned position embeddings, LayerNorm, GELU, and the existing 2-layer model stay as-is.

## Roadmap

### Milestone 0 — Establish the baseline

Run:

```bash
cargo test
cargo run --release --bin train -- data/input.txt
```

The training binary is currently only a forward-pass smoke test. It should report an initial loss around `ln(vocab_size)` for random weights.

Do not change anything until the existing tests are understood and green.

### Milestone 1 — Finish primitive derivatives

Start with exactly one function:

```text
Tensor::matmul_backward
```

For `Z = A @ B` and incoming gradient `dZ`, derive and test:

```text
dA = dZ @ B^T
dB = A^T @ dZ
```

Then add a finite-difference gradient check for small matrices. Only after that passes should the other existing backward helpers be audited the same way.

**Stop here after the first successful session.** A small verified step is the intended pace.

### Milestone 2 — Build a tiny reverse-mode autodiff graph

Once the primitive derivatives are trusted, design the missing abstraction:

- which operation produced a tensor
- which parent tensors fed that operation
- accumulated gradient for each value that requires gradients
- reverse topological traversal from a scalar loss

The target API can eventually feel roughly like:

```text
loss.backward()
```

but the representation should be designed deliberately rather than copied wholesale from a framework.

### Milestone 3 — Prove autograd on toy math

Before touching the Transformer, train something embarrassingly small:

- scalar expressions such as `y = x*x + x`
- a linear layer on a few points
- a 2-layer MLP that memorizes a tiny dataset

If a ten-example MLP cannot overfit, the bug is still below the Transformer.

### Milestone 4 — Train one attention unit

Use a synthetic sequence task small enough to inspect by hand. Verify that attention parameters receive gradients and that loss falls.

### Milestone 5 — Reconnect the GPT

Let the existing forward model become a composition of autograd-aware tensor operations. Avoid writing a giant bespoke `gpt_backward`; reverse-mode autodiff should make the layer-level backward pass emerge from the primitive operations.

First target: memorize a microscopic corpus such as a few repeated lines.

### Milestone 6 — Tiny Shakespeare

Only after the model can intentionally overfit tiny data should Tiny Shakespeare become useful. At that point generated text is an integration test, not the primary debugging signal.

## Architecture

The existing learning model is intentionally small:

| Hyperparameter | Value |
| --- | ---: |
| Layers | 2 |
| Heads | 2 |
| Embedding width | 128 |
| MLP ratio | 4× |
| Context length | 256 |
| Tokenizer | character-level |

There is no reason to make it larger while the goal is understanding gradients.

## Project layout

```text
src/
  tensor.rs             hand-written tensor operations + local derivatives
  tokenizer/
    tokenizer.rs        character tokenizer
    dataset.rs          sliding-window dataset
  model/
    config.rs           model dimensions
    attention.rs        multi-head causal attention
    mlp.rs              feed-forward network
    block.rs            Transformer block
    gpt.rs              full GPT forward pass
  training.rs           cross-entropy, backward helper, Adam, training TODO
  inference.rs          sampling and generation
  bin/
    train.rs            current forward/loss smoke test
    generate.rs         inference with random weights

tests/
  tensor_tests.rs
  model_tests.rs
  tokenizer_tests.rs
  training_tests.rs
```

## Useful references

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)
- [micrograd](https://github.com/karpathy/micrograd) — useful as a conceptual reference, not code to copy
- [nanoGPT](https://github.com/karpathy/nanoGPT) — useful later as a behavioral reference

The next task is intentionally narrow: **understand and implement `matmul_backward`, then prove it with finite differences.**
