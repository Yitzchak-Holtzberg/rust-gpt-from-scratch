# rust-gpt-from-scratch

An educational GPT implementation in Rust, trained on Tiny Shakespeare.

This project started genuinely from scratch: a hand-written tensor type, matrix operations, attention, MLPs, layer norm, loss, sampling, and the local backward derivatives for the tensor operations. The final step — wiring a complete Transformer backward graph through attention, residuals, normalization, embeddings, and every parameter — was where the project deliberately switched to [Candle](https://github.com/huggingface/candle) for autograd.

The hand-rolled implementation is still preserved in git history. The last pre-Candle version is [`1ebcd6d`](https://github.com/Yitzchak-Holtzberg/rust-gpt-from-scratch/tree/1ebcd6dfe93d835dbb29c28e879398d66cc4f51c); the Candle migration is [`ff9148c`](https://github.com/Yitzchak-Holtzberg/rust-gpt-from-scratch/commit/ff9148ceb66dbed0e8d4ec4fe170fe35bd8f6ad3).

## Current model

`Config::nano()` is an approximately 11M-parameter decoder-only Transformer:

| Setting | Value |
| --- | ---: |
| Layers | 6 |
| Attention heads | 6 |
| Embedding width | 384 |
| Context length | 256 |
| MLP | SwiGLU, hidden size 1024 |
| Normalization | RMSNorm, pre-norm |
| Position encoding | RoPE |
| Dropout | 0.1 |
| LM head | tied to token embeddings |

The server path trains a byte-level BPE tokenizer with a 1,024-token vocabulary, which gives about 11.0M parameters. The CLI and WASM path retain the simpler character tokenizer.

## What is actually “from scratch”?

The project has two eras:

1. **Hand-rolled engine** — custom `Tensor`, matmul, softmax, GELU, layer norm, causal attention, model forward pass, cross-entropy, sampling, and tensor-level backward operations. This is the part written to learn the mechanics.
2. **Candle era** — the model architecture, tokenizer plumbing, training loop, sampling, server, checkpointing, and UI remain project code, while Candle owns tensors and automatic differentiation.

Switching to Candle was not a failure of the tensor implementation. The old code reached a complete forward/inference pipeline and implemented the primitive backward derivatives; it stopped before the much larger layer-level backward chain required for actual end-to-end training.

## Why the early generations were bad

There were several distinct experiments, and they should not be conflated.

### 1. The hand-rolled model never had full end-to-end backprop

The pre-Candle executable could run tokenize → forward → loss → sample → decode, but the weights were still random because `attention_backward`, `block_backward`, `gpt_backward`, and the full parameter update chain did not exist yet. Gibberish from that version was expected.

### 2. The first Candle run was drastically undertrained

The initial Candle trainer defaulted to 500 steps with a single 256-token sample per step: roughly **128,000 token predictions**.

The next version added batching, but the UI still defaulted to 500 steps × batch 16 × context 256: roughly **2.05 million token predictions**.

For comparison, Karpathy's published nanoGPT Shakespeare character preset uses 5,000 steps × batch 64 × context 256: roughly **81.9 million token predictions** — about 40× the training signal of that second setup. See [`config/train_shakespeare_char.py`](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py).

That difference alone is enough to explain a large amount of the “Shakespeare-flavored word salad.”

### 3. Too many variables changed at once

The debugging path then changed model size, batch size, learning rate, AdamW settings, CUDA, tokenizer type, and eventually the architecture itself. The current model uses RoPE + RMSNorm + SwiGLU + weight tying, while the earlier baseline used learned position embeddings + LayerNorm + GELU.

Those are reasonable experiments, but once several change together, a bad sample no longer tells you which change helped or hurt.

### 4. Repeated UI training runs are not true continuous training

Each `/api/train` request currently constructs a fresh AdamW optimizer. The model weights survive between runs, but AdamW's first- and second-moment state does not. The learning-rate schedule also restarts on every request.

That means ten short runs are **not equivalent** to one long run. For serious training, prefer one sufficiently long run until optimizer state is made persistent/checkpointable.

### 5. There is no validation split

The project reports training loss and generated samples, but not held-out validation loss. On a corpus this small, that makes it hard to distinguish undertraining, successful learning, and pure memorization. Adding a deterministic train/validation split is the most useful next measurement improvement.

## A sane debugging baseline

Before changing architecture again, establish one boring, reproducible baseline.

Karpathy's Shakespeare character preset is:

```text
layers        6
heads         6
embedding     384
context       256
batch         64
steps         5000
learning rate 1e-3
beta2         0.99
warmup        100
min LR        1e-4
dropout       0.2
```

The current model is not an exact nanoGPT reproduction, so this is a training reference rather than a claim of apples-to-apples equivalence. The clean experiment is: character tokenizer first, fixed seed, train/validation loss, one uninterrupted run, then change one thing at a time.

## Quick start

Download Tiny Shakespeare:

```bash
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

Run tests:

```bash
cargo test
```

### CLI: character-level training

```bash
cargo run --release --bin train -- data/input.txt 5000 64
cargo run --release --bin generate -- "ROMEO:" 200
```

The CLI runs on CPU and saves `weights.safetensors` plus `vocab.json`.

### Local training dashboard

CPU:

```bash
cargo run --release --bin serve
```

CUDA:

```bash
cargo run --release --features cuda --bin serve
```

Then open `http://127.0.0.1:9000/`.

The server path uses the BPE tokenizer, supports live loss/generation/probability monitoring, and can save/load model checkpoints.

## Repository layout

```text
src/
  gpt.rs                 model, RoPE, attention, SwiGLU, sampling
  tokenizer.rs           tokenizer module surface
  tokenizer/
    tokenizer.rs         character tokenizer
    bpe.rs               byte-level BPE tokenizer
    dataset.rs           sliding-window dataset
  server.rs              Axum server/router
  server/
    state.rs             shared model/training state
    handlers.rs          HTTP endpoints + checkpoint I/O
    training.rs          AdamW loop, LR schedule, grad clipping
    monitoring.rs        samples, token probabilities, GPU stats
  bin/
    train.rs             character-level CLI trainer
    generate.rs          character-level CLI inference
    serve.rs             server entry point

wasm-inference/           browser inference wrapper
web/                      local dashboard + design mockups
docs/lesson-attention.md  attention notes from the hand-rolled phase
```

## Known rough edges

- Server training recreates AdamW state for each training request.
- Checkpoints persist model/tokenizer/dataset, not optimizer state.
- CLI and server intentionally use different tokenizers, which makes their losses and step counts non-comparable.
- There is no train/validation split or reproducible random seed yet.
- The WASM path is character-tokenizer based and is separate from the BPE server checkpoint format.

Those are more important to fix than further architecture ornamentation.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Candle](https://github.com/huggingface/candle)
- [Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
