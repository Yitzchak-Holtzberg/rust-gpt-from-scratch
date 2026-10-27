# rust-gpt-from-scratch

An educational GPT implementation in Rust, trained on Tiny Shakespeare.

The part of this project that mattered to the original exercise was genuinely built from scratch: a hand-written tensor type, matrix operations, attention, MLPs, layer norm, loss, sampling, and local backward derivatives for the tensor operations. The project reached a complete forward/inference pipeline before switching to [Candle](https://github.com/huggingface/candle) for tensors and autograd.

That switch is also the natural boundary between two different projects in practice. The hand-rolled implementation was the learning exercise. After the Candle migration, later model tuning, BPE/CUDA work, dashboard work, and architecture experiments were largely agent-driven continuation rather than the part the author was personally trying to understand or optimize.

The last pre-Candle version is [`1ebcd6d`](https://github.com/Yitzchak-Holtzberg/rust-gpt-from-scratch/tree/1ebcd6dfe93d835dbb29c28e879398d66cc4f51c); the Candle migration is [`ff9148c`](https://github.com/Yitzchak-Holtzberg/rust-gpt-from-scratch/commit/ff9148ceb66dbed0e8d4ec4fe170fe35bd8f6ad3).

## What was actually built from scratch?

The hand-rolled phase implemented:

- a custom `Tensor` type and tensor operations
- matrix multiplication
- slicing and concatenation
- softmax, GELU, and layer normalization
- token and position embeddings
- causal multi-head self-attention
- transformer blocks and a GPT forward pass
- cross-entropy loss
- temperature/top-k sampling and autoregressive generation
- backward derivatives for the primitive tensor operations and loss

The stopping point was not a broken tensor implementation. The missing piece was the full layer-level backward graph through attention, MLPs, residual paths, normalization, embeddings, and every parameter. At that point the project switched to Candle rather than continuing toward a home-grown autograd engine.

The old implementation remains in git history and is arguably the most interesting part of the repository if the goal is to understand how a Transformer actually works below the framework layer.

## Current model

The repository later evolved into a more conventional Candle-based training playground. `Config::nano()` is now an approximately 11M-parameter decoder-only Transformer:

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

The server path trains a byte-level BPE tokenizer with a 1,024-token vocabulary, giving roughly 11M parameters. The CLI and WASM path retain the simpler character tokenizer.

## Why were the generated results so weak?

There are two different answers depending on which phase is being discussed.

### Hand-rolled phase

The hand-rolled model never had complete end-to-end training. It could tokenize, run a full Transformer forward pass, calculate loss, sample, and decode, but the full parameter-level backward chain was unfinished. Generated text from random or effectively untrained weights was therefore expected to be poor.

### Later Candle/agent phase

After the framework migration, the repository did gain real autograd training, but the early training setups were much weaker than the nanoGPT Shakespeare reference recipe.

The first Candle trainer defaulted to 500 steps with one 256-token sample per step: about **128,000 token predictions**.

The next version used batch size 16 but still defaulted to 500 steps: about **2.05 million token predictions**.

Karpathy's published nanoGPT Shakespeare character preset uses 5,000 steps × batch 64 × context 256: about **81.9 million token predictions**. See [`config/train_shakespeare_char.py`](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py).

So the later agent-driven experiments spent substantial time tuning a setup that was initially undertrained by a very large margin. They also changed several variables together—model size, batch size, learning rate, optimizer settings, tokenizer, CUDA, and eventually architecture—which made the results difficult to diagnose cleanly.

One training-loop rough edge remains today: every `/api/train` request creates a fresh AdamW optimizer. Model weights survive between runs, but AdamW's moment state does not, and the learning-rate schedule restarts. Multiple short UI runs therefore are not equivalent to one continuous run.

There is also no held-out validation split, so the dashboard cannot cleanly distinguish undertraining from memorization or overfitting.

These are observations about the later repository state, not a reconstruction of a deliberate personal training program.

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
  gpt.rs                 current Candle model, RoPE, attention, SwiGLU, sampling
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

## Historical note

If the repository is being read as a learning project, start with the pre-Candle history rather than the current `main` tree. The current code is cleaner and more capable, but it hides most of the machinery that motivated the project in the first place.

Useful historical commits:

- [`9148f85`](https://github.com/Yitzchak-Holtzberg/rust-gpt-from-scratch/commit/9148f85b597b5fb4a2a3e47ebffe904735c27d05) — custom Tensor methods and tests
- [`dea8892`](https://github.com/Yitzchak-Holtzberg/rust-gpt-from-scratch/commit/dea8892e227106793a6eca1da167cedb7a9b1738) — model components, tensor ops, sampling/generation
- [`e937210`](https://github.com/Yitzchak-Holtzberg/rust-gpt-from-scratch/commit/e9372109acb2f38232ee833fe80d0666284133ec) — tensor backward methods and cross-entropy backward
- [`1ebcd6d`](https://github.com/Yitzchak-Holtzberg/rust-gpt-from-scratch/commit/1ebcd6dfe93d835dbb29c28e879398d66cc4f51c) — complete hand-rolled forward/inference pipeline
- [`ff9148c`](https://github.com/Yitzchak-Holtzberg/rust-gpt-from-scratch/commit/ff9148ceb66dbed0e8d4ec4fe170fe35bd8f6ad3) — migration to Candle

## Known rough edges in the current implementation

- Server training recreates AdamW state for each training request.
- Checkpoints persist model/tokenizer/dataset, not optimizer state.
- CLI and server use different tokenizers, so their losses and step counts are not directly comparable.
- There is no train/validation split or reproducible random seed yet.
- The WASM path is character-tokenizer based and separate from the BPE server checkpoint format.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Candle](https://github.com/huggingface/candle)
- [Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
