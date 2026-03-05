# nano-gpt-rs

A ~2M-parameter GPT built from scratch in pure Rust — no Python, no PyTorch,
no magic frameworks. Every operation is a loop over `Vec<f32>`.

---

## What this project is

This is an educational implementation of a decoder-only transformer (the
architecture behind GPT-2, GPT-3, LLaMA, etc.). The goal is to understand
exactly how these models work by building every piece by hand.

After completing all phases, you will have:
- A model that can generate plausible text character by character
- A deep understanding of why every design choice in the architecture exists
- Direct experience with the pathologies that make training hard

---

## Architecture

| Hyperparameter | Value | Why |
|---|---|---|
| Layers | 2 | Depth gives the model hierarchy; 2 is enough to learn |
| Heads | 2 | Multiple "perspectives" in attention; must divide embed_dim |
| Embedding dim | 128 | Width of every representation vector |
| MLP ratio | 4x | Hidden MLP layer is 4x wider — standard since 2017 |
| Context length | 256 | Max tokens the model sees at once; memory scales as T^2 |
| Tokenizer | Character-level | ~65-token vocab from Tiny Shakespeare |

Approximate parameter count: ~840K-2M depending on vocab size.

---

## How a transformer works — concept by concept

Read these in order. Each section maps to a file in this repo.

### 1. Tensors (`src/tensor/mod.rs`)

A tensor is just a multi-dimensional array of floating point numbers with a
shape that says how those numbers are arranged. A `[4, 128]` tensor is 4 rows
of 128 values each.

Every operation in a neural network is a mathematical operation on tensors:
matrix multiplication, element-wise addition, softmax, etc. This file
implements all of them from scratch.

**The critical op: matrix multiplication** (`matmul`)
- Takes `[M, K]` x `[K, N]` -> `[M, N]`
- This is the core of every "linear layer"
- O(M*K*N) — the performance bottleneck of every forward pass

**Softmax**: converts raw scores to probabilities summing to 1. Includes a
numerical stability trick (subtract the max before exp).

**Layer normalization**: normalises activations to mean=0, std=1, then
re-scales with learned parameters. Prevents activations from exploding or
vanishing through deep networks.
Paper: https://arxiv.org/abs/1607.06450

**GELU**: smooth activation function used instead of ReLU. Better gradient
flow in transformers.
Paper: https://arxiv.org/abs/1606.08415

**Causal mask**: fills the upper triangle of the attention score matrix with
-infinity so the model cannot see future tokens. This is what makes generation
possible — the model only knows the past.

---

### 2. Tokenization (`src/tokenizer/mod.rs`)

Neural networks work on integers, not text. Every character maps to an integer
token ID. The token ID indexes into the embedding table to retrieve the
learned vector for that character.

**Character-level**: simplest possible approach. One token per unique character.
Tiny Shakespeare -> ~65 unique characters -> vocab_size = 65.

**The language modelling objective**: given tokens [t0, t1, ..., tn], the
model is trained to predict t_{i+1} given t_i and everything before it. The
dataset consists of overlapping windows of length T from the training corpus,
each shifted by one position to produce (input, target) pairs.

**Alternatives (stretch goals)**:
- Byte Pair Encoding (BPE): iteratively merge frequent character pairs into
  subword tokens. GPT-2 uses ~50k BPE tokens. Sequences are shorter; the
  model sees whole words.
- SentencePiece: similar idea, used in LLaMA and T5.

---

### 3. Self-Attention (`src/model/attention.rs`)

This is the core mechanism of every transformer. It lets each token position
gather information from all previous positions in a data-dependent way.

**The three vectors**:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What information do I contain?"
- **V (Value)**: "What should I pass on if selected?"

Each is a learned linear projection of the input. The attention weight between
position i and position j is:

```
score(i, j) = dot(Q_i, K_j) / sqrt(head_dim)
```

After masking the future positions to -inf and applying softmax, these scores
become weights. The output for position i is a weighted sum of all V vectors.

**Why divide by sqrt(head_dim)?**
Dot products grow with dimension. Without scaling, softmax saturates (one
weight -> 1.0, rest -> 0.0), killing gradient flow.

**Why multiple heads?**
Each head learns to attend to different relationships — one might track
subject-verb dependencies, another coreference, another positional patterns.
They run in parallel on different linear projections of the same input.

**Memory complexity**: O(T^2) per layer. Doubling context length quadruples
memory. This is why long contexts are expensive.

Original paper: https://arxiv.org/abs/1706.03762
Flash Attention (efficient implementation): https://arxiv.org/abs/2205.14135

---

### 4. MLP (`src/model/mlp.rs`)

The second sub-layer in each block. After attention mixes information across
positions, the MLP processes each position independently.

```
x -> Linear(D -> 4D) -> GELU -> Linear(4D -> D)
```

The 4x expansion ("MLP ratio") is a deliberate capacity choice from the
original transformer paper. The wide hidden layer gives the model room to
compute complex per-token transformations.

Most of the model's "factual knowledge" is believed to be stored in the MLP
weights. Attention decides where to look; the MLP decides what to do with it.

---

### 5. Transformer Block (`src/model/block.rs`)

One block = attention + MLP, each with a residual connection and layer norm.

**Residual connections**: instead of replacing x, each sub-layer adds to it:
```
x = x + attention(layernorm(x))
x = x + mlp(layernorm(x))
```
This lets gradients flow directly through the addition during backprop,
bypassing every weight matrix. Without this, deep networks cannot be trained.
Original residual paper: https://arxiv.org/abs/1512.03385

**Pre-LN vs Post-LN**: this code normalises before the sub-layer (Pre-LN),
which GPT-2 introduced. It trains more stably than the original Post-LN.

Stacking blocks: early blocks learn local syntax; later blocks learn broader
semantics. This is why depth matters.

---

### 6. Full GPT Forward Pass (`src/model/gpt.rs`)

```
tokens -> token_embedding + positional_embedding
       -> block_0 -> block_1
       -> final_layernorm
       -> lm_head (linear, D -> vocab_size)
       -> logits [T, vocab_size]
```

**Token embedding**: each token ID is a row index into a `[vocab_size, D]`
matrix. After training, similar tokens end up with similar vectors.

**Positional embedding**: transformers have no built-in sense of order.
A learned `[context_len, D]` table gives each position a unique vector.
Alternative: RoPE (Rotary Position Embeddings) used in LLaMA — encodes
position in the angle of the Q/K vectors rather than by adding a table entry.
Paper: https://arxiv.org/abs/2104.09864

**LM head**: a final linear layer that projects D -> vocab_size, producing one
score per vocabulary token. These scores are "logits". Softmax turns them into
probabilities; argmax picks the most likely token.

**Weight tying**: GPT-2 shares the token embedding matrix with the LM head.
This halves parameters and often improves performance. Not implemented here
but noted as an optimisation.

---

### 7. Loss and Training (`src/training/mod.rs`)

**Cross-entropy loss**: measures how much probability the model puts on the
correct token. If the model is completely random, loss = ln(vocab_size) ~ 4.17
for a 65-character vocab. A well-trained model on Tiny Shakespeare reaches ~1.3.

**Adam optimizer**: adaptive gradient descent. Tracks a running mean of
gradients (momentum) and a running mean of squared gradients (velocity). Uses
their ratio to normalise the effective learning rate per parameter.
Paper: https://arxiv.org/abs/1412.6980

**The missing piece — backpropagation**: the optimizer needs the gradient of
the loss with respect to every parameter. Computing those gradients is the
backward pass. The forward pass and loss are implemented; the backward pass is
the Phase 5 TODO. See `src/training/mod.rs` for two implementation paths:
manual backprop vs. candle autodiff.

**Gradient clipping**: before applying Adam, clip the global gradient norm to
<=1.0. Without this, early steps produce enormous gradients that produce NaN.

---

### 8. Inference (`src/inference/mod.rs`)

Autoregressive generation: run the forward pass, sample the next token from
the last position's logits, append it, repeat.

**Temperature**: divide logits by `t` before softmax. `t < 1` -> sharper
(more greedy). `t > 1` -> flatter (more random). `t = 0` -> argmax.

**Top-k**: keep only the k highest-probability tokens and renormalise. Prevents
the model from ever picking garbage low-probability tokens.

**KV cache** (stretch goal): during generation, the keys and values for past
positions never change. Caching them turns each new step from O(T^2) to O(T).
Without this, generating 500 tokens requires 500 complete forward passes.

Further reading: https://arxiv.org/abs/1904.09751

---

## Project structure

```
src/
  lib.rs                    — Module declarations
  tensor/
    mod.rs                  — Tensor type and all math operations
  model/
    config.rs               — Hyperparameter struct
    attention.rs            — Causal multi-head self-attention
    mlp.rs                  — Feed-forward MLP block
    block.rs                — Transformer block (pre-norm + residuals)
    gpt.rs                  — Full GPT forward pass
  tokenizer/
    mod.rs                  — Character tokenizer and sliding-window dataset
  training/
    mod.rs                  — Cross-entropy loss, Adam optimizer (grads: TODO)
  inference/
    mod.rs                  — Top-k temperature sampling, generation loop
  bin/
    train.rs                — Training entry point
    generate.rs             — Generation entry point
data/
  README.md                 — How to get training data
```

---

## Quickstart

```bash
# 1. Get training data (Tiny Shakespeare, ~1MB)
curl -o data/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# 2. Build and run the forward pass / loss sanity check
cargo run --release --bin train -- data/input.txt

# Expected output:
#   Vocab size: 65
#   Dataset samples: 1115094
#   Model parameters: ~840000
#   Initial loss (random weights): 4.17xx
#   Expected loss for random init: 4.1744  <- these should match

# 3. Generation (garbage output until training is implemented)
cargo run --release --bin generate -- "To be or not"
```

---

## Implementation status

| Phase | Component | Status |
|---|---|---|
| 1 | Tensor engine (matmul, softmax, layer_norm, gelu) | Done |
| 2 | Causal single-head attention | Done |
| 3 | Multi-head attention + residuals | Done |
| 4 | Full GPT forward pass | Done |
| 4 | Character tokenizer + dataset | Done |
| 5 | Cross-entropy loss | Done |
| 5 | Adam optimizer state | Done |
| **5** | **Backpropagation** | **TODO** |
| **5** | **Training loop** | **TODO** |
| 6 | Top-k temperature sampling | Done |
| 6 | Autoregressive generation loop | Done |
| 6 | Weight save/load | TODO |
| Stretch | KV cache | TODO |
| Stretch | RoPE positional embeddings | TODO |
| Stretch | Weight tying (embed <-> lm_head) | TODO |
| Stretch | Gradient clipping | TODO |

---

## Common failure modes

| Symptom | Likely cause |
|---|---|
| Loss stays at ln(vocab_size) forever | Gradients not flowing — optimizer not stepping |
| Loss explodes to NaN | Learning rate too high; add gradient clipping |
| Loss decreases then plateaus early | LR too low, or too little data |
| Shape assertion panics | embed_dim % num_heads != 0 — check config |
| Generated text is single character repeated | Temperature too low (greedy), or model not trained |
| Generated text is pure noise | Temperature too high, or model not trained |

---

## Key papers

- **Attention Is All You Need** (the original transformer):
  https://arxiv.org/abs/1706.03762

- **GPT-2** (language models are unsupervised multitask learners):
  https://openai.com/research/language-unsupervised

- **Layer Normalization**:
  https://arxiv.org/abs/1607.06450

- **GELU activation**:
  https://arxiv.org/abs/1606.08415

- **Adam optimizer**:
  https://arxiv.org/abs/1412.6980

- **Residual connections** (ResNets):
  https://arxiv.org/abs/1512.03385

- **Flash Attention** (efficient O(T) memory attention):
  https://arxiv.org/abs/2205.14135

- **RoPE positional embeddings** (used in LLaMA):
  https://arxiv.org/abs/2104.09864

- **Sampling strategies** (temperature, top-k, top-p):
  https://arxiv.org/abs/1904.09751

---

## Recommended resources

- Andrej Karpathy's nanoGPT (the Python version this is inspired by):
  https://github.com/karpathy/nanoGPT

- Karpathy's "Let's build GPT from scratch" video (3 hours, very thorough):
  https://www.youtube.com/watch?v=kCc8FmEb1nY

- "The Illustrated Transformer" (visual explanation of every component):
  https://jalammar.github.io/illustrated-transformer/

- candle (Rust ML framework, useful for Option B backprop):
  https://github.com/huggingface/candle
