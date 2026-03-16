# rust-gpt-from-scratch

## Where to start

Work through the phases in order. Each phase builds on the previous one.
The file to open is listed — everything else is already stubbed out.

---

### Phase 1 — Tensor engine ✅ COMPLETE
**File:** `src/tensor/mod.rs`

All methods implemented and tested: `new`, `zeros`, `ones`, `reshape`,
`transpose_2d`, `add`, `mul`, `scale`, `matmul`, `softmax`, `layer_norm`,
`gelu`, `causal_mask`.

All tests in `tests/tensor_tests.rs` passing.

**Read:**
- Layer norm paper: https://arxiv.org/abs/1607.06450
- GELU paper: https://arxiv.org/abs/1606.08415
- Visual intro to matrix multiply: https://matrixmultiplication.xyz

---

### Phase 2 — Tokenizer and dataset 🔄 IN PROGRESS
**File:** `src/tokenizer/mod.rs`

Tests written in `tests/tokenizer_tests.rs`. Implement:
- [x] Tests written (TDD — tests exist, implementations are `todo!()`)
- [ ] `CharTokenizer::from_text` — collect unique chars, sort, build `HashMap`
- [ ] `vocab_size` — return `self.vocab.len()`
- [ ] `encode` — map each char to its ID, drop unknowns
- [ ] `decode` — map each ID back to char, join into String
- [ ] `Dataset::new`, `num_samples`, `get` — sliding window over token IDs

The dataset returns sliding windows of (input, target) token pairs
where target is input shifted right by one position.

**Get training data:**
```
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

**Check your work:** encode a short string, decode it back, assert it matches.

---

### Phase 3 — Attention
**File:** `src/model/attention.rs`

Implement `AttentionWeights::zeros` and `attention_forward`.

The steps are in the function's doc comment. The shape flow is:
```
[T, D] -> project to [T, 3D] -> split Q/K/V [T, D] each
       -> reshape to [H, T, Dh]
       -> scores = Q @ K.T / sqrt(Dh) + causal_mask  [T, T]
       -> softmax(scores) @ V  [T, Dh]
       -> concat heads [T, D] -> output projection [T, D]
```

**The most common bug:** getting the head reshape wrong. Draw the tensor
dimensions on paper before writing the loop.

**Read:**
- Original transformer paper (section 3.2): https://arxiv.org/abs/1706.03762
- Illustrated Transformer (visual): https://jalammar.github.io/illustrated-transformer/

---

### Phase 4 — MLP, Block, and full forward pass
**Files:** `src/model/mlp.rs`, `src/model/block.rs`, `src/model/gpt.rs`

Do these in order:

1. `mlp_forward` — two matmuls with a GELU in between
2. `block_forward` — call layernorm, then attention, then add residual; repeat for MLP
3. `GptWeights::zeros` and `gpt_forward` — embed tokens + positions, run blocks, final norm, lm_head

**Check your work:** `cargo run --bin train -- data/input.txt`

If the tokenizer and tensor engine are correct, the initial loss should be
close to `ln(vocab_size)` — for 65 chars that's ~4.17. If it's wildly
different, something in the forward pass is wrong.

**Read:**
- Residual connections: https://arxiv.org/abs/1512.03385

---

### Phase 5 — Training
**File:** `src/training/mod.rs`, then wire up `src/bin/train.rs`

Two parts:

**5a. Loss** — implement `cross_entropy_loss`. Use the log-sum-exp trick
(subtract the row max before exp) for numerical stability.

**5b. Optimizer** — implement `Adam::new` and `Adam::step`. The update rule
is in the function's doc comment.

**5c. Backprop** — the hard part. Two options:

- **Manual:** derive and implement `dLoss/dW` for each op working backwards
  from the loss. Start with the gradient of cross-entropy + softmax combined
  (it simplifies cleanly to `softmax(logits) - one_hot(target)`), then work
  back through lm_head, layernorm, MLP, attention, embeddings.

- **candle autodiff:** add `candle-core` to `Cargo.toml`, swap `Tensor` for
  `candle_core::Tensor`, call `.backward()` on the loss, collect `.grad()`
  per parameter, pass to `Adam::step`.
  Docs: https://github.com/huggingface/candle

**5d. Training loop** — in `src/bin/train.rs`: sample batches, forward pass,
loss, backward, step. Log loss every N steps.

**Expected loss curve:** starts at ~4.17, should fall to ~1.3-1.5 after
enough steps on Tiny Shakespeare.

**Common failures:**
| Symptom | Cause |
|---|---|
| Loss stays at 4.17 | Gradients not flowing / optimizer not stepping |
| Loss goes to NaN | LR too high — clip gradients to norm 1.0 |
| Loss falls then plateaus | LR too low or batch too small |

**Read:**
- Adam paper: https://arxiv.org/abs/1412.6980
- Karpathy's nanoGPT (reference implementation in Python): https://github.com/karpathy/nanoGPT
- Karpathy's "build GPT from scratch" video: https://www.youtube.com/watch?v=kCc8FmEb1nY

---

### Phase 6 — Inference
**Files:** `src/inference/mod.rs`, `src/bin/generate.rs`

Implement `sample_token` (temperature scaling + top-k filter + multinomial
sample) and `generate` (the autoregressive loop).

Then wire up `generate.rs` to load saved weights and vocab from disk, encode
the prompt, run `generate()`, and print the result.

**Read:**
- Sampling strategies (temperature, top-k, top-p): https://arxiv.org/abs/1904.09751

---

### Stretch goals
- **KV cache** — cache K and V from past positions so each generation step
  is O(T×D) instead of O(T²×D)
- **Gradient clipping** — clip global gradient norm to 1.0 before optimizer step
- **Weight tying** — share `tok_embed` and `lm_head` weights (halves params)
- **RoPE** — replace learned positional embeddings with rotary encodings:
  https://arxiv.org/abs/2104.09864

---

## Architecture

| Hyperparameter | Value |
|---|---|
| Layers | 2 |
| Heads | 2 |
| Embedding dim | 128 |
| MLP ratio | 4x |
| Context length | 256 |
| Tokenizer | Character-level |

---

## Project structure

```
src/
  tensor/mod.rs       Phase 1 — all math ops
  tokenizer/mod.rs    Phase 2 — char tokenizer, dataset
  model/
    config.rs         Hyperparameters (already filled in)
    attention.rs      Phase 3 — multi-head self-attention
    mlp.rs            Phase 4 — feed-forward block
    block.rs          Phase 4 — transformer block
    gpt.rs            Phase 4 — full forward pass
  training/mod.rs     Phase 5 — loss, optimizer, backprop
  inference/mod.rs    Phase 6 — sampling, generation loop
  bin/
    train.rs          Phase 5 — training entry point
    generate.rs       Phase 6 — generation entry point
data/
  README.md           How to get training data
```
