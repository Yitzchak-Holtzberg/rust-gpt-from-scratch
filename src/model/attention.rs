// =============================================================================
// src/model/attention.rs — Causal multi-head self-attention
// =============================================================================
//
// WHAT THIS FILE IS
//   The core mechanism of every transformer. Self-attention lets each token
//   "look at" every other token (that came before it) and decide how much
//   each one matters for predicting the next token.
//
// THE BIG IDEA
//   For each token position, we compute three vectors:
//     Q (Query)  — "what am I looking for?"
//     K (Key)    — "what do I contain?"
//     V (Value)  — "what should I pass on if selected?"
//
//   Attention score between position i and j = dot(Q_i, K_j) / sqrt(head_dim)
//   We then softmax those scores (after masking the future) to get weights,
//   and take a weighted sum of V vectors. This is the output for position i.
//
// WHY MULTIPLE HEADS?
//   Different heads learn to attend to different kinds of relationships:
//   one head might track syntactic dependencies, another coreference, etc.
//   Each head gets embed_dim / num_heads dimensions to work with.
//
// WHY SCALE BY sqrt(head_dim)?
//   Dot products grow in magnitude with dimension. Without scaling, softmax
//   saturates (one weight → 1.0, rest → 0.0), killing gradient flow.
//
// CAUSAL MASK
//   We add -inf to all positions j > i before softmax. After exp(-inf) = 0,
//   position i only ever attends to positions <= i. This is what makes the
//   model autoregressive — it can only see the past.
//
// COMPLEXITY
//   O(T² × D) per layer. This is why context length is expensive.
//   Flash Attention (a stretch goal) reduces memory to O(T) by tiling.
//   Paper: https://arxiv.org/abs/1706.03762 (original transformer)
//   Paper: https://arxiv.org/abs/2205.14135 (Flash Attention)

use crate::tensor::Tensor;
use crate::model::config::Config;

/// Weights for one causal multi-head self-attention block.
pub struct AttentionWeights {
    pub w_qkv: Tensor, // [embed_dim, 3 * embed_dim] — fused Q, K, V projection
    pub b_qkv: Tensor, // [3 * embed_dim]
    pub w_proj: Tensor, // [embed_dim, embed_dim] — output projection
    pub b_proj: Tensor, // [embed_dim]
}

impl AttentionWeights {
    pub fn zeros(cfg: &Config) -> Self {
        let d = cfg.embed_dim;
        AttentionWeights {
            w_qkv: Tensor::zeros(vec![d, 3 * d]),
            b_qkv: Tensor::zeros(vec![3 * d]),
            w_proj: Tensor::zeros(vec![d, d]),
            b_proj: Tensor::zeros(vec![d]),
        }
    }
}

/// Forward pass for causal multi-head self-attention.
/// input: [T, D]  returns: [T, D]
pub fn attention_forward(x: &Tensor, w: &AttentionWeights, cfg: &Config) -> Tensor {
    let t = x.shape[0];
    let d = cfg.embed_dim;
    let h = cfg.num_heads;
    let dh = cfg.head_dim();

    // Step 1: Project input to Q, K, V all at once (fused for efficiency).
    // [T, D] × [D, 3D] → [T, 3D]
    let qkv = x.matmul(&w.w_qkv).add(&broadcast_add_row(&w.b_qkv, t));

    // Step 2: Split the [T, 3D] tensor into three [T, D] tensors.
    let q = slice_cols(&qkv, 0, d);
    let k = slice_cols(&qkv, d, 2 * d);
    let v = slice_cols(&qkv, 2 * d, 3 * d);

    // Step 3: Reshape [T, D] → [H, T, Dh] so each head gets its own slice.
    let q = reshape_heads(&q, h, dh);
    let k = reshape_heads(&k, h, dh);
    let v = reshape_heads(&v, h, dh);

    let scale = (dh as f32).sqrt().recip(); // 1 / sqrt(head_dim)
    let mask = Tensor::causal_mask(t); // [T, T] upper-triangle of -inf

    // Step 4: Run scaled dot-product attention for each head independently.
    let mut head_outputs: Vec<Tensor> = Vec::with_capacity(h);
    for hi in 0..h {
        let qi = head_slice(&q, hi, t, dh); // [T, Dh]
        let ki = head_slice(&k, hi, t, dh); // [T, Dh]
        let vi = head_slice(&v, hi, t, dh); // [T, Dh]

        // Attention scores: [T, Dh] × [Dh, T] → [T, T]
        let scores = qi.matmul(&ki.transpose_2d()).scale(scale).add(&mask);
        let attn = scores.softmax(); // [T, T] — rows sum to 1

        // Weighted sum of values: [T, T] × [T, Dh] → [T, Dh]
        head_outputs.push(attn.matmul(&vi));
    }

    // Step 5: Concatenate all heads back to [T, D], then apply output projection.
    let concat = concat_heads(&head_outputs, t, d);
    concat.matmul(&w.w_proj).add(&broadcast_add_row(&w.b_proj, t))
}

// ---- helpers ----------------------------------------------------------------

fn broadcast_add_row(bias: &Tensor, t: usize) -> Tensor {
    let d = bias.numel();
    let mut data = Vec::with_capacity(t * d);
    for _ in 0..t {
        data.extend_from_slice(&bias.data);
    }
    Tensor::new(data, vec![t, d])
}

fn slice_cols(x: &Tensor, start: usize, end: usize) -> Tensor {
    let t = x.shape[0];
    let d_full = x.shape[1];
    let d_out = end - start;
    let mut data = Vec::with_capacity(t * d_out);
    for row in 0..t {
        data.extend_from_slice(&x.data[row * d_full + start..row * d_full + end]);
    }
    Tensor::new(data, vec![t, d_out])
}

fn reshape_heads(x: &Tensor, h: usize, dh: usize) -> Tensor {
    // [T, H*Dh] → [H, T, Dh]: interleave head dimensions
    let t = x.shape[0];
    let mut data = vec![0.0f32; h * t * dh];
    for ti in 0..t {
        for hi in 0..h {
            for di in 0..dh {
                data[hi * t * dh + ti * dh + di] = x.data[ti * h * dh + hi * dh + di];
            }
        }
    }
    Tensor::new(data, vec![h, t, dh])
}

fn head_slice(x: &Tensor, hi: usize, t: usize, dh: usize) -> Tensor {
    let offset = hi * t * dh;
    Tensor::new(x.data[offset..offset + t * dh].to_vec(), vec![t, dh])
}

fn concat_heads(heads: &[Tensor], t: usize, d: usize) -> Tensor {
    // Inverse of reshape_heads: [H, T, Dh] → [T, D]
    let h = heads.len();
    let dh = d / h;
    let mut data = vec![0.0f32; t * d];
    for (hi, head) in heads.iter().enumerate() {
        for ti in 0..t {
            for di in 0..dh {
                data[ti * d + hi * dh + di] = head.data[ti * dh + di];
            }
        }
    }
    Tensor::new(data, vec![t, d])
}
