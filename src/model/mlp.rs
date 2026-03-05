// =============================================================================
// src/model/mlp.rs — Feed-forward MLP (position-wise)
// =============================================================================
//
// WHAT THIS FILE IS
//   The second sub-layer inside every transformer block. After attention mixes
//   information across positions, the MLP processes each position independently
//   and in parallel.
//
// THE BIG IDEA
//   Attention decides *where* to gather information from.
//   The MLP decides *what to do* with it.
//
//   Structure: Linear(D → 4D) → GELU → Linear(4D → D)
//
//   The 4× expansion is a deliberate design choice from the original
//   "Attention is All You Need" paper. The wide hidden layer gives the model
//   capacity to learn complex per-token transformations.
//
//   Because it operates per-token (no cross-position interaction), the MLP is
//   easily parallelised and is where most of the model's "factual knowledge"
//   is believed to be stored.
//
// WHY GELU INSTEAD OF RELU?
//   ReLU hard-zeros negative inputs (not differentiable at 0). GELU is smooth,
//   which gives better gradient flow and empirically performs better in
//   transformers. See tensor/mod.rs for the formula.
//
// FURTHER READING
//   "Attention Is All You Need" (Vaswani et al. 2017) §3.3:
//   https://arxiv.org/abs/1706.03762

use crate::tensor::Tensor;
use crate::model::config::Config;

/// Feed-forward MLP weights for one transformer block.
pub struct MlpWeights {
    pub w_fc: Tensor,   // [embed_dim, embed_dim * mlp_ratio] — expand
    pub b_fc: Tensor,   // [embed_dim * mlp_ratio]
    pub w_proj: Tensor, // [embed_dim * mlp_ratio, embed_dim] — project back
    pub b_proj: Tensor, // [embed_dim]
}

impl MlpWeights {
    pub fn zeros(cfg: &Config) -> Self {
        let d = cfg.embed_dim;
        let h = d * cfg.mlp_ratio;
        MlpWeights {
            w_fc: Tensor::zeros(vec![d, h]),
            b_fc: Tensor::zeros(vec![h]),
            w_proj: Tensor::zeros(vec![h, d]),
            b_proj: Tensor::zeros(vec![d]),
        }
    }
}

/// Forward: x [T, D] → hidden [T, 4D] → out [T, D]
pub fn mlp_forward(x: &Tensor, w: &MlpWeights, cfg: &Config) -> Tensor {
    let t = x.shape[0];
    let h = cfg.embed_dim * cfg.mlp_ratio;

    // Expand: [T, D] × [D, 4D] + bias → [T, 4D], then apply GELU
    let hidden = x
        .matmul(&w.w_fc)
        .add(&broadcast_bias(&w.b_fc, t, h))
        .gelu();

    // Project back: [T, 4D] × [4D, D] + bias → [T, D]
    hidden
        .matmul(&w.w_proj)
        .add(&broadcast_bias(&w.b_proj, t, cfg.embed_dim))
}

fn broadcast_bias(bias: &Tensor, t: usize, d: usize) -> Tensor {
    let mut data = Vec::with_capacity(t * d);
    for _ in 0..t {
        data.extend_from_slice(&bias.data);
    }
    Tensor::new(data, vec![t, d])
}
