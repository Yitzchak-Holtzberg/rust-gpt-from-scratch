// =============================================================================
// src/model/block.rs — One transformer block
// =============================================================================
//
// WHAT THIS FILE IS
//   A transformer block is one "layer" of the model. It combines self-attention
//   (for cross-token communication) and an MLP (for per-token transformation),
//   each wrapped in a residual connection and layer normalisation.
//
// THE RESIDUAL CONNECTION
//   Instead of replacing x, each sublayer adds to it: x = x + sublayer(x)
//
//   Why this matters: gradients during backprop can flow directly through the
//   addition — they don't have to pass through every weight matrix. This is
//   what makes very deep networks trainable. Without residuals, GPT-2's 96
//   layers would not converge.
//   Paper: https://arxiv.org/abs/1512.03385 (ResNets, original residual idea)
//
// PRE-NORM vs POST-NORM
//   This code uses Pre-LN: LayerNorm → sublayer → residual add.
//   Original transformer used Post-LN: sublayer → residual add → LayerNorm.
//   Pre-LN trains more stably without learning rate warmup.
//   GPT-2 and later models use Pre-LN.
//
// STRUCTURE OF ONE BLOCK
//   x → LayerNorm → Self-Attention → + x   (residual)
//     → LayerNorm → MLP            → + x   (residual)
//
// STACKING BLOCKS
//   Each block refines the representation. Early blocks tend to learn local
//   syntax; later blocks learn broader semantics. This is why depth matters.

use crate::tensor::Tensor;
use crate::model::config::Config;
use crate::model::attention::{AttentionWeights, attention_forward};
use crate::model::mlp::{MlpWeights, mlp_forward};

/// Weights for one transformer block.
pub struct BlockWeights {
    pub ln1_w: Tensor,    // LayerNorm scale before attention
    pub ln1_b: Tensor,    // LayerNorm bias before attention
    pub attn: AttentionWeights,
    pub ln2_w: Tensor,    // LayerNorm scale before MLP
    pub ln2_b: Tensor,    // LayerNorm bias before MLP
    pub mlp: MlpWeights,
}

impl BlockWeights {
    pub fn zeros(cfg: &Config) -> Self {
        let d = cfg.embed_dim;
        BlockWeights {
            ln1_w: Tensor::ones(vec![d]),  // init to 1 so LN is identity at start
            ln1_b: Tensor::zeros(vec![d]),
            attn: AttentionWeights::zeros(cfg),
            ln2_w: Tensor::ones(vec![d]),
            ln2_b: Tensor::zeros(vec![d]),
            mlp: MlpWeights::zeros(cfg),
        }
    }
}

/// Forward: x [T, D] → [T, D]
pub fn block_forward(x: &Tensor, w: &BlockWeights, cfg: &Config) -> Tensor {
    // Pre-norm attention sub-layer with residual
    let normed = x.layer_norm(&w.ln1_w, &w.ln1_b);
    let attn_out = attention_forward(&normed, &w.attn, cfg);
    let x = x.add(&attn_out);

    // Pre-norm MLP sub-layer with residual
    let normed = x.layer_norm(&w.ln2_w, &w.ln2_b);
    let mlp_out = mlp_forward(&normed, &w.mlp, cfg);
    x.add(&mlp_out)
}
