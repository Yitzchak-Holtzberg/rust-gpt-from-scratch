use crate::model::attention::{attention_forward, AttentionWeights};
use crate::model::config::Config;
use crate::model::mlp::{mlp_forward, MlpWeights};
use crate::tensor::Tensor;

pub struct BlockWeights {
    pub ln1_weight: Tensor,
    pub ln1_bias: Tensor,
    pub attn: AttentionWeights,
    pub ln2_weight: Tensor,
    pub ln2_bias: Tensor,
    pub mlp: MlpWeights,
}

impl BlockWeights {
    pub fn zeros(cfg: &Config) -> Self {
        BlockWeights {
            ln1_weight: Tensor::ones(vec![cfg.embed_dim]),
            ln1_bias: Tensor::zeros(vec![cfg.embed_dim]),
            attn: AttentionWeights::zeros(cfg),
            ln2_weight: Tensor::ones(vec![cfg.embed_dim]),
            ln2_bias: Tensor::zeros(vec![cfg.embed_dim]),
            mlp: MlpWeights::zeros(cfg),
        }
    }
}

/// One transformer block: pre-norm attention + pre-norm MLP, both with residuals.
/// input: [T, D] -> output: [T, D]
///
/// Steps:
///   x = x + attention(layer_norm(x, ln1_w, ln1_b))
///   x = x + mlp(layer_norm(x, ln2_w, ln2_b))
pub fn block_forward(x: &Tensor, w: &BlockWeights, cfg: &Config) -> Tensor {
    let normalised = x.layer_norm(&w.ln1_weight, &w.ln1_bias);
    let attention_out = attention_forward(&normalised, &w.attn, cfg);
    let x = x.add(&attention_out);

    let normalised = x.layer_norm(&w.ln2_weight, &w.ln2_bias);
    let mlp_out = mlp_forward(&normalised, &w.mlp);
    x.add(&mlp_out)
}
