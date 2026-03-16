use crate::model::attention::{attention_forward, AttentionWeights};
use crate::model::config::Config;
use crate::model::mlp::{mlp_forward, MlpWeights};
use crate::tensor::Tensor;

pub struct BlockWeights {
    pub ln1_w: Tensor,
    pub ln1_b: Tensor,
    pub attn: AttentionWeights,
    pub ln2_w: Tensor,
    pub ln2_b: Tensor,
    pub mlp: MlpWeights,
}

impl BlockWeights {
    pub fn zeros(cfg: &Config) -> Self {
        BlockWeights {
            ln1_w: Tensor::ones(vec![cfg.embed_dim]),
            ln1_b: Tensor::zeros(vec![cfg.embed_dim]),
            attn: AttentionWeights::zeros(cfg),
            ln2_w: Tensor::ones(vec![cfg.embed_dim]),
            ln2_b: Tensor::zeros(vec![cfg.embed_dim]),
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
    todo!()
}
