use crate::tensor::Tensor;
use crate::model::config::Config;
use crate::model::attention::{AttentionWeights, attention_forward};
use crate::model::mlp::{MlpWeights, mlp_forward};

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
        todo!()
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
