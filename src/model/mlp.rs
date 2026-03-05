use crate::tensor::Tensor;
use crate::model::config::Config;

pub struct MlpWeights {
    pub w_fc: Tensor,   // [embed_dim, embed_dim * mlp_ratio]
    pub b_fc: Tensor,   // [embed_dim * mlp_ratio]
    pub w_proj: Tensor, // [embed_dim * mlp_ratio, embed_dim]
    pub b_proj: Tensor, // [embed_dim]
}

impl MlpWeights {
    pub fn zeros(cfg: &Config) -> Self {
        todo!()
    }
}

/// Position-wise feed-forward MLP.
/// input: [T, D] -> hidden: [T, 4D] -> output: [T, D]
///
/// Steps:
///   1. x @ w_fc + b_fc  -> [T, 4D]
///   2. gelu(...)
///   3. x @ w_proj + b_proj -> [T, D]
pub fn mlp_forward(x: &Tensor, w: &MlpWeights, cfg: &Config) -> Tensor {
    todo!()
}
