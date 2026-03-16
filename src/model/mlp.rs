use crate::model::config::Config;
use crate::tensor::Tensor;

pub struct MlpWeights {
    pub w_fc: Tensor,   // [embed_dim, embed_dim * mlp_ratio]
    pub b_fc: Tensor,   // [embed_dim * mlp_ratio]
    pub w_proj: Tensor, // [embed_dim * mlp_ratio, embed_dim]
    pub b_proj: Tensor, // [embed_dim]
}

impl MlpWeights {
    pub fn zeros(cfg: &Config) -> Self {
        let embed_dim = cfg.embed_dim;
        let mlp_dim = embed_dim * cfg.mlp_ratio;
        MlpWeights {
            w_fc: Tensor::zeros(vec![embed_dim, mlp_dim]),
            b_fc: Tensor::zeros(vec![mlp_dim]),
            w_proj: Tensor::zeros(vec![mlp_dim, embed_dim]),
            b_proj: Tensor::zeros(vec![embed_dim]),
        }
    }
}

/// Position-wise feed-forward MLP.
/// input: [T, D] -> hidden: [T, 4D] -> output: [T, D]
///
/// Steps:
///   1. x @ w_fc + b_fc  -> [T, 4D]
///   2. gelu(...)
///   3. x @ w_proj + b_proj -> [T, D]
pub fn mlp_forward(x: &Tensor, w: &MlpWeights) -> Tensor {
    x
        .matmul(&w.w_fc)
        .add_bias(&w.b_fc)
        .gelu()
        .matmul(&w.w_proj)
        .add_bias(&w.b_proj)
}
