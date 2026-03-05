use crate::tensor::Tensor;
use crate::model::config::Config;

pub struct AttentionWeights {
    pub w_qkv: Tensor,  // [embed_dim, 3 * embed_dim]
    pub b_qkv: Tensor,  // [3 * embed_dim]
    pub w_proj: Tensor, // [embed_dim, embed_dim]
    pub b_proj: Tensor, // [embed_dim]
}

impl AttentionWeights {
    pub fn zeros(cfg: &Config) -> Self {
        todo!()
    }
}

/// Causal multi-head self-attention.
/// input: [T, D] -> output: [T, D]
///
/// Steps:
///   1. Project input to Q, K, V via w_qkv  ([T,D] x [D,3D] -> [T,3D])
///   2. Split into three [T, D] tensors
///   3. Reshape each to [H, T, head_dim]
///   4. For each head: scores = Q @ K.T / sqrt(head_dim) + causal_mask
///   5. attn = softmax(scores)
///   6. out = attn @ V
///   7. Concat all heads -> [T, D]
///   8. Final projection via w_proj
pub fn attention_forward(x: &Tensor, w: &AttentionWeights, cfg: &Config) -> Tensor {
    todo!()
}
