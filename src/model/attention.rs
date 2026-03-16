use crate::model::config::Config;
use crate::tensor::Tensor;

pub struct AttentionWeights {
    pub w_qkv: Tensor,  // [embed_dim, 3 * embed_dim]
    pub b_qkv: Tensor,  // [3 * embed_dim]
    pub w_proj: Tensor, // [embed_dim, embed_dim]
    pub b_proj: Tensor, // [embed_dim]
}

impl AttentionWeights {
    pub fn zeros(cfg: &Config) -> Self {
        let embed_dim = cfg.embed_dim;
        AttentionWeights {
            w_qkv: Tensor::zeros(vec![embed_dim, 3 * embed_dim]),
            b_qkv: Tensor::zeros(vec![3 * embed_dim]),
            w_proj: Tensor::zeros(vec![embed_dim, embed_dim]),
            b_proj: Tensor::zeros(vec![embed_dim]),
        }
    }
}

/// Causal multi-head self-attention.
/// input: [T, D] -> output: [T, D]
///whic hfile?

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
    let embed_dim = cfg.embed_dim;
    let head_dim = cfg.head_dim();

    // Project input to queries, keys, values all at once: [T, 3D]
    let qkv = x.matmul(&w.w_qkv).add_bias(&w.b_qkv);

    // Split into three separate [T, D] tensors
    let queries = qkv.slice_cols(0, embed_dim);
    let keys = qkv.slice_cols(embed_dim, 2 * embed_dim);
    let values = qkv.slice_cols(2 * embed_dim, 3 * embed_dim);

    // Each head processes a slice of the embedding dimension
    let mut head_outputs: Vec<Tensor> = Vec::new();

    for head_idx in 0..cfg.num_heads {
        let col_start = head_idx * head_dim;
        let col_end = col_start + head_dim;

        let query_head = queries.slice_cols(col_start, col_end); // [T, head_dim]
        let key_head = keys.slice_cols(col_start, col_end); // [T, head_dim]
        let value_head = values.slice_cols(col_start, col_end); // [T, head_dim]
        let head_result = single_head_attention(&query_head, &key_head, &value_head); // [T, head_dim]
        head_outputs.push(head_result);
    }
    let for_result = head_outputs[0].clone();
    let combined = head_outputs
        .iter()
        .skip(1)
        .fold(for_result, |acc, head| acc.concat_cols(head)); // concatenate head_outputs and apply final projection

    combined.matmul(&w.w_proj).add_bias(&w.b_proj)
}

/// Single head attention: takes query, key, value each [T, head_dim]
/// and returns the attended output [T, head_dim].
fn single_head_attention(query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
    let head_dim = query.shape[1];
    query
        .matmul(&key.transpose_2d())
        .scale(1.0 / (head_dim as f32).sqrt())
        .add(&Tensor::causal_mask(query.shape[0]))
        .softmax_rows()
        .matmul(value)
}
