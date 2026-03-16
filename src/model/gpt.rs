use crate::model::block::{block_forward, BlockWeights};
use crate::model::config::Config;
use crate::tensor::Tensor;

pub struct GptWeights {
    pub token_embeddings: Tensor,    // [vocab_size, embed_dim]
    pub position_embeddings: Tensor, // [context_len, embed_dim]
    pub blocks: Vec<BlockWeights>,
    pub final_norm_weight: Tensor, // [embed_dim]
    pub final_norm_bias: Tensor,   // [embed_dim]
    pub output_projection: Tensor, // [embed_dim, vocab_size]
}

impl GptWeights {
    pub fn zeros(cfg: &Config) -> Self {
        GptWeights {
            token_embeddings: Tensor::zeros(vec![cfg.vocab_size, cfg.embed_dim]),
            position_embeddings: Tensor::zeros(vec![cfg.context_len, cfg.embed_dim]),
            blocks: (0..cfg.num_layers)
                .map(|_| BlockWeights::zeros(cfg))
                .collect(),
            final_norm_weight: Tensor::ones(vec![cfg.embed_dim]),
            final_norm_bias: Tensor::zeros(vec![cfg.embed_dim]),
            output_projection: Tensor::zeros(vec![cfg.embed_dim, cfg.vocab_size]),
        }
    }

    /// Total number of scalar parameters in the model.
    pub fn num_params(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| {
                b.ln1_weight.numel()
                    + b.ln1_bias.numel()
                    + b.attn.w_qkv.numel()
                    + b.attn.b_qkv.numel()
                    + b.attn.w_proj.numel()
                    + b.attn.b_proj.numel()
                    + b.ln2_weight.numel()
                    + b.ln2_bias.numel()
                    + b.mlp.w_fc.numel()
                    + b.mlp.b_fc.numel()
                    + b.mlp.w_proj.numel()
                    + b.mlp.b_proj.numel()
            })
            .sum::<usize>()
            + self.token_embeddings.numel()
            + self.position_embeddings.numel()
            + self.final_norm_weight.numel()
            + self.final_norm_bias.numel()
            + self.output_projection.numel()
    }
}

/// Full GPT forward pass.
/// tokens: &[usize] of length T
/// returns logits: [T, vocab_size]
///
/// Steps:
///   1. Look up token_embeddings[token] + position_embeddings[position] for each position
///   2. Pass through each block in self.blocks
///   3. Apply final layer norm (final_norm_weight, final_norm_bias)
///   4. Project to vocab: x @ output_projection -> [T, vocab_size]
pub fn gpt_forward(tokens: &[usize], w: &GptWeights, cfg: &Config) -> Tensor {
    let positions: Vec<usize> = (0..tokens.len()).collect();

    let mut x = w
        .token_embeddings
        .slice_rows(tokens)
        .add(&w.position_embeddings.slice_rows(&positions));

    for block in &w.blocks {
        x = block_forward(&x, block, cfg);
    }

    let x = x.layer_norm(&w.final_norm_weight, &w.final_norm_bias);
    x.matmul(&w.output_projection)
}
