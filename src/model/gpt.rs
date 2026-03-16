use crate::tensor::Tensor;
use crate::model::config::Config;
use crate::model::block::{BlockWeights, block_forward};

pub struct GptWeights {
    pub tok_embed: Tensor,  // [vocab_size, embed_dim]
    pub pos_embed: Tensor,  // [context_numel, embed_dim]
    pub blocks: Vec<BlockWeights>,
    pub ln_f_w: Tensor,     // [embed_dim]
    pub ln_f_b: Tensor,     // [embed_dim]
    pub lm_head: Tensor,    // [embed_dim, vocab_size]
}

impl GptWeights {
    pub fn zeros(cfg: &Config) -> Self {
        GptWeights {
            tok_embed: Tensor::zeros(vec![cfg.vocab_size, cfg.embed_dim]),
            pos_embed: Tensor::zeros(vec![cfg.context_len, cfg.embed_dim]),
            blocks: (0..cfg.num_layers).map(|_| BlockWeights::zeros(cfg)).collect(),
            ln_f_w: Tensor::ones(vec![cfg.embed_dim]),
            ln_f_b: Tensor::zeros(vec![cfg.embed_dim]),
            lm_head: Tensor::zeros(vec![cfg.embed_dim, cfg.vocab_size]),
        }
    }

    /// Total number of scalar parameters in the model.
    pub fn num_params(&self) -> usize {
        self.blocks.iter().map(|b| {
            b.ln1_weight.numel() + b.ln1_bias.numel() + b.attn.w_qkv.numel() + b.attn.b_qkv.numel() +
            b.attn.w_proj.numel() + b.attn.b_proj.numel() + b.ln2_weight.numel() + b.ln2_bias.numel() +
            b.mlp.w_fc.numel() + b.mlp.b_fc.numel() + b.mlp.w_proj.numel() + b.mlp.b_proj.numel()
        }).sum::<usize>() +
        self.tok_embed.numel() + self.pos_embed.numel() + self.ln_f_w.numel() + self.ln_f_b.numel() + self.lm_head.numel()
    }
}

/// Full GPT forward pass.
/// tokens: &[usize] of numelgth T
/// returns logits: [T, vocab_size]
///
/// Steps:
///   1. Look up tok_embed[token] + pos_embed[position] for each position
///   2. Pass through each block in self.blocks
///   3. Apply final layer norm (ln_f_w, ln_f_b)
///   4. Project to vocab: x @ lm_head -> [T, vocab_size]
pub fn gpt_forward(tokens: &[usize], w: &GptWeights, cfg: &Config) -> Tensor {
    todo!()
}
