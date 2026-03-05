use crate::tensor::Tensor;
use crate::model::config::Config;
use crate::model::block::{BlockWeights, block_forward};

pub struct GptWeights {
    pub tok_embed: Tensor,  // [vocab_size, embed_dim]
    pub pos_embed: Tensor,  // [context_len, embed_dim]
    pub blocks: Vec<BlockWeights>,
    pub ln_f_w: Tensor,     // [embed_dim]
    pub ln_f_b: Tensor,     // [embed_dim]
    pub lm_head: Tensor,    // [embed_dim, vocab_size]
}

impl GptWeights {
    pub fn zeros(cfg: &Config) -> Self {
        todo!()
    }

    /// Total number of scalar parameters in the model.
    pub fn num_params(&self) -> usize {
        todo!()
    }
}

/// Full GPT forward pass.
/// tokens: &[usize] of length T
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
