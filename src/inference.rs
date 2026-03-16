use crate::tensor::Tensor;
use crate::model::config::Config;
use crate::model::gpt::{GptWeights, gpt_forward};

/// Sample one token from logits at the last sequence position.
/// temperature: >1 more random, <1 more greedy
/// top_k: 0 = disabled; otherwise keep only top k logits before sampling
pub fn sample_token(logits: &Tensor, temperature: f32, top_k: usize) -> usize {
    todo!()
}

/// Run the model autoregressively for max_new_tokens steps.
/// Returns only the newly generated token IDs (not the prompt).
pub fn generate(
    prompt: &[usize],
    weights: &GptWeights,
    cfg: &Config,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
) -> Vec<usize> {
    todo!()
}
