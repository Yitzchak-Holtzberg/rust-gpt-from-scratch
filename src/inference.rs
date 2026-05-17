use crate::tensor::Tensor;
use crate::model::config::Config;
use crate::model::gpt::{GptWeights, gpt_forward};
use rand::Rng;

/// Sample one token from logits at the last sequence position.
/// temperature: >1 more random, <1 more greedy
/// top_k: 0 = disabled; otherwise keep only top k logits before sampling
pub fn sample_token(logits: &Tensor, temperature: f32, top_k: usize) -> usize {
    let last_row_start = (logits.shape[0] - 1) * logits.shape[1];
    let vocab_size = logits.shape[1];
    let mut scaled: Vec<f32> = logits.data[last_row_start..last_row_start + vocab_size]
        .iter()
        .map(|logit| logit / temperature.max(1e-6))
        .collect();

    if top_k > 0 && top_k < vocab_size {
        let mut sorted = scaled.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let threshold = sorted[top_k - 1];
        for value in scaled.iter_mut() {
            if *value < threshold {
                *value = f32::NEG_INFINITY;
            }
        }
    }

    let max_logit = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = scaled.iter().map(|v| (v - max_logit).exp()).collect();
    let sum: f32 = exp_values.iter().sum();
    let probabilities: Vec<f32> = exp_values.iter().map(|e| e / sum).collect();

    let mut rng = rand::thread_rng();
    let random_threshold: f32 = rng.gen();
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if random_threshold < cumulative {
            return index;
        }
    }
    vocab_size - 1
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
    let mut tokens: Vec<usize> = prompt.to_vec();
    let mut generated = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let context_start = tokens.len().saturating_sub(cfg.context_len);
        let context_window = &tokens[context_start..];
        let logits = gpt_forward(context_window, weights, cfg);
        let next_token = sample_token(&logits, temperature, top_k);
        tokens.push(next_token);
        generated.push(next_token);
    }
    generated
}
