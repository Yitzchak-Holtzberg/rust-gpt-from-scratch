// =============================================================================
// src/inference/mod.rs — Autoregressive text generation
// =============================================================================
//
// WHAT THIS FILE IS
//   Once the model is trained, this is how you get it to produce text. It
//   implements the generation loop and sampling strategies that control the
//   randomness of the output.
//
// AUTOREGRESSIVE GENERATION
//   The model always produces one token at a time. Given a sequence of tokens
//   so far, it runs a forward pass and looks only at the logits for the last
//   position (which predicts the next token). We sample from that distribution,
//   append the result, and repeat.
//
//   This means an N-step generation loop runs N full forward passes. The KV
//   cache (stretch goal) optimises this by caching attention keys and values
//   from past positions so each new step is O(T × D) instead of O(T² × D).
//
// TEMPERATURE SAMPLING
//   Before softmax, divide all logits by `temperature`:
//     temperature = 1.0 → unchanged distribution
//     temperature < 1.0 → sharper distribution (more greedy, repetitive)
//     temperature > 1.0 → flatter distribution (more random, creative)
//
//   Setting temperature = 0 collapses to argmax (always picks the top token).
//
// TOP-K SAMPLING
//   Keep only the k highest-probability tokens and renormalise. This prevents
//   the model from ever picking very low-probability tokens (garbage characters
//   that would derail generation). top_k = 40 is a common default.
//
// FURTHER READING
//   "The Curious Case of Neural Text Degeneration" (Holtzman et al. 2019)
//   discusses temperature, top-k, and introduces nucleus (top-p) sampling:
//   https://arxiv.org/abs/1904.09751

use rand::Rng;
use crate::tensor::Tensor;
use crate::model::config::Config;
use crate::model::gpt::{GptWeights, gpt_forward};

/// Sample one token from the logits at the last sequence position.
/// temperature > 1 → more random; < 1 → more greedy.
/// top_k = 0 disables top-k filtering (uses full vocab).
pub fn sample_token(logits: &Tensor, temperature: f32, top_k: usize) -> usize {
    let v = logits.shape[1];
    let t = logits.shape[0];

    // Extract logits for the last position (the "next token" prediction)
    let last_logits: Vec<f32> = logits.data[(t - 1) * v..t * v].to_vec();

    // Apply temperature scaling
    let mut scaled: Vec<(usize, f32)> = last_logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, l / temperature))
        .collect();

    // Top-k: keep only the k highest logits, drop the rest
    if top_k > 0 && top_k < v {
        scaled.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scaled.truncate(top_k);
    }

    // Softmax over the retained logits → probability distribution
    let max = scaled.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = scaled
        .iter()
        .map(|&(i, l)| (i, (l - max).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    for (_, p) in probs.iter_mut() {
        *p /= sum;
    }

    // Multinomial sample: walk the CDF, return the token at the sample point
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0f32;
    let last_idx = probs.last().map(|(i, _)| *i).unwrap_or(0);
    for &(idx, prob) in &probs {
        cumsum += prob;
        if r <= cumsum {
            return idx;
        }
    }
    last_idx // fallback for floating point edge case
}

/// Autoregressive generation loop.
/// Runs max_new_tokens forward passes and returns the generated token IDs.
pub fn generate(
    prompt: &[usize],
    weights: &GptWeights,
    cfg: &Config,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
) -> Vec<usize> {
    let mut tokens = prompt.to_vec();

    for _ in 0..max_new_tokens {
        // Crop to the model's context window if the sequence has grown too long
        let start = tokens.len().saturating_sub(cfg.context_len);
        let ctx = &tokens[start..];

        // Full forward pass — O(T² × D) per step (KV cache would make this O(T × D))
        let logits = gpt_forward(ctx, weights, cfg);
        let next = sample_token(&logits, temperature, top_k);
        tokens.push(next);
    }

    // Return only the newly generated tokens, not the prompt
    tokens[prompt.len()..].to_vec()
}
