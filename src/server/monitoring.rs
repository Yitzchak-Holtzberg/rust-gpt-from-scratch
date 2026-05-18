//! Live-monitoring helpers consumed by the training loop:
//! - peek at next-token probabilities from a single forward pass
//! - shell out to `nvidia-smi` for GPU utilization and VRAM
//! - format token labels and choose a sensible default sample prompt
//!
//! Kept separate from `training.rs` so the training loop stays focused on
//! the optimizer step itself.

use std::process::Command;

use candle_core::{Device, IndexOp, Tensor};
use candle_nn::ops::softmax_last_dim;

use crate::gpt::Gpt;
use crate::server::state::{GpuStats, TokenProb};
use crate::tokenizer::BpeTokenizer;

/// Convert a decoded BPE token into a human-readable label for the
/// probability bars. Multi-char tokens are shown as-is (e.g. `" the"`,
/// `"Romeo"`), with control characters escaped so they don't break the
/// row layout.
pub(crate) fn token_label(token: &str) -> String {
    if token.is_empty() {
        return "∅".to_string();
    }
    token
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
}

/// Run a single forward pass on the prompt and return the top-K next-token
/// probabilities — cheap (one forward, no sampling, no autoregressive loop).
pub(crate) fn compute_next_token_probs(
    model: &Gpt,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    top_k: usize,
    device: &Device,
) -> candle_core::Result<Vec<TokenProb>> {
    let encoded = tokenizer.encode(prompt);
    if encoded.is_empty() {
        return Ok(Vec::new());
    }
    let context_start = encoded.len().saturating_sub(model.config.context_len);
    let context_window: Vec<u32> = encoded[context_start..]
        .iter()
        .map(|id| *id as u32)
        .collect();

    let input_tensor =
        Tensor::from_vec(context_window.clone(), (1, context_window.len()), device)?;
    let logits = model.forward(&input_tensor)?; // [1, T, V]
    let (_batch, sequence_length, _vocab) = logits.dims3()?;
    let last_logits = logits.i((0, sequence_length - 1))?; // [V]
    let probabilities: Vec<f32> = softmax_last_dim(&last_logits)?.to_vec1()?;

    let mut indexed: Vec<(usize, f32)> = probabilities
        .iter()
        .enumerate()
        .map(|(id, &probability)| (id, probability))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k);

    Ok(indexed
        .into_iter()
        .map(|(id, probability)| {
            let decoded = tokenizer.decode(&[id]);
            TokenProb {
                label: token_label(&decoded),
                probability,
            }
        })
        .collect())
}

/// Query GPU utilization and VRAM via `nvidia-smi`. Returns `None` if the
/// binary isn't on PATH or the output is malformed — call sites just don't
/// render the GPU tile in that case.
pub(crate) fn query_gpu_stats() -> Option<GpuStats> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let first_line = text.lines().next()?;
    let parts: Vec<&str> = first_line.split(',').map(|s| s.trim()).collect();
    if parts.len() < 3 {
        return None;
    }
    Some(GpuStats {
        utilization_percent: parts[0].parse().ok()?,
        vram_used_mib: parts[1].parse().ok()?,
        vram_total_mib: parts[2].parse().ok()?,
    })
}

/// Pick a sample prompt for the "watching it learn" panel. Prefer the
/// user-supplied prompt if it encodes to a non-empty token sequence,
/// otherwise fall back to the canonical `"ROMEO:"`, then `"\n"`, then any
/// single character.
pub(crate) fn choose_sample_prompt(
    requested: Option<String>,
    tokenizer: &BpeTokenizer,
) -> String {
    let candidates = requested
        .into_iter()
        .chain(std::iter::once("ROMEO:".to_string()))
        .chain(std::iter::once("\n".to_string()));
    for candidate in candidates {
        if !candidate.is_empty() && !tokenizer.encode(&candidate).is_empty() {
            return candidate;
        }
    }
    "a".to_string()
}
