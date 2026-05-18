//! Shared application state for the localhost training server.
//!
//! `AppState` is what every Axum handler gets injected with. The interior
//! types live behind RwLocks so the training task can hold the model write
//! lock for the duration of a run while the status endpoint streams progress
//! from a separate, cheaper read lock.
//!
//! All `TrainingStatus` sub-types implement `Serialize` so the live UI can
//! deserialize them via `/api/status` and render without any backend-side
//! HTML templating.

use std::sync::Arc;

use candle_core::Device;
use candle_nn::VarMap;
use serde::Serialize;
use tokio::sync::RwLock;

use crate::gpt::{Config, Gpt};
use crate::tokenizer::{BpeTokenizer, Dataset};

/// Shared application state. Cheap to clone — interior fields are Arc'd.
#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) inner: Arc<RwLock<Inner>>,
    pub(crate) training_status: Arc<RwLock<TrainingStatus>>,
    /// Device used for all tensors. Tries CUDA on startup, falls back to CPU.
    pub(crate) device: Device,
}

/// Model + tokenizer + dataset. None until a corpus has been uploaded.
pub(crate) struct Inner {
    pub(crate) tokenizer: Option<BpeTokenizer>,
    pub(crate) model: Option<Gpt>,
    pub(crate) parameter_map: Option<VarMap>,
    pub(crate) dataset: Option<Dataset>,
    pub(crate) config: Config,
}

/// One snapshot of generated text taken mid-training, for the timeline scrubber.
#[derive(Clone, Serialize, Default)]
pub(crate) struct GenerationSample {
    pub(crate) step: usize,
    pub(crate) text: String,
}

/// One row in the "next-token probability" bars panel.
#[derive(Clone, Serialize, Default)]
pub(crate) struct TokenProb {
    /// Human-readable label for the token — multi-char BPE tokens with
    /// newlines/tabs escaped (e.g. `" the"`, `"\\n"`, `"ROMEO"`).
    pub(crate) label: String,
    pub(crate) probability: f32,
}

/// Snapshot of GPU stats from `nvidia-smi`. None if the binary isn't available.
#[derive(Clone, Serialize, Default)]
pub(crate) struct GpuStats {
    pub(crate) utilization_percent: u32,
    pub(crate) vram_used_mib: u32,
    pub(crate) vram_total_mib: u32,
}

/// Everything the UI polls at /api/status — both training progress and the
/// live-monitoring panels (samples, probabilities, GPU).
#[derive(Clone, Serialize, Default)]
pub(crate) struct TrainingStatus {
    pub(crate) is_training: bool,
    pub(crate) current_step: usize,
    pub(crate) target_steps: usize,
    /// Recent loss values, sampled every few steps for the line chart.
    pub(crate) recent_losses: Vec<f32>,
    pub(crate) last_loss: Option<f32>,
    pub(crate) best_loss: Option<f32>,
    pub(crate) steps_per_second: f32,
    pub(crate) tokens_per_second: f32,
    pub(crate) elapsed_seconds: f32,
    pub(crate) eta_seconds: f32,
    /// Rolling history of generations from a fixed prompt — drives the
    /// "watching it learn" timeline scrubber in the UI.
    pub(crate) samples: Vec<GenerationSample>,
    pub(crate) sample_prompt: String,
    /// Top-K next-token probabilities from a single forward pass on
    /// `sample_prompt`. Refreshed more often than `samples` since it's cheap.
    pub(crate) next_token_probs: Vec<TokenProb>,
    pub(crate) gpu: Option<GpuStats>,
    pub(crate) corpus_loaded: bool,
    pub(crate) vocab_size: usize,
    pub(crate) total_parameters: usize,
    pub(crate) total_samples: usize,
}
