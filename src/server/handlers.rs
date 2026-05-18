//! Axum route handlers and their request/response types.
//!
//! Handlers stay thin — they pull state out of locks, validate, and delegate
//! the actual work to `training::run_training` (for /train) or model methods
//! (for /generate). The shared `load_corpus` helper is also used at server
//! startup to preload `data/input.txt` if it exists.

use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    Json,
};
use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

use crate::gpt::{generate, Config, Gpt};
use crate::server::state::{AppState, Inner, TrainingStatus};
use crate::server::training::run_training;
use crate::tokenizer::{BpeTokenizer, Dataset};

// ---------- upload ----------

#[derive(Serialize)]
pub(crate) struct UploadResponse {
    pub(crate) vocab_size: usize,
    pub(crate) total_parameters: usize,
    pub(crate) total_samples: usize,
}

/// Receives a raw text body, builds the tokenizer + a fresh model, and stores
/// everything in shared state. Replaces any previous model.
pub(crate) async fn handle_corpus_upload(
    State(state): State<AppState>,
    corpus_text: String,
) -> Result<Json<UploadResponse>, (StatusCode, String)> {
    let response = load_corpus(&state, &corpus_text).await?;
    Ok(Json(response))
}

/// Shared corpus loading used by both the HTTP upload route and the optional
/// auto-load of `data/input.txt` at server startup.
pub(crate) async fn load_corpus(
    state: &AppState,
    corpus_text: &str,
) -> Result<UploadResponse, (StatusCode, String)> {
    if corpus_text.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Empty corpus".into()));
    }

    let tokenizer = BpeTokenizer::from_text(corpus_text);
    let mut config = Config::nano();
    config.vocab_size = tokenizer.vocab_size();

    let parameter_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&parameter_map, DType::F32, &state.device);
    let model = Gpt::new(config.clone(), var_builder)
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    let total_parameter_count: usize = parameter_map
        .all_vars()
        .iter()
        .map(|parameter| parameter.elem_count())
        .sum();

    let encoded_tokens = tokenizer.encode(corpus_text);
    let dataset = Dataset::new(encoded_tokens, config.context_len);
    let total_samples = dataset.num_samples();
    let vocab_size = tokenizer.vocab_size();

    {
        let mut inner = state.inner.write().await;
        *inner = Inner {
            tokenizer: Some(tokenizer),
            model: Some(model),
            parameter_map: Some(parameter_map),
            dataset: Some(dataset),
            config,
        };
    }
    {
        let mut status = state.training_status.write().await;
        *status = TrainingStatus {
            corpus_loaded: true,
            vocab_size,
            total_parameters: total_parameter_count,
            total_samples,
            ..Default::default()
        };
    }

    Ok(UploadResponse {
        vocab_size,
        total_parameters: total_parameter_count,
        total_samples,
    })
}

// ---------- train ----------

#[derive(Deserialize, Clone)]
pub(crate) struct TrainRequest {
    pub(crate) steps: usize,
    #[serde(default = "default_learning_rate")]
    pub(crate) learning_rate: f32,
    #[serde(default = "default_batch_size")]
    pub(crate) batch_size: usize,
    /// Prompt for the live sample text + next-token probability bars.
    /// Defaults to "ROMEO:" if vocab supports it.
    #[serde(default)]
    pub(crate) sample_prompt: Option<String>,
    /// Generate a sample (and refresh probs) every N steps.
    #[serde(default = "default_sample_every")]
    pub(crate) sample_every: usize,
    /// How many tokens to generate in each periodic sample.
    #[serde(default = "default_max_sample_tokens")]
    pub(crate) max_sample_tokens: usize,
}

fn default_learning_rate() -> f32 {
    // nanoGPT char-level uses 1e-3; below this the model plateaus at "bigram"
    // territory (loss ~2.5) without breaking through to word formation.
    1e-3
}
fn default_batch_size() -> usize { 32 }
fn default_sample_every() -> usize { 200 }
fn default_max_sample_tokens() -> usize { 60 }

/// Spawns a blocking task that runs `steps` AdamW updates. The task holds
/// the inner write-lock for the duration; the status endpoint reads from a
/// separate lock so the UI can poll progress live.
pub(crate) async fn handle_train_request(
    State(state): State<AppState>,
    Json(request): Json<TrainRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    if state.training_status.read().await.is_training {
        return Err((StatusCode::CONFLICT, "Already training".into()));
    }
    if state.inner.read().await.tokenizer.is_none() {
        return Err((StatusCode::BAD_REQUEST, "No corpus uploaded".into()));
    }

    let state_for_task = state.clone();
    tokio::task::spawn_blocking(move || run_training(state_for_task, request));

    Ok(Json(serde_json::json!({ "started": true })))
}

// ---------- status ----------

pub(crate) async fn handle_status_request(
    State(state): State<AppState>,
) -> Json<TrainingStatus> {
    Json(state.training_status.read().await.clone())
}

// ---------- generate ----------

#[derive(Deserialize)]
pub(crate) struct GenerateRequest {
    prompt: String,
    max_new_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_temperature() -> f32 { 1.0 }
fn default_top_k() -> usize { 40 }

#[derive(Serialize)]
pub(crate) struct GenerateResponse {
    output: String,
}

pub(crate) async fn handle_generate_request(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let inner = state.inner.read().await;
    let tokenizer = inner
        .tokenizer
        .as_ref()
        .ok_or((StatusCode::BAD_REQUEST, "No corpus uploaded yet".to_string()))?;
    let model = inner
        .model
        .as_ref()
        .ok_or((StatusCode::BAD_REQUEST, "No model available".to_string()))?;

    let prompt_token_ids: Vec<u32> = tokenizer
        .encode(&request.prompt)
        .iter()
        .map(|id| *id as u32)
        .collect();
    if prompt_token_ids.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Prompt contains no characters from the training vocabulary".to_string(),
        ));
    }

    let top_k = if request.top_k == 0 { None } else { Some(request.top_k) };
    let mut rng = rand::thread_rng();
    let generated_ids = generate(
        model,
        &prompt_token_ids,
        request.max_new_tokens,
        request.temperature,
        top_k,
        &state.device,
        &mut rng,
    )
    .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    let generated_ids_as_usize: Vec<usize> =
        generated_ids.iter().map(|id| *id as usize).collect();
    let generated_text = tokenizer.decode(&generated_ids_as_usize);
    Ok(Json(GenerateResponse {
        output: format!("{}{}", request.prompt, generated_text),
    }))
}

// ---------- checkpoint save / load ----------

/// File paths for a checkpoint. Three files keep each piece in its native
/// format (binary weights via safetensors, binary tokens for compactness,
/// human-readable JSON for tokenizer + config).
const CHECKPOINT_WEIGHTS: &str = "data/checkpoint.safetensors";
const CHECKPOINT_META: &str = "data/checkpoint.json";
const CHECKPOINT_TOKENS: &str = "data/checkpoint.tokens.bin";

#[derive(Serialize, Deserialize)]
struct CheckpointMeta {
    /// HuggingFace tokenizer serialized as JSON — round-trips via
    /// `BpeTokenizer::from_json`. Matches the standard `tokenizer.json`
    /// format so artifacts are portable.
    tokenizer_json: String,
    config: Config,
}

#[derive(Serialize)]
pub(crate) struct CheckpointResponse {
    vocab_size: usize,
    total_parameters: usize,
    total_samples: usize,
}

/// Persist the current model, tokenizer, and encoded dataset to `data/`.
/// Fails with 4xx if nothing has been loaded/trained yet.
pub(crate) async fn handle_save_checkpoint(
    State(state): State<AppState>,
) -> Result<Json<CheckpointResponse>, (StatusCode, String)> {
    let inner = state.inner.read().await;
    let parameter_map = inner.parameter_map.as_ref().ok_or((
        StatusCode::BAD_REQUEST,
        "No model loaded yet".to_string(),
    ))?;
    let tokenizer = inner.tokenizer.as_ref().ok_or((
        StatusCode::BAD_REQUEST,
        "No tokenizer loaded yet".to_string(),
    ))?;
    let dataset = inner.dataset.as_ref().ok_or((
        StatusCode::BAD_REQUEST,
        "No dataset loaded yet".to_string(),
    ))?;

    std::fs::create_dir_all("data")
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, format!("mkdir: {err}")))?;

    parameter_map
        .save(CHECKPOINT_WEIGHTS)
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, format!("save weights: {err}")))?;

    let meta = CheckpointMeta {
        tokenizer_json: tokenizer
            .to_json()
            .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, format!("tokenizer json: {err}")))?,
        config: inner.config.clone(),
    };
    let meta_json = serde_json::to_string(&meta)
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, format!("meta json: {err}")))?;
    std::fs::write(CHECKPOINT_META, meta_json)
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, format!("write meta: {err}")))?;

    // Dataset tokens as little-endian u32 bytes — compact and trivial to parse.
    let tokens_bytes: Vec<u8> = dataset
        .data
        .iter()
        .flat_map(|&token_id| (token_id as u32).to_le_bytes())
        .collect();
    std::fs::write(CHECKPOINT_TOKENS, tokens_bytes)
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, format!("write tokens: {err}")))?;

    Ok(Json(CheckpointResponse {
        vocab_size: tokenizer.vocab_size(),
        total_parameters: parameter_map.all_vars().iter().map(|v| v.elem_count()).sum(),
        total_samples: dataset.num_samples(),
    }))
}

/// Restore model + tokenizer + dataset from the most recent checkpoint.
/// Generation works immediately after load; training resumes too because
/// the dataset comes back from disk.
pub(crate) async fn handle_load_checkpoint(
    State(state): State<AppState>,
) -> Result<Json<CheckpointResponse>, (StatusCode, String)> {
    let meta_text = std::fs::read_to_string(CHECKPOINT_META).map_err(|err| {
        (StatusCode::NOT_FOUND, format!("read checkpoint meta: {err}"))
    })?;
    let meta: CheckpointMeta = serde_json::from_str(&meta_text)
        .map_err(|err| (StatusCode::BAD_REQUEST, format!("parse meta: {err}")))?;

    let tokenizer = BpeTokenizer::from_json(&meta.tokenizer_json)
        .map_err(|err| (StatusCode::BAD_REQUEST, format!("parse tokenizer: {err}")))?;

    // Build a fresh VarMap → VarBuilder → Gpt with the saved config so the
    // parameter names match the safetensors file, then load weights in-place.
    let mut parameter_map = VarMap::new();
    let var_builder =
        VarBuilder::from_varmap(&parameter_map, DType::F32, &state.device);
    let model = Gpt::new(meta.config.clone(), var_builder)
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, format!("build model: {err}")))?;
    parameter_map
        .load(CHECKPOINT_WEIGHTS)
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, format!("load weights: {err}")))?;

    let tokens_bytes = std::fs::read(CHECKPOINT_TOKENS)
        .map_err(|err| (StatusCode::NOT_FOUND, format!("read tokens: {err}")))?;
    let dataset_tokens: Vec<usize> = tokens_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as usize)
        .collect();
    let dataset = Dataset::new(dataset_tokens, meta.config.context_len);

    let total_parameter_count: usize =
        parameter_map.all_vars().iter().map(|v| v.elem_count()).sum();
    let vocab_size = tokenizer.vocab_size();
    let total_samples = dataset.num_samples();

    {
        let mut inner = state.inner.write().await;
        *inner = Inner {
            tokenizer: Some(tokenizer),
            model: Some(model),
            parameter_map: Some(parameter_map),
            dataset: Some(dataset),
            config: meta.config,
        };
    }
    {
        let mut status = state.training_status.write().await;
        *status = TrainingStatus {
            corpus_loaded: true,
            vocab_size,
            total_parameters: total_parameter_count,
            total_samples,
            ..Default::default()
        };
    }

    Ok(Json(CheckpointResponse {
        vocab_size,
        total_parameters: total_parameter_count,
        total_samples,
    }))
}

// ---------- index page ----------

pub(crate) async fn serve_index_page() -> impl IntoResponse {
    Html(INDEX_HTML)
}

const INDEX_HTML: &str = include_str!("../../web/index.html");
