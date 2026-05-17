//! Localhost UI for training and querying a character-level GPT.
//!
//! Run with: `cargo run --release --bin serve`
//! Then open http://127.0.0.1:8080/ in a browser.
//!
//! Flow:
//!   1. Drop a `.txt` file onto the page (uploads the corpus and builds a model)
//!   2. Click "Train" to run N training steps (progress streams back live)
//!   3. Type a prompt and click "Generate" to sample text from the trained model

use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss::cross_entropy, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use nano_gpt_rs::gpt::{generate, Config, Gpt};
use nano_gpt_rs::tokenizer::{CharTokenizer, Dataset};

/// Shared application state. Cheap to clone — interior fields are Arc'd.
#[derive(Clone)]
struct AppState {
    inner: Arc<RwLock<Inner>>,
    training_status: Arc<RwLock<TrainingStatus>>,
}

/// Model + tokenizer + dataset. None until a corpus has been uploaded.
struct Inner {
    tokenizer: Option<CharTokenizer>,
    model: Option<Gpt>,
    parameter_map: Option<VarMap>,
    dataset: Option<Dataset>,
    config: Config,
    total_parameter_count: usize,
}

#[derive(Clone, Serialize, Default)]
struct TrainingStatus {
    is_training: bool,
    current_step: usize,
    target_steps: usize,
    /// Recent loss values, sampled every few steps for the line chart.
    recent_losses: Vec<f32>,
    last_loss: Option<f32>,
    steps_per_second: f32,
    corpus_loaded: bool,
    vocab_size: usize,
    total_parameters: usize,
    total_samples: usize,
}

#[tokio::main]
async fn main() {
    let app_state = AppState {
        inner: Arc::new(RwLock::new(Inner {
            tokenizer: None,
            model: None,
            parameter_map: None,
            dataset: None,
            config: Config::nano(),
            total_parameter_count: 0,
        })),
        training_status: Arc::new(RwLock::new(TrainingStatus::default())),
    };

    // Convenience: if data/input.txt is present, preload it so the UI is
    // ready to train immediately. This is the standard Tiny Shakespeare path.
    let default_corpus_path = "data/input.txt";
    if let Ok(default_corpus) = std::fs::read_to_string(default_corpus_path) {
        match load_corpus(&app_state, &default_corpus).await {
            Ok(summary) => println!(
                "Preloaded {default_corpus_path} (vocab {}, params {}, samples {})",
                summary.vocab_size, summary.total_parameters, summary.total_samples
            ),
            Err((_, message)) => eprintln!("Could not preload {default_corpus_path}: {message}"),
        }
    }

    let router = Router::new()
        .route("/", get(serve_index_page))
        .route("/api/upload", post(handle_corpus_upload))
        .route("/api/train", post(handle_train_request))
        .route("/api/status", get(handle_status_request))
        .route("/api/generate", post(handle_generate_request))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024)) // allow up to 64 MB corpus
        .with_state(app_state);

    let bind_address = "127.0.0.1:8080";
    println!("Server listening on http://{bind_address}/");
    let listener = tokio::net::TcpListener::bind(bind_address).await.unwrap();
    axum::serve(listener, router).await.unwrap();
}

// ----- handlers -----

#[derive(Serialize)]
struct UploadResponse {
    vocab_size: usize,
    total_parameters: usize,
    total_samples: usize,
}

/// Receives a raw text body, builds the tokenizer + a fresh model, and stores
/// everything in shared state. Replaces any previous model.
async fn handle_corpus_upload(
    State(state): State<AppState>,
    corpus_text: String,
) -> Result<Json<UploadResponse>, (StatusCode, String)> {
    let response = load_corpus(&state, &corpus_text).await?;
    Ok(Json(response))
}

/// Shared corpus loading used by both the HTTP upload route and the optional
/// auto-load of `data/input.txt` at server startup.
async fn load_corpus(
    state: &AppState,
    corpus_text: &str,
) -> Result<UploadResponse, (StatusCode, String)> {
    if corpus_text.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Empty corpus".into()));
    }

    let tokenizer = CharTokenizer::from_text(corpus_text);
    let mut config = Config::nano();
    config.vocab_size = tokenizer.vocab_size();

    let parameter_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&parameter_map, DType::F32, &Device::Cpu);
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
        inner.tokenizer = Some(tokenizer);
        inner.model = Some(model);
        inner.parameter_map = Some(parameter_map);
        inner.dataset = Some(dataset);
        inner.config = config;
        inner.total_parameter_count = total_parameter_count;
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

#[derive(Deserialize)]
struct TrainRequest {
    steps: usize,
    #[serde(default = "default_learning_rate")]
    learning_rate: f32,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
}

fn default_learning_rate() -> f32 {
    3e-4
}

fn default_batch_size() -> usize {
    16
}

/// Spawns a background task that runs `steps` AdamW updates. The task holds
/// the inner write-lock for the duration; the status endpoint reads from a
/// separate lock so the UI can poll progress live.
async fn handle_train_request(
    State(state): State<AppState>,
    Json(request): Json<TrainRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    {
        let already_training = state.training_status.read().await.is_training;
        if already_training {
            return Err((StatusCode::CONFLICT, "Already training".into()));
        }
    }
    {
        let inner = state.inner.read().await;
        if inner.tokenizer.is_none() {
            return Err((StatusCode::BAD_REQUEST, "No corpus uploaded".into()));
        }
    }

    let state_for_task = state.clone();
    tokio::task::spawn_blocking(move || run_training(state_for_task, request));

    Ok(Json(serde_json::json!({ "started": true })))
}

fn run_training(state: AppState, request: TrainRequest) {
    let runtime = tokio::runtime::Handle::current();

    runtime.block_on(async {
        let mut status = state.training_status.write().await;
        status.is_training = true;
        status.current_step = 0;
        status.target_steps = request.steps;
        status.recent_losses.clear();
        status.last_loss = None;
        status.steps_per_second = 0.0;
    });

    let inner = runtime.block_on(state.inner.write());

    let dataset = inner.dataset.as_ref().expect("dataset must exist");
    let model = inner.model.as_ref().expect("model must exist");
    let parameter_map = inner.parameter_map.as_ref().expect("parameter_map must exist");
    let config = inner.config.clone();

    let optimizer_params = ParamsAdamW {
        lr: request.learning_rate as f64,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(parameter_map.all_vars(), optimizer_params)
        .expect("AdamW init");

    let device = Device::Cpu;
    let mut rng = rand::thread_rng();
    let all_sample_indices: Vec<usize> = (0..dataset.num_samples()).collect();
    let training_started_at = Instant::now();
    let batch_size = request.batch_size.max(1);
    let context_len = config.context_len;
    let vocab_size = config.vocab_size;

    for step in 1..=request.steps {
        // Stack `batch_size` random sliding windows into a single [B, T] batch
        let mut input_buffer: Vec<u32> = Vec::with_capacity(batch_size * context_len);
        let mut target_buffer: Vec<u32> = Vec::with_capacity(batch_size * context_len);
        for _ in 0..batch_size {
            let &sample_index = all_sample_indices.choose(&mut rng).unwrap();
            let (input_token_ids, target_token_ids) = dataset.get(sample_index);
            input_buffer.extend(input_token_ids.iter().map(|id| *id as u32));
            target_buffer.extend(target_token_ids.iter().map(|id| *id as u32));
        }

        let input_tensor = Tensor::from_vec(input_buffer, (batch_size, context_len), &device)
            .expect("input tensor");
        let target_tensor =
            Tensor::from_vec(target_buffer, (batch_size * context_len,), &device)
                .expect("target tensor");

        let logits = model.forward(&input_tensor).expect("forward");
        let logits_flat = logits
            .reshape((batch_size * context_len, vocab_size))
            .expect("reshape logits");
        let loss = cross_entropy(&logits_flat, &target_tensor).expect("loss");
        optimizer.backward_step(&loss).expect("backward step");

        let loss_value = loss.to_scalar::<f32>().expect("loss to scalar");
        let elapsed = training_started_at.elapsed().as_secs_f32();
        let steps_per_second = step as f32 / elapsed.max(1e-3);

        // Update progress every 5 steps (or on first/last) to keep UI snappy
        if step == 1 || step % 5 == 0 || step == request.steps {
            runtime.block_on(async {
                let mut status = state.training_status.write().await;
                status.current_step = step;
                status.last_loss = Some(loss_value);
                status.steps_per_second = steps_per_second;
                status.recent_losses.push(loss_value);
                // Cap history length so the response stays small
                if status.recent_losses.len() > 200 {
                    let drop_count = status.recent_losses.len() - 200;
                    status.recent_losses.drain(0..drop_count);
                }
            });
        }
    }

    runtime.block_on(async {
        let mut status = state.training_status.write().await;
        status.is_training = false;
    });

    drop(inner); // release the lock so generation can proceed
}

async fn handle_status_request(State(state): State<AppState>) -> Json<TrainingStatus> {
    let status = state.training_status.read().await.clone();
    Json(status)
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
    max_new_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_k() -> usize {
    40
}

#[derive(Serialize)]
struct GenerateResponse {
    output: String,
}

async fn handle_generate_request(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let inner = state.inner.read().await;
    let tokenizer = inner.tokenizer.as_ref().ok_or((
        StatusCode::BAD_REQUEST,
        "No corpus uploaded yet".to_string(),
    ))?;
    let model = inner.model.as_ref().ok_or((
        StatusCode::BAD_REQUEST,
        "No model available".to_string(),
    ))?;

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
        &Device::Cpu,
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

async fn serve_index_page() -> impl IntoResponse {
    Html(INDEX_HTML)
}

const INDEX_HTML: &str = include_str!("../../web/index.html");
