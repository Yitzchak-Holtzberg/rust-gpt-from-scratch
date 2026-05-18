//! Localhost HTTP server for training and querying the GPT.
//!
//! Run from the binary entry point at [src/bin/serve.rs](../bin/serve.rs):
//!
//! ```ignore
//! fn main() { nano_gpt_rs::server::run(); }
//! ```
//!
//! Submodule layout:
//! - [`state`]: shared `AppState`, training status + child types
//! - [`handlers`]: Axum route handlers + their request/response types
//! - [`training`]: the optimizer loop + LR schedule + grad clipping
//! - [`monitoring`]: live probes (next-token probs, GPU stats, labels)

mod handlers;
mod monitoring;
mod state;
mod training;

use std::sync::Arc;

use axum::{
    extract::DefaultBodyLimit,
    routing::{get, post},
    Router,
};
use candle_core::Device;
use tokio::sync::RwLock;

use crate::gpt::Config;
use crate::server::handlers::{
    handle_corpus_upload, handle_generate_request, handle_load_checkpoint,
    handle_save_checkpoint, handle_status_request, handle_train_request, load_corpus,
    serve_index_page,
};
use crate::server::state::{AppState, Inner, TrainingStatus};

/// Start the localhost server. Blocks until the listener is shut down.
/// Wraps tokio runtime setup so the binary entry can stay one line.
#[tokio::main]
pub async fn run() {
    let device = pick_device();
    let app_state = AppState {
        inner: Arc::new(RwLock::new(Inner {
            tokenizer: None,
            model: None,
            parameter_map: None,
            dataset: None,
            config: Config::nano(),
        })),
        training_status: Arc::new(RwLock::new(TrainingStatus::default())),
        device,
    };

    preload_default_corpus(&app_state).await;

    let router = Router::new()
        .route("/", get(serve_index_page))
        .route("/api/upload", post(handle_corpus_upload))
        .route("/api/train", post(handle_train_request))
        .route("/api/status", get(handle_status_request))
        .route("/api/generate", post(handle_generate_request))
        .route("/api/save_checkpoint", post(handle_save_checkpoint))
        .route("/api/load_checkpoint", post(handle_load_checkpoint))
        .layer(DefaultBodyLimit::max(64 * 1024 * 1024)) // up to 64 MB corpora
        .with_state(app_state);

    let bind_address = "127.0.0.1:9000";
    println!("Server listening on http://{bind_address}/");
    let listener = tokio::net::TcpListener::bind(bind_address).await.unwrap();
    axum::serve(listener, router).await.unwrap();
}

/// Try to initialize CUDA device 0; fall back to CPU and print the reason.
/// CUDA support is compile-time gated by the `cuda` feature in Cargo.toml.
fn pick_device() -> Device {
    match Device::new_cuda(0) {
        Ok(device) => {
            println!("Using CUDA device 0");
            device
        }
        Err(err) => {
            println!("CUDA unavailable ({err}); falling back to CPU");
            Device::Cpu
        }
    }
}

/// Preload `data/input.txt` (Tiny Shakespeare) if it exists, so the UI is
/// ready to train immediately without requiring an upload.
async fn preload_default_corpus(state: &AppState) {
    let path = "data/input.txt";
    let Ok(corpus_text) = std::fs::read_to_string(path) else {
        return;
    };
    match load_corpus(state, &corpus_text).await {
        Ok(summary) => println!(
            "Preloaded {path} (vocab {}, params {}, samples {})",
            summary.vocab_size, summary.total_parameters, summary.total_samples
        ),
        Err((_, message)) => eprintln!("Could not preload {path}: {message}"),
    }
}
