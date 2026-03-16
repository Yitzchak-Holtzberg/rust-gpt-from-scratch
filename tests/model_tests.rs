use nano_gpt_rs::model::attention::{attention_forward, AttentionWeights};
use nano_gpt_rs::model::block::{block_forward, BlockWeights};
use nano_gpt_rs::model::config::Config;
use nano_gpt_rs::model::mlp::{mlp_forward, MlpWeights};
use nano_gpt_rs::tensor::Tensor;


/// Output shape should be [T, D] — same as input.
#[test]
fn test_mlp_output_shape() {
    let cfg = Config::nano();
    let w = MlpWeights::zeros(&cfg);
    let t = 3; // 3 tokens
    let x = Tensor::zeros(vec![t, cfg.embed_dim]);

    let out = mlp_forward(&x, &w);

    assert_eq!(out.shape, vec![t, cfg.embed_dim]);
}

// --- attention_forward ---

/// Output shape must be [T, D] — same as input.
#[test]
fn test_attention_output_shape() {
    let cfg = Config::nano();
    let w = AttentionWeights::zeros(&cfg);
    let t = 4;
    let x = Tensor::zeros(vec![t, cfg.embed_dim]);

    let out = attention_forward(&x, &w, &cfg);

    assert_eq!(out.shape, vec![t, cfg.embed_dim]);
}

// --- block_forward ---

/// Output shape must be [T, D] — same as input.
#[test]
fn test_block_output_shape() {
    let cfg = Config::nano();
    let w = BlockWeights::zeros(&cfg);
    let t = 4;
    let x = Tensor::zeros(vec![t, cfg.embed_dim]);

    let out = block_forward(&x, &w, &cfg);

    assert_eq!(out.shape, vec![t, cfg.embed_dim]);
}

// --- mlp_forward ---

/// With all-zero weights and all-zero input, every output value should be 0.
/// Reason: zero input @ zero weights + zero bias = zero at every step.
#[test]
fn test_mlp_zero_weights_zero_output() {
    let cfg = Config::nano();
    let w = MlpWeights::zeros(&cfg);
    let x = Tensor::zeros(vec![2, cfg.embed_dim]);

    let out = mlp_forward(&x, &w);

    assert!(out.data.iter().all(|&v| v == 0.0),
        "Expected all zeros, got: {:?}", &out.data[..4]);
}
