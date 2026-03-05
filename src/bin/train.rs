/// Training entry point.
///
/// Usage:
///   cargo run --release --bin train -- data/input.txt
///
/// This wires together: tokenizer -> dataset -> forward pass -> loss.
/// Backpropagation is the TODO: see src/training/mod.rs for guidance.

use std::env;
use std::fs;

use nano_gpt_rs::model::config::Config;
use nano_gpt_rs::model::gpt::{GptWeights, gpt_forward};
use nano_gpt_rs::tokenizer::{CharTokenizer, Dataset};
use nano_gpt_rs::training::{cross_entropy_loss, Adam};

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("data/input.txt");

    let text = fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Could not read training file: {}", path));

    println!("Loaded {} chars from {}", text.len(), path);

    let tokenizer = CharTokenizer::from_text(&text);
    println!("Vocab size: {}", tokenizer.vocab_size());

    let mut cfg = Config::nano();
    cfg.vocab_size = tokenizer.vocab_size();

    let tokens = tokenizer.encode(&text);
    let dataset = Dataset::new(tokens, cfg.context_len);
    println!("Dataset samples: {}", dataset.num_samples());

    let weights = GptWeights::zeros(&cfg);
    println!("Model parameters: {}", weights.num_params());

    // Sanity-check: forward pass on first sample
    let (input, targets) = dataset.get(0);
    let logits = gpt_forward(&input, &weights, &cfg);
    let loss = cross_entropy_loss(&logits, &targets);
    println!("Initial loss (random weights): {:.4}", loss);
    println!(
        "Expected loss for random init: {:.4}",
        (cfg.vocab_size as f32).ln()
    );

    // TODO: implement the training loop here.
    // 1. For each step, sample a random batch from dataset.
    // 2. Run gpt_forward -> cross_entropy_loss.
    // 3. Compute gradients (manual or via candle autodiff).
    // 4. Call optimizer.step(params, grads).
    // 5. Log loss every N steps, save weights periodically.
    println!("\nTraining loop not yet implemented. See src/training/mod.rs.");

    let _ = Adam::new(weights.num_params(), 3e-4);
}
