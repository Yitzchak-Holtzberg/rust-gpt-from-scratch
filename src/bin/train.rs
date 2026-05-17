/// Training entry point.
/// Usage: cargo run --release --bin train -- data/input.txt
///
/// NOTE: This currently runs the FORWARD pass and computes loss, but cannot
/// actually train the model — the layer-level backward chain
/// (gpt_backward / block_backward / attention_backward / mlp_backward) has not
/// been implemented. See the TODO comment at the bottom of src/training.rs.
///
/// For now this serves as a smoke test that the forward pipeline works.
use std::env;
use std::fs;

use nano_gpt_rs::model::config::Config;
use nano_gpt_rs::model::gpt::{gpt_forward, GptWeights};
use nano_gpt_rs::tokenizer::{CharTokenizer, Dataset};
use nano_gpt_rs::training::cross_entropy_loss;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("data/input.txt");

    let text = fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Could not read {path}"));

    let tokenizer = CharTokenizer::from_text(&text);
    let mut cfg = Config::nano();
    cfg.vocab_size = tokenizer.vocab_size();

    let tokens = tokenizer.encode(&text);
    let dataset = Dataset::new(tokens, cfg.context_len);
    let weights = GptWeights::random(&cfg);

    println!("Vocab size: {}", tokenizer.vocab_size());
    println!("Total params: {}", weights.num_params());
    println!("Total samples: {}", dataset.num_samples());

    // Forward-pass smoke test on the first few samples.
    let samples_to_check = 3.min(dataset.num_samples());
    for sample_index in 0..samples_to_check {
        let (input_ids, target_ids) = dataset.get(sample_index);
        let logits = gpt_forward(&input_ids, &weights, &cfg);
        let loss = cross_entropy_loss(&logits, &target_ids);
        println!("sample {sample_index}: loss = {loss:.4}");
    }

    println!("\nFull training requires backward passes through the model layers.");
    println!("See the TODO comment at the bottom of src/training.rs.");
}
