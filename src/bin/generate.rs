/// Generation entry point.
///
/// Usage (after training and saving weights):
///   cargo run --release --bin generate -- "To be or not"
///
/// Currently uses zero-initialized weights (outputs garbage).
/// Wire in weight loading once training is implemented.

use std::env;

use nano_gpt_rs::model::config::Config;
use nano_gpt_rs::model::gpt::GptWeights;
use nano_gpt_rs::tokenizer::CharTokenizer;
use nano_gpt_rs::inference::generate;

fn main() {
    let args: Vec<String> = env::args().collect();
    let prompt_str = args.get(1).map(String::as_str).unwrap_or("Hello");

    // TODO: load vocab and weights from disk after training.
    // For now, build a tiny demo vocab from the prompt itself.
    let tokenizer = CharTokenizer::from_text(prompt_str);
    let mut cfg = Config::nano();
    cfg.vocab_size = tokenizer.vocab_size();

    let weights = GptWeights::zeros(&cfg);

    let prompt_tokens = tokenizer.encode(prompt_str);
    if prompt_tokens.is_empty() {
        eprintln!("Prompt contains no known tokens.");
        return;
    }

    let generated = generate(&prompt_tokens, &weights, &cfg, 200, 0.8, 40);
    let output = tokenizer.decode(&generated);

    println!("Prompt:    {}", prompt_str);
    println!("Generated: {}", output);
    println!("\n(Note: weights are zero-initialized until training is complete.)");
}
