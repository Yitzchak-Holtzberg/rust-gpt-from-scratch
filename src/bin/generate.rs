/// Generation entry point.
/// Usage: cargo run --release --bin generate -- "To be or not"
///
/// NOTE: This uses random-initialized weights (no trained model exists yet —
/// the layer-level backward passes haven't been implemented). Output will be
/// gibberish, but the full pipeline runs end-to-end.
use std::env;
use std::fs;

use nano_gpt_rs::model::config::Config;
use nano_gpt_rs::model::gpt::GptWeights;
use nano_gpt_rs::tokenizer::CharTokenizer;
use nano_gpt_rs::inference::generate;

fn main() {
    let args: Vec<String> = env::args().collect();
    let prompt_str = args.get(1).map(String::as_str).unwrap_or("Hello");

    let corpus_path = "data/input.txt";
    let text = fs::read_to_string(corpus_path)
        .unwrap_or_else(|_| panic!("Could not read {corpus_path} (download Tiny Shakespeare per data/README.md)"));

    let tokenizer = CharTokenizer::from_text(&text);
    let mut cfg = Config::nano();
    cfg.vocab_size = tokenizer.vocab_size();

    let weights = GptWeights::random(&cfg);
    let prompt_ids = tokenizer.encode(prompt_str);
    if prompt_ids.is_empty() {
        eprintln!("Prompt contained no characters from the training vocabulary");
        return;
    }

    let generated_ids = generate(&prompt_ids, &weights, &cfg, 100, 1.0, 40);
    let output_text = tokenizer.decode(&generated_ids);

    println!("Prompt: {prompt_str}");
    println!("Output: {output_text}");
}
