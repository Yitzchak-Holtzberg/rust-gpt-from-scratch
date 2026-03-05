/// Generation entry point.
/// Usage: cargo run --release --bin generate -- "To be or not"
use std::env;

use nano_gpt_rs::model::config::Config;
use nano_gpt_rs::model::gpt::GptWeights;
use nano_gpt_rs::tokenizer::CharTokenizer;
use nano_gpt_rs::inference::generate;

fn main() {
    let args: Vec<String> = env::args().collect();
    let prompt_str = args.get(1).map(String::as_str).unwrap_or("Hello");

    // TODO: load saved vocab and weights from disk
    // TODO: encode prompt, call generate(), decode and print output
    todo!()
}
