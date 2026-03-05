/// Training entry point.
/// Usage: cargo run --release --bin train -- data/input.txt
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
        .unwrap_or_else(|_| panic!("Could not read {}", path));

    // TODO: build tokenizer, dataset, config, weights, optimizer
    // TODO: training loop — forward, loss, backward, step
    todo!()
}
