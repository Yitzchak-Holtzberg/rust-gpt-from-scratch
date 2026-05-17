//! Generation entry point.
//!
//! Usage:
//!   cargo run --release --bin generate -- "ROMEO:" [max_new_tokens]
//!
//! Loads the trained weights from `weights.safetensors` and the vocabulary
//! from `vocab.json`, then autoregressively samples new characters after the
//! supplied prompt.

use std::env;
use std::fs;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

use nano_gpt_rs::gpt::{generate, Config, Gpt};
use nano_gpt_rs::tokenizer::CharTokenizer;

fn main() -> candle_core::Result<()> {
    let arguments: Vec<String> = env::args().collect();
    let prompt_text = arguments.get(1).map(String::as_str).unwrap_or("Hello");
    let max_new_tokens: usize = arguments
        .get(2)
        .and_then(|argument| argument.parse().ok())
        .unwrap_or(200);

    // Restore vocabulary so we can encode the prompt and decode model output
    let vocabulary_as_json = fs::read_to_string("vocab.json")
        .expect("vocab.json not found - run `cargo run --release --bin train` first");
    let vocabulary: Vec<char> =
        serde_json::from_str(&vocabulary_as_json).expect("parse vocab.json");
    let tokenizer = CharTokenizer::from_vocab(vocabulary);

    let device = Device::Cpu;
    let mut config = Config::nano();
    config.vocab_size = tokenizer.vocab_size();

    // Construct the model topology, then load saved weights into it
    let mut parameter_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&parameter_map, DType::F32, &device);
    let model = Gpt::new(config.clone(), var_builder)?;
    parameter_map
        .load("weights.safetensors")
        .expect("weights.safetensors not found - run `cargo run --release --bin train` first");

    let prompt_token_ids: Vec<u32> = tokenizer
        .encode(prompt_text)
        .iter()
        .map(|id| *id as u32)
        .collect();
    if prompt_token_ids.is_empty() {
        eprintln!("Prompt contained no characters from the training vocabulary");
        return Ok(());
    }

    let mut rng = rand::thread_rng();
    let generated_token_ids = generate(
        &model,
        &prompt_token_ids,
        max_new_tokens,
        1.0,
        Some(40),
        &device,
        &mut rng,
    )?;

    let generated_token_ids_as_usize: Vec<usize> =
        generated_token_ids.iter().map(|id| *id as usize).collect();
    let generated_text = tokenizer.decode(&generated_token_ids_as_usize);
    println!("Prompt: {prompt_text}");
    println!("Output: {prompt_text}{generated_text}");
    Ok(())
}
