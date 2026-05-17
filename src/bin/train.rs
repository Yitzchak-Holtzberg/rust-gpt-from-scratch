//! Training entry point.
//!
//! Usage:
//!   cargo run --release --bin train -- data/input.txt [num_steps]
//!
//! Trains a small GPT on the supplied text file, saving the learned weights
//! to `weights.safetensors` and the character vocabulary to `vocab.json`.

use std::env;
use std::fs;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::{loss::cross_entropy, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::seq::SliceRandom;

use nano_gpt_rs::gpt::{Config, Gpt};
use nano_gpt_rs::tokenizer::{CharTokenizer, Dataset};

fn main() -> candle_core::Result<()> {
    let arguments: Vec<String> = env::args().collect();
    let training_text_path = arguments
        .get(1)
        .map(String::as_str)
        .unwrap_or("data/input.txt");
    let number_of_training_steps: usize = arguments
        .get(2)
        .and_then(|argument| argument.parse().ok())
        .unwrap_or(500);
    let batch_size: usize = arguments
        .get(3)
        .and_then(|argument| argument.parse().ok())
        .unwrap_or(16);

    let training_text = fs::read_to_string(training_text_path)
        .unwrap_or_else(|_| panic!("Could not read {training_text_path}"));

    // Build tokenizer from training corpus
    let tokenizer = CharTokenizer::from_text(&training_text);
    let mut config = Config::nano();
    config.vocab_size = tokenizer.vocab_size();

    // Encode entire corpus once, then expose it as sliding (input, target) windows
    let encoded_tokens: Vec<usize> = tokenizer.encode(&training_text);
    let dataset = Dataset::new(encoded_tokens, config.context_len);

    // Set up the model and the AdamW optimizer (autograd tracks the parameters
    // registered through `var_builder` into `parameter_map`).
    let device = Device::Cpu;
    let parameter_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&parameter_map, DType::F32, &device);
    let model = Gpt::new(config.clone(), var_builder)?;

    let optimizer_params = ParamsAdamW {
        lr: 3e-4,
        ..Default::default()
    };
    let mut optimizer = AdamW::new(parameter_map.all_vars(), optimizer_params)?;

    let total_parameter_count: usize = parameter_map
        .all_vars()
        .iter()
        .map(|parameter| parameter.elem_count())
        .sum();
    println!("Vocab size: {}", tokenizer.vocab_size());
    println!("Total parameters: {total_parameter_count}");
    println!("Total training samples: {}", dataset.num_samples());
    println!("Context length: {}", config.context_len);
    println!("Batch size: {batch_size}");
    println!("Training for {number_of_training_steps} steps\n");

    let mut rng = rand::thread_rng();
    let all_sample_indices: Vec<usize> = (0..dataset.num_samples()).collect();
    let training_started_at = Instant::now();
    let vocab_size = config.vocab_size;
    let context_len = config.context_len;

    for step in 1..=number_of_training_steps {
        // Stack `batch_size` random sliding windows into a single batch.
        // Layout: input_buffer is row-major [B, T] flattened.
        let mut input_buffer: Vec<u32> = Vec::with_capacity(batch_size * context_len);
        let mut target_buffer: Vec<u32> = Vec::with_capacity(batch_size * context_len);
        for _ in 0..batch_size {
            let &sample_index = all_sample_indices.choose(&mut rng).unwrap();
            let (input_token_ids, target_token_ids) = dataset.get(sample_index);
            input_buffer.extend(input_token_ids.iter().map(|id| *id as u32));
            target_buffer.extend(target_token_ids.iter().map(|id| *id as u32));
        }

        let input_tensor = Tensor::from_vec(input_buffer, (batch_size, context_len), &device)?;
        // cross_entropy expects flat targets [N], so we leave the target buffer flat.
        let target_tensor =
            Tensor::from_vec(target_buffer, (batch_size * context_len,), &device)?;

        // Forward [B, T] → [B, T, V] → flatten to [B*T, V] for the loss
        let logits = model.forward(&input_tensor)?;
        let logits_flat = logits.reshape((batch_size * context_len, vocab_size))?;
        let loss = cross_entropy(&logits_flat, &target_tensor)?;
        optimizer.backward_step(&loss)?;

        if step == 1 || step % 25 == 0 || step == number_of_training_steps {
            let loss_value = loss.to_scalar::<f32>()?;
            let elapsed_seconds = training_started_at.elapsed().as_secs_f64();
            let steps_per_second = step as f64 / elapsed_seconds;
            println!(
                "step {step:>4} | loss {loss_value:.4} | {steps_per_second:.2} steps/s"
            );
        }
    }

    // Persist weights and the vocabulary so `generate` can load them later
    let weights_path = "weights.safetensors";
    parameter_map.save(weights_path)?;
    println!("\nWeights saved to {weights_path}");

    let vocabulary_as_json = serde_json::to_string(&tokenizer.vocab).expect("serialize vocab");
    fs::write("vocab.json", vocabulary_as_json).expect("write vocab.json");
    println!("Vocabulary saved to vocab.json");

    Ok(())
}
