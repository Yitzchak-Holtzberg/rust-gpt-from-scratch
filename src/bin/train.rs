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

    // Llama / GPT-3 style AdamW: β2=0.95 (default 0.999 is too slow to adapt for
    // LM training), weight_decay=0.1 (decoupled L2 — the main regularizer paired
    // with dropout). These matter more on a 120K-token corpus where overfitting
    // would otherwise dominate.
    let optimizer_params = ParamsAdamW {
        lr: 3e-4,
        beta1: 0.9,
        beta2: 0.95,
        weight_decay: 0.1,
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

    let base_learning_rate = 3e-4_f64;
    let total_steps = number_of_training_steps;

    for step in 1..=number_of_training_steps {
        // Linear warmup → cosine decay
        let current_lr = scheduled_learning_rate(step, total_steps, base_learning_rate);
        optimizer.set_learning_rate(current_lr);

        // Stack `batch_size` random sliding windows into a single batch.
        let mut input_buffer: Vec<u32> = Vec::with_capacity(batch_size * context_len);
        let mut target_buffer: Vec<u32> = Vec::with_capacity(batch_size * context_len);
        for _ in 0..batch_size {
            let &sample_index = all_sample_indices.choose(&mut rng).unwrap();
            let (input_token_ids, target_token_ids) = dataset.get(sample_index);
            input_buffer.extend(input_token_ids.iter().map(|id| *id as u32));
            target_buffer.extend(target_token_ids.iter().map(|id| *id as u32));
        }

        let input_tensor = Tensor::from_vec(input_buffer, (batch_size, context_len), &device)?;
        let target_tensor =
            Tensor::from_vec(target_buffer, (batch_size * context_len,), &device)?;

        // Forward [B, T] → [B, T, V] → flatten to [B*T, V] for the loss.
        // `forward_train` enables dropout — critical regularization on a 120K-token corpus.
        let logits = model.forward_train(&input_tensor)?;
        let logits_flat = logits.reshape((batch_size * context_len, vocab_size))?;
        let loss = cross_entropy(&logits_flat, &target_tensor)?;

        // Backward + global-norm gradient clipping → optimizer step
        let raw_grads = loss.backward()?;
        let clipped_grads = clip_grads_by_global_norm(&parameter_map, raw_grads, 1.0)?;
        optimizer.step(&clipped_grads)?;

        if step == 1 || step % 25 == 0 || step == number_of_training_steps {
            let loss_value = loss.to_scalar::<f32>()?;
            let elapsed_seconds = training_started_at.elapsed().as_secs_f64();
            let steps_per_second = step as f64 / elapsed_seconds;
            println!(
                "step {step:>4} | loss {loss_value:.4} | lr {current_lr:.6} | {steps_per_second:.2} steps/s"
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

/// Linear warmup → cosine decay from `base_lr` down to 10% of `base_lr`.
fn scheduled_learning_rate(step: usize, total_steps: usize, base_lr: f64) -> f64 {
    let warmup_steps = 100usize.min(total_steps / 10).max(1);
    if step <= warmup_steps {
        return base_lr * (step as f64 / warmup_steps as f64);
    }
    let cosine_steps_remaining = (total_steps - warmup_steps).max(1) as f64;
    let progress = ((step - warmup_steps) as f64 / cosine_steps_remaining).clamp(0.0, 1.0);
    let cosine_factor = 0.5 * (1.0 + (progress * std::f64::consts::PI).cos());
    base_lr * (0.1 + 0.9 * cosine_factor)
}

/// Cap the global L2 norm of all parameter gradients at `max_norm`.
fn clip_grads_by_global_norm(
    parameter_map: &VarMap,
    mut raw_grads: candle_core::backprop::GradStore,
    max_norm: f32,
) -> candle_core::Result<candle_core::backprop::GradStore> {
    let mut total_squared_norm = 0.0f64;
    for variable in parameter_map.all_vars() {
        if let Some(gradient) = raw_grads.get(variable.as_tensor()) {
            let squared_sum = gradient.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            total_squared_norm += squared_sum;
        }
    }
    let global_norm = total_squared_norm.sqrt() as f32;
    if global_norm <= max_norm || !global_norm.is_finite() {
        return Ok(raw_grads);
    }

    let scale = (max_norm / global_norm) as f64;
    for variable in parameter_map.all_vars() {
        let var_tensor = variable.as_tensor();
        if let Some(gradient) = raw_grads.get(var_tensor) {
            let scaled = (gradient * scale)?;
            raw_grads.insert(var_tensor, scaled);
        }
    }
    Ok(raw_grads)
}
