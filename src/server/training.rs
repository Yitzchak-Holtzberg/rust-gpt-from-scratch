//! The training loop: one `run_training` call drives N AdamW steps on the
//! currently-loaded model while pushing live progress (loss, samples,
//! probabilities, GPU stats) into the shared `TrainingStatus`.
//!
//! Helpers in this file are training-specific (LR schedule, gradient
//! clipping). Probability probes and GPU monitoring live in `monitoring.rs`.

use std::time::Instant;

use candle_core::{Tensor, backprop::GradStore};
use candle_nn::{loss::cross_entropy, AdamW, Optimizer, ParamsAdamW, VarMap};
use rand::seq::SliceRandom;

use crate::gpt::generate;
use crate::server::handlers::TrainRequest;
use crate::server::monitoring::{
    choose_sample_prompt, compute_next_token_probs, query_gpu_stats,
};
use crate::server::state::{AppState, GenerationSample};

/// Cap on how many historical loss values to keep — limits the size of
/// `/api/status` responses while still giving the line chart enough points.
const LOSS_HISTORY_CAP: usize = 400;
/// Cap on how many timeline samples to keep so the scrubber doesn't grow
/// indefinitely on long training sessions.
const SAMPLE_HISTORY_CAP: usize = 30;
/// Push the status snapshot every N steps (or on probe-triggered steps).
const STATUS_UPDATE_INTERVAL: usize = 5;
/// Probability probe cadence — one forward pass on the sample prompt.
const PROBS_INTERVAL: usize = 25;
/// GPU stats poll cadence — shells out to `nvidia-smi` (~100ms each).
const GPU_INTERVAL: usize = 60;

/// Drive `request.steps` optimizer updates on the model held in `state`.
/// Holds the inner write-lock for the entire run; status updates use a
/// separate lock so the UI can poll progress freely.
pub(crate) fn run_training(state: AppState, request: TrainRequest) {
    let runtime = tokio::runtime::Handle::current();

    let inner = runtime.block_on(state.inner.write());
    let dataset = inner.dataset.as_ref().expect("dataset must exist");
    let model = inner.model.as_ref().expect("model must exist");
    let tokenizer = inner.tokenizer.as_ref().expect("tokenizer must exist");
    let parameter_map = inner.parameter_map.as_ref().expect("parameter_map must exist");
    let config = inner.config.clone();

    let sample_prompt = choose_sample_prompt(request.sample_prompt.clone(), tokenizer);
    let sample_every = request.sample_every.max(1);
    let max_sample_tokens = request.max_sample_tokens;

    // Reset live-tracking state for this run.
    runtime.block_on(async {
        let mut status = state.training_status.write().await;
        status.is_training = true;
        status.current_step = 0;
        status.target_steps = request.steps;
        status.recent_losses.clear();
        status.last_loss = None;
        status.best_loss = None;
        status.steps_per_second = 0.0;
        status.tokens_per_second = 0.0;
        status.elapsed_seconds = 0.0;
        status.eta_seconds = 0.0;
        status.samples.clear();
        status.next_token_probs.clear();
        status.sample_prompt = sample_prompt.clone();
    });

    // Match nanoGPT char-level hyperparameters:
    // - beta2=0.95 (vs candle default 0.999) helps the second-moment estimate
    //   adapt faster for shallow transformers on small data.
    // - weight_decay=0.1 (vs candle default 0.01) regularizes the model away
    //   from the early "stuck on bigram statistics" plateau.
    let base_lr = request.learning_rate as f64;
    let mut optimizer = AdamW::new(
        parameter_map.all_vars(),
        ParamsAdamW {
            lr: base_lr,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
        },
    )
    .expect("AdamW init");

    let device = state.device.clone();
    let mut rng = rand::thread_rng();
    let all_sample_indices: Vec<usize> = (0..dataset.num_samples()).collect();
    let started_at = Instant::now();
    let batch_size = request.batch_size.max(1);
    let context_len = config.context_len;
    let vocab_size = config.vocab_size;
    let total_steps = request.steps;

    let sample_prompt_token_ids: Vec<u32> = tokenizer
        .encode(&sample_prompt)
        .iter()
        .map(|id| *id as u32)
        .collect();

    for step in 1..=total_steps {
        optimizer.set_learning_rate(scheduled_learning_rate(step, total_steps, base_lr));

        // Stack `batch_size` random sliding windows into one [B, T] batch.
        let mut input_buffer: Vec<u32> = Vec::with_capacity(batch_size * context_len);
        let mut target_buffer: Vec<u32> = Vec::with_capacity(batch_size * context_len);
        for _ in 0..batch_size {
            let &sample_index = all_sample_indices.choose(&mut rng).unwrap();
            let (input_ids, target_ids) = dataset.get(sample_index);
            input_buffer.extend(input_ids.iter().map(|id| *id as u32));
            target_buffer.extend(target_ids.iter().map(|id| *id as u32));
        }

        let input_tensor =
            Tensor::from_vec(input_buffer, (batch_size, context_len), &device)
                .expect("input tensor");
        let target_tensor =
            Tensor::from_vec(target_buffer, (batch_size * context_len,), &device)
                .expect("target tensor");

        let logits = model.forward(&input_tensor).expect("forward");
        let logits_flat = logits
            .reshape((batch_size * context_len, vocab_size))
            .expect("reshape logits");
        let loss = cross_entropy(&logits_flat, &target_tensor).expect("loss");

        // Global-norm gradient clipping prevents outlier batches from
        // destabilizing training — standard transformer practice.
        let raw_grads = loss.backward().expect("backward");
        let clipped_grads =
            clip_grads_by_global_norm(parameter_map, raw_grads, 1.0).expect("clip grads");
        optimizer.step(&clipped_grads).expect("optimizer step");

        let loss_value = loss.to_scalar::<f32>().expect("loss to scalar");
        let elapsed = started_at.elapsed().as_secs_f32();
        let steps_per_second = step as f32 / elapsed.max(1e-3);
        let tokens_per_second = steps_per_second * batch_size as f32 * context_len as f32;
        let eta_seconds = if step < total_steps {
            (total_steps - step) as f32 / steps_per_second.max(1e-3)
        } else {
            0.0
        };

        let probs_update = if step == 1 || step % PROBS_INTERVAL == 0 {
            compute_next_token_probs(model, tokenizer, &sample_prompt, 10, &device).ok()
        } else {
            None
        };
        let gpu_update = if step == 1 || step % GPU_INTERVAL == 0 {
            query_gpu_stats()
        } else {
            None
        };
        let sample_update = if step == 1 || step % sample_every == 0 || step == total_steps {
            generate(
                model,
                &sample_prompt_token_ids,
                max_sample_tokens,
                0.8,
                Some(40),
                &device,
                &mut rng,
            )
            .ok()
            .map(|generated_ids| {
                let ids_as_usize: Vec<usize> =
                    generated_ids.iter().map(|id| *id as usize).collect();
                let suffix = tokenizer.decode(&ids_as_usize);
                GenerationSample {
                    step,
                    text: format!("{sample_prompt}{suffix}"),
                }
            })
        } else {
            None
        };

        let needs_status_update = step == 1
            || step % STATUS_UPDATE_INTERVAL == 0
            || step == total_steps
            || probs_update.is_some()
            || gpu_update.is_some()
            || sample_update.is_some();
        if needs_status_update {
            runtime.block_on(async {
                let mut status = state.training_status.write().await;
                status.current_step = step;
                status.last_loss = Some(loss_value);
                status.best_loss = Some(
                    status
                        .best_loss
                        .map(|best| best.min(loss_value))
                        .unwrap_or(loss_value),
                );
                status.steps_per_second = steps_per_second;
                status.tokens_per_second = tokens_per_second;
                status.elapsed_seconds = elapsed;
                status.eta_seconds = eta_seconds;
                status.recent_losses.push(loss_value);
                if status.recent_losses.len() > LOSS_HISTORY_CAP {
                    let drop = status.recent_losses.len() - LOSS_HISTORY_CAP;
                    status.recent_losses.drain(0..drop);
                }
                if let Some(probs) = probs_update {
                    status.next_token_probs = probs;
                }
                if let Some(gpu) = gpu_update {
                    status.gpu = Some(gpu);
                }
                if let Some(sample) = sample_update {
                    status.samples.push(sample);
                    if status.samples.len() > SAMPLE_HISTORY_CAP {
                        status.samples.remove(0);
                    }
                }
            });
        }
    }

    runtime.block_on(async {
        let mut status = state.training_status.write().await;
        status.is_training = false;
    });
    drop(inner); // release the lock so generation can proceed
}

/// Linear warmup → cosine decay, restarted each training session.
///
/// - First `warmup_steps` (= min(100, total/10)): LR ramps 0 → base_lr
/// - Then cosine-decay from base_lr down to 10% of base_lr by the final step
///
/// Restarting per-session gives "cosine restarts" (SGDR-style) when the user
/// clicks Train multiple times — each click escapes any plateau by briefly
/// re-raising the LR, then decaying again.
fn scheduled_learning_rate(step: usize, total_steps: usize, base_lr: f64) -> f64 {
    let warmup_steps = 100usize.min(total_steps / 10).max(1);
    if step <= warmup_steps {
        return base_lr * (step as f64 / warmup_steps as f64);
    }
    let cosine_steps = (total_steps - warmup_steps).max(1) as f64;
    let progress = ((step - warmup_steps) as f64 / cosine_steps).clamp(0.0, 1.0);
    let cosine_factor = 0.5 * (1.0 + (progress * std::f64::consts::PI).cos());
    base_lr * (0.1 + 0.9 * cosine_factor)
}

/// Cap the global L2 norm of all parameter gradients at `max_norm`. Returns
/// the (possibly scaled) gradient store. Standard transformer practice —
/// without this, an outlier batch can produce gradients large enough to
/// destabilize training (loss spikes upward).
fn clip_grads_by_global_norm(
    parameter_map: &VarMap,
    mut grads: GradStore,
    max_norm: f32,
) -> candle_core::Result<GradStore> {
    let mut total_squared_norm = 0.0f64;
    for variable in parameter_map.all_vars() {
        if let Some(gradient) = grads.get(variable.as_tensor()) {
            total_squared_norm += gradient.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
    }
    let global_norm = total_squared_norm.sqrt() as f32;
    if global_norm <= max_norm || !global_norm.is_finite() {
        return Ok(grads);
    }

    let scale = (max_norm / global_norm) as f64;
    for variable in parameter_map.all_vars() {
        let var_tensor = variable.as_tensor();
        if let Some(gradient) = grads.get(var_tensor) {
            let scaled = (gradient * scale)?;
            grads.insert(var_tensor, scaled);
        }
    }
    Ok(grads)
}
