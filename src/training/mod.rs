// =============================================================================
// src/training/mod.rs — Loss function and optimizer
// =============================================================================
//
// WHAT THIS FILE IS
//   Two things: (1) how to measure how wrong the model is, and (2) how to
//   update the weights to make it less wrong.
//
// CROSS-ENTROPY LOSS
//   After the forward pass, we have logits: [T, V] — one score per vocab token
//   per position. The target is a single token ID per position. The loss is:
//
//     loss = -mean( log( softmax(logits)[t, target[t]] ) )  for t in 0..T
//
//   Intuition: we want P(correct token) = 1.0. Cross-entropy penalises
//   low probability on the correct token. If the model is completely random
//   (uniform distribution over V tokens), the loss is ln(V). For V=65 chars:
//   ln(65) ≈ 4.17. That's what you should see before any training.
//
//   After good training on Tiny Shakespeare, loss should fall to ~1.3–1.5 nats.
//
// ADAM OPTIMIZER
//   Gradient descent with adaptive per-parameter learning rates.
//   Plain SGD: param -= lr * grad
//   Adam:      tracks a running mean of gradients (m, "momentum") and a running
//              mean of squared gradients (v, "velocity"). It uses the ratio
//              m / sqrt(v) as the effective gradient, which normalises by
//              gradient scale and adds momentum. This makes training much faster
//              and more stable than plain SGD.
//
//   Key hyperparameters:
//     lr          Learning rate — how big a step to take each iteration.
//                 Too high → loss explodes. Too low → slow convergence.
//                 ~3e-4 is a good default for small models.
//     beta1=0.9   Exponential decay for first moment (momentum).
//     beta2=0.999 Exponential decay for second moment (adaptive scaling).
//     weight_decay Adds a small L2 penalty for large weights (regularisation).
//
//   Paper: https://arxiv.org/abs/1412.6980
//
// GRADIENT CLIPPING (recommended addition)
//   Clip the global gradient norm to e.g. 1.0 before calling optimizer.step().
//   Without this, early training steps can produce enormous gradients that
//   send weights to NaN. Formula: scale all grads by min(1, clip/norm).
//
// THE MISSING PIECE: BACKPROPAGATION
//   The optimizer needs gradients. Computing them requires the backward pass.
//   See the large TODO block at the bottom of this file for implementation paths.

use crate::tensor::Tensor;

/// Cross-entropy loss over a sequence.
/// logits: [T, V], targets: &[usize] of length T.
/// Returns scalar mean loss (in nats).
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> f32 {
    let t = logits.shape[0];
    let v = logits.shape[1];
    assert_eq!(t, targets.len());

    let mut total = 0.0f32;
    for i in 0..t {
        let row = &logits.data[i * v..(i + 1) * v];
        // Numerically stable log-softmax: subtract max before exp to prevent overflow
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum = row.iter().map(|x| (x - max).exp()).sum::<f32>().ln();
        let log_prob = row[targets[i]] - max - log_sum;
        total -= log_prob;
    }
    total / t as f32
}

/// Adam optimizer. Holds running statistics for every parameter.
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub t: usize,        // step counter (for bias correction)
    pub m: Vec<f32>,     // first moment (running mean of gradients)
    pub v: Vec<f32>,     // second moment (running mean of squared gradients)
}

impl Adam {
    pub fn new(num_params: usize, lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.1,
            t: 0,
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
        }
    }

    /// Apply one Adam gradient step to `params` using `grads`.
    /// Both slices must have length == num_params passed to Adam::new.
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;
        // Bias correction: early steps have small m/v estimates, scale them up
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..params.len() {
            // AdamW: apply weight decay directly to params, not to grad
            let g = grads[i] + self.weight_decay * params[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// =============================================================================
// TODO — Implement backpropagation (Phase 5)
// =============================================================================
//
// OPTION A: Manual backprop
//   Derive and implement dLoss/dW for each weight matrix. Educational but hard.
//   Gradient derivations to implement (in reverse order of forward pass):
//     1. dLoss/d_logits  — derivative of cross-entropy + softmax (they combine
//                          cleanly: grad = softmax(logits) - one_hot(target))
//     2. d_logits/d_lm_head, d_logits/d_x_final  (matmul backward)
//     3. LayerNorm backward  (see https://arxiv.org/abs/1607.06450 appendix)
//     4. MLP backward: GELU grad, two matmul backwards, two bias grads
//     5. Attention backward: softmax grad, matmul grads, causal mask is free
//     6. Embedding backward: scatter grad into embedding rows
//
//   Add a `grad: Vec<f32>` field to Tensor and accumulate during backward().
//
// OPTION B: Use candle autodiff (practical path)
//   1. Add to Cargo.toml:
//        candle-core = { version = "0.8" }
//   2. Replace Tensor with candle_core::Tensor throughout.
//   3. Forward ops use candle (which records the computation graph automatically).
//   4. Call loss_tensor.backward()? after the forward pass.
//   5. Collect param.grad()? for each parameter.
//   6. Pass collected grads to Adam::step above.
//   candle docs: https://github.com/huggingface/candle
// =============================================================================
