use crate::tensor::Tensor;

/// Cross-entropy loss over a sequence.
/// logits: [T, vocab_size], targets: token IDs of length T
/// returns scalar mean loss
///
/// For each position: loss = -log( softmax(logits[t])[target[t]] )
/// Use the log-sum-exp trick for numerical stability.
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> f32 {
    todo!()
}

/// AdamW optimizer state. One m and v entry per parameter.
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub t: usize,
    pub m: Vec<f32>,
    pub v: Vec<f32>,
}

impl Adam {
    pub fn new(num_params: usize, lr: f32) -> Self {
        todo!()
    }

    /// Apply one gradient step.
    /// params and grads must be the same length.
    ///
    /// Steps:
    ///   g = grad + weight_decay * param   (AdamW decoupled decay)
    ///   m = beta1 * m + (1 - beta1) * g
    ///   v = beta2 * v + (1 - beta2) * g^2
    ///   m_hat = m / (1 - beta1^t)
    ///   v_hat = v / (1 - beta2^t)
    ///   param -= lr * m_hat / (sqrt(v_hat) + eps)
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        todo!()
    }
}

// TODO Phase 5: backpropagation
//
// Option A — manual: derive dLoss/dW for each op and implement backward()
//   on Tensor. Work backwards: lm_head -> final_ln -> blocks -> embeddings.
//
// Option B — candle: add candle-core to Cargo.toml, swap Tensor for
//   candle_core::Tensor, call loss.backward(), collect .grad() per param,
//   pass to Adam::step.
