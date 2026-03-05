// =============================================================================
// src/tensor/mod.rs — The foundational math layer
// =============================================================================
//
// WHAT THIS FILE IS
//   Everything in a neural network is a tensor (a multi-dimensional array of
//   numbers). This file defines that data structure and every mathematical
//   operation the model needs. Nothing here is magic — it's just loops over
//   Vec<f32>.
//
// WHY IT EXISTS SEPARATELY
//   By isolating all math here, the model code (attention, MLP, etc.) can stay
//   conceptually clean. When a bug produces a NaN, you trace it back to an op
//   in this file.
//
// KEY CONCEPTS
//   • Shape:     describes how data is arranged, e.g. [4, 128] = 4 rows of 128
//   • Matmul:    the core operation of every linear layer — O(M*K*N) naive
//   • Softmax:   turns raw scores into probabilities that sum to 1
//   • LayerNorm: normalises each row to mean=0, std=1, then re-scales
//                → stabilises training, especially in deep networks
//   • GELU:      smooth activation function used in GPT-2/3 instead of ReLU
//   • Causal mask: sets future positions to -inf before softmax so the model
//                  cannot "look ahead" during generation
//
// AUTOGRAD NOTE
//   This file implements FORWARD ops only. To train, you also need the
//   backward pass (gradients). See src/training/mod.rs for the TODO.

/// Core tensor type: row-major, f32, arbitrary shape.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(data.len(), total, "data length {} != shape product {}", data.len(), total);
        Tensor { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Tensor { data: vec![0.0; n], shape }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Tensor { data: vec![1.0; n], shape }
    }

    pub fn from_scalar(val: f32) -> Self {
        Tensor { data: vec![val], shape: vec![1] }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Reshape without copying data.
    /// Concept: the same numbers, viewed as a different grid. No math happens.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let n: usize = new_shape.iter().product();
        assert_eq!(n, self.numel(), "reshape: element count mismatch");
        Tensor { data: self.data.clone(), shape: new_shape }
    }

    /// 2-D transpose [M, N] -> [N, M].
    /// Concept: swap rows and columns. Required to align matrices before matmul.
    pub fn transpose_2d(&self) -> Self {
        assert_eq!(self.ndim(), 2, "transpose_2d requires 2-D tensor");
        let (m, n) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = self.data[i * n + j];
            }
        }
        Tensor::new(out, vec![n, m])
    }

    /// Elementwise add — same shape required.
    /// Used for residual connections: x = x + sublayer(x)
    pub fn add(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "add: shape mismatch {:?} vs {:?}", self.shape, other.shape);
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Elementwise multiply.
    pub fn mul(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "mul: shape mismatch");
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a * b).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Scale all elements by a scalar constant.
    /// Used to divide attention scores by sqrt(head_dim) for numerical stability.
    pub fn scale(&self, s: f32) -> Self {
        let data = self.data.iter().map(|x| x * s).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Matrix multiply: self [M, K] x other [K, N] -> [M, N].
    ///
    /// Concept: the core of every "Linear" layer. Given an input vector x
    /// and a weight matrix W, matmul(x, W) produces the layer's output.
    /// Complexity is O(M*K*N) — this is the bottleneck of every forward pass.
    pub fn matmul(&self, other: &Tensor) -> Self {
        assert_eq!(self.ndim(), 2);
        assert_eq!(other.ndim(), 2);
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k, k2, "matmul: inner dim mismatch {} vs {}", k, k2);

        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                for j in 0..n {
                    out[i * n + j] += a * other.data[p * n + j];
                }
            }
        }
        Tensor::new(out, vec![m, n])
    }

    /// Batched matmul: self [B, M, K] x other [B, K, N] -> [B, M, N].
    /// Runs matmul independently for each element of the batch dimension.
    pub fn bmm(&self, other: &Tensor) -> Self {
        assert_eq!(self.ndim(), 3);
        assert_eq!(other.ndim(), 3);
        let (b, m, k) = (self.shape[0], self.shape[1], self.shape[2]);
        let (b2, k2, n) = (other.shape[0], other.shape[1], other.shape[2]);
        assert_eq!(b, b2);
        assert_eq!(k, k2);

        let mut out = vec![0.0f32; b * m * n];
        for bi in 0..b {
            for i in 0..m {
                for p in 0..k {
                    let a = self.data[bi * m * k + i * k + p];
                    for j in 0..n {
                        out[bi * m * n + i * n + j] += a * other.data[bi * k * n + p * n + j];
                    }
                }
            }
        }
        Tensor::new(out, vec![b, m, n])
    }

    /// Softmax over last dimension.
    ///
    /// Concept: converts raw scores ("logits") into a probability distribution
    /// that sums to 1. The -max trick (log-sum-exp stabilisation) prevents
    /// exp() from overflowing for large values.
    pub fn softmax(&self) -> Self {
        assert!(self.ndim() >= 1);
        let last = *self.shape.last().unwrap();
        let rows = self.numel() / last;
        let mut out = self.data.clone();
        for r in 0..rows {
            let slice = &mut out[r * last..(r + 1) * last];
            let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for x in slice.iter_mut() {
                *x = (*x - max).exp();
                sum += *x;
            }
            for x in slice.iter_mut() {
                *x /= sum;
            }
        }
        Tensor::new(out, self.shape.clone())
    }

    /// Layer normalization over last dimension. eps = 1e-5.
    ///
    /// Concept: for each token position, subtract the mean and divide by the
    /// standard deviation, making the activations unit-normal. Then re-scale
    /// with learned weight (gamma) and bias (beta). This prevents activations
    /// from exploding or vanishing as they pass through many layers.
    /// Paper: https://arxiv.org/abs/1607.06450
    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor) -> Self {
        let last = *self.shape.last().unwrap();
        assert_eq!(weight.numel(), last);
        assert_eq!(bias.numel(), last);
        let rows = self.numel() / last;
        let mut out = self.data.clone();
        for r in 0..rows {
            let slice = &mut out[r * last..(r + 1) * last];
            let mean = slice.iter().sum::<f32>() / last as f32;
            let var = slice.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / last as f32;
            let std = (var + 1e-5).sqrt();
            for (i, x) in slice.iter_mut().enumerate() {
                *x = (*x - mean) / std * weight.data[i] + bias.data[i];
            }
        }
        Tensor::new(out, self.shape.clone())
    }

    /// GELU activation (tanh approximation).
    ///
    /// Concept: a smooth, non-linear activation function. Unlike ReLU (which
    /// hard-zeros negatives), GELU gates values probabilistically, which works
    /// better in transformers. Used in GPT-2, BERT, and most modern LLMs.
    /// Paper: https://arxiv.org/abs/1606.08415
    pub fn gelu(&self) -> Self {
        let data = self.data.iter().map(|&x| {
            0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        }).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Build a [T, T] causal attention mask.
    ///
    /// Concept: in an autoregressive model, position i must NOT see positions
    /// j > i (the future). Setting those scores to -infinity before softmax
    /// drives their attention weights to 0. This is what makes GPT "causal".
    pub fn causal_mask(t: usize) -> Self {
        let mut data = vec![0.0f32; t * t];
        for i in 0..t {
            for j in (i + 1)..t {
                data[i * t + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::new(data, vec![t, t])
    }
}
