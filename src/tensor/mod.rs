/// Row-major, f32, arbitrary shape.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len = shape.iter().product();
        assert_eq!(data.len(), expected_len, "Data length doesn't match shape");
        Tensor { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let number_of_zeros = shape.iter().product();
        let zero_contents = vec![0.0f32; number_of_zeros];
        Tensor::new(zero_contents, shape)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let number_of_ones = shape.iter().product();
        let zero_contents = vec![1.0f32; number_of_ones];
        Tensor::new(zero_contents, shape)
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Reinterpret data as a different shape. No data copy.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        assert_eq!(
            new_shape.iter().product::<usize>(),
            self.numel(),
            "shape mismatch"
        );
        Tensor {
            data: self.data.clone(),
            shape: new_shape,
        }
    }

    /// [M, N] -> [N, M]
    pub fn transpose_2d(&self) -> Self {
        assert_eq!(self.shape.len(), 2, "tensor is not 2D");
        let row_count = self.shape[0];
        let column_count = self.shape[1];
        let mut new_data = vec![0.0; row_count * column_count];

        // Data is flat. Element at (i, j) lives at i*cols+j in the original.
        // After transposing, that element belongs at (j, i), which is j*rows+i in the output.
        for i in 0..row_count {
            for j in 0..column_count {
                new_data[j * row_count + i] = self.data[i * column_count + j];
            }
        }

        Tensor {
            data: new_data,
            shape: vec![column_count, row_count],
        }
    }

    /// Elementwise add. Shapes must match.
    pub fn add(&self, other: &Tensor) -> Self {
        assert_eq!(
            self.shape,
            other.shape,
            "vectors to add don't match"
        );

        let new_totaled_data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor {
            data: new_totaled_data,
            shape: self.shape.clone(),
        }
    }

    /// Elementwise multiply. Shapes must match.
    pub fn mul(&self, other: &Tensor) -> Self {
        assert_eq!(
            self.shape,
            other.shape,
            "vectors to multiply don't match"
        );

        let new_totaled_data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor {
            data: new_totaled_data,
            shape: self.shape.clone(),
        }
    }

    /// Multiply every element by a scalar.
    pub fn scale(&self, s: f32) -> Self {
        let new_totaled_data = self.data.iter().map(|a| a * s).collect();

        Tensor {
            data: new_totaled_data,
            shape: self.shape.clone(),
        }
    }

    /// [M, K] x [K, N] -> [M, N]
    pub fn matmul(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape[1], other.shape[0], "inner dimensions must match");
        let mut result = vec![0.0f32; self.shape[0] * other.shape[1]];
        let length_i = self.shape[0];
        let length_j = other.shape[1];
        let length_k = self.shape[1];
        let mut current_value = 0.0f32;
        for i in 0..length_i {
            for j in 0..length_j {
                for k in 0..length_k {
                    current_value += self.data[i * length_k + k] * other.data[k * length_j + j];
                }
                result[i * length_j + j] = current_value;
                current_value = 0.0;
            }
        }
        Tensor {
            data: result,
            shape: vec![length_i, length_j],
        }
    }

    /// Softmax over the last dimension.
    pub fn softmax(&self) -> Self {
        let exps: Vec<f32> = self.data.iter().map(|a| a.exp()).collect();
        let sum: f32 = exps.iter().sum();
        let softmaxed_data: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        Tensor {
            data: softmaxed_data,
            shape: self.shape.clone(),
        }
    }

    /// Layer norm over the last dimension with learned weight and bias.
    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor) -> Self {
        let mean = self.data.iter().sum::<f32>() / self.numel() as f32;
        let variance =
            self.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.numel() as f32;
        let sqrt_variance = variance.sqrt();
        let normalized: Vec<f32> = self
            .data
            .iter()
            .map(|x| (x - mean) / (sqrt_variance + 1e-5))
            .collect();
        let final_data = normalized
            .iter()
            .zip(weight.data.iter())
            .zip(bias.data.iter())
            .map(|((n, w), b)| n * w + b)
            .collect();
        Tensor {
            data: final_data,
            shape: self.shape.clone(),
        }
    }

    /// GELU activation (tanh approximation).
    pub fn gelu(&self) -> Self {
        let final_data: Vec<f32> = self
            .data
            .iter()
            .map(|x| {
                let cubed = x.powi(3);
                let inner = x + 0.044715 * cubed;
                let scale = 0.797_884_6_f32 * inner;
                0.5 * x * (1.0 + scale.tanh())
            })
            .collect();
        Tensor {
            data: final_data,
            shape: self.shape.clone(),
        }
    }

    /// Returns a [T, T] matrix with 0 on/below diagonal and -inf above.
    /// Used to prevent attention from looking at future positions.
    pub fn causal_mask(t: usize) -> Self {
        let mut result = vec![0.0f32; t * t];
        for row in 0..t {
            for column in 0..t {
                if column > row {
                    result[row * t + column] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor {
            data: result,
            shape: vec![t, t],
        }
    }
}
