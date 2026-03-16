/// Row-major, f32, arbitrary shape.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
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
        let one_contents = vec![1.0f32; number_of_ones];
        Tensor::new(one_contents, shape)
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get row i as a slice. Only valid for 2D tensors.
    /// Returns a slice representing row `i`.
    pub fn row(&self, i: usize) -> &[f32] {
        assert_eq!(self.ndim(), 2, "row() is only valid for 2D tensors");
        self.data
            .chunks_exact(self.shape[1])
            .nth(i)
            .expect("row index out of bounds")
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

    /// Select rows by index from a 2D tensor [N, D] -> [T, D].
    pub fn slice_rows(&self, indices: &[usize]) -> Self {
        let mut result_data = Vec::new();
        for &index in indices {
            result_data.extend_from_slice(self.row(index));
        }
        Tensor {
            data: result_data,
            shape: vec![indices.len(), self.shape[1]],
        }
    }

    /// Extract a column range from a 2D tensor [T, D] -> [T, end-start].
    pub fn slice_cols(&self, start: usize, end: usize) -> Self {
        let rows = self.shape[0];
        let mut new_data = Vec::new();
        for i in 0..rows {
            new_data.extend_from_slice(&self.row(i)[start..end]);
        }
        Tensor {
            data: new_data,
            shape: vec![rows, end - start],
        }
    }

    /// Concatenate columns of two 2D tensors [T, A] and [T, B] -> [T, A+B].
    pub fn concat_cols(&self, other: &Tensor) -> Self {
        let rows = self.shape[0];
        let mut new_data = Vec::new();
        // For each row, take all columns from self then all columns from other.
        for i in 0..rows {
            new_data.extend_from_slice(self.row(i));
            new_data.extend_from_slice(other.row(i));
        }
        Tensor {
            data: new_data,
            shape: vec![rows, self.shape[1] + other.shape[1]],
        }
    }

    /// Add a 1D bias [N] to every row of a 2D tensor [rows, N].
    pub fn add_bias(&self, bias: &Tensor) -> Self {
        assert_eq!(self.ndim(), 2, "tensor must be 2D");
        assert_eq!(bias.ndim(), 1, "bias must be 1D");
        assert_eq!(
            self.shape[1], bias.shape[0],
            "bias length must match number of columns"
        );
        let rows = self.shape[0];
        let mut new_data = Vec::new();
        for i in 0..rows {
            let updated_row: Vec<f32> = self
                .row(i)
                .iter()
                .zip(bias.data.iter())
                .map(|(a, b)| a + b)
                .collect();
            new_data.extend_from_slice(&updated_row);
        }
        Tensor {
            shape: self.shape.clone(),
            data: new_data,
        }
    }

    /// Elementwise add. Shapes must match.
    pub fn add(&self, other: &Tensor) -> Self {
        assert_eq!(self.shape, other.shape, "vectors to add don't match");

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
        assert_eq!(self.shape, other.shape, "vectors to multiply don't match");

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

    /// Softmax independently per row of a 2D tensor [rows, N].
    /// Each row's values will sum to 1.0.
    pub fn softmax_rows(&self) -> Self {
        assert_eq!(self.ndim(), 2, "tensor must be 2D");
        let rows = self.shape[0];
        let mut result = Vec::new();
        for i in 0..rows {
            let exps: Vec<f32> = self.row(i).iter().map(|a| a.exp()).collect();
            let sum: f32 = exps.iter().sum();
            result.extend(exps.iter().map(|e| e / sum));
        }

        Tensor {
            shape: self.shape.clone(),
            data: result,
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
        // Require 2D input — we normalise each token (row) independently
        let [rows, cols] = self.shape[..] else {
            panic!("expected 2D tensor")
        };
        let mut out_data = Vec::with_capacity(rows * cols);

        for i in 0..rows {
            let row = self.row(i);

            // Compute mean and variance across this token's embedding vector
            let mean = row.iter().sum::<f32>() / cols as f32;
            let variance = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / cols as f32;
            let std = (variance + 1e-5).sqrt(); // 1e-5 prevents division by zero

            // Normalise, then apply learned scale (weight) and shift (bias) per dimension
            out_data
                .extend((0..cols).map(|j| (row[j] - mean) / std * weight.data[j] + bias.data[j]));
        }

        Tensor {
            data: out_data,
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
