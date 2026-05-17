const GELU_COEFF: f32 = 0.797_884_6;
const LAYER_NORM_EPS: f32 = 1e-5;

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

    /// Random uniform init in [-scale, scale]. Useful for quick prototyping
    /// when proper Xavier/Kaiming init isn't needed.
    pub fn randn(shape: Vec<usize>, scale: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let count = shape.iter().product();
        let data: Vec<f32> = (0..count).map(|_| rng.gen_range(-scale..scale)).collect();
        Tensor::new(data, shape)
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
            let std = (variance + LAYER_NORM_EPS).sqrt();

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
                let scale = GELU_COEFF * inner;
                0.5 * x * (1.0 + scale.tanh())
            })
            .collect();
        Tensor {
            data: final_data,
            shape: self.shape.clone(),
        }
    }

    /// Backward pass for matmul: z = self @ other
    ///
    /// Steps:
    ///   grad_self  = grad_out @ other^T
    ///   grad_other = self^T @ grad_out
    ///   return (grad_self, grad_other)
    pub fn matmul_backward(&self, other: &Tensor, grad_out: &Tensor) -> (Tensor, Tensor) {
        assert_eq!(self.ndim(), 2, "lhs must be 2D");
        assert_eq!(other.ndim(), 2, "rhs must be 2D");
        assert_eq!(grad_out.ndim(), 2, "grad_out must be 2D");
        todo!()
    }

    /// Backward pass for element-wise add: z = self + other
    ///
    /// Steps:
    ///   grad_self  = grad_out
    ///   grad_other = grad_out
    ///   return (grad_self, grad_other)
    pub fn add_backward(&self, other: &Tensor, grad_out: &Tensor) -> (Tensor, Tensor) {
        assert_eq!(self.shape, other.shape, "vectors to add don't match");
        assert_eq!(self.shape, grad_out.shape, "shape mismatch");
        (grad_out.clone(), grad_out.clone())
    }

    /// Backward pass for add_bias: z = self + bias (bias is 1D, broadcast across rows)
    ///
    /// Steps:
    ///   grad_self = grad_out
    ///   grad_bias = for each column: sum grad_out across all rows
    ///   return (grad_self, grad_bias)
    pub fn add_bias_backward(&self, bias: &Tensor, grad_out: &Tensor) -> (Tensor, Tensor) {
        assert_eq!(self.ndim(), 2, "tensor must be 2D");
        assert_eq!(bias.ndim(), 1, "bias must be 1D");
        assert_eq!(self.shape, grad_out.shape, "shape mismatch");
        assert_eq!(
            self.shape[1], bias.shape[0],
            "bias length must match number of columns"
        );
        let cols = grad_out.shape[1];
        let mut second_half = vec![0.0_f32; cols];

        for row in grad_out.data.chunks(cols) {
            for (j, value) in row.iter().enumerate() {
                second_half[j] += value;
            }
        }

        (
            Tensor {
                data: grad_out.data.clone(),
                shape: grad_out.shape.clone(),
            },
            Tensor {
                data: second_half,
                shape: vec![cols],
            },
        )
    }

    /// Backward pass for slice_rows: z = self.slice_rows(indices)
    ///
    /// Steps:
    ///   grad = zeros with same shape as self
    ///   for each i, grad[indices[i]] += grad_out[i]   (scatter-add back)
    ///   return grad
    pub fn slice_rows_backward(&self, indices: &[usize], grad_out: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "tensor must be 2D");
        assert_eq!(grad_out.ndim(), 2, "grad_out must be 2D");
        assert_eq!(
            grad_out.shape[0],
            indices.len(),
            "indices length must match grad_out rows"
        );
        assert_eq!(grad_out.shape[1], self.shape[1], "column count mismatch");
        let num_columns = self.shape[1];
        let mut gradient = Tensor::zeros(self.shape.clone());
        for (grad_out_row_index, &original_row_index) in indices.iter().enumerate() {
            let incoming_gradient_row = grad_out.row(grad_out_row_index);
            for (column_index, &incoming_gradient_value) in incoming_gradient_row.iter().enumerate() {
                gradient.data[original_row_index * num_columns + column_index] += incoming_gradient_value;
            }
        }
        gradient
    }

    /// Backward pass for slice_cols: z = self.slice_cols(start, end)
    ///
    /// Steps:
    ///   grad = zeros with same shape as self
    ///   for each row: copy grad_out columns into grad[row][start..end]
    ///   return grad
    pub fn slice_cols_backward(&self, start: usize, end: usize, grad_out: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "tensor must be 2D");
        assert_eq!(grad_out.ndim(), 2, "grad_out must be 2D");
        assert_eq!(end - start, grad_out.shape[1], "slice width mismatch");
        assert_eq!(grad_out.shape[0], self.shape[0], "row count mismatch");
        let num_columns = self.shape[1];
        let mut gradient = Tensor::zeros(self.shape.clone());
        for (row_index, incoming_gradient_row) in grad_out.data.chunks(end - start).enumerate() {
            let dest_start = row_index * num_columns + start;
            gradient.data[dest_start..dest_start + (end - start)].copy_from_slice(incoming_gradient_row);
        }
        gradient
    }

    /// Backward pass for concat_cols: z = self | other (side by side)
    ///
    /// Steps:
    ///   grad_self  = grad_out.slice_cols(0, self.cols)
    ///   grad_other = grad_out.slice_cols(self.cols, self.cols + other.cols)
    ///   return (grad_self, grad_other)
    pub fn concat_cols_backward(&self, other: &Tensor, grad_out: &Tensor) -> (Tensor, Tensor) {
        assert_eq!(self.ndim(), 2, "lhs must be 2D");
        assert_eq!(other.ndim(), 2, "rhs must be 2D");
        assert_eq!(grad_out.ndim(), 2, "grad_out must be 2D");
        assert_eq!(
            grad_out.shape[1],
            self.shape[1] + other.shape[1],
            "column count mismatch"
        );
        let left_tensor_column_count = self.shape[1];
        let right_tensor_end_column = left_tensor_column_count + other.shape[1];
        (
            grad_out.slice_cols(0, left_tensor_column_count),
            grad_out.slice_cols(left_tensor_column_count, right_tensor_end_column),
        )
    }

    /// Backward pass for transpose_2d: z = self^T
    ///
    /// Steps:
    ///   grad_self = grad_out^T
    pub fn transpose_2d_backward(&self, grad_out: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "tensor is not 2D");
        assert_eq!(grad_out.ndim(), 2, "grad_out must be 2D");
        grad_out.transpose_2d()
    }

    /// Backward pass for scale: z = self * s
    ///
    /// Steps:
    ///   grad_self = grad_out * s
    pub fn scale_backward(&self, s: f32, grad_out: &Tensor) -> Tensor {
        assert_eq!(self.shape, grad_out.shape, "shape mismatch");
        grad_out.scale(s)
    }

    /// Backward pass for GELU: z = gelu(self)
    /// self is the original input (needed to compute the derivative).
    ///
    /// Steps:
    ///   for each element x = self[i]:
    ///     inner     = 0.797884_6 * (x + 0.044715 * x^3)
    ///     tanh_val  = tanh(inner)
    ///     sech2     = 1 - tanh_val^2
    ///     d_inner   = 0.797884_6 * (1 + 3 * 0.044715 * x^2)
    ///     gelu_grad = 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * d_inner
    ///     grad[i]   = grad_out[i] * gelu_grad
    pub fn gelu_backward(&self, grad_out: &Tensor) -> Tensor {
        assert_eq!(self.shape, grad_out.shape, "shape mismatch");
        let output_data: Vec<f32> = self.data.iter().zip(grad_out.data.iter()).map(|(&input_value, &upstream_gradient)| {
            let tanh_argument = GELU_COEFF * (input_value + 0.044715 * input_value.powi(3));
            let tanh_value = tanh_argument.tanh();
            let sech_squared = 1.0 - tanh_value.powi(2);
            let tanh_argument_derivative = GELU_COEFF * (1.0 + 3.0 * 0.044715 * input_value.powi(2));
            let gelu_local_derivative = 0.5 * (1.0 + tanh_value) + 0.5 * input_value * sech_squared * tanh_argument_derivative;
            upstream_gradient * gelu_local_derivative
        }).collect();
        self.with_data(output_data)
    }

    /// Backward pass for softmax_rows.
    /// self is the softmax output (probabilities).
    ///
    /// Steps:
    ///   for each row:
    ///     dot       = sum(self[row][j] * grad_out[row][j])  for all j
    ///     grad[row][j] = self[row][j] * (grad_out[row][j] - dot)
    pub fn softmax_rows_backward(&self, grad_out: &Tensor) -> Tensor {
        assert_eq!(self.shape, grad_out.shape, "shape mismatch");
        assert_eq!(self.ndim(), 2, "softmax_out must be 2D");
        let num_columns = self.shape[1];
        let output_data: Vec<f32> = self.data.chunks(num_columns)
            .zip(grad_out.data.chunks(num_columns))
            .flat_map(|(probability_row, upstream_gradient_row)| {
                let dot_product: f32 = probability_row.iter()
                    .zip(upstream_gradient_row.iter())
                    .map(|(probability, upstream_gradient)| probability * upstream_gradient)
                    .sum();
                probability_row.iter()
                    .zip(upstream_gradient_row.iter())
                    .map(move |(&probability, &upstream_gradient)| probability * (upstream_gradient - dot_product))
                    .collect::<Vec<_>>()
            })
            .collect();
        self.with_data(output_data)
    }

    /// Backward pass for layer_norm.
    /// self is the original input before normalization.
    /// Returns (grad_input, grad_weight, grad_bias).
    ///
    /// Steps:
    ///   for each row:
    ///     mean      = mean of self[row]
    ///     variance  = mean of (self[row] - mean)^2
    ///     inv_std   = 1 / sqrt(variance + eps)
    ///     x_hat[j]  = (self[row][j] - mean) * inv_std        (normalized value)
    ///     dx_hat[j] = grad_out[row][j] * weight[j]
    ///     grad_weight[j] += grad_out[row][j] * x_hat[j]      (accumulate across rows)
    ///     grad_bias[j]   += grad_out[row][j]                  (accumulate across rows)
    ///     sum_dxh         = sum of dx_hat[j]
    ///     sum_dxh_xh      = sum of dx_hat[j] * x_hat[j]
    ///     grad_input[row][j] = inv_std / cols * (cols * dx_hat[j] - sum_dxh - x_hat[j] * sum_dxh_xh)
    fn normalize_row_with_stats(row: &[f32]) -> (f32, Vec<f32>) {
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / row.len() as f32;
        let inverse_std_dev = 1.0 / (variance + LAYER_NORM_EPS).sqrt();
        let normalized_values = row.iter().map(|value| (value - mean) * inverse_std_dev).collect();
        (inverse_std_dev, normalized_values)
    }

    fn with_data(&self, data: Vec<f32>) -> Tensor {
        Tensor { data, shape: self.shape.clone() }
    }

    pub fn layer_norm_backward(
        &self,
        weight: &Tensor,
        grad_out: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        assert_eq!(self.shape, grad_out.shape, "shape mismatch");
        assert_eq!(self.ndim(), 2, "input must be 2D");
        assert_eq!(weight.ndim(), 1, "weight must be 1D");
        let embedding_dim = self.shape[1];
        let embedding_dim_f = embedding_dim as f32;
        let mut input_gradient_data = vec![0.0f32; self.data.len()];
        let mut weight_gradient = vec![0.0f32; embedding_dim];
        let mut bias_gradient = vec![0.0f32; embedding_dim];

        for (row_index, (input_values, upstream_gradient_row)) in self
            .data.chunks(embedding_dim)
            .zip(grad_out.data.chunks(embedding_dim))
            .enumerate()
        {
            let (inverse_std_dev, normalized_values) = Self::normalize_row_with_stats(input_values);

            let mut sum_weighted_grad = 0.0f32;
            let mut sum_weighted_grad_times_normed = 0.0f32;

            for col_index in 0..embedding_dim {
                let grad_scaled_by_weight = upstream_gradient_row[col_index] * weight.data[col_index];
                weight_gradient[col_index] += upstream_gradient_row[col_index] * normalized_values[col_index];
                bias_gradient[col_index] += upstream_gradient_row[col_index];
                sum_weighted_grad += grad_scaled_by_weight;
                sum_weighted_grad_times_normed += grad_scaled_by_weight * normalized_values[col_index];
            }

            for col_index in 0..embedding_dim {
                let grad_scaled_by_weight = upstream_gradient_row[col_index] * weight.data[col_index];
                input_gradient_data[row_index * embedding_dim + col_index] = inverse_std_dev / embedding_dim_f
                    * (embedding_dim_f * grad_scaled_by_weight - sum_weighted_grad - normalized_values[col_index] * sum_weighted_grad_times_normed);
            }
        }
        (
            self.with_data(input_gradient_data),
            Tensor { data: weight_gradient, shape: vec![embedding_dim] },
            Tensor { data: bias_gradient, shape: vec![embedding_dim] },
        )
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
