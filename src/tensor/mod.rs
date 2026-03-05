/// Row-major, f32, arbitrary shape.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        todo!()
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        todo!()
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        todo!()
    }

    pub fn numel(&self) -> usize {
        todo!()
    }

    pub fn ndim(&self) -> usize {
        todo!()
    }

    /// Reinterpret data as a different shape. No data copy.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        todo!()
    }

    /// [M, N] -> [N, M]
    pub fn transpose_2d(&self) -> Self {
        todo!()
    }

    /// Elementwise add. Shapes must match.
    pub fn add(&self, other: &Tensor) -> Self {
        todo!()
    }

    /// Elementwise multiply. Shapes must match.
    pub fn mul(&self, other: &Tensor) -> Self {
        todo!()
    }

    /// Multiply every element by a scalar.
    pub fn scale(&self, s: f32) -> Self {
        todo!()
    }

    /// [M, K] x [K, N] -> [M, N]
    pub fn matmul(&self, other: &Tensor) -> Self {
        todo!()
    }

    /// Softmax over the last dimension.
    pub fn softmax(&self) -> Self {
        todo!()
    }

    /// Layer norm over the last dimension with learned weight and bias.
    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor) -> Self {
        todo!()
    }

    /// GELU activation (tanh approximation).
    pub fn gelu(&self) -> Self {
        todo!()
    }

    /// Returns a [T, T] matrix with 0 on/below diagonal and -inf above.
    /// Used to prevent attention from looking at future positions.
    pub fn causal_mask(t: usize) -> Self {
        todo!()
    }
}
