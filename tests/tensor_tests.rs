use nano_gpt_rs::tensor::Tensor;

// --- Tensor::new ---

#[test]
fn test_new_valid() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(t.shape, vec![2, 2]);
}

#[test]
#[should_panic]
fn test_new_shape_mismatch_panics() {
    Tensor::new(vec![1.0, 2.0], vec![3, 3]);
}
#[test]
fn test_create_zeros_works() {
    let t = Tensor::zeros(vec![5, 6]);
    assert_eq!(t.shape, vec![5, 6]);
    assert_eq!(t.data, vec![0.0f32; 30]);
}

#[test]
fn test_create_ones_works() {
    let t = Tensor::ones(vec![5, 6]);
    assert_eq!(t.shape, vec![5, 6]);
    assert_eq!(t.data, vec![1.0f32; 30]);
}

#[test]
fn test_numel_works() {
    let t = Tensor::ones(vec![5, 6]);
    let numel_result = Tensor::numel(&t);
    assert_eq!(numel_result, 30);
}

#[test]
fn test_ndim_works() {
    let t = Tensor::ones(vec![5, 6]);
    let ndim_result = Tensor::ndim(&t);
    assert_eq!(ndim_result, 2);
}
// --- Tensor::transpose_2d ---

#[test]
fn test_transpose_2d_shape() {
    // [2, 3] -> [3, 2]
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let result = t.transpose_2d();
    assert_eq!(result.shape, vec![3, 2]);
}

#[test]
fn test_transpose_2d_values() {
    // Original:        Transposed:
    // 1 2 3            1 4
    // 4 5 6            2 5
    //                  3 6
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let result = t.transpose_2d();
    assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
#[should_panic]
fn test_transpose_2d_non_2d_panics() {
    let t = Tensor::ones(vec![2, 3, 4]);
    t.transpose_2d();
}

// --- Tensor::add ---

#[test]
fn test_add_values() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
    let result = a.add(&b);
    assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
}

#[test]
#[should_panic]
fn test_add_shape_mismatch_panics() {
    let a = Tensor::ones(vec![2, 3]);
    let b = Tensor::ones(vec![3, 2]);
    a.add(&b);
}

// --- Tensor::mul ---

#[test]
fn test_mul_values() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
    let result = a.mul(&b);
    assert_eq!(result.data, vec![4.0, 10.0, 18.0]);
}

#[test]
#[should_panic]
fn test_mul_shape_mismatch_panics() {
    let a = Tensor::ones(vec![2, 3]);
    let b = Tensor::ones(vec![3, 2]);
    a.mul(&b);
}

// --- Tensor::scale ---

#[test]
fn test_scale_values() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let result = t.scale(3.0);
    assert_eq!(result.data, vec![3.0, 6.0, 9.0]);
}

#[test]
fn test_scale_by_zero() {
    let t = Tensor::ones(vec![4]);
    let result = t.scale(0.0);
    assert_eq!(result.data, vec![0.0; 4]);
}

// --- Tensor::matmul ---

#[test]
fn test_matmul_shape() {
    // [2, 3] x [3, 4] -> [2, 4]
    let a = Tensor::ones(vec![2, 3]);
    let b = Tensor::ones(vec![3, 4]);
    let result = a.matmul(&b);
    assert_eq!(result.shape, vec![2, 4]);
}

#[test]
fn test_matmul_values() {
    // A:        B:        output:
    // 1 2       5 6       19 22
    // 3 4       7 8       43 50
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let result = a.matmul(&b);
    assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_matmul_non_square() {
    // [2, 3] x [3, 2] -> [2, 2]
    // A:          B:        output:
    // 1 2 3       7  8      58  64
    // 4 5 6       9  10     139 154
    //             11 12
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
    let result = a.matmul(&b);
    assert_eq!(result.shape, vec![2, 2]);
    assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
#[should_panic]
fn test_matmul_inner_dim_mismatch_panics() {
    let a = Tensor::ones(vec![2, 3]);
    let b = Tensor::ones(vec![4, 2]);
    a.matmul(&b);
}

// --- Tensor::softmax ---

#[test]
fn test_softmax_output_sums_to_one() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let result = t.softmax();
    let sum: f32 = result.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1");
}

#[test]
fn test_softmax_shape_preserved() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let result = t.softmax();
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn test_softmax_largest_gets_highest_prob() {
    let t = Tensor::new(vec![1.0, 2.0, 10.0], vec![3]);
    let result = t.softmax();
    assert!(result.data[2] > result.data[1]);
    assert!(result.data[1] > result.data[0]);
}

// --- Tensor::layer_norm ---

#[test]
fn test_layer_norm_shape_preserved() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let weight = Tensor::ones(vec![4]);
    let bias = Tensor::zeros(vec![4]);
    let result = t.layer_norm(&weight, &bias);
    assert_eq!(result.shape, vec![4]);
}

#[test]
fn test_layer_norm_zero_mean() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let weight = Tensor::ones(vec![4]);
    let bias = Tensor::zeros(vec![4]);
    let result = t.layer_norm(&weight, &bias);
    let mean: f32 = result.data.iter().sum::<f32>() / result.data.len() as f32;
    assert!(mean.abs() < 1e-5, "layer norm output should have ~zero mean");
}

// --- Tensor::gelu ---

#[test]
fn test_gelu_shape_preserved() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let result = t.gelu();
    assert_eq!(result.shape, vec![3]);
}

#[test]
fn test_gelu_positive_input_stays_positive() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    let result = t.gelu();
    for v in &result.data {
        assert!(*v > 0.0);
    }
}

#[test]
fn test_gelu_zero_input_is_zero() {
    let t = Tensor::new(vec![0.0], vec![1]);
    let result = t.gelu();
    assert!((result.data[0]).abs() < 1e-6);
}

// --- Tensor::causal_mask ---

#[test]
fn test_causal_mask_shape() {
    let mask = Tensor::causal_mask(4);
    assert_eq!(mask.shape, vec![4, 4]);
}

#[test]
fn test_causal_mask_diagonal_is_zero() {
    let mask = Tensor::causal_mask(3);
    // diagonal and below should be 0
    assert_eq!(mask.data[0], 0.0); // [0,0]
    assert_eq!(mask.data[4], 0.0); // [1,1]
    assert_eq!(mask.data[8], 0.0); // [2,2]
}

#[test]
fn test_causal_mask_above_diagonal_is_neg_inf() {
    let mask = Tensor::causal_mask(3);
    // above diagonal should be -inf
    assert_eq!(mask.data[1], f32::NEG_INFINITY); // [0,1]
    assert_eq!(mask.data[2], f32::NEG_INFINITY); // [0,2]
    assert_eq!(mask.data[5], f32::NEG_INFINITY); // [1,2]
}

// --- Tensor::ones ---
// Write your test here before implementing ones!
