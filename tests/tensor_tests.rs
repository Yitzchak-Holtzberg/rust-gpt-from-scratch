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
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let weight = Tensor::ones(vec![4]);
    let bias = Tensor::zeros(vec![4]);
    let result = t.layer_norm(&weight, &bias);
    assert_eq!(result.shape, vec![1, 4]);
}

#[test]
fn test_layer_norm_zero_mean() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
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

// --- Tensor::softmax_rows ---

#[test]
fn test_softmax_rows_shape() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let result = t.softmax_rows();
    assert_eq!(result.shape, vec![2, 3]);
}

#[test]
fn test_softmax_rows_each_row_sums_to_one() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let result = t.softmax_rows();
    // row 0 should sum to 1.0
    let row0_sum: f32 = result.data[0..3].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "row 0 sum was {}", row0_sum);
    // row 1 should sum to 1.0
    let row1_sum: f32 = result.data[3..6].iter().sum();
    assert!((row1_sum - 1.0).abs() < 1e-5, "row 1 sum was {}", row1_sum);
}

#[test]
fn test_softmax_rows_larger_values_get_higher_weight() {
    let t = Tensor::new(vec![0.0, 0.0, 10.0, 10.0, 0.0, 0.0], vec![2, 3]);
    let result = t.softmax_rows();
    // In row 0, column 2 (value 10) should dominate
    assert!(result.data[2] > result.data[0]);
    // In row 1, column 0 (value 10) should dominate
    assert!(result.data[3] > result.data[5]);
}

#[test]
fn test_softmax_rows_independent_per_row() {
    // Two identical rows should produce identical results
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], vec![2, 3]);
    let result = t.softmax_rows();
    assert!((result.data[0] - result.data[3]).abs() < 1e-6);
    assert!((result.data[1] - result.data[4]).abs() < 1e-6);
    assert!((result.data[2] - result.data[5]).abs() < 1e-6);
}

// --- Tensor::add_bias ---

#[test]
fn test_add_bias_shape() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let bias = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let result = t.add_bias(&bias);
    assert_eq!(result.shape, vec![2, 3]);
}

#[test]
fn test_add_bias_values() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let bias = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
    let result = t.add_bias(&bias);
    // row 0: [1+10, 2+20, 3+30] = [11, 22, 33]
    // row 1: [4+10, 5+20, 6+30] = [14, 25, 36]
    assert_eq!(result.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn test_add_bias_single_row() {
    let t = Tensor::new(vec![5.0, 6.0], vec![1, 2]);
    let bias = Tensor::new(vec![0.5, 0.5], vec![2]);
    let result = t.add_bias(&bias);
    assert_eq!(result.data, vec![5.5, 6.5]);
}

#[test]
#[should_panic]
fn test_add_bias_wrong_size_panics() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let bias = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
    t.add_bias(&bias);
}

// --- Tensor::slice_cols ---

/// Slicing columns 1..3 from a [2, 4] tensor should give [2, 2].
#[test]
fn test_slice_cols_shape() {
    // Row 0: [1, 2, 3, 4]
    // Row 1: [5, 6, 7, 8]
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
    let sliced = t.slice_cols(1, 3);
    assert_eq!(sliced.shape, vec![2, 2]);
}

/// Values should be columns 1 and 2 from each row.
#[test]
fn test_slice_cols_values() {
    // Row 0: [1, 2, 3, 4]  → cols 1..3 → [2, 3]
    // Row 1: [5, 6, 7, 8]  → cols 1..3 → [6, 7]
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 4]);
    let sliced = t.slice_cols(1, 3);
    assert_eq!(sliced.data, vec![2.0, 3.0, 6.0, 7.0]);
}

// --- Tensor::concat_cols ---

/// Concatenating [2, 2] and [2, 2] should give [2, 4].
#[test]
fn test_concat_cols_shape() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let result = a.concat_cols(&b);
    assert_eq!(result.shape, vec![2, 4]);
}

/// Row 0 of a=[1,2], row 0 of b=[5,6] → combined row 0 = [1,2,5,6].
#[test]
fn test_concat_cols_values() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let result = a.concat_cols(&b);
    assert_eq!(result.data, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
}

// --- slice_rows ---

/// Picking rows 1 and 0 (in that order) from a [3, 2] tensor returns [2, 2].
#[test]
fn test_slice_rows_shape() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let result = t.slice_rows(&[1, 0]);
    assert_eq!(result.shape, vec![2, 2]);
}

/// Row 1 is [3,4], row 0 is [1,2] — result should be [[3,4],[1,2]].
#[test]
fn test_slice_rows_values() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let result = t.slice_rows(&[1, 0]);
    assert_eq!(result.data, vec![3.0, 4.0, 1.0, 2.0]);
}

// --- Tensor::ones ---
// Write your test here before implementing ones!
