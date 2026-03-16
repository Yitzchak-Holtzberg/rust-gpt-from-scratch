use nano_gpt_rs::tensor::Tensor;
use nano_gpt_rs::training::cross_entropy_loss;

#[test]
fn test_cross_entropy_uniform_distribution() {
    // All-zero logits → uniform softmax → loss should equal ln(vocab_size)
    let vocab_size = 10;
    let t = 3;
    let logits = Tensor::zeros(vec![t, vocab_size]);
    let targets = vec![0, 5, 9];

    let loss = cross_entropy_loss(&logits, &targets);
    let expected = (vocab_size as f32).ln(); // ln(10) ≈ 2.3026

    assert!(
        (loss - expected).abs() < 1e-5,
        "expected {expected}, got {loss}"
    );
}

#[test]
fn test_cross_entropy_confident_prediction() {
    // If one logit is much higher than the rest, loss should be near zero
    let mut data = vec![0.0f32; 3 * 5]; // [3, 5]
    data[2] = 100.0;  // row 0: target is index 2, logit is 100
    data[8] = 100.0;  // row 1: target is index 3, logit is 100
    data[10] = 100.0; // row 2: target is index 0, logit is 100

    let logits = Tensor::new(data, vec![3, 5]);
    let targets = vec![2, 3, 0];

    let loss = cross_entropy_loss(&logits, &targets);
    assert!(loss < 0.01, "expected near-zero loss, got {loss}");
}
