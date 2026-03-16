use nano_gpt_rs::tokenizer::{CharTokenizer, Dataset};

// --- CharTokenizer::from_text / vocab_size ---

#[test]
fn test_vocab_contains_all_unique_chars() {
    let tok = CharTokenizer::from_text("hello");
    let mut chars: Vec<char> = tok.vocab.clone();
    chars.sort();
    chars.dedup();
    assert_eq!(chars.len(), tok.vocab.len());
}

#[test]
fn test_vocab_size_matches_unique_chars() {
    let tok = CharTokenizer::from_text("aabbcc");
    assert_eq!(tok.vocab_size(), 3);
}

#[test]
fn test_vocab_is_sorted() {
    let tok = CharTokenizer::from_text("cba");
    assert_eq!(tok.vocab, vec!['a', 'b', 'c']);
}

// --- CharTokenizer::encode ---

#[test]
fn test_encode_length_matches_input() {
    let tok = CharTokenizer::from_text("hello");
    let ids = tok.encode("hello");
    assert_eq!(ids.len(), 5);
}

#[test]
fn test_encode_same_char_same_id() {
    let tok = CharTokenizer::from_text("hello");
    let ids = tok.encode("ll");
    assert_eq!(ids[0], ids[1]);
}

#[test]
fn test_encode_unknown_char_dropped() {
    let tok = CharTokenizer::from_text("abc");
    let ids = tok.encode("axb"); // 'x' not in vocab
    assert_eq!(ids.len(), 2);
}

// --- CharTokenizer::decode ---

#[test]
fn test_decode_roundtrip() {
    let tok = CharTokenizer::from_text("hello world");
    let ids = tok.encode("hello");
    let result = tok.decode(&ids);
    assert_eq!(result, "hello");
}

#[test]
fn test_encode_decode_roundtrip_full() {
    let tok = CharTokenizer::from_text("hello world");
    let original = "world";
    let ids = tok.encode(original);
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, original);
}

// --- Dataset ---

#[test]
fn test_dataset_stores_data() {
    let ds = Dataset::new(vec![1, 2, 3, 4, 5], 3);
    assert_eq!(ds.data, vec![1, 2, 3, 4, 5]);
    assert_eq!(ds.block_size, 3);
}

#[test]
fn test_dataset_num_samples() {
    let ds = Dataset::new(vec![1, 2, 3, 4, 5, 6], 3);
    assert_eq!(ds.num_samples(), 3); // 6 - 3
}

#[test]
fn test_dataset_get_first_window() {
    let ds = Dataset::new(vec![1, 2, 3, 4, 5], 3);
    let (input, target) = ds.get(0);
    assert_eq!(input, vec![1, 2, 3]);
    assert_eq!(target, vec![2, 3, 4]);
}

#[test]
fn test_dataset_get_middle_window() {
    let ds = Dataset::new(vec![1, 2, 3, 4, 5], 3);
    let (input, target) = ds.get(1);
    assert_eq!(input, vec![2, 3, 4]);
    assert_eq!(target, vec![3, 4, 5]);
}

#[test]
#[should_panic]
fn test_dataset_get_out_of_bounds_panics() {
    let ds = Dataset::new(vec![1, 2, 3, 4, 5], 3);
    ds.get(99);
}
