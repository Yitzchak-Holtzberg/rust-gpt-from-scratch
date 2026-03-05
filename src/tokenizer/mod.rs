use std::collections::HashMap;

/// Character-level tokenizer built from a training corpus.
pub struct CharTokenizer {
    pub vocab: Vec<char>,
    char_to_id: HashMap<char, usize>,
}

impl CharTokenizer {
    /// Collect all unique chars from text, sort them, assign sequential IDs.
    pub fn from_text(text: &str) -> Self {
        todo!()
    }

    pub fn vocab_size(&self) -> usize {
        todo!()
    }

    /// text -> token IDs. Unknown chars are dropped silently.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        todo!()
    }

    /// token IDs -> text.
    pub fn decode(&self, ids: &[usize]) -> String {
        todo!()
    }
}

/// Sliding-window dataset.
/// Each sample: input = tokens[i..i+T], target = tokens[i+1..i+T+1]
pub struct Dataset {
    pub data: Vec<usize>,
    pub block_size: usize,
}

impl Dataset {
    pub fn new(tokens: Vec<usize>, block_size: usize) -> Self {
        todo!()
    }

    pub fn num_samples(&self) -> usize {
        todo!()
    }

    /// Returns (input_ids, target_ids) both of length block_size.
    pub fn get(&self, idx: usize) -> (Vec<usize>, Vec<usize>) {
        todo!()
    }
}
