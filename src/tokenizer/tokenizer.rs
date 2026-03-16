use std::collections::{HashMap, HashSet};

/// Character-level tokenizer built from a training corpus.
pub struct CharTokenizer {
    pub vocab: Vec<char>,
    pub(super) char_to_id: HashMap<char, usize>,
}

impl CharTokenizer {
    /// Collect all unique chars from text, sort them, assign sequential IDs.
    pub fn from_text(text: &str) -> Self {
        let unique_chars: HashSet<char> = text.chars().collect();
        let mut vocab: Vec<char> = unique_chars.into_iter().collect();
        vocab.sort_unstable();
        let char_to_id = vocab.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        CharTokenizer { vocab, char_to_id }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// text -> token IDs. Unknown chars are dropped silently.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().filter_map(|c| self.char_to_id.get(&c).copied()).collect()
    }

    /// token IDs -> text.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|id| self.vocab[*id]).collect()
    }
}
