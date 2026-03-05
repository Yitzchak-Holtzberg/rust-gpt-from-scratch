// =============================================================================
// src/tokenizer/mod.rs — Character tokenizer and sliding-window dataset
// =============================================================================
//
// WHAT THIS FILE IS
//   Converts raw text into integer token IDs (encoding) and back (decoding),
//   and packages the training corpus into (input, target) pairs.
//
// WHY TOKENIZATION MATTERS
//   Neural networks work on numbers, not strings. Every character or word must
//   map to an integer that indexes the embedding table.
//
// CHARACTER-LEVEL TOKENIZATION
//   The simplest possible approach: each unique character in the training text
//   is one token. Tiny Shakespeare has ~65 unique characters, so vocab_size=65.
//
//   Pros:  tiny vocab, no out-of-vocabulary (OOV) problem, simple to implement
//   Cons:  sequences are long (one token per character), model must learn to
//          spell; carries no prior knowledge of word structure
//
// ALTERNATIVES (stretch goals)
//   Byte Pair Encoding (BPE): merge frequent character pairs iteratively to
//   build a vocabulary of common subwords. GPT-2 uses ~50k BPE tokens.
//   This shortens sequences dramatically and lets the model see whole words.
//   Library: tiktoken (Python), or implement from scratch — ~200 lines.
//
//   SentencePiece: similar idea, used in LLaMA.
//
// SLIDING WINDOW DATASET
//   Given a long token sequence [t0, t1, t2, ..., tN], we extract overlapping
//   windows of length block_size for training. The input is [ti..ti+T] and
//   the target is [ti+1..ti+T+1] — shifted by one position. The model learns
//   to predict each next token given its prefix. This is the language modelling
//   objective: maximise P(t_{i+1} | t_0..t_i) for all i.

use std::collections::HashMap;

/// Character-level tokenizer. Vocab is built from the training corpus.
pub struct CharTokenizer {
    pub vocab: Vec<char>,       // vocab[id] = character
    char_to_id: HashMap<char, usize>, // character → token id
}

impl CharTokenizer {
    /// Build vocab by collecting all unique characters, sorted for
    /// deterministic ordering across runs.
    pub fn from_text(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        chars.sort();

        let char_to_id: HashMap<char, usize> = chars.iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();

        CharTokenizer { vocab: chars, char_to_id }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Text → token IDs. Unknown characters are silently dropped.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .filter_map(|c| self.char_to_id.get(&c).copied())
            .collect()
    }

    /// Token IDs → text. Out-of-range IDs are silently dropped.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocab.get(id).copied())
            .collect()
    }
}

/// Sliding-window dataset for language model training.
/// Each sample is (input_tokens[0..T], target_tokens[1..T+1]).
pub struct Dataset {
    pub data: Vec<usize>,
    pub block_size: usize, // T — the context window length
}

impl Dataset {
    pub fn new(tokens: Vec<usize>, block_size: usize) -> Self {
        Dataset { data: tokens, block_size }
    }

    /// Number of valid (input, target) pairs.
    pub fn num_samples(&self) -> usize {
        if self.data.len() > self.block_size {
            self.data.len() - self.block_size
        } else {
            0
        }
    }

    /// Returns (input, target) both of length block_size.
    /// input[i] predicts target[i] = input[i+1].
    pub fn get(&self, idx: usize) -> (Vec<usize>, Vec<usize>) {
        let input = self.data[idx..idx + self.block_size].to_vec();
        let target = self.data[idx + 1..idx + self.block_size + 1].to_vec();
        (input, target)
    }
}
