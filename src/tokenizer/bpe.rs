//! BPE (byte-pair encoding) tokenizer backed by HuggingFace's `tokenizers`
//! crate. Replaces the char-level tokenizer for server-side training and
//! inference — gives ~3–5× shorter sequences and treats common multi-char
//! patterns as atomic tokens, which unblocks the model from "bigram plateau"
//! territory.
//!
//! Drop-in surface for the previous [`CharTokenizer`]: same `from_text` /
//! `vocab_size` / `encode` / `decode` method names so callers don't need to
//! be generic over which tokenizer they hold.

use std::collections::HashSet;
use std::io::Write;

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

/// Wraps a trained `tokenizers::Tokenizer` so callers get a small, focused API
/// that matches the existing char-tokenizer surface.
pub struct BpeTokenizer {
    inner: Tokenizer,
}

impl BpeTokenizer {
    /// Train a fresh BPE tokenizer on `text`. Vocab size defaults to 1024 —
    /// large enough that common words like " the", " and", "ROMEO" become
    /// single tokens, small enough that the embedding matrix stays cheap.
    pub fn from_text(text: &str) -> Self {
        Self::from_text_with_vocab_size(text, 1024)
    }

    /// Same as [`from_text`] but lets the caller pick the vocab size.
    pub fn from_text_with_vocab_size(text: &str, vocab_size: usize) -> Self {
        // The HF training API only exposes `train_from_files` for the
        // top-level `Tokenizer` — so we write the corpus to a temp file,
        // train against it, then drop the file. For a Shakespeare-sized
        // corpus this is a one-time ~1s cost.
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join(format!(
            "nano-gpt-rs-corpus-{}.txt",
            std::process::id()
        ));
        {
            let mut file = std::fs::File::create(&temp_path).expect("create temp corpus file");
            file.write_all(text.as_bytes())
                .expect("write temp corpus file");
        }

        // ByteLevel pre-tokenizer + decoder is the GPT-2 / Llama-style recipe:
        //   - pre-tokenizer maps each byte to a unique unicode glyph so BPE
        //     merges operate on a clean stream (spaces become "Ġ", etc.)
        //   - decoder reverses that mapping when we go back to text
        // This handles any input byte without UNKs and round-trips perfectly.
        let mut tokenizer = Tokenizer::new(BPE::default());
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

        // `ByteLevel::alphabet()` returns the crate's internal `AHashSet`, but
        // `initial_alphabet` wants a `std::collections::HashSet`. Roundtrip.
        let alphabet: HashSet<char> = ByteLevel::alphabet().into_iter().collect();
        let bpe_trainer = BpeTrainerBuilder::new()
            .vocab_size(vocab_size)
            .min_frequency(2)
            .initial_alphabet(alphabet)
            .special_tokens(vec![AddedToken::from(String::from("<|endoftext|>"), true)])
            .show_progress(false)
            .build();

        // The top-level `Tokenizer` wraps `ModelWrapper`, so the trainer must
        // be a `TrainerWrapper` (an enum over BPE / WordPiece / Unigram / etc).
        let mut trainer: TrainerWrapper = bpe_trainer.into();

        tokenizer
            .train_from_files(&mut trainer, vec![temp_path.to_string_lossy().into_owned()])
            .expect("BPE training failed");

        // Best-effort cleanup; ignore if it was already deleted.
        let _ = std::fs::remove_file(&temp_path);

        Self { inner: tokenizer }
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// text -> token IDs (returned as usize to match the prior API).
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.inner.encode(text, false) {
            Ok(encoding) => encoding.get_ids().iter().map(|id| *id as usize).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// token IDs -> text. Errors round-trip as empty strings — this matches
    /// the existing char tokenizer's lenient behavior on bad input.
    pub fn decode(&self, ids: &[usize]) -> String {
        let ids_u32: Vec<u32> = ids.iter().map(|id| *id as u32).collect();
        self.inner.decode(&ids_u32, false).unwrap_or_default()
    }

    /// Serialize the trained tokenizer to a JSON string. Round-trips losslessly
    /// via [`from_json`]; the format matches HuggingFace's standard
    /// `tokenizer.json` so artifacts are portable to Python.
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(&self.inner)
    }

    /// Rebuild a tokenizer from a JSON string produced by [`to_json`].
    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        let inner: Tokenizer = serde_json::from_str(json)?;
        Ok(Self { inner })
    }
}
