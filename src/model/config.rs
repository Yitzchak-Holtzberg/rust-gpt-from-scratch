// =============================================================================
// src/model/config.rs — Model hyperparameters
// =============================================================================
//
// WHAT THIS FILE IS
//   A single struct that holds every number you'd tune when changing the model
//   size. Passing Config around avoids magic constants scattered through the
//   codebase.
//
// KEY HYPERPARAMETERS EXPLAINED
//   vocab_size   Number of unique tokens. For character-level on Shakespeare
//                this is ~65. Larger vocab = bigger embedding table.
//
//   context_len  Maximum sequence length the model can process in one forward
//                pass. Memory for attention scales as O(T²), so this is
//                expensive to increase.
//
//   embed_dim    Width of every representation vector. Every token becomes a
//                vector of this size. All intermediate activations are also
//                this size. Doubling this ≈ 4× the parameters.
//
//   num_heads    How many attention "perspectives" run in parallel. Each head
//                sees a different linear projection of Q/K/V. Requires
//                embed_dim % num_heads == 0.
//
//   num_layers   How many transformer blocks are stacked. Depth gives the
//                model capacity to compose features hierarchically.
//
//   mlp_ratio    The MLP hidden layer is embed_dim * mlp_ratio wide. 4× is
//                standard (from the original "Attention is All You Need").
//
// PARAMETER COUNT ESTIMATE (nano config)
//   tok_embed:   vocab_size × 128          ≈   8 320  (at 65-char vocab)
//   pos_embed:   256 × 128                 =  32 768
//   per block:   ~2 × (4 × 128² + 128² + small biases) ≈ 394 752
//   2 blocks:                              ≈ 789 504
//   lm_head:     128 × vocab_size          ≈   8 320
//   Total                                  ≈ ~840 K params (< 2M)
//
// To hit 2M params, increase embed_dim to 256 or add more layers.

/// Hyperparameters for the nano-GPT.
#[derive(Clone, Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub context_len: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub mlp_ratio: usize,
    pub dropout: f32, // stored for future use; not applied during inference
}

impl Config {
    /// 2M-parameter configuration. Set vocab_size after building the tokenizer.
    pub fn nano() -> Self {
        Config {
            vocab_size: 0,
            context_len: 256,
            embed_dim: 128,
            num_heads: 2,
            num_layers: 2,
            mlp_ratio: 4,
            dropout: 0.1,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
}
