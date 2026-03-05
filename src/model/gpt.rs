// =============================================================================
// src/model/gpt.rs — Full GPT forward pass
// =============================================================================
//
// WHAT THIS FILE IS
//   The top-level model. It ties together all the components into one
//   complete forward pass: tokens in → logits out.
//
// THE FORWARD PASS STEP BY STEP
//
//   1. TOKEN EMBEDDING
//      Each integer token ID is looked up in a table to get a D-dimensional
//      vector. This is a learned representation — similar tokens end up with
//      similar vectors after training.
//
//   2. POSITIONAL EMBEDDING
//      Transformers have no built-in sense of order (attention is a set
//      operation). We add a learned vector for each position 0..T, so the
//      model can distinguish "the word at position 3" from "the word at
//      position 10" even if they're the same token.
//      Alternative: RoPE (Rotary Position Embeddings), used in LLaMA.
//      Paper: https://arxiv.org/abs/2104.09864
//
//   3. TRANSFORMER BLOCKS
//      The embedding passes through num_layers blocks. Each refines the
//      representation using attention + MLP.
//
//   4. FINAL LAYER NORM
//      Stabilises the output before the classification head.
//
//   5. LM HEAD (language model head)
//      A final linear projection from D → vocab_size. The output is called
//      "logits" — raw scores for each vocabulary token. To get probabilities,
//      apply softmax. During training, we compute cross-entropy loss against
//      the target token. During generation, we sample from the distribution.
//
// NOTE ON WEIGHT TYING
//   GPT-2 ties tok_embed and lm_head (they share the same matrix). This
//   halves parameters and often improves performance. This implementation
//   keeps them separate for clarity; tying is a TODO optimisation.

use crate::tensor::Tensor;
use crate::model::config::Config;
use crate::model::block::{BlockWeights, block_forward};

/// All learnable parameters for the full GPT model.
pub struct GptWeights {
    pub tok_embed: Tensor,  // [vocab_size, embed_dim]  — token embedding table
    pub pos_embed: Tensor,  // [context_len, embed_dim] — positional embedding table
    pub blocks: Vec<BlockWeights>,
    pub ln_f_w: Tensor,     // [embed_dim] — final layer norm scale
    pub ln_f_b: Tensor,     // [embed_dim] — final layer norm bias
    pub lm_head: Tensor,    // [embed_dim, vocab_size] — projects to token logits
}

impl GptWeights {
    pub fn zeros(cfg: &Config) -> Self {
        let d = cfg.embed_dim;
        let v = cfg.vocab_size;
        let t = cfg.context_len;
        GptWeights {
            tok_embed: Tensor::zeros(vec![v, d]),
            pos_embed: Tensor::zeros(vec![t, d]),
            blocks: (0..cfg.num_layers).map(|_| BlockWeights::zeros(cfg)).collect(),
            ln_f_w: Tensor::ones(vec![d]),
            ln_f_b: Tensor::zeros(vec![d]),
            lm_head: Tensor::zeros(vec![d, v]),
        }
    }

    /// Count total trainable parameters.
    pub fn num_params(&self) -> usize {
        let mut n = self.tok_embed.numel()
            + self.pos_embed.numel()
            + self.ln_f_w.numel()
            + self.ln_f_b.numel()
            + self.lm_head.numel();
        for b in &self.blocks {
            n += b.ln1_w.numel() + b.ln1_b.numel();
            n += b.attn.w_qkv.numel() + b.attn.b_qkv.numel();
            n += b.attn.w_proj.numel() + b.attn.b_proj.numel();
            n += b.ln2_w.numel() + b.ln2_b.numel();
            n += b.mlp.w_fc.numel() + b.mlp.b_fc.numel();
            n += b.mlp.w_proj.numel() + b.mlp.b_proj.numel();
        }
        n
    }
}

/// Full GPT forward pass.
/// tokens: &[usize] of length T (must be <= context_len)
/// Returns logits: [T, vocab_size]
pub fn gpt_forward(tokens: &[usize], w: &GptWeights, cfg: &Config) -> Tensor {
    let t = tokens.len();
    assert!(t <= cfg.context_len, "sequence length {} exceeds context_len {}", t, cfg.context_len);

    let d = cfg.embed_dim;

    // Step 1+2: Token embedding + positional embedding (summed)
    let mut x_data = vec![0.0f32; t * d];
    for (i, &tok) in tokens.iter().enumerate() {
        for j in 0..d {
            x_data[i * d + j] = w.tok_embed.data[tok * d + j] + w.pos_embed.data[i * d + j];
        }
    }
    let mut x = Tensor::new(x_data, vec![t, d]);

    // Step 3: Run through each transformer block
    for block in &w.blocks {
        x = block_forward(&x, block, cfg);
    }

    // Step 4: Final layer norm
    x = x.layer_norm(&w.ln_f_w, &w.ln_f_b);

    // Step 5: Project to vocabulary logits: [T, D] × [D, V] → [T, V]
    x.matmul(&w.lm_head)
}
