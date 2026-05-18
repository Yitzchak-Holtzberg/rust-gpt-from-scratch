//! GPT model implementation using the `candle` ML framework.
//!
//! Modern architecture (configurable via `Config::nano()`):
//!   - Token embeddings (no learned positional embeddings — RoPE is applied
//!     inside attention to Q and K instead)
//!   - N transformer blocks (pre-norm: RMSNorm → CausalAttention → residual,
//!     then RMSNorm → SwiGLU feedforward → residual)
//!   - Final RMSNorm
//!   - Tied language-model head: the token-embedding matrix is reused as the
//!     output projection (`x @ wte.T`), so the embedding and the LM head share
//!     a single weight tensor (~400K fewer params, acts as regularization).
//!   - Dropout on embeddings, attention output, and MLP output.
//!
//! All parameters are owned by a `candle_nn::VarMap` and updated via autograd.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    init::Init, ops::softmax_last_dim, rms_norm, Dropout, Embedding, Linear, Module, RmsNorm,
    VarBuilder,
};
use serde::{Deserialize, Serialize};

/// GPT-2 / Llama-style init recipe: a small Gaussian for every weight.
///
/// `embed_std` is 0.02 (the canonical Karpathy/GPT-2 number — small enough that
/// initial logits land in the linear regime of softmax, so initial loss is
/// close to `ln(vocab)` instead of the dozens you get with `N(0,1)`).
///
/// Residual output projections (attention `proj` and MLP `down`) get a
/// depth-scaled `0.02 / sqrt(2 * num_layers)` so the residual stream variance
/// doesn't blow up as depth grows — critical once we have weight tying that
/// reuses the embedding init for the LM head.
const EMBED_INIT_STD: f64 = 0.02;
fn residual_proj_init_std(num_layers: usize) -> f64 {
    EMBED_INIT_STD / ((2.0 * num_layers as f64).sqrt())
}

fn linear_with_std(
    in_features: usize,
    out_features: usize,
    init_std: f64,
    var_builder: VarBuilder,
) -> Result<Linear> {
    let weight = var_builder.get_with_hints(
        (out_features, in_features),
        "weight",
        Init::Randn {
            mean: 0.0,
            stdev: init_std,
        },
    )?;
    Ok(Linear::new(weight, None))
}

fn default_dropout() -> f32 {
    0.0
}
fn default_rope_theta() -> f32 {
    10000.0
}

/// Hyperparameters that fully define the model architecture. Serializable so
/// it can be saved alongside weights in a checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub context_len: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    /// Used to size the SwiGLU hidden dim. Llama-style: hidden ≈ (2/3) × mlp_ratio × embed_dim,
    /// rounded up to a multiple of 64.
    pub mlp_ratio: usize,
    /// Dropout probability applied at embeddings, attention output, and MLP output.
    /// `#[serde(default)]` keeps older checkpoints loadable.
    #[serde(default = "default_dropout")]
    pub dropout: f32,
    /// RoPE base frequency. 10000 is the GPT-NeoX / Llama default.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

impl Config {
    /// ~11M-parameter config tuned for Tiny Shakespeare with BPE(1024).
    /// Same capacity as the original LayerNorm/GELU/learned-pos baseline, but
    /// every architecture lever is now a modern Llama-style choice. With
    /// dropout=0.1 and weight tying acting as regularizers, this configuration
    /// resists the memorization regime that the older arch fell into on
    /// such a small corpus.
    pub fn nano() -> Self {
        Self {
            vocab_size: 0,
            context_len: 256,
            embed_dim: 384,
            num_heads: 6,
            num_layers: 6,
            mlp_ratio: 4,
            dropout: 0.1,
            rope_theta: 10000.0,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    /// Hidden dim for the SwiGLU MLP, rounded to a multiple of 64 so matmul
    /// kernels stay well-aligned.
    pub fn swiglu_hidden_dim(&self) -> usize {
        let raw = (2 * self.mlp_ratio * self.embed_dim) / 3;
        ((raw + 63) / 64) * 64
    }
}

/// Precomputed cosine/sine tables for Rotary Position Embeddings (RoPE).
///
/// RoPE encodes absolute position by rotating pairs of feature dimensions of
/// Q and K by an angle proportional to the token's position. Because rotations
/// preserve dot products, the relative-position term emerges naturally in
/// `QK^T` without any learned positional embedding. Tables are shared across
/// every layer and every batch, computed once at model construction.
struct RopeCache {
    /// `[context_len, head_dim/2]`
    cosines: Tensor,
    /// `[context_len, head_dim/2]`
    sines: Tensor,
}

fn build_rope_cache(
    context_len: usize,
    head_dim: usize,
    theta: f32,
    device: &Device,
) -> Result<RopeCache> {
    assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
    let half = head_dim / 2;

    // inv_freq[i] = 1 / theta^(2i / head_dim) for i in 0..half
    let mut inv_freq = Vec::with_capacity(half);
    for i in 0..half {
        let exponent = (2 * i) as f32 / head_dim as f32;
        inv_freq.push(theta.powf(-exponent));
    }

    // angles[t, i] = t * inv_freq[i]
    let mut cosines = Vec::with_capacity(context_len * half);
    let mut sines = Vec::with_capacity(context_len * half);
    for position in 0..context_len {
        for i in 0..half {
            let angle = position as f32 * inv_freq[i];
            cosines.push(angle.cos());
            sines.push(angle.sin());
        }
    }

    Ok(RopeCache {
        cosines: Tensor::from_vec(cosines, (context_len, half), device)?,
        sines: Tensor::from_vec(sines, (context_len, half), device)?,
    })
}

/// Apply RoPE to a `[batch, num_heads, seq_len, head_dim]` tensor.
///
/// Uses the GPT-NeoX / HuggingFace split convention: the first half of the
/// `head_dim` axis is rotated against the second half, rather than adjacent
/// pairs being rotated against each other (the Llama original convention).
/// The split form is easier to express with vectorized tensor ops.
fn apply_rope(input: &Tensor, cache: &RopeCache) -> Result<Tensor> {
    let (_batch, _heads, sequence_length, head_dim) = input.dims4()?;
    let half = head_dim / 2;

    // x1 = first half, x2 = second half  ->  shape [B, H, T, half]
    let first_half = input.narrow(D::Minus1, 0, half)?;
    let second_half = input.narrow(D::Minus1, half, half)?;

    // Slice cos/sin tables down to current sequence length, then reshape so
    // they broadcast over batch and head dims: [T, half] -> [1, 1, T, half]
    let cosines = cache
        .cosines
        .narrow(0, 0, sequence_length)?
        .reshape((1, 1, sequence_length, half))?;
    let sines = cache
        .sines
        .narrow(0, 0, sequence_length)?
        .reshape((1, 1, sequence_length, half))?;

    let rotated_first = (first_half.broadcast_mul(&cosines)?
        - second_half.broadcast_mul(&sines)?)?;
    let rotated_second = (first_half.broadcast_mul(&sines)?
        + second_half.broadcast_mul(&cosines)?)?;

    Tensor::cat(&[&rotated_first, &rotated_second], D::Minus1)
}

/// Multi-head causal self-attention with RoPE.
///
/// Each token can attend to itself and earlier tokens, but never to future
/// tokens — enforced by adding `-inf` to the upper triangle of the attention
/// score matrix before the softmax. Position information enters through RoPE
/// applied to Q and K (not to V).
struct CausalSelfAttention {
    query_key_value_projection: Linear,
    output_projection: Linear,
    output_dropout: Dropout,
    num_heads: usize,
    head_dim: usize,
    /// Pre-built `[1, 1, context_len, context_len]` mask, shared by every
    /// forward call so we don't reallocate it per step.
    causal_mask: Tensor,
}

impl CausalSelfAttention {
    fn new(config: &Config, var_builder: VarBuilder) -> Result<Self> {
        // No biases anywhere (Llama-style): cleaner gradients, marginally fewer params.
        // The output projection feeds the residual stream, so it gets the
        // depth-scaled init to keep activation magnitudes stable through depth.
        let query_key_value_projection = linear_with_std(
            config.embed_dim,
            3 * config.embed_dim,
            EMBED_INIT_STD,
            var_builder.pp("qkv"),
        )?;
        let output_projection = linear_with_std(
            config.embed_dim,
            config.embed_dim,
            residual_proj_init_std(config.num_layers),
            var_builder.pp("proj"),
        )?;
        let causal_mask = build_causal_mask(config.context_len, var_builder.device())?
            .reshape((1, 1, config.context_len, config.context_len))?;
        Ok(Self {
            query_key_value_projection,
            output_projection,
            output_dropout: Dropout::new(config.dropout),
            num_heads: config.num_heads,
            head_dim: config.head_dim(),
            causal_mask,
        })
    }

    fn forward(&self, input_embeddings: &Tensor, rope: &RopeCache, train: bool) -> Result<Tensor> {
        // input_embeddings: [batch, seq_len, embed_dim]
        let (batch_size, sequence_length, embed_dim) = input_embeddings.dims3()?;

        // One big linear projection produces Q, K, V concatenated along the feature axis
        let query_key_value = self
            .query_key_value_projection
            .forward(input_embeddings)?;
        let queries = query_key_value.narrow(2, 0, embed_dim)?;
        let keys = query_key_value.narrow(2, embed_dim, embed_dim)?;
        let values = query_key_value.narrow(2, 2 * embed_dim, embed_dim)?;

        // [B, T, C] → [B, T, H, head_dim] → [B, H, T, head_dim]
        let split_into_heads = |tensor: Tensor| -> Result<Tensor> {
            tensor
                .reshape((batch_size, sequence_length, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let queries = split_into_heads(queries)?;
        let keys = split_into_heads(keys)?;
        let values = split_into_heads(values)?;

        // RoPE on Q and K (not V): position information enters via rotated dot products.
        let queries = apply_rope(&queries, rope)?;
        let keys = apply_rope(&keys, rope)?;

        // Scaled dot-product attention scores: [B, H, T, T]
        let attention_scale = (self.head_dim as f64).sqrt();
        let attention_scores =
            (queries.matmul(&keys.transpose(D::Minus2, D::Minus1)?)? / attention_scale)?;

        // Causal mask is cached at [1, 1, context_len, context_len]; slice to
        // the current sequence length so generation (which uses shorter
        // contexts at the start) still works.
        let causal_mask_4d = self
            .causal_mask
            .i((.., .., ..sequence_length, ..sequence_length))?;
        let masked_attention_scores = attention_scores.broadcast_add(&causal_mask_4d)?;

        let attention_weights = softmax_last_dim(&masked_attention_scores)?;
        let weighted_values = attention_weights.matmul(&values)?;

        // Merge heads: [B, H, T, head_dim] → [B, T, H, head_dim] → [B, T, C]
        let merged = weighted_values
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, sequence_length, embed_dim))?;

        let projected = self.output_projection.forward(&merged)?;
        self.output_dropout.forward(&projected, train)
    }
}

/// Builds an `[N, N]` mask with zeros on/below the diagonal and `-inf` above,
/// to be added to attention scores so future positions get zero probability
/// after softmax.
fn build_causal_mask(sequence_length: usize, device: &Device) -> Result<Tensor> {
    let mut mask_values = vec![0f32; sequence_length * sequence_length];
    for row in 0..sequence_length {
        for column in (row + 1)..sequence_length {
            mask_values[row * sequence_length + column] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(mask_values, (sequence_length, sequence_length), device)
}

/// SwiGLU feedforward network: `down(silu(gate(x)) * up(x))`.
///
/// SwiGLU (Shazeer 2020, popularized by PaLM/Llama) is a gated MLP. The `gate`
/// path is activated by SiLU/Swish, then element-wise multiplied with the
/// linear `up` path. The multiplicative gating gives the MLP more
/// expressive power per parameter than a plain GELU FFN.
struct Feedforward {
    gate_projection: Linear,
    up_projection: Linear,
    down_projection: Linear,
    output_dropout: Dropout,
}

impl Feedforward {
    fn new(config: &Config, var_builder: VarBuilder) -> Result<Self> {
        let hidden_dim = config.swiglu_hidden_dim();
        // `down` feeds the residual stream → depth-scaled init.
        let gate_projection = linear_with_std(
            config.embed_dim,
            hidden_dim,
            EMBED_INIT_STD,
            var_builder.pp("gate"),
        )?;
        let up_projection = linear_with_std(
            config.embed_dim,
            hidden_dim,
            EMBED_INIT_STD,
            var_builder.pp("up"),
        )?;
        let down_projection = linear_with_std(
            hidden_dim,
            config.embed_dim,
            residual_proj_init_std(config.num_layers),
            var_builder.pp("down"),
        )?;
        Ok(Self {
            gate_projection,
            up_projection,
            down_projection,
            output_dropout: Dropout::new(config.dropout),
        })
    }

    fn forward(&self, input_embeddings: &Tensor, train: bool) -> Result<Tensor> {
        let gate = self.gate_projection.forward(input_embeddings)?.silu()?;
        let up = self.up_projection.forward(input_embeddings)?;
        let hidden = (gate * up)?;
        let projected = self.down_projection.forward(&hidden)?;
        self.output_dropout.forward(&projected, train)
    }
}

/// One transformer block: pre-norm attention with residual, then pre-norm
/// feedforward with residual. Pre-norm (rather than post-norm) gives more
/// stable training and is what nanoGPT, GPT-2, and Llama all use.
struct TransformerBlock {
    pre_attention_norm: RmsNorm,
    attention: CausalSelfAttention,
    pre_feedforward_norm: RmsNorm,
    feedforward: Feedforward,
}

impl TransformerBlock {
    fn new(config: &Config, var_builder: VarBuilder) -> Result<Self> {
        let pre_attention_norm = rms_norm(config.embed_dim, 1e-5, var_builder.pp("ln1"))?;
        let attention = CausalSelfAttention::new(config, var_builder.pp("attn"))?;
        let pre_feedforward_norm = rms_norm(config.embed_dim, 1e-5, var_builder.pp("ln2"))?;
        let feedforward = Feedforward::new(config, var_builder.pp("mlp"))?;
        Ok(Self {
            pre_attention_norm,
            attention,
            pre_feedforward_norm,
            feedforward,
        })
    }

    fn forward(&self, input_embeddings: &Tensor, rope: &RopeCache, train: bool) -> Result<Tensor> {
        let after_attention = (input_embeddings
            + self.attention.forward(
                &self.pre_attention_norm.forward(input_embeddings)?,
                rope,
                train,
            )?)?;
        let after_feedforward = (&after_attention
            + self
                .feedforward
                .forward(&self.pre_feedforward_norm.forward(&after_attention)?, train)?)?;
        Ok(after_feedforward)
    }
}

/// The full GPT model.
pub struct Gpt {
    token_embeddings: Embedding,
    embedding_dropout: Dropout,
    transformer_blocks: Vec<TransformerBlock>,
    final_norm: RmsNorm,
    /// Tied LM head: shares its weight tensor with `token_embeddings`. Stored
    /// as a `Linear` (which does `x @ W.T`) so we still get standard matmul.
    language_model_head: Linear,
    rope_cache: RopeCache,
    pub config: Config,
}

impl Gpt {
    pub fn new(config: Config, var_builder: VarBuilder) -> Result<Self> {
        // Build the token-embedding weight ourselves so we can (a) initialize it
        // with the small N(0, 0.02) std required by tied-head training, and
        // (b) share the exact same Var with the LM head below.
        let embedding_var_builder = var_builder.pp("tok_embed");
        let embedding_weight = embedding_var_builder.get_with_hints(
            (config.vocab_size, config.embed_dim),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: EMBED_INIT_STD,
            },
        )?;
        let token_embeddings = Embedding::new(embedding_weight.clone(), config.embed_dim);

        let transformer_blocks = (0..config.num_layers)
            .map(|layer_index| {
                TransformerBlock::new(&config, var_builder.pp(format!("block_{layer_index}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let final_norm = rms_norm(config.embed_dim, 1e-5, var_builder.pp("final_norm"))?;

        // Weight tying: the LM head reuses the same Tensor as the embedding.
        // `Linear` does `x @ W.T`, and `embedding_weight` has shape
        // `[vocab, embed_dim]` — exactly what a Linear from embed_dim→vocab
        // wants. Both modules reference one underlying Var, so gradients from
        // the embedding lookup *and* the LM-head matmul accumulate into a
        // single parameter (no separate `lm_head` weight is registered).
        let language_model_head = Linear::new(embedding_weight, None);

        let rope_cache = build_rope_cache(
            config.context_len,
            config.head_dim(),
            config.rope_theta,
            var_builder.device(),
        )?;

        Ok(Self {
            token_embeddings,
            embedding_dropout: Dropout::new(config.dropout),
            transformer_blocks,
            final_norm,
            language_model_head,
            rope_cache,
            config,
        })
    }

    /// Inference forward pass. Dropout layers are no-ops because `train=false`.
    /// Used by `generate`, the HTTP server's preview sampling, and the WASM
    /// inference build — no callers had to change.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.forward_impl(token_ids, false)
    }

    /// Training forward pass — same computation, but dropout is active.
    pub fn forward_train(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.forward_impl(token_ids, true)
    }

    fn forward_impl(&self, token_ids: &Tensor, train: bool) -> Result<Tensor> {
        // [B, T] → [B, T, C]
        let token_vectors = self.token_embeddings.forward(token_ids)?;
        // No positional embeddings — RoPE handles position inside attention.
        let mut hidden_states = self.embedding_dropout.forward(&token_vectors, train)?;

        for block in &self.transformer_blocks {
            hidden_states = block.forward(&hidden_states, &self.rope_cache, train)?;
        }

        let normalized = self.final_norm.forward(&hidden_states)?;
        self.language_model_head.forward(&normalized)
    }
}

/// Autoregressively generate `max_new_tokens` token IDs after the prompt.
///
/// The model is given a sliding window of the most recent `context_len` tokens
/// at each step. Returns only the newly generated tokens.
pub fn generate(
    model: &Gpt,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    device: &Device,
    rng: &mut impl rand::Rng,
) -> Result<Vec<u32>> {
    let mut all_tokens: Vec<u32> = prompt_token_ids.to_vec();
    let mut newly_generated = Vec::with_capacity(max_new_tokens);

    for _ in 0..max_new_tokens {
        let context_start = all_tokens.len().saturating_sub(model.config.context_len);
        let context_window = &all_tokens[context_start..];

        // Wrap the single context in a batch of size 1: [T] → [1, T]
        let context_tensor =
            Tensor::from_slice(context_window, (1, context_window.len()), device)?;
        let logits = model.forward(&context_tensor)?; // [1, T, V]

        // We only need the prediction for the LAST position
        let (_batch, seq_len, _vocab) = logits.dims3()?;
        let last_position_logits = logits.i((0, seq_len - 1))?; // [V]

        let next_token = sample_from_logits(&last_position_logits, temperature, top_k, rng)?;
        all_tokens.push(next_token);
        newly_generated.push(next_token);
    }
    Ok(newly_generated)
}

/// Convert a 1-D logits tensor into a sampled token ID.
///
/// Steps:
///   1. Divide logits by `temperature` (higher = more random, lower = more greedy)
///   2. If `top_k` is set, mask logits outside the top-K to `-inf`
///   3. Softmax → multinomial sample
fn sample_from_logits(
    logits: &Tensor,
    temperature: f32,
    top_k: Option<usize>,
    rng: &mut impl rand::Rng,
) -> Result<u32> {
    let mut scaled_logits: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
    let safe_temperature = temperature.max(1e-6);
    for value in scaled_logits.iter_mut() {
        *value /= safe_temperature;
    }

    if let Some(k) = top_k {
        if k < scaled_logits.len() {
            let mut sorted_logits = scaled_logits.clone();
            sorted_logits.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let kth_largest = sorted_logits[k - 1];
            for value in scaled_logits.iter_mut() {
                if *value < kth_largest {
                    *value = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Numerically stable softmax: subtract max before exp
    let max_logit = scaled_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exponentiated: Vec<f32> = scaled_logits.iter().map(|v| (v - max_logit).exp()).collect();
    let normalizer: f32 = exponentiated.iter().sum();
    let probabilities: Vec<f32> = exponentiated.iter().map(|e| e / normalizer).collect();

    // Inverse-CDF sampling
    let random_threshold: f32 = rng.gen();
    let mut cumulative_probability = 0.0;
    for (token_id, probability) in probabilities.iter().enumerate() {
        cumulative_probability += probability;
        if random_threshold < cumulative_probability {
            return Ok(token_id as u32);
        }
    }
    Ok((probabilities.len() - 1) as u32)
}
