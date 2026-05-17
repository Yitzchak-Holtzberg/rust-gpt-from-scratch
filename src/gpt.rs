//! GPT model implementation using the `candle` ML framework.
//!
//! Architecture (configurable via `Config::nano()`):
//!   - Token + learned positional embeddings
//!   - N transformer blocks (pre-norm: LayerNorm → CausalAttention → residual,
//!     then LayerNorm → Feedforward → residual)
//!   - Final LayerNorm
//!   - Linear language-model head projecting embeddings to vocabulary logits
//!
//! All parameters are owned by a `candle_nn::VarMap` and updated via autograd.

use candle_core::{Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm, linear, linear_no_bias, ops::softmax_last_dim, Embedding, LayerNorm,
    Linear, Module, VarBuilder,
};

/// Hyperparameters that fully define the model architecture.
#[derive(Clone, Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub context_len: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub mlp_ratio: usize,
}

impl Config {
    /// ~450K-parameter default suitable for character-level Shakespeare.
    /// `vocab_size` must be set after building the tokenizer.
    pub fn nano() -> Self {
        Self {
            vocab_size: 0,
            context_len: 256,
            embed_dim: 128,
            num_heads: 2,
            num_layers: 2,
            mlp_ratio: 4,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
}

/// Multi-head causal self-attention.
///
/// Each token can attend to itself and earlier tokens, but never to future
/// tokens — enforced by adding `-inf` to the upper triangle of the attention
/// score matrix before the softmax.
struct CausalSelfAttention {
    query_key_value_projection: Linear,
    output_projection: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl CausalSelfAttention {
    fn new(config: &Config, var_builder: VarBuilder) -> Result<Self> {
        let query_key_value_projection = linear(
            config.embed_dim,
            3 * config.embed_dim,
            var_builder.pp("qkv"),
        )?;
        let output_projection = linear(config.embed_dim, config.embed_dim, var_builder.pp("proj"))?;
        Ok(Self {
            query_key_value_projection,
            output_projection,
            num_heads: config.num_heads,
            head_dim: config.head_dim(),
        })
    }

    fn forward(&self, input_embeddings: &Tensor) -> Result<Tensor> {
        let (sequence_length, embed_dim) = input_embeddings.dims2()?;

        // One big linear projection produces Q, K, V concatenated along the feature axis
        let query_key_value = self
            .query_key_value_projection
            .forward(input_embeddings)?;
        let queries = query_key_value.narrow(1, 0, embed_dim)?;
        let keys = query_key_value.narrow(1, embed_dim, embed_dim)?;
        let values = query_key_value.narrow(1, 2 * embed_dim, embed_dim)?;

        // Split the embedding dimension into separate attention heads
        // and reshape so the head axis comes first: [seq, heads, head_dim] → [heads, seq, head_dim]
        let split_into_heads = |tensor: Tensor| -> Result<Tensor> {
            tensor
                .reshape((sequence_length, self.num_heads, self.head_dim))?
                .transpose(0, 1)?
                .contiguous()
        };
        let queries = split_into_heads(queries)?;
        let keys = split_into_heads(keys)?;
        let values = split_into_heads(values)?;

        // Scaled dot-product attention scores: how much each token attends to every other token
        let attention_scale = (self.head_dim as f64).sqrt();
        let attention_scores =
            (queries.matmul(&keys.transpose(D::Minus2, D::Minus1)?)? / attention_scale)?;

        // Apply causal mask so position t can only see positions <= t
        let causal = build_causal_mask(sequence_length, input_embeddings.device())?;
        let masked_attention_scores = attention_scores.broadcast_add(&causal)?;

        let attention_weights = softmax_last_dim(&masked_attention_scores)?;
        let weighted_values = attention_weights.matmul(&values)?;

        // Merge heads back into a single embedding dimension
        let merged = weighted_values
            .transpose(0, 1)?
            .contiguous()?
            .reshape((sequence_length, embed_dim))?;

        self.output_projection.forward(&merged)
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

/// Position-wise feedforward network: project up, GELU, project down.
struct Feedforward {
    project_up: Linear,
    project_down: Linear,
}

impl Feedforward {
    fn new(config: &Config, var_builder: VarBuilder) -> Result<Self> {
        let hidden_dim = config.embed_dim * config.mlp_ratio;
        let project_up = linear(config.embed_dim, hidden_dim, var_builder.pp("fc"))?;
        let project_down = linear(hidden_dim, config.embed_dim, var_builder.pp("proj"))?;
        Ok(Self {
            project_up,
            project_down,
        })
    }

    fn forward(&self, input_embeddings: &Tensor) -> Result<Tensor> {
        let activated = self.project_up.forward(input_embeddings)?.gelu()?;
        self.project_down.forward(&activated)
    }
}

/// One transformer block: pre-norm attention with residual, then pre-norm
/// feedforward with residual. Pre-norm (rather than post-norm) gives more
/// stable training and is what nanoGPT, GPT-2, and Llama all use.
struct TransformerBlock {
    pre_attention_norm: LayerNorm,
    attention: CausalSelfAttention,
    pre_feedforward_norm: LayerNorm,
    feedforward: Feedforward,
}

impl TransformerBlock {
    fn new(config: &Config, var_builder: VarBuilder) -> Result<Self> {
        let pre_attention_norm = layer_norm(config.embed_dim, 1e-5, var_builder.pp("ln1"))?;
        let attention = CausalSelfAttention::new(config, var_builder.pp("attn"))?;
        let pre_feedforward_norm = layer_norm(config.embed_dim, 1e-5, var_builder.pp("ln2"))?;
        let feedforward = Feedforward::new(config, var_builder.pp("mlp"))?;
        Ok(Self {
            pre_attention_norm,
            attention,
            pre_feedforward_norm,
            feedforward,
        })
    }

    fn forward(&self, input_embeddings: &Tensor) -> Result<Tensor> {
        let after_attention = (input_embeddings
            + self
                .attention
                .forward(&self.pre_attention_norm.forward(input_embeddings)?)?)?;
        let after_feedforward = (&after_attention
            + self
                .feedforward
                .forward(&self.pre_feedforward_norm.forward(&after_attention)?)?)?;
        Ok(after_feedforward)
    }
}

/// The full GPT model.
pub struct Gpt {
    token_embeddings: Embedding,
    position_embeddings: Embedding,
    transformer_blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    language_model_head: Linear,
    pub config: Config,
}

impl Gpt {
    pub fn new(config: Config, var_builder: VarBuilder) -> Result<Self> {
        let token_embeddings = embedding(
            config.vocab_size,
            config.embed_dim,
            var_builder.pp("tok_embed"),
        )?;
        let position_embeddings = embedding(
            config.context_len,
            config.embed_dim,
            var_builder.pp("pos_embed"),
        )?;
        let transformer_blocks = (0..config.num_layers)
            .map(|layer_index| {
                TransformerBlock::new(&config, var_builder.pp(format!("block_{layer_index}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let final_norm = layer_norm(config.embed_dim, 1e-5, var_builder.pp("final_norm"))?;
        let language_model_head =
            linear_no_bias(config.embed_dim, config.vocab_size, var_builder.pp("lm_head"))?;
        Ok(Self {
            token_embeddings,
            position_embeddings,
            transformer_blocks,
            final_norm,
            language_model_head,
            config,
        })
    }

    /// Forward pass.
    ///
    /// `token_ids`: shape `[sequence_length]`, integer tensor on the model's device.
    /// Returns logits of shape `[sequence_length, vocab_size]`.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let sequence_length = token_ids.dims1()?;
        let position_ids = Tensor::arange(0u32, sequence_length as u32, token_ids.device())?;

        let token_vectors = self.token_embeddings.forward(token_ids)?;
        let position_vectors = self.position_embeddings.forward(&position_ids)?;
        let mut hidden_states = (token_vectors + position_vectors)?;

        for block in &self.transformer_blocks {
            hidden_states = block.forward(&hidden_states)?;
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

        let context_tensor =
            Tensor::from_slice(context_window, (context_window.len(),), device)?;
        let logits = model.forward(&context_tensor)?;

        // We only need the prediction for the LAST position
        let last_position_index = logits.dims2()?.0 - 1;
        let last_position_logits = logits.i(last_position_index)?;

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
    let mut scaled_logits: Vec<f32> = logits.to_vec1()?;
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
