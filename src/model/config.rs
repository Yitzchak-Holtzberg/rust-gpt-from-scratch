#[derive(Clone, Debug)]
pub struct Config {
    pub vocab_size: usize,
    pub context_len: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub mlp_ratio: usize,
    pub dropout: f32,
}

impl Config {
    /// 2M-parameter default. Set vocab_size after building the tokenizer.
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
