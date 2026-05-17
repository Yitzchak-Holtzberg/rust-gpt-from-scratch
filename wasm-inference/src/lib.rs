//! WASM-bindgen wrapper around `nano_gpt_rs::gpt` for running inference in the browser.

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use wasm_bindgen::prelude::*;

use nano_gpt_rs::gpt::{generate, Config, Gpt};
use nano_gpt_rs::tokenizer::CharTokenizer;

#[wasm_bindgen(start)]
pub fn install_panic_hook() {
    console_error_panic_hook::set_once();
}

/// One inference engine = one loaded model + tokenizer. Reused across many
/// generate() calls.
#[wasm_bindgen]
pub struct InferenceEngine {
    model: Gpt,
    tokenizer: CharTokenizer,
}

#[wasm_bindgen]
impl InferenceEngine {
    /// Build an engine from raw `.safetensors` bytes and the vocabulary JSON
    /// produced by the trainer.
    #[wasm_bindgen(constructor)]
    pub fn new(
        weights_bytes: &[u8],
        vocab_json: &str,
    ) -> Result<InferenceEngine, JsError> {
        let vocab: Vec<char> = serde_json::from_str(vocab_json)
            .map_err(|err| JsError::new(&format!("vocab json parse error: {err}")))?;
        let tokenizer = CharTokenizer::from_vocab(vocab);

        let mut config = Config::nano();
        config.vocab_size = tokenizer.vocab_size();

        let device = Device::Cpu;
        let parameter_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&parameter_map, DType::F32, &device);
        let model = Gpt::new(config, var_builder)
            .map_err(|err| JsError::new(&format!("model init: {err}")))?;

        // Load weights from the supplied buffer into the existing VarMap.
        let safetensors_buffer =
            candle_core::safetensors::load_buffer(weights_bytes, &device)
                .map_err(|err| JsError::new(&format!("safetensors load: {err}")))?;
        let parameter_data = parameter_map.data();
        let mut variables = parameter_data
            .lock()
            .map_err(|_| JsError::new("parameter map lock poisoned"))?;
        for (parameter_name, parameter) in variables.iter_mut() {
            let loaded_tensor = safetensors_buffer
                .get(parameter_name)
                .ok_or_else(|| {
                    JsError::new(&format!("missing weight in safetensors: {parameter_name}"))
                })?;
            parameter
                .set(loaded_tensor)
                .map_err(|err| JsError::new(&format!("set {parameter_name}: {err}")))?;
        }
        drop(variables);

        Ok(InferenceEngine { model, tokenizer })
    }

    /// Autoregressively sample `max_new_tokens` characters after `prompt`.
    /// `top_k = 0` disables top-K filtering.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> Result<String, JsError> {
        let prompt_token_ids: Vec<u32> = self
            .tokenizer
            .encode(prompt)
            .iter()
            .map(|id| *id as u32)
            .collect();
        if prompt_token_ids.is_empty() {
            return Err(JsError::new(
                "prompt contains no characters from the training vocabulary",
            ));
        }
        let top_k_option = if top_k == 0 { None } else { Some(top_k) };
        let mut rng = rand::thread_rng();
        let generated_ids = generate(
            &self.model,
            &prompt_token_ids,
            max_new_tokens,
            temperature,
            top_k_option,
            &Device::Cpu,
            &mut rng,
        )
        .map_err(|err| JsError::new(&format!("generation failed: {err}")))?;
        let generated_ids_as_usize: Vec<usize> =
            generated_ids.iter().map(|id| *id as usize).collect();
        let generated_text = self.tokenizer.decode(&generated_ids_as_usize);
        Ok(format!("{prompt}{generated_text}"))
    }

    /// Vocab size — useful for the UI to show stats.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}
