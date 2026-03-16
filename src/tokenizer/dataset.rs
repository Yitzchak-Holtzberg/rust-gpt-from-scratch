/// Sliding-window dataset.
/// Each sample: input = tokens[i..i+T], target = tokens[i+1..i+T+1]
pub struct Dataset {
    pub data: Vec<usize>,
    pub block_size: usize,
}

impl Dataset {
    pub fn new(tokens: Vec<usize>, block_size: usize) -> Self {
        Dataset {
            data: tokens,
            block_size,
        }
    }

    pub fn num_samples(&self) -> usize {
        self.data.len() - self.block_size
    }

    /// Returns (input_ids, target_ids) both of length block_size.
    pub fn get(&self, idx: usize) -> (Vec<usize>, Vec<usize>) {
        assert!(idx < self.num_samples(), "index out of bounds");
        let input_ids = self.data[idx..idx + self.block_size].to_vec();
        let target_ids = self.data[idx + 1..idx + self.block_size + 1].to_vec();
        (input_ids, target_ids)
    }
}
