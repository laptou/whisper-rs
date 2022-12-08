use tch::{Device, Tensor};

use super::LogitFilter;

#[derive(Debug)]
pub struct SuppressTokens {
    suppress_indices: Tensor,
}

impl SuppressTokens {
    pub(in crate::decode) fn new(token_ids: &[u32], device: Device) -> Self {
        let token_ids: Vec<_> = token_ids.into_iter().map(|i| *i as i64).collect();

        Self {
            suppress_indices: Tensor::of_slice(&token_ids[..]).to_device(device),
        }
    }
}

impl LogitFilter for SuppressTokens {
    fn apply(&self, logits: &mut Tensor, _tokens: &Tensor) {
      let _ = logits.index_fill_(1, &self.suppress_indices, f64::NEG_INFINITY);
    }
}
