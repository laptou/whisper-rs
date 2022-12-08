use std::{cell::RefCell, rc::Rc};

use tch::{Device, Tensor};

use crate::{decode::DecodePrompt, tokenize::Tokenizer};

use super::LogitFilter;

#[derive(Debug)]
pub struct SuppressBlank {
    suppress_indices: Tensor,
    prompt: Rc<RefCell<DecodePrompt>>,
}

impl SuppressBlank {
    pub(in crate::decode) fn new(
        tokenizer: &Tokenizer,
        prompt: Rc<RefCell<DecodePrompt>>,
        device: Device,
    ) -> Self {
        let token_id_space = tokenizer
            .encode(" ", true)
            .unwrap()
            .get_ids()
            .first()
            .copied()
            .unwrap();
        let token_id_eot = tokenizer.token_id_eot;

        Self {
            suppress_indices: Tensor::of_slice(&[token_id_space as i64, token_id_eot as i64])
                .to_device(device),
            prompt,
        }
    }
}

impl LogitFilter for SuppressBlank {
    fn apply(&self, logits: &mut Tensor, tokens: &Tensor) {
        if tokens.size()[1] == self.prompt.borrow().sample_begin {
            logits.index_fill_(1, &self.suppress_indices, f64::NEG_INFINITY);
        }
    }
}
