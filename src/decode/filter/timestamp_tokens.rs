use std::{cell::RefCell, rc::Rc};

use tch::{Device, IndexOp, Kind, Tensor};

use crate::{decode::DecodePrompt, tokenize::Tokenizer};

use super::LogitFilter;

#[derive(Debug)]
pub struct TimestampTokens {
    max_initial_timestamp_index: Option<i64>,
    tokenizer: Rc<Tokenizer>,
    device: Device,
    prompt: Rc<RefCell<DecodePrompt>>,
}

impl TimestampTokens {
    pub(in crate::decode) fn new(
        tokenizer: Rc<Tokenizer>,
        prompt: Rc<RefCell<DecodePrompt>>,
        max_initial_timestamp_index: Option<i64>,
        device: Device,
    ) -> Self {
        Self {
            device,
            tokenizer,
            prompt,
            max_initial_timestamp_index,
        }
    }
}

impl LogitFilter for TimestampTokens {
    fn apply(&self, logits: &mut Tensor, tokens: &Tensor) {
        let sample_begin = self.prompt.borrow().sample_begin;

        let _ = logits.index_fill_(
            1,
            &Tensor::from(self.tokenizer.token_id_no_timestamps as i64).to_device(self.device),
            f64::NEG_INFINITY,
        );

        let token_id_timestampbegin = self.tokenizer.token_id_ts_begin as i64;

        // timestamps have to appear in pairs, except directly before EOT; mask
        // logits accordingly
        for k in 0..tokens.size()[0] {
            let seq: Vec<i64> = tokens.i((k, sample_begin..)).into();
            let last_was_timestamp = match seq.last() {
                Some(&last) => last >= token_id_timestampbegin,
                None => false,
            };
            let second_last_was_timestamp =
                seq.len() < 2 || seq[seq.len() - 2] >= token_id_timestampbegin;

            if last_was_timestamp {
                if second_last_was_timestamp {
                    // has to be non-timestamp
                    let _ = logits
                        .i((k, token_id_timestampbegin..))
                        .fill_(f64::NEG_INFINITY);
                } else {
                    // cannot be normal text tokens
                    let _ = logits
                        .i((k, ..self.tokenizer.token_id_eot as i64))
                        .fill_(f64::NEG_INFINITY);
                }
            }
        }

        if tokens.size()[1] == sample_begin {
            // suppress generating non-timestamp tokens at the beginning
            let _ = logits
                .i((.., ..token_id_timestampbegin))
                .fill_(f64::NEG_INFINITY);

            if let Some(max_initial_timestamp_index) = self.max_initial_timestamp_index {
                let last_allowed = token_id_timestampbegin + max_initial_timestamp_index;
                let _ = logits.i((.., last_allowed + 1..)).fill_(f64::NEG_INFINITY);
            }
        }

        // if sum of probability over timestamps is above any other token,
        // sample timestamp
        let logprobs = logits.log_softmax(-1, Kind::Float);
        for k in 0..tokens.size()[0] {
            let timestamp_logprob: f64 = logprobs
                .i((k, token_id_timestampbegin..))
                .logsumexp(&[-1], false)
                .into();
            let max_text_token_logprob: f64 =
                logprobs.i((k, ..token_id_timestampbegin)).max().into();

            if timestamp_logprob > max_text_token_logprob {
              let _ = logits
                    .i((k, ..token_id_timestampbegin))
                    .fill_(f64::NEG_INFINITY);
            }
        }
    }
}
