use tch::{Device, IndexOp, Kind, NewAxis, Tensor};

use crate::model::Whisper;

use super::TokenExtractor;

#[derive(Debug)]
pub struct GreedyTokenExtractor {
    token_id_eot: u32,
    temperature: f32,
    group_size: usize,
}

impl TokenExtractor for GreedyTokenExtractor {
    fn reset(&mut self) {}

    fn update(
        &mut self,
        _model: &Whisper,
        tokens: Tensor,
        logits: Tensor,
        sum_logprobs: &mut Tensor,
    ) -> (Tensor, bool) {
        let mut next_tokens = if self.temperature == 0.0 {
            logits.argmax(Some(-1), false)
        } else {
            unimplemented!()
            // todo: tch doesn't have Categorical?
        };

        let logprobs = logits.log_softmax(-1, Kind::Float);
        let current_logprobs = logprobs.i((
            &Tensor::arange(logprobs.size()[0], (Kind::Float, Device::Cpu)),
            &next_tokens,
        ));

        let last_tokens = tokens.i((.., -1));
        let last_tokens_eot = last_tokens.eq(self.token_id_eot as i64);
        *sum_logprobs += current_logprobs * last_tokens_eot.logical_not();

        let _ = next_tokens.index_put_(
            &[Some(last_tokens_eot)],
            &Tensor::from(self.token_id_eot as i64),
            false,
        );

        let tokens = Tensor::cat(&[&tokens, &next_tokens.i((.., NewAxis))], -1);
        let completed = tokens.eq(self.token_id_eot as i64).all().into();

        (tokens, completed)
    }

    fn finalize(&mut self, _tokens: Tensor, _sum_logprobs: Tensor) -> (Vec<Vec<Tensor>>, Tensor) {
        // (
        //     // make sure each sequence has at least one EOT token at the end
        //     tokens.pad(&[0, 1], "constant", self.token_id_eot as f64),
        //     sum_logprobs,
        // )

        todo!()
    }

    fn group_size(&self) -> usize {
        self.group_size
    }
}
