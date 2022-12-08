use std::fmt::Debug;
use tch::Tensor;

use crate::model::Whisper;

mod beam_search;
mod greedy;

pub use beam_search::*;
pub use greedy::*;

#[derive(Debug, Clone, Copy)]
pub enum TokenExtractMode {
    Greedy { group_sze: usize },
    BeamSearch { beam_size: usize, patience: f32 },
}

/// This is called `TokenDecoder` in original Whisper source. Is used to
/// determine a token sequence from a set of log probabilities.
pub trait TokenExtractor: Debug {
    /// Initialize any stateful variables for decoding a new sequence.
    fn reset(&mut self);

    /// Specify how to select the next token, based on the current trace and logits
    ///
    /// Parameters
    /// ----------
    /// tokens : Tensor, shape = (n_batch, current_sequence_length)
    ///     all tokens in the context so far, including the prefix and sot_sequence tokens
    ///
    /// logits : Tensor, shape = (n_batch, vocab_size)
    ///     per-token logits of the probability distribution at the current step
    ///
    /// sum_logprobs : Tensor, shape = (n_batch)
    ///     cumulative log probabilities for each sequence
    ///
    /// Returns
    /// -------
    /// tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
    ///     the tokens, appended with the selected next token
    ///
    /// completed : bool
    ///     True if all sequences has reached the end of text
    fn update(
        &mut self,
        model: &Whisper,
        tokens: Tensor,
        logits: Tensor,
        sum_logprobs: &mut Tensor,
    ) -> (Tensor, bool);

    /// Finalize search and return the final candidate sequences
    ///
    /// Parameters
    /// ----------
    /// tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
    ///     all tokens in the context so far, including the prefix and sot_sequence
    ///
    /// sum_logprobs : Tensor, shape = (n_audio, n_group)
    ///     cumulative log probabilities for each sequence
    ///
    /// Returns
    /// -------
    /// tokens : Sequence[Sequence[Tensor]], length = n_audio
    ///     sequence of Tensors containing candidate token sequences, for each audio input
    ///
    /// sum_logprobs : List[List[float]], length = n_audio
    ///     sequence of cumulative log probabilities corresponding to the above
    ///
    fn finalize(&mut self, tokens: Tensor, sum_logprobs: Tensor) -> (Vec<Vec<Tensor>>, Tensor);

    fn group_size(&self) -> usize;
}
