use std::fmt::Debug;
use tch::Tensor;

mod max_likelihood;

pub use max_likelihood::*;

pub trait SequenceRanker: Debug {
    /// Given a list of groups of samples and their cumulative log probabilities,
    /// return the indices of the samples in each group to select as the final result
    fn rank(&self, tokens: &Vec<Vec<Tensor>>, sum_logprobs: &Tensor) -> Vec<i64>;
}
