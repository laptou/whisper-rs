use tch::{IndexOp, Tensor};

use super::SequenceRanker;

#[derive(Debug)]
pub struct MaximumLikelihoodRanker {
    pub length_penalty: Option<f64>,
}

impl SequenceRanker for MaximumLikelihoodRanker {
    /// Select the sample with the highest log probabilities, penalized using either
    /// a simple length normalization or Google NMT paper's length penalty
    fn rank(&self, tokens: &Vec<Vec<Tensor>>, sum_logprobs: &Tensor) -> Vec<i64> {
        let lengths = tokens.iter().map(|s| s.iter().map(|t| t.size()[0]));

        (0..sum_logprobs.size()[0])
            .map(|i| sum_logprobs.i(i))
            .zip(lengths)
            .map(|(probs, lengths)| {
                let mut scores = vec![];
                for (logprob, length) in probs.iter::<f64>().unwrap().zip(lengths) {
                    let penalty = match self.length_penalty {
                        // from the Google NMT paper
                        Some(p) => f64::powf((5. + length as f64) / 6., p),
                        None => length as f64,
                    };
                    scores.push(logprob / penalty);
                }

                let penalties = Tensor::of_slice(&scores[..]);
                penalties.argmax(None, false).into()
            })
            .collect()
    }
}
