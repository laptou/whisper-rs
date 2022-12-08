use std::collections::HashMap;

use tch::{Device, IndexOp, Kind, NewAxis, Tensor};

use crate::model::Whisper;

use super::TokenExtractor;

#[derive(Debug)]
pub struct BeamSearchTokenExtractor {
    pub token_id_eot: u32,
    pub beam_size: usize,
    pub finished_sequences: Option<Vec<HashMap<Vec<i64>, f64>>>,
    pub patience: f32,
}

impl TokenExtractor for BeamSearchTokenExtractor {
    fn reset(&mut self) {
        self.finished_sequences = None;
    }

    fn update(
        &mut self,
        model: &Whisper,
        tokens: Tensor,
        logits: Tensor,
        sum_logprobs: &mut Tensor,
    ) -> (Tensor, bool) {
        debug_assert_eq!(tokens.size()[0] % (self.beam_size as i64), 0);

        let n_audio = (tokens.size()[0] as usize) / self.beam_size;

        if self.finished_sequences.is_none() {
            self.finished_sequences = Some(vec![HashMap::new(); n_audio]);
        }

        let logprobs = logits.log_softmax(-1, Kind::Float);

        let mut next_tokens = vec![];
        let mut source_indices = vec![];
        let mut finished_sequences = vec![];

        // copy the tokens into vec all at once to avoid several copies from GPU
        // (expensive!)
        let tokens_shape = tokens.size();
        // dbg!(&tokens_shape);
        let prefixes: Vec<i64> = tokens.view(-1).into();

        for i in 0..n_audio {
            // map of sequence -> (score, source)
            let mut scores_sources = HashMap::new();
            let mut finished = HashMap::new();

            // STEP 1: calculate the cumulative log probabilities for possible
            // candidates
            for j in 0..self.beam_size {
                let idx = i * self.beam_size + j;
                let prefix =
                    &prefixes[idx * tokens_shape[1] as usize..(idx + 1) * tokens_shape[1] as usize];

                let idx = idx as i64;
                let (logprobs, tokens) =
                    logprobs
                        .i(idx)
                        .topk((self.beam_size + 1) as i64, -1, true, true);

                let len = logprobs.size1().unwrap();

                for i in 0..len {
                    let logprob = logprobs.double_value(&[i]);
                    let new_logprob: f64 = sum_logprobs.double_value(&[idx]) + logprob;

                    let token = tokens.int64_value(&[i]);
                    let mut sequence = Vec::from(prefix);
                    sequence.push(token);

                    scores_sources.insert(sequence, (new_logprob, idx));
                }
            }

            // STEP 2: rank the candidates and keep the top beam_size sequences
            // for each audio
            let mut saved = 0;
            // sort by score
            let mut scores_sources: Vec<_> = scores_sources.into_iter().collect();
            scores_sources.sort_by(|a, b| b.1 .0.partial_cmp(&a.1 .0).unwrap());
            for (sequence, (score, source)) in scores_sources {
                if sequence.last() == Some(&(self.token_id_eot as i64)) {
                    finished.insert(sequence, score);
                } else {
                    let _ = sum_logprobs.index_put_(
                        &[Some(Tensor::from(next_tokens.len() as i64))],
                        &Tensor::from(score as f32),
                        false,
                    );
                    next_tokens.push(sequence);
                    source_indices.push(source);

                    saved += 1;
                    if saved == self.beam_size {
                        break;
                    }
                }
            }

            finished_sequences.push(finished);
        }

        let tokens = Tensor::of_slice2(&next_tokens[..]);

        let sources_indices = Tensor::of_slice(&source_indices[..]);
        // update caches in multi-head attn blocks to match the beams that we
        // selected
        model.update_cache(&sources_indices);

        // add newly finished sequences to self.finished_sequences
        debug_assert_eq!(
            self.finished_sequences.as_ref().unwrap().len(),
            finished_sequences.len()
        );

        let max_candidates = f32::round(self.patience * self.beam_size as f32) as usize;

        for (previously_finished, newly_finished) in self
            .finished_sequences
            .as_deref_mut()
            .unwrap()
            .iter_mut()
            .zip(&finished_sequences)
        {
            for (new_seq, new_score) in newly_finished {
                if previously_finished.len() >= max_candidates {
                    break; // the candidate list is full
                }

                previously_finished.insert(new_seq.clone(), *new_score);
            }
        }

        // mark as completed if all audio has enough number of samples
        let completed = self
            .finished_sequences
            .as_ref()
            .unwrap()
            .iter()
            .all(|seq| seq.len() >= max_candidates);

        (tokens, completed)
    }

    fn finalize(&mut self, tokens: Tensor, sum_logprobs: Tensor) -> (Vec<Vec<Tensor>>, Tensor) {
        let device = tokens.device();

        // collect all finished sequences, including patience, and add unfinished ones if not enough

        for (i, seqs) in self
            .finished_sequences
            .as_deref_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            let i = i as i64;

            if seqs.len() < self.beam_size {
                // when not enough sequences are finished

                let slp = Vec::<i64>::from(sum_logprobs.i(i as i64).argsort(-1, false));
                for j in slp.into_iter().rev() {
                    let mut seq = Vec::from(tokens.i((i, j)));
                    seq.push(self.token_id_eot as i64);

                    seqs.insert(seq, sum_logprobs.i((i, j)).into());
                    if seqs.len() >= self.beam_size {
                        break;
                    }
                }
            }
        }

        let tokens: Vec<_> = self
            .finished_sequences
            .as_ref()
            .unwrap()
            .iter()
            .map(|seqs| {
                seqs.keys()
                    .map(|seq| Tensor::of_slice(seq).to_device(device))
                    .collect()
            })
            .collect();

        let sum_logprobs: Vec<Tensor> = self
            .finished_sequences
            .as_ref()
            .unwrap()
            .iter()
            .map(|seqs| {
                let t: Vec<f64> = seqs.values().copied().collect();
                Tensor::of_slice(&t[..]).to_device(device)
            })
            .collect();
        let sum_logprobs = Tensor::stack(&sum_logprobs[..], 0);

        (tokens, sum_logprobs)
    }

    fn group_size(&self) -> usize {
        self.beam_size
    }
}
