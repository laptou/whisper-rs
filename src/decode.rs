use std::fmt::Debug;
use std::{collections::HashMap, sync::Arc};

use anyhow::Context;
use tch::{nn::Module, Device, IndexOp, Kind, NewAxis, Tensor};

use crate::{
    model::Whisper,
    tokenize::{Task, Tokenizer},
    util::tensor_dbg,
};

#[derive(Debug)]
pub struct DecodeTask<'a> {
    model: &'a Whisper,
    options: DecodeOptions,
    device: Device,

    sample_len: i64,
    sot_idx: usize,
    initial_tokens: Vec<u32>,
    tokenizer: Tokenizer,
    logit_filters: Vec<Box<dyn LogitFilter>>,
    token_extractor: Box<dyn TokenExtractor>,
    sequence_ranker: Box<dyn SequenceRanker>,
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeOptions {
    pub task: Task,
    pub sample_len: Option<usize>,
    pub token_extract_mode: TokenExtractMode,
    pub len_penalty: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub enum TokenExtractMode {
    Greedy(usize),
    BeamSearch { beam_size: usize, patience: f32 },
}

#[derive(Debug)]
pub struct DecodingResult {
    pub audio_features: Tensor,
    pub language: String,
    pub language_probs: Option<HashMap<String, f32>>,
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    pub temperature: f32,
    pub compression_ratio: f32,
}

trait LogitFilter: Debug {
    fn apply(&self, logits: &mut Tensor, tokens: &Tensor);
}

/// This is called `TokenDecoder` in original Whisper source. Is used to
/// determine a token sequence from a set of log probabilities.
trait TokenExtractor: Debug {
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

#[derive(Debug)]
struct GreedyTokenExtractor {
    token_id_eot: u32,
    temperature: f32,
}

impl TokenExtractor for GreedyTokenExtractor {
    fn reset(&mut self) {}

    fn update(
        &mut self,
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
        unimplemented!()
    }
}

#[derive(Debug)]
struct BeamSearchTokenExtractor {
    token_id_eot: u32,
    beam_size: usize,
    finished_sequences: Option<Vec<HashMap<Vec<i64>, f64>>>,
    patience: f32,
}

impl TokenExtractor for BeamSearchTokenExtractor {
    fn reset(&mut self) {
        self.finished_sequences = None;
    }

    fn update(
        &mut self,
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
            .iter()
            .all(|seq| seq.len() >= max_candidates);

        (tokens, completed)
    }

    fn finalize(&mut self, tokens: Tensor, sum_logprobs: Tensor) -> (Vec<Vec<Tensor>>, Tensor) {
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
                let tmp: Vec<Tensor> = seqs.keys().map(|seq| Tensor::of_slice(seq)).collect();
                // Tensor::stack(&tmp[..], 0)
                tmp
            })
            .collect();
        // let tokens = Tensor::stack(&tokens[..], 0);

        let sum_logprobs: Vec<Tensor> = self
            .finished_sequences
            .as_ref()
            .unwrap()
            .iter()
            .map(|seqs| {
                let t: Vec<f64> = seqs.values().copied().collect();
                Tensor::of_slice(&t[..])
            })
            .collect();
        let sum_logprobs = Tensor::stack(&sum_logprobs[..], 0);

        (tokens, sum_logprobs)
    }

    fn group_size(&self) -> usize {
        self.beam_size
    }
}

trait SequenceRanker: Debug {
    /// Given a list of groups of samples and their cumulative log probabilities,
    /// return the indices of the samples in each group to select as the final result
    fn rank(&self, tokens: &Vec<Vec<Tensor>>, sum_logprobs: &Tensor) -> Vec<i64>;
}

#[derive(Debug)]
struct MaximumLikelihoodRanker {
    length_penalty: Option<f64>,
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

#[derive(Debug)]
struct SuppressBlank {
    suppress_indices: Tensor,
    sample_begin: i64,
}

impl SuppressBlank {
    fn new(tokenizer: &Tokenizer, sample_begin: i64) -> Self {
        let token_id_space = tokenizer
            .encode(" ", true)
            .unwrap()
            .get_ids()
            .first()
            .copied()
            .unwrap();
        let token_id_eot = tokenizer.token_id_eot;

        Self {
            suppress_indices: Tensor::of_slice(&[token_id_space as i64, token_id_eot as i64]),
            sample_begin,
        }
    }
}

impl LogitFilter for SuppressBlank {
    fn apply(&self, logits: &mut Tensor, tokens: &Tensor) {
        if tokens.size()[1] == self.sample_begin {
            logits.index_fill_(1, &self.suppress_indices, f64::NEG_INFINITY);
        }
    }
}

#[derive(Debug)]
struct SuppressTokens {
    suppress_indices: Tensor,
}

impl SuppressTokens {
    fn new(token_ids: &[u32]) -> Self {
        let token_ids: Vec<_> = token_ids.into_iter().map(|i| *i as i64).collect();

        Self {
            suppress_indices: Tensor::of_slice(&token_ids[..]),
        }
    }
}

impl LogitFilter for SuppressTokens {
    fn apply(&self, logits: &mut Tensor, tokens: &Tensor) {
        logits.index_fill_(1, &self.suppress_indices, f64::NEG_INFINITY);
    }
}

#[derive(Debug)]
struct TimestampTokens {
    max_initial_timestamp_index: Option<i64>,
    sample_begin: i64,
    tokenizer: Arc<Tokenizer>,
}

impl TimestampTokens {
    fn new(
        tokenizer: Arc<Tokenizer>,
        sample_begin: i64,
        max_initial_timestamp_index: Option<i64>,
    ) -> Self {
        Self {
            tokenizer,
            sample_begin,
            max_initial_timestamp_index,
        }
    }
}

impl LogitFilter for TimestampTokens {
    fn apply(&self, logits: &mut Tensor, tokens: &Tensor) {
        logits.index_fill_(
            1,
            &Tensor::from(self.tokenizer.token_id_notimestamps as i64),
            f64::NEG_INFINITY,
        );

        let token_id_timestampbegin = self.tokenizer.token_id_timestampbegin as i64;

        // timestamps have to appear in pairs, except directly before EOT; mask
        // logits accordingly
        for k in 0..tokens.size()[0] {
            let seq: Vec<i64> = tokens.i((k, self.sample_begin..)).into();
            let last_was_timestamp = match seq.last() {
                Some(&last) => last >= token_id_timestampbegin,
                None => false,
            };
            let second_last_was_timestamp =
                seq.len() < 2 || seq[seq.len() - 2] >= token_id_timestampbegin;

            if last_was_timestamp {
                if second_last_was_timestamp {
                    // has to be non-timestamp
                    logits
                        .i((k, token_id_timestampbegin..))
                        .fill_(f64::NEG_INFINITY);
                } else {
                    // cannot be normal text tokens
                    logits
                        .i((k, ..self.tokenizer.token_id_eot as i64))
                        .fill_(f64::NEG_INFINITY);
                }
            }
        }

        if tokens.size()[1] == self.sample_begin {
            // suppress generating non-timestamp tokens at the beginning
            logits
                .i((.., ..token_id_timestampbegin))
                .fill_(f64::NEG_INFINITY);

            if let Some(max_initial_timestamp_index) = self.max_initial_timestamp_index {
                let last_allowed = token_id_timestampbegin + max_initial_timestamp_index;
                logits.i((.., last_allowed + 1..)).fill_(f64::NEG_INFINITY);
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
                logits
                    .i((k, ..token_id_timestampbegin))
                    .fill_(f64::NEG_INFINITY);
            }
        }
    }
}

impl<'a> DecodeTask<'a> {
    pub fn new(model: &'a Whisper, options: DecodeOptions) -> anyhow::Result<Self> {
        let sample_len = options
            .sample_len
            .map_or(model.dims.n_text_ctxs / 2, |s| s as i64);

        let tokenizer = Tokenizer::new(Task::Transcribe).context("failed to create tokenizer")?;

        let initial_tokens = tokenizer.sequence_sot();
        let sot_idx = initial_tokens
            .iter()
            .position(|&t| t == tokenizer.token_id_sot)
            .expect("sot sequence doesn't contain sot token");

        Ok(Self {
            device: model.device(),
            options,
            model,

            token_extractor: match options.token_extract_mode {
                TokenExtractMode::Greedy(_n) => todo!(),
                TokenExtractMode::BeamSearch {
                    beam_size,
                    patience,
                } => Box::new(BeamSearchTokenExtractor {
                    beam_size,
                    token_id_eot: tokenizer.token_id_eot,
                    finished_sequences: None,
                    patience,
                }),
            },

            sequence_ranker: Box::new(MaximumLikelihoodRanker {
                length_penalty: options.len_penalty,
            }),

            tokenizer,
            sot_idx,
            sample_len,
            initial_tokens,
            logit_filters: vec![],
        })
    }

    fn main_loop(
        &mut self,
        audio_features: &Tensor,
        mut tokens: Tensor,
    ) -> (Tensor, Tensor, Vec<f32>) {
        let n_batch = tokens.size()[0];
        debug_assert_eq!(audio_features.size()[0], n_batch);

        let dtype = (audio_features.kind(), audio_features.device());
        let mut sum_logprobs = Tensor::zeros(&[n_batch], dtype);
        let mut no_speech_probs = vec![f32::NAN; n_batch as usize];

        for i in 0..self.sample_len {
            let logits = {
                let mut tokens = tokens.i(..);

                if *tokens.size().last().unwrap() as usize > self.initial_tokens.len() {
                    // only need to use the last token except in the first forward pass
                    tokens = tokens.slice(-1, -1, None, 1);
                }

                self.model.decoder.forward_ext(&tokens, &audio_features, i)
            };

            if i == 0 {
                let probs_at_sot = logits.i((.., self.sot_idx as i64)).softmax(-1, dtype.0);
                probs_at_sot
                    .i((.., self.tokenizer.token_id_nospeech as i64))
                    .copy_data(&mut no_speech_probs[..], n_batch as usize);
            }

            let mut logits = logits.i((.., -1));

            for filter in &self.logit_filters {
                filter.apply(&mut logits, &tokens);
            }

            let (new_tokens, completed) =
                self.token_extractor
                    .update(tokens, logits, &mut sum_logprobs);
            tokens = new_tokens.to_device(self.device);

            if completed || *tokens.size().last().unwrap() > self.model.dims.n_text_ctxs {
                break;
            }
        }

        (tokens, sum_logprobs, no_speech_probs)
    }

    pub fn run(&mut self, mel: Tensor) -> anyhow::Result<Vec<DecodingResult>> {
        // without no_grad, pytorch will save the result of every operation for
        // calculating gradients which will cause the program to run out of
        // memory immediately
        tch::no_grad(move || self.run_inner(mel.to_device(self.device)))
    }

    fn run_inner(&mut self, mel: Tensor) -> anyhow::Result<Vec<DecodingResult>> {
        let n_audio = mel.size()[0];

        let initial_tokens: Vec<_> = self.initial_tokens.iter().map(|t| *t as i64).collect();
        // dbg!(&initial_tokens);

        let audio_features = self.model.encoder.forward(&mel);
        let repeated_tokens = Tensor::of_slice(&initial_tokens[..])
            .repeat(&[n_audio, 1])
            .to_device(self.device);

        // tensor_dbg!(&audio_features);
        // tensor_dbg!(&repeated_tokens);

        let n_group = match self.options.token_extract_mode {
            TokenExtractMode::Greedy(n) => n,
            TokenExtractMode::BeamSearch { beam_size, .. } => beam_size,
        } as i64;

        // dbg!(&n_group);

        // repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        let audio_features = audio_features.repeat_interleave_self_int(n_group, 0, None);
        let repeated_tokens = repeated_tokens.repeat_interleave_self_int(n_group, 0, None);

        // tensor_dbg!(&audio_features);
        // tensor_dbg!(&repeated_tokens);

        // call the main sampling loop
        let (tokens, sum_logprobs, no_speech_probs) =
            self.main_loop(&audio_features, repeated_tokens);

        // tensor_dbg!(&tokens);
        // dbg!(&sum_logprobs);
        // dbg!(&no_speech_probs);

        // reshape the tensors to have (n_audio, n_group) as the first two dimensions
        let audio_features = audio_features.slice(0, None, None, n_group);
        let no_speech_probs = Tensor::of_slice(&no_speech_probs[..]);
        let no_speech_probs = no_speech_probs.slice(0, None, None, n_group);

        debug_assert_eq!(audio_features.size()[0], n_audio);
        debug_assert_eq!(no_speech_probs.size()[0], n_audio);

        // tensor_dbg!(&audio_features);
        // tensor_dbg!(&no_speech_probs);

        let tokens = tokens.reshape(&[n_audio, n_group, -1]);
        let sum_logprobs = sum_logprobs.reshape(&[n_audio, n_group]);

        // tensor_dbg!(&tokens);
        // tensor_dbg!(&sum_logprobs);

        // get the final candidates for each group, and slice between the first sampled token and EOT
        let (tokens, sum_logprobs) = self.token_extractor.finalize(tokens, sum_logprobs);
        let tokens: Vec<Vec<Tensor>> = tokens
            .into_iter()
            .map(|s| {
                s.into_iter()
                    .map(|t| {
                        let end = t
                            .eq(self.tokenizer.token_id_eot as i64)
                            .nonzero()
                            .int64_value(&[0, 0]);

                        t.i(self.initial_tokens.len() as i64..end)
                    })
                    .collect()
            })
            .collect();

        dbg!(&tokens);
        tensor_dbg!(&sum_logprobs);

        // select the top-ranked sample in each group
        let selected = self.sequence_ranker.rank(&tokens, &sum_logprobs);
        let tokens: Vec<_> = tokens
            .into_iter()
            .zip(&selected)
            .map(|(mut t, i)| t.remove(*i as usize))
            .collect();
        let texts: Vec<_> = tokens.iter().map(|t| self.tokenizer.decode(t)).collect();

        dbg!(&selected);
        dbg!(&tokens);
        dbg!(&texts);

        let sum_logprobs = selected
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, selection)| f64::from(sum_logprobs.i((idx as i64, selection))));
        let avg_logprobs: Vec<_> = sum_logprobs
            .zip(&tokens)
            .map(|(lp, t)| lp / (t.size()[0] as f64 + 1.))
            .collect();

        println!("tokens = {tokens:?}");
        println!("texts = {texts:?}");
        println!("avg_logprobs = {avg_logprobs:?}");

        Ok(vec![])

        // fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
        // if len(set(map(len, fields))) != 1:
        //     raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        // Ok(DecodingResult {
        //     audio_features,
        //     language,
        //     tokens,
        //     text,
        //     avg_logprob,
        //     no_speech_prob,
        //     temperature,
        //     compression_ratio: 0.,
        // })

        // return [
        //     DecodingResult(
        //         audio_features=features,
        //         language=language,
        //         tokens=tokens,
        //         text=text,
        //         avg_logprob=avg_logprob,
        //         no_speech_prob=no_speech_prob,
        //         temperature=self.options.temperature,
        //         compression_ratio=compression_ratio(text),
        //     )
        //     for text, language, tokens, features, avg_logprob, no_speech_prob in zip(*fields)
        // ]
    }
}
