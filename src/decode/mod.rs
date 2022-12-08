use std::cell::RefCell;

use std::fmt::Debug;
use std::rc::Rc;

use tch::{nn::Module, Device, IndexOp, Kind, Tensor};

use crate::audio;
use crate::util::tensor_dbg;
use crate::{
    model::Whisper,
    tokenize::{Task, Tokenizer},
};

pub use self::extractor::TokenExtractMode;
use self::extractor::*;
use self::filter::*;
use self::sequencer::*;

mod extractor;
mod filter;
mod sequencer;

#[derive(Debug)]
pub struct DecodeTask<'a> {
    model: &'a Whisper,
    options: DecodeOptions,
    device: Device,

    /// Information related to the prompt given to the model
    // Stored in an `Rc<Cell<>>` b/c it may change at runtime
    prompt: Rc<RefCell<DecodePrompt>>,

    tokenizer: Rc<Tokenizer>,
    logit_filters: Vec<Box<dyn LogitFilter>>,
    token_extractor: Box<dyn TokenExtractor>,
    sequence_ranker: Box<dyn SequenceRanker>,
    sample_len: i64,
}

#[derive(Debug)]
struct DecodePrompt {
    sample_begin: i64,
    sot_idx: usize,
    initial_tokens: Tensor,
}

impl DecodePrompt {
    pub fn new(prompt: Option<&Tensor>, tokenizer: &Tokenizer, device: Device) -> Self {
        let seq_sot: Vec<_> = tokenizer
            .sequence_sot()
            .into_iter()
            .map(|t| t as i64)
            .collect();

        let initial_tokens = if let Some(prompt) = prompt {
            // if we have a prompt, concat the start-of-previous sequence, the
            // prompt, and the start-of-transcript sequence

            let initial_tokens = Tensor::empty(
                &[1 + prompt.size1().unwrap() + seq_sot.len() as i64],
                (Kind::Int64, device),
            );

            initial_tokens
                .narrow(0, 0, 1)
                .copy_(&Tensor::from(tokenizer.token_id_startofprev as i64));

            initial_tokens
                .narrow(0, 1, prompt.size1().unwrap())
                .copy_(&prompt);

            initial_tokens
                .narrow(0, 1 + prompt.size1().unwrap(), seq_sot.len() as i64)
                .copy_(&Tensor::of_slice(seq_sot.as_slice()));

            tensor_dbg!(prompt);

            initial_tokens
        } else {
            Tensor::of_slice(seq_sot.as_slice())
        };

        tensor_dbg!(initial_tokens);

        let sample_begin = initial_tokens.size1().unwrap();

        dbg!(sample_begin);

        let sot_idx = initial_tokens
            .eq(tokenizer.token_id_sot as i64)
            .nonzero()
            .int64_value(&[0]) as usize;

        Self {
            sample_begin,
            sot_idx,
            initial_tokens,
        }
    }
}

#[derive(Debug)]
pub struct DecodeOptions {
    pub task: Task,
    pub sample_len: Option<usize>,
    pub token_extract_mode: TokenExtractMode,
    pub len_penalty: Option<f64>,
    pub max_initial_timestamp: Option<f64>,
    pub timestamps: bool,
    pub suppress_blank: bool,
    pub suppress_non_speech: bool,
    pub suppress_tokens: Option<Vec<u32>>,
    pub prompt: Option<Tensor>,
}

#[derive(Debug)]
pub struct DecodeOutput {
    pub audio_features: Tensor,
    // pub language: String,
    // pub language_probs: Option<HashMap<String, f32>>,
    pub tokens: Tensor,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    // pub temperature: f32,
    // pub compression_ratio: f32,
}

impl<'a> DecodeTask<'a> {
    pub fn new(
        model: &'a Whisper,
        tokenizer: Rc<Tokenizer>,
        options: DecodeOptions,
    ) -> anyhow::Result<Self> {
        let device = model.device();

        let sample_len = options
            .sample_len
            .map_or(model.dims.n_text_ctxs / 2, |s| s as i64);

        let token_extractor = match options.token_extract_mode {
            TokenExtractMode::Greedy { .. } => todo!(),
            TokenExtractMode::BeamSearch {
                beam_size,
                patience,
            } => Box::new(BeamSearchTokenExtractor {
                beam_size,
                token_id_eot: tokenizer.token_id_eot,
                finished_sequences: None,
                patience,
            }),
        };

        let sequence_ranker = Box::new(MaximumLikelihoodRanker {
            length_penalty: options.len_penalty,
        });

        let prompt = Rc::new(RefCell::new(DecodePrompt::new(
            options.prompt.as_ref(),
            &*tokenizer,
            device,
        )));

        let mut logit_filters: Vec<Box<dyn LogitFilter>> = vec![];

        if options.suppress_blank {
            logit_filters.push(Box::new(SuppressBlank::new(
                &tokenizer,
                prompt.clone(),
                device,
            )));
        }

        if options.suppress_tokens.is_some() || options.suppress_non_speech {
            let mut suppress_tokens = options.suppress_tokens.clone().unwrap_or(vec![]);

            if options.suppress_non_speech {
                suppress_tokens.extend(tokenizer.non_speech_tokens());
            }

            logit_filters.push(Box::new(SuppressTokens::new(&suppress_tokens[..], device)));
        }

        if options.timestamps {
            // usually 0.02 seconds
            let precision = crate::audio::CHUNK_LENGTH as f64 / model.dims.n_audio_ctxs as f64;
            let max_initial_timestamp_index = match options.max_initial_timestamp {
                Some(mit) => Some(f64::round(mit / precision) as i64),
                None => None,
            };

            logit_filters.push(Box::new(TimestampTokens::new(
                tokenizer.clone(),
                prompt.clone(),
                max_initial_timestamp_index,
                device,
            )));
        }

        Ok(Self {
            device,
            options,
            model,

            token_extractor,
            sequence_ranker,

            tokenizer,
            prompt,
            sample_len,
            logit_filters,
        })
    }

    pub fn set_prompt(&mut self, prompt: Option<&Tensor>) {
        // replace empty tensor w/ None
        let prompt = prompt.and_then(|p| {
            if p.size1().unwrap() > 0 {
                Some(p)
            } else {
                None
            }
        });

        self.prompt
            .replace(DecodePrompt::new(prompt, &*self.tokenizer, self.device));
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

        let prompt = &*self.prompt.borrow();

        for i in 0..self.sample_len {
            let logits = {
                let mut tokens = tokens.i(..);

                if *tokens.size().last().unwrap() > prompt.sample_begin {
                    // only need to use the last token except in the first forward pass
                    tokens = tokens.slice(-1, -1, None, 1);
                }

                // tensor_dbg!(tokens);
                self.model.decoder.forward_ext(&tokens, &audio_features, i)
            };

            // tensor_dbg!(logits);
            // tensor_dbg!(tokens);

            if i == 0 {
                let probs_at_sot = logits.i((.., prompt.sot_idx as i64)).softmax(-1, dtype.0);
                probs_at_sot
                    .i((.., self.tokenizer.token_id_no_speech as i64))
                    .copy_data(&mut no_speech_probs[..], n_batch as usize);
            }

            // now we need to consider the logits at the last token only
            let mut logits = logits.i((.., -1));

            // apply the logit filters, e.g. for suppressing or applying penalty to
            for filter in &self.logit_filters {
                filter.apply(&mut logits, &tokens);
                // tensor_dbg!(logits);
            }

            // expand the tokens tensor with the selected next tokens
            let (new_tokens, completed) =
                self.token_extractor
                    .update(&self.model, tokens, logits, &mut sum_logprobs);
            tokens = new_tokens.to_device(self.device);

            // tensor_dbg!(tokens);

            if completed || *tokens.size().last().unwrap() > self.model.dims.n_text_ctxs {
                break;
            }
        }

        self.model.clear_cache();

        (tokens, sum_logprobs, no_speech_probs)
    }

    pub fn run(&mut self, mel_audio: &Tensor) -> anyhow::Result<Vec<DecodeOutput>> {
        // without no_grad, pytorch will save the result of every operation for
        // calculating gradients which will cause the program to run out of
        // memory immediately
        tch::no_grad(move || self.run_inner(mel_audio.to_device(self.device).unsqueeze(0)))
    }

    fn run_inner(&mut self, mel_audio: Tensor) -> anyhow::Result<Vec<DecodeOutput>> {
        let (n_audio, _, n_frames) = mel_audio.size3().expect("2d mel spectrogram");

        debug_assert_eq!(n_frames, audio::N_FRAMES);

        let audio_features = self.model.encoder.forward(&mel_audio);
        let repeated_tokens = self
            .prompt
            .borrow()
            .initial_tokens
            .repeat(&[n_audio, 1])
            .to_device(self.device);

        // tensor_dbg!(&audio_features);
        // tensor_dbg!(&repeated_tokens);

        let n_group = self.token_extractor.group_size() as i64;

        // dbg!(&n_group);

        // repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        let audio_features = audio_features.repeat_interleave_self_int(n_group, 0, None);
        let repeated_tokens = repeated_tokens.repeat_interleave_self_int(n_group, 0, None);

        // tensor_dbg!(&audio_features);
        // tensor_dbg!(&repeated_tokens);

        // this is necessary b/c we run the encoder, which causes values to be
        // persisted into its cache which messes things up later when beam
        // search tries to update the cache

        // TODO: solve this properly by removing caches in the encoder? doesn't
        // seem like they are used anyway
        self.model.clear_cache();
        self.token_extractor.reset();

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

        let sample_begin = self.prompt.borrow().sample_begin;

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

                        tensor_dbg!(t);

                        t.i(sample_begin..end)
                    })
                    .collect()
            })
            .collect();

        // dbg!(&tokens);
        // tensor_dbg!(&sum_logprobs);

        // select the top-ranked sample in each group
        let selected = self.sequence_ranker.rank(&tokens, &sum_logprobs);
        let tokens: Vec<_> = tokens
            .into_iter()
            .zip(&selected)
            .map(|(mut t, i)| t.remove(*i as usize))
            .collect();
        let texts: Result<Vec<_>, _> = tokens.iter().map(|t| self.tokenizer.decode(t)).collect();
        let texts = texts?;

        // dbg!(&selected);
        // dbg!(&tokens);
        // dbg!(&texts);

        let sum_logprobs = selected
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, selection)| f64::from(sum_logprobs.i((idx as i64, selection))));
        let avg_logprobs: Vec<_> = sum_logprobs
            .zip(&tokens)
            .map(|(lp, t)| lp / (t.size()[0] as f64 + 1.))
            .collect();

        let mut results = vec![];

        for (i, (((tokens, text), avg_logprob), no_speech_prob)) in tokens
            .into_iter()
            .zip(texts)
            .zip(avg_logprobs)
            .zip(Vec::<f64>::from(&no_speech_probs))
            .enumerate()
        {
            results.push(DecodeOutput {
                audio_features: audio_features.i(i as i64),
                tokens,
                text,
                avg_logprob,
                no_speech_prob,
            });
        }

        Ok(results)
    }
}
