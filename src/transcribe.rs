use std::rc::Rc;

use anyhow::Context;
use tch::{IndexOp, Tensor};

use crate::{
    audio::{self, N_FRAMES},
    decode::{DecodeOptions, DecodeTask, TokenExtractMode},
    model::Whisper,
    tokenize::{Task, Tokenizer},
};

#[derive(Debug)]
pub struct TranscribeTask<'a> {
    model: &'a Whisper,
    decode_task: DecodeTask<'a>,
    tokenizer: Rc<Tokenizer>,

    prompt: TranscribePrompt,

    device: tch::Device,
}

#[derive(Debug)]
pub enum TranscribePrompt {
    Pretokenized(Vec<u32>),
    Text(String),
    None { condition_on_prev_text: bool },
}

#[derive(Debug)]
pub struct TranscribeOptions {
    // decode options
    pub sample_len: Option<usize>,
    pub token_extract_mode: TokenExtractMode,
    pub len_penalty: Option<f64>,
    pub max_initial_timestamp: Option<f64>,
    pub timestamps: bool,
    pub suppress_blank: bool,
    pub suppress_non_speech: bool,
    pub suppress_tokens: Option<Vec<u32>>,

    // transcribe options
    /// Initial prompt for transcription, which is prepended to the input and to
    /// the output.
    pub prompt: TranscribePrompt,
}

#[derive(Debug)]
pub struct TranscribeOutput {
    pub tokens: Tensor,
    pub text: String,
    pub segments: Vec<TranscribeOutputSegment>,
}

#[derive(Debug)]
pub struct TranscribeOutputSegment {
    pub seek: i64,
    pub start_time: f64,
    pub end_time: f64,
    pub start_token: i64,
    pub end_token: i64,
    pub text: String,
}

impl<'a> TranscribeTask<'a> {
    pub fn new(model: &'a Whisper, options: TranscribeOptions) -> anyhow::Result<Self> {
        let device = model.device();

        let tokenizer =
            Rc::new(Tokenizer::new(Task::Transcribe).context("failed to create tokenizer")?);

        let decode_task = DecodeTask::new(
            model,
            tokenizer.clone(),
            DecodeOptions {
                task: Task::Transcribe,
                sample_len: options.sample_len,
                token_extract_mode: options.token_extract_mode,
                len_penalty: options.len_penalty,
                max_initial_timestamp: options.max_initial_timestamp,
                timestamps: options.timestamps,
                suppress_blank: options.suppress_blank,
                suppress_tokens: options.suppress_tokens,
                suppress_non_speech: options.suppress_non_speech,
                // prompt is set at runtime loop
                prompt: None,
            },
        )
        .context("failed to create decode task")?;

        Ok(Self {
            device,
            model,
            tokenizer,
            decode_task,
            prompt: options.prompt,
        })
    }

    pub fn run(&mut self, audio: &Tensor) -> anyhow::Result<TranscribeOutput> {
        // without no_grad, pytorch will save the result of every operation for
        // calculating gradients which will cause the program to run out of
        // memory immediately
        tch::no_grad(move || self.run_inner(audio))
    }

    fn run_inner(&mut self, audio: &Tensor) -> anyhow::Result<TranscribeOutput> {
        let mel_filter = audio::mel_filter();
        let mel_audio = audio::log_mel_spectrogram(audio, &mel_filter);

        let (_, n_frames) = mel_audio.size2().unwrap();

        // smallest amount of time that is discernable by this model
        const QUANTUM_LENGTH: f64 = audio::HOP_LENGTH as f64 / audio::SAMPLE_RATE as f64;

        // mel frames per output token: 2
        let input_stride: i64 = audio::N_FRAMES / self.model.dims.n_audio_ctxs;
        // time per output token: 0.02 (seconds)
        let time_precision = input_stride as f64 * QUANTUM_LENGTH;

        let mut seek = 0;

        let mut tokens = {
            // if we have an initial prompt, prepend it to the list of
            // tokens we generate, so that it is fed into the decoder
            let tokens = match &self.prompt {
                TranscribePrompt::Pretokenized(t) => t.iter().copied().map(|t| t as i64).collect(),
                TranscribePrompt::Text(t) => {
                    let tokens = self.tokenizer.encode(t.as_str(), true).unwrap();
                    let tokens = Vec::from(tokens.get_ids());
                    tokens.into_iter().map(|t| t as i64).collect()
                }
                TranscribePrompt::None { .. } => vec![],
            };

            Tensor::of_slice(&tokens[..]).to_device(self.device)
        };

        let condition_on_prev_text = match &self.prompt {
            TranscribePrompt::None {
                condition_on_prev_text,
            } => *condition_on_prev_text,
            _ => true,
        };

        let mut segments = vec![];

        while seek < n_frames {
            let mel_audio_segment = audio::pad_or_trim(&mel_audio.i((.., seek..)), audio::N_FRAMES);
            let mut segment_duration = audio::CHUNK_LENGTH as f64;

            if condition_on_prev_text {
                // put the tokens we have so far into the decoder
                self.decode_task.set_prompt(Some(&tokens));
            }

            dbg!(&mel_audio_segment);

            // we are only putting one audio track in there at once, so it should just return one result
            let mut results = self.decode_task.run(&mel_audio_segment)?;
            let result = results.remove(0);
            let segment_tokens = result.tokens;

            // TODO: no speech detection

            let ts_tokens = segment_tokens.ge(self.tokenizer.token_id_ts_begin as i64);
            let ts_tokens_consecutive = Tensor::where_(
                &ts_tokens
                    .slice(0, None, -1, 1)
                    .logical_and(&ts_tokens.slice(0, 1, None, 1)),
            )
            .into_iter()
            .nth(0)
            .unwrap()
                + 1;

            let ts_offset = seek as f64 * QUANTUM_LENGTH;
            let token_id_ts_begin = self.tokenizer.token_id_ts_begin as i64;

            if ts_tokens_consecutive.size1()? > 0 {
                // output contains two consecutive timestamp tokens

                let mut last_slice = 0;

                for current_slice in ts_tokens_consecutive.iter::<i64>()? {
                    let sliced_tokens = segment_tokens.slice(0, last_slice, current_slice, 1);
                    let start_ts_pos = sliced_tokens.int64_value(&[0]) - token_id_ts_begin;
                    let end_ts_pos = sliced_tokens.int64_value(&[-1]) - token_id_ts_begin;

                    let token_offset = tokens.size1()?;

                    segments.push(TranscribeOutputSegment {
                        start_time: ts_offset + start_ts_pos as f64 * time_precision,
                        end_time: ts_offset + end_ts_pos as f64 * time_precision,
                        start_token: token_offset + last_slice + 1,
                        end_token: token_offset + current_slice,
                        seek,
                        text: self.tokenizer.decode(&sliced_tokens)?,
                    });

                    last_slice = current_slice;
                }

                let last_ts_pos = segment_tokens.int64_value(&[last_slice - 1]) - token_id_ts_begin;
                seek += last_ts_pos * input_stride;
                tokens = Tensor::cat(
                    &[tokens, segment_tokens.slice(0, None, last_slice + 1, 1)],
                    -1,
                );
            } else {
                let timestamps = segment_tokens.index(&[Some(ts_tokens.nonzero().flatten(0, -1))]);

                let last_token_id = i64::from(timestamps.i(-1));
                if timestamps.size()[0] > 0 && last_token_id != token_id_ts_begin {
                    // no consecutive timestamps but it has a timestamp; use the last one.
                    // single timestamp at the end means no speech after the last timestamp.
                    let last_ts_pos = last_token_id - token_id_ts_begin;
                    segment_duration = last_ts_pos as f64 * time_precision;
                }

                segments.push(TranscribeOutputSegment {
                    start_time: ts_offset,
                    end_time: ts_offset + segment_duration,
                    start_token: 0,
                    end_token: segment_tokens.size()[0],
                    seek,
                    text: result.text,
                });

                seek += N_FRAMES;
                tokens = Tensor::cat(&[tokens, segment_tokens], -1);
            }
        }

        let text = self.tokenizer.decode(&tokens)?;

        Ok(TranscribeOutput {
            tokens,
            text,
            segments,
        })
    }
}
