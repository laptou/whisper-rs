use tch::{nn::VarStore, Device};

use tracing::info;
use whisper::{
    self,
    model::{ModelDims, Whisper},
    tokenize::{Task, Tokenizer},
    transcribe::{TranscribeOptions, TranscribePrompt, TranscribeTask},
};

pub fn main() {
    tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::builder()
            .pretty()
            .finish(),
    )
    .unwrap();

    let device = Device::Cpu;
    info!("using device {device:?}");

    let mut vars = VarStore::new(device);
    let model = Whisper::new(
        vars.root(),
        ModelDims {
            n_vocab: 51864,
            n_audio_ctxs: 1500,
            n_audio_states: 512,
            n_audio_heads: 8,
            n_audio_layers: 6,
            n_text_ctxs: 448,
            n_text_states: 512,
            n_text_heads: 8,
            n_text_layers: 6,
        },
    );
    vars.load("weights.ot").unwrap();
    info!("loaded model weights");

    let tokenizer = Tokenizer::new_from_file(Task::Transcribe, "tokenizer/gpt2.json").unwrap();

    let mut task = TranscribeTask::new(
        &model,
        TranscribeOptions {
            sample_len: None,
            token_extract_mode: whisper::decode::TokenExtractMode::BeamSearch {
                beam_size: 5,
                patience: 1.0,
            },
            len_penalty: None,
            max_initial_timestamp: Some(1.0),
            suppress_blank: true,
            suppress_non_speech: true,
            suppress_tokens: None,
            timestamps: true,
            prompt: TranscribePrompt::None {
                condition_on_prev_text: true,
            },
            tokenizer,
        },
    )
    .unwrap();
    info!("initialized transcription task");

    // mp3 loader is crazy slow in dev mode, so we load a preconverted wav for
    // convenience
    let audio = whisper::audio::load_audio("test/data/jfk_resampled.wav").unwrap();
    info!("loaded audio");

    let output = task.run(&audio).unwrap();
    info!("transcribed audio, output: {output:#?}");
}
