use tch::{nn::VarStore, Device};

use whisper::{
    self,
    model::{ModelDims, Whisper},
    transcribe::{TranscribeOptions, TranscribePrompt, TranscribeTask},
};

pub fn main() {
    let device = Device::cuda_if_available();
    dbg!(device);

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
    vars.set_device(device);

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
            suppress_tokens: Some(vec![]),
            timestamps: true,
            prompt: TranscribePrompt::None {
                condition_on_prev_text: true,
            },
        },
    )
    .unwrap();

    let audio = whisper::audio::load_audio("test/data/export_resampled.wav").unwrap();
    let output = task.run(&audio).unwrap();
    println!("{output:?}");
}
