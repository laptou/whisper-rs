use tch::{nn::VarStore, Device};

use whisper::{
    self,
    decode::{DecodeOptions, DecodeTask},
    model::{ModelDims, Whisper},
};

pub fn main() {
    // let tensors = tch::Tensor::read_npz("oof.npz").unwrap();
    // dbg!(tensors);
    // Tensor::save_multi(&tensors[..], "oof.ot").unwrap();

    let mut vars = VarStore::new(Device::cuda_if_available());
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

    vars.load("oof.ot").unwrap();
    // println!("vocab size = {}", tok.get_vocab_size(true));

    let mut dt = DecodeTask::new(
        &model,
        DecodeOptions {
            task: whisper::tokenize::Task::Transcribe,
            sample_len: None,
            token_extract_mode: whisper::decode::TokenExtractMode::BeamSearch {
                beam_size: 5,
                patience: 1.0,
            },
            len_penalty: None,
        },
    )
    .unwrap();

    let audio = whisper::audio::load_audio("test/data/jfk_resampled.wav").unwrap();
    println!("audio = {audio}");

    let mel_filter = whisper::audio::mel_filter();

    let mel_audio = whisper::audio::log_mel_spectrogram(&audio, &mel_filter);
    println!("mel_audio = {mel_audio}");
    let mel_audio = whisper::audio::pad_or_trim(&mel_audio, whisper::audio::N_FRAMES);
    println!("mel_audio = {mel_audio}");
    let mel_audio = mel_audio.unsqueeze(0);

    dt.run(mel_audio).unwrap();
}
