use tch::{Kind, Tensor};

// hard-coded audio hyperparameters
const SAMPLE_RATE: i64 = 16000;
const N_FFT: i64 = 400;
const N_MELS: i64 = 80;
const HOP_LENGTH: i64 = 160;
const CHUNK_LENGTH: i64 = 30;
const N_SAMPLES: i64 = CHUNK_LENGTH * SAMPLE_RATE; // 480000: number of samples in a chunk
const N_FRAMES: i64 = N_SAMPLES / HOP_LENGTH; // 3000: number of frames in a mel spectrogram input

fn mel_filter() -> Tensor {
    use mel_filter::{mel, NormalizationFactor};
    let ret = mel::<f32>(
        SAMPLE_RATE as usize,
        N_FFT as usize,
        Some(N_MELS as usize),
        None,
        None,
        false,
        NormalizationFactor::One,
    );

    Tensor::of_slice2(&ret[..])
}

pub fn log_mel_spectrogram(audio: &Tensor, mel_filter: &Tensor) -> Tensor {
    let dev = audio.device();

    let window = Tensor::hann_window(N_FFT, (Kind::Float, dev));
    // compute sliding window fourier transform to get frequency components
    // changing over time
    let stft = audio.stft(N_FFT, HOP_LENGTH, None, Some(&window), false, true, true);
    let magnitudes = stft.slice(1, None, Some(-1), 1).abs().pow_tensor_scalar(2);

    let mel_filter = mel_filter.to_device(dev);
    // calculate mel spectrogram by multiplying mel filter w/ magnitudes at
    // given frequencies
    let mel_spec = mel_filter.matmul(&magnitudes);
    // remove zeros to avoid nan, then take log
    let log_spec = mel_spec.clamp_min(1e-10).log10();
    // force range to 8 by flooring out values much smaller than the maximum
    let log_spec = log_spec.maximum(&(log_spec.max() - 8.0));
    // scale
    (log_spec + 4.0) / 4.0
}
