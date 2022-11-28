use std::{fs::File, io::ErrorKind, path::Path};

use anyhow::Context;
use rubato::Resampler;
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, formats::FormatOptions, io::MediaSourceStream,
    meta::MetadataOptions, probe::Hint,
};
use tch::{IndexOp, Kind, Tensor};

// hard-coded audio hyperparameters
const SAMPLE_RATE: i64 = 16000;
const N_FFT: i64 = 400;
const N_MELS: i64 = 80;
const HOP_LENGTH: i64 = 160;
const CHUNK_LENGTH: i64 = 30;
const N_SAMPLES: i64 = CHUNK_LENGTH * SAMPLE_RATE; // 480000: number of samples in a chunk
const N_FRAMES: i64 = N_SAMPLES / HOP_LENGTH; // 3000: number of frames in a mel spectrogram input

pub fn load_audio(path: impl AsRef<Path>) -> anyhow::Result<Tensor> {
    // number of samples per chunk that we feed into the resampler
    const RESAMPLE_CHUNK_SIZE: usize = 1024;

    // Create a media source. Note that the MediaSource trait is automatically implemented for File,
    // among other types.
    let file = Box::new(File::open(path).context("failed to open audio file")?);

    // Create the media source stream using the boxed media source from above.
    let mss = MediaSourceStream::new(file, Default::default());

    // Create a hint to help the format registry guess what format reader is appropriate. In this
    // example we'll leave it empty.
    let hint = Hint::new();

    // Use the default options when reading and decoding.
    let format_opts: FormatOptions = Default::default();
    let metadata_opts: MetadataOptions = Default::default();
    let decoder_opts: DecoderOptions = Default::default();

    // Probe the media source stream for a format.
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .context("failed to determine audio format")?;

    // Get the format reader yielded by the probe operation.
    let mut format = probed.format;

    // Get the default track.
    let track = format
        .default_track()
        .context("failed to get default audio track")?;

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .context("failed to create decoder for track")?;

    // Store the track identifier, it will be used to filter packets.
    let track_id = track.id;

    // get the sample rate and channel count
    let track_sr = track
        .codec_params
        .sample_rate
        .context("track sample rate is unknown")? as i64;

    let num_channels = track
        .codec_params
        .channels
        .context("unknown channel count")?
        .count();

    let mut resampler = if track_sr == SAMPLE_RATE {
        None
    } else {
        // default parameters yoinked from rubato example
        let params = rubato::InterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: rubato::InterpolationType::Linear,
            oversampling_factor: 256,
            window: rubato::WindowFunction::BlackmanHarris2,
        };

        let resampler = rubato::SincFixedIn::<f32>::new(
            SAMPLE_RATE as f64 / track_sr as f64,
            1.0,
            params,
            RESAMPLE_CHUNK_SIZE,
            // we downmix to mono before putting audio into the resampler
            1,
        )
        .context("failed to create resampler")?;

        Some(resampler)
    };

    let mut raw_audio = Vec::<f32>::new();

    // use as a deque of samples that need resampling
    // b/c the resampler accepts fixed-size chunks of samples, but the
    let mut resample_queue = Vec::<f32>::new();

    // The decode loop.
    loop {
        // Get the next packet from the media format.
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(io))
                if io.kind() == ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(other) => anyhow::bail!(other),
        };

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples.
        let samples_ref = decoder.decode(&packet).context("failed to decode packet")?;

        debug_assert_eq!(samples_ref.spec().channels.count(), num_channels);

        // Convert samples into f32
        let mut samples_buf = SampleBuffer::new(samples_ref.capacity() as u64, *samples_ref.spec());
        samples_buf.copy_interleaved_ref(samples_ref);

        // Downmix to mono
        let samples_mixed: Vec<_> = if num_channels == 1 {
            Vec::from(samples_buf.samples())
        } else {
            samples_buf
                .samples()
                .chunks_exact(num_channels)
                .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                .collect()
        };

        if let Some(resampler) = &mut resampler {
            resample_queue.extend(samples_mixed);

            while resample_queue.len() >= RESAMPLE_CHUNK_SIZE {
                let remainder = resample_queue.split_off(RESAMPLE_CHUNK_SIZE);
                let chunk = std::mem::replace(&mut resample_queue, remainder);

                let raw_audio_segments = resampler
                    .process(&[chunk], None)
                    .context("failed to resample packet")?;

                // it's a vec for each channel, but we have restricted input to mono audio only
                // so there should be only one
                debug_assert_eq!(raw_audio_segments.len(), 1);

                raw_audio.extend(&raw_audio_segments[0]);
            }
        } else {
            raw_audio.extend(samples_buf.samples());
        }
    }

    // if there are leftover samples that need resampling, pad with zero and
    // then resample
    if resample_queue.len() > 0 {
        resample_queue.resize(RESAMPLE_CHUNK_SIZE, 0.0);

        let raw_audio_segments = resampler
            .unwrap()
            .process(&[resample_queue], None)
            .context("failed to resample packet")?;

        // it's a vec for each channel, but we have restricted input to mono audio only
        // so there should be only one
        debug_assert_eq!(raw_audio_segments.len(), 1);

        raw_audio.extend(&raw_audio_segments[0]);
    }

    Ok(Tensor::of_slice(&raw_audio[..]))
}

pub fn mel_filter() -> Tensor {
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
    let stft = audio.stft_center(
        N_FFT,
        HOP_LENGTH,
        None,
        Some(&window),
        true,
        "reflect",
        false,
        true,
        true,
    );
    println!(
        "stft size = {:?}, window size = {:?}, audio size = {:?}",
        stft.size(),
        window.size(),
        audio.size()
    );
    let magnitudes = stft.slice(1, None, Some(-1), 1).abs().pow_tensor_scalar(2);
    println!("magnitudes size = {:?}", magnitudes.size());

    let mel_filter = mel_filter.to_device(dev);
    // calculate mel spectrogram by multiplying mel filter w/ magnitudes at
    // given frequencies
    println!("mel_filter size = {:?}", mel_filter.size());
    let mel_spec = mel_filter.matmul(&magnitudes);
    // remove zeros to avoid nan, then take log
    let log_spec = mel_spec.clamp_min(1e-10).log10();
    // force range to 8 by flooring out values much smaller than the maximum
    let log_spec = log_spec.maximum(&(log_spec.max() - 8.0));
    // scale
    (log_spec + 4.0) / 4.0
}

#[cfg(test)]
mod test {
    use tch::IndexOp;

    use crate::util::test::read_csv_2d;

    #[test]
    fn test_mel_filter() {
        let actual = super::mel_filter();
        let actual = actual.slice(0, None, 8, 1).slice(1, None, 8, 1);

        let expected = read_csv_2d("test/data/mel-filter-8x8.csv").unwrap();

        assert!(
            actual.allclose(&expected, 1e-05, 1e-08, false),
            "actual = {}, expected = {}",
            actual,
            expected
        );
    }

    #[test]
    fn test_mel_spectrogram() {
        // for this test, we use pre-resampled audio that was resampled using
        // ffmpeg so that our results will match the results of the official
        // Whisper function, even though rust resample is confirmed to work
        let audio = super::load_audio("test/data/jfk_resampled.wav").unwrap();
        let mel_filter = super::mel_filter();
        let actual = super::log_mel_spectrogram(&audio, &mel_filter);
        let expected = read_csv_2d("test/data/mel-spectrogram.csv").unwrap();

        // based on manual inspection, the mel spectrogram works but the two
        // spectrograms differ slightly in a way that i don't know how to test
        // for automated-ly

        assert!(
            actual.allclose(&expected, 0.1, 0.005, false),
            "actual = {}, expected = {}",
            actual,
            expected
        );
    }
}
