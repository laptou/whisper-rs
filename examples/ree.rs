use tch::{nn::VarStore, Device, Tensor, Kind};
use whisper::{self, model::Whisper};

pub fn main() {
    let tensors = tch::Tensor::read_npz("oof.npz").unwrap();
    Tensor::save_multi(&tensors[..], "oof.ot").unwrap();
    let mut vars = VarStore::new(Device::cuda_if_available());
    // ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=512,
    // n_audio_head=8, n_audio_layer=6, n_vocab=51864, n_text_ctx=448,
    // n_text_state=512, n_text_head=8, n_text_layer=6)
    let model = Whisper::new(
        vars.root(),
        51864,
        1500,
        512,
        8,
        6,
        448,
        512,
        8,
        6,
        Tensor::empty(&[1, 1], (Kind::Float, Device::cuda_if_available())),
    );
    vars.load("oof.ot").unwrap();
    
    println!("{:?}", vars.variables());
}
