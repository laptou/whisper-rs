use std::fmt::Debug;
use tch::Tensor;

mod suppress_blanks;
mod suppress_tokens;
mod timestamp_tokens;

pub use suppress_blanks::*;
pub use suppress_tokens::*;
pub use timestamp_tokens::*;

pub trait LogitFilter: Debug {
    fn apply(&self, logits: &mut Tensor, tokens: &Tensor);
}
