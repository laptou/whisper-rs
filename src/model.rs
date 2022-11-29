use tch::{
    nn::{self, Module},
    Device, IndexOp, Kind, NewAxis, Tensor,
};

use crate::audio::N_MELS;

#[derive(Debug)]
pub struct LayerNorm(nn::LayerNorm);

impl nn::Module for LayerNorm {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        self.0
            .forward(&xs.to_dtype(tch::Kind::Float, false, false))
            .to_dtype(xs.kind(), false, false)
    }
}

#[derive(Debug)]
pub struct Linear(nn::Linear);

impl nn::Module for Linear {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        xs.linear(
            &self.0.ws.to_dtype(xs.kind(), false, false),
            self.0
                .bs
                .as_ref()
                .map(|bs| bs.to_dtype(xs.kind(), false, false)),
        )
    }
}

pub fn default_dtype() -> (Kind, Device) {
    (Kind::Float, Device::cuda_if_available())
}

/// Returns sinusoids for positional embedding
/// `max_timescale` defaults to 10000.
pub fn sinsoids(length: i64, channels: i64, max_timescale: Option<f32>) -> Tensor {
    debug_assert!(channels % 2 == 0, "number of channels must be even");
    let log_timescale_increment =
        f32::ln(max_timescale.unwrap_or(10000.0)) / (channels / 2 - 1) as f32;
    let inv_timescales =
        (-log_timescale_increment * Tensor::arange((channels / 2) as i64, default_dtype())).exp();
    let scaled_time = Tensor::arange(length as i64, default_dtype()).i((.., NewAxis))
        * inv_timescales.i((NewAxis, ..));
    Tensor::cat(&[scaled_time.sin(), scaled_time.cos()], 1)
}

#[derive(Debug)]
pub struct MultiHeadAttention {
    pub n_heads: i64,
    pub query: nn::Linear,
    pub key: nn::Linear,
    pub value: nn::Linear,
    pub out: nn::Linear,
}

impl MultiHeadAttention {
    pub fn new<'a>(vs: nn::Path<'a>, n_states: i64, n_heads: i64) -> Self {
        Self {
            n_heads,
            query: nn::linear(
                &vs / "query",
                n_states,
                n_states,
                nn::LinearConfig::default(),
            ),
            key: nn::linear(
                &vs / "key",
                n_states,
                n_states,
                nn::LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            value: nn::linear(
                &vs / "value",
                n_states,
                n_states,
                nn::LinearConfig::default(),
            ),
            out: nn::linear(&vs / "out", n_states, n_states, nn::LinearConfig::default()),
        }
    }

    pub fn forward_ext(&self, xs: &Tensor, xa: Option<&Tensor>, mask: Option<&Tensor>) -> Tensor {
        let q = self.query.forward(&xs);
        let k = self.key.forward(xa.unwrap_or(&xs));
        let v = self.value.forward(xa.unwrap_or(&xs));

        let (_n_batch, n_ctx, n_state) = q.size3().unwrap();
        let scale = f64::powf((n_state / self.n_heads) as f64, -0.25);

        let q_size = q.size3().unwrap();
        let k_size = k.size3().unwrap();
        let v_size = v.size3().unwrap();

        let q = q
            .view([q_size.0, q_size.1, self.n_heads, -1])
            .permute(&[0, 2, 1, 3])
            * scale;
        let k = k
            .view([k_size.0, k_size.1, self.n_heads, -1])
            .permute(&[0, 2, 3, 1])
            * scale;
        let v = q
            .view([v_size.0, v_size.1, self.n_heads, -1])
            .permute(&[0, 2, 1, 3]);

        let mut qk = q.matmul(&k);
        if let Some(mask) = mask {
            qk += mask.i((..n_ctx, ..n_ctx));
        }

        let w = qk.softmax(-1, q.kind());
        let wv = w.matmul(&v).permute(&[0, 2, 1, 3]).flatten(2, -1);

        self.out.forward(&wv)
    }
}

impl nn::Module for MultiHeadAttention {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward_ext(xs, None, None)
    }
}

#[derive(Debug)]
pub struct GELU;

impl nn::Module for GELU {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.gelu("none")
    }
}

#[derive(Debug)]
pub struct ResidualAttentionBlock {
    pub attn: MultiHeadAttention,
    pub attn_ln: nn::LayerNorm,
    pub cross_attn: Option<MultiHeadAttention>,
    pub cross_attn_ln: Option<nn::LayerNorm>,
    pub mlp: nn::Sequential,
    pub mlp_ln: nn::LayerNorm,
}

impl ResidualAttentionBlock {
    pub fn new<'a>(vs: nn::Path<'a>, n_states: i64, n_heads: i64, cross_attn: bool) -> Self {
        let n_mlps = n_states * 4;
        Self {
            attn: MultiHeadAttention::new(&vs / "attn", n_states, n_heads),
            attn_ln: nn::layer_norm(
                &vs / "attn_ln",
                vec![n_states],
                nn::LayerNormConfig::default(),
            ),
            cross_attn: if cross_attn {
                Some(MultiHeadAttention::new(
                    &vs / "cross_attn",
                    n_states,
                    n_heads,
                ))
            } else {
                None
            },
            cross_attn_ln: if cross_attn {
                Some(nn::layer_norm(
                    &vs / "cross_attn_ln",
                    vec![n_states],
                    nn::LayerNormConfig::default(),
                ))
            } else {
                None
            },
            mlp: nn::seq()
                .add(nn::linear(
                    &vs / "mlp" / 0,
                    n_states,
                    n_mlps,
                    nn::LinearConfig::default(),
                ))
                .add(GELU)
                .add(nn::linear(
                    &vs / "mlp" / 2,
                    n_mlps,
                    n_states,
                    nn::LinearConfig::default(),
                )),
            mlp_ln: nn::layer_norm(
                &vs / "mlp_ln",
                vec![n_states],
                nn::LayerNormConfig::default(),
            ),
        }
    }

    pub fn forward_ext(&self, xs: &Tensor, xa: Option<&Tensor>, mask: Option<&Tensor>) -> Tensor {
        let mut x = xs
            + self
                .attn
                .forward_ext(&self.attn_ln.forward(&xs), None, mask);

        if let (Some(cross_attn), Some(cross_attn_ln)) = (&self.cross_attn, &self.cross_attn_ln) {
            x += cross_attn.forward_ext(&cross_attn_ln.forward(&x), xa, None);
        }

        x += self.mlp.forward(&self.mlp_ln.forward(&x));
        x
    }
}

impl Module for ResidualAttentionBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward_ext(xs, None, None)
    }
}

#[derive(Debug)]
pub struct AudioEncoder {
    pub conv1: nn::Conv1D,
    pub conv2: nn::Conv1D,
    pub blocks: Vec<ResidualAttentionBlock>,
    pub position_emb: Tensor,
    pub ln_post: nn::LayerNorm,
}

impl AudioEncoder {
    pub fn new<'a>(
        vs: nn::Path<'a>,
        n_ctxs: i64,
        n_states: i64,
        n_heads: i64,
        n_layers: i64,
    ) -> Self {
        Self {
            conv1: nn::conv1d(
                &vs / "conv1",
                N_MELS,
                n_states,
                3,
                nn::ConvConfig {
                    padding: 1,
                    ..Default::default()
                },
            ),
            conv2: nn::conv1d(
                &vs / "conv2",
                n_states,
                n_states,
                3,
                nn::ConvConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
            ),
            blocks: (0..n_layers)
                .map(|i| ResidualAttentionBlock::new(&vs / "blocks" / i, n_states, n_heads, false))
                .collect(),
            position_emb: sinsoids(n_ctxs, n_states, None),
            ln_post: nn::layer_norm(&vs / "ln_post", vec![n_states], nn::LayerNormConfig::default()),
        }
    }
}

impl Module for AudioEncoder {
    /// xs: shape = (batch_size, n_mels, n_ctx)
    ///     the mel spectrogram of the audio
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = self.conv1.forward(xs).gelu("none");
        let xs = self.conv2.forward(&xs).gelu("none").permute(&[0, 2, 1]);

        debug_assert_eq!(
            &xs.size()[1..],
            &self.position_emb.size()[..],
            "incorrect audio shape"
        );
        let mut xs = xs + &self.position_emb;

        for block in &self.blocks {
            xs = block.forward(&xs);
        }

        self.ln_post.forward(&xs)
    }
}

#[derive(Debug)]
pub struct TextDecoder {
    pub token_emb: nn::Embedding,
    pub position_emb: Tensor,
    pub mask: Tensor,
    pub blocks: Vec<ResidualAttentionBlock>,
    pub ln_pre: nn::LayerNorm,
}

impl TextDecoder {
    pub fn new<'a>(
        vs: nn::Path<'a>,
        n_vocab: i64,
        n_ctxs: i64,
        n_states: i64,
        n_heads: i64,
        n_layers: i64,
        position_emb: Tensor,
    ) -> Self {
        Self {
            token_emb: nn::embedding(
                &vs / "token_embedding",
                n_vocab,
                n_states,
                nn::EmbeddingConfig::default(),
            ),
            position_emb,
            mask: Tensor::empty(&[n_ctxs, n_ctxs], default_dtype())
                .fill_(f64::NEG_INFINITY)
                .triu_(1),
            blocks: (0..n_layers)
                .map(|i| ResidualAttentionBlock::new(&vs / "blocks" / i, n_states, n_heads, true))
                .collect(),
            ln_pre: nn::layer_norm(&vs / "ln", vec![n_states], nn::LayerNormConfig::default()),
        }
    }

    /// x : shape = (batch_size, <= n_ctx)
    ///     the text tokens
    /// xa: shape = (batch_size, n_mels, n_audio_ctx)
    ///     the encoded audio features to be attended onxs: shape = (batch_size, n_mels, n_ctx)
    fn forward_ext(&self, xs: &Tensor, xa: &Tensor) -> Tensor {
        // offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        // x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        // x = x.to(xa.dtype)

        // for block in self.blocks:
        //     x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        // x = self.ln(x)
        // logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        // return logits

        let offset = 0;
        let mut x = self.token_emb.forward(xs)
            + self
                .position_emb
                .i(offset..offset + *xs.size().last().unwrap());

        for block in &self.blocks {
            x = block.forward_ext(&x, Some(xa), Some(&self.mask));
        }

        x = self.ln_pre.forward(&x);

        x.matmul(&self.token_emb.ws.transpose(0, 1))
    }
}

#[derive(Debug)]
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
}

impl Whisper {
    pub fn new<'a>(
        vs: nn::Path<'a>,
        n_vocab: i64,
        n_audio_ctxs: i64,
        n_audio_states: i64,
        n_audio_heads: i64,
        n_audio_layers: i64,
        n_text_ctxs: i64,
        n_text_states: i64,
        n_text_heads: i64,
        n_text_layers: i64,
        position_emb: Tensor,
    ) -> Self {
        Self {
            encoder: AudioEncoder::new(
                &vs / "encoder",
                n_audio_ctxs,
                n_audio_states,
                n_audio_heads,
                n_audio_layers,
            ),
            decoder: TextDecoder::new(
                &vs / "decoder",
                n_vocab,
                n_text_ctxs,
                n_text_states,
                n_text_heads,
                n_text_layers,
                position_emb,
            ),
        }
    }

    pub fn forward_ext(&self, mel: &Tensor, tokens: &Tensor) -> Tensor {
        self.decoder.forward_ext(tokens, &self.encoder.forward(mel))
    }
}
