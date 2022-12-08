use std::cell::Cell;

use tch::{
    nn::{self, Module},
    Device, IndexOp, Kind, NewAxis, Tensor,
};

use crate::{audio::N_MELS, util::tensor_dbg};

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

/// This implements the kv_cache function from whisper src
pub struct Cached<T: nn::Module> {
    inner: T,
    threshold: i64,
    cache: Cell<Option<Tensor>>,
}

impl<T: nn::Module> Cached<T> {
    pub fn new(inner: T, threshold: i64) -> Self {
        Self {
            inner,
            threshold,
            cache: Cell::new(None),
        }
    }

    /// Used during beam search to update the cache to match the selected beams.
    pub fn update_cache(&self, sources_indices: &Tensor) {
        self.cache.set(match self.cache.take() {
            Some(cache) => {
                // tensor_dbg!(cache);
                // tensor_dbg!(sources_indices);
                Some(cache.index(&[Some(sources_indices)]))
            }
            None => None,
        })
    }

    pub fn clear_cache(&self) {
        self.cache.set(None);
    }
}

impl<T: nn::Module> std::fmt::Debug for Cached<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cached")
            .field("inner", &self.inner)
            .finish()
    }
}

impl<T: nn::Module> nn::Module for Cached<T> {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        let output = self.inner.forward(xs);

        // tensor_dbg!(output);

        let output = match self.cache.take() {
            Some(cache) if xs.size()[1] <= self.threshold => {
                // tensor_dbg!(cache);
                Tensor::cat(&[cache, output], 1).detach()
            }

            // save as-is, for the first token or cross attention
            _ => output,
        };

        // tensor_dbg!(output);

        self.cache.set(Some(output.i(..)));
        output
    }
}

/// Returns sinusoids for positional embedding
/// `max_timescale` defaults to 10000.
pub fn sinusoids(length: i64, channels: i64, max_timescale: Option<f32>, device: Device) -> Tensor {
    debug_assert!(channels % 2 == 0, "number of channels must be even");
    let log_timescale_increment =
        f32::ln(max_timescale.unwrap_or(10000.0)) / (channels / 2 - 1) as f32;
    let inv_timescales = (-log_timescale_increment
        * Tensor::arange((channels / 2) as i64, (Kind::Float, device)))
    .exp();
    let scaled_time = Tensor::arange(length as i64, (Kind::Float, device)).i((.., NewAxis))
        * inv_timescales.i((NewAxis, ..));
    Tensor::cat(&[scaled_time.sin(), scaled_time.cos()], 1)
}

#[derive(Debug)]
pub struct MultiHeadAttention {
    pub n_heads: i64,
    pub query: nn::Linear,
    pub key: Cached<nn::Linear>,
    pub value: Cached<nn::Linear>,
    pub out: nn::Linear,
}

impl MultiHeadAttention {
    pub fn new<'a>(vs: nn::Path<'a>, n_states: i64, n_heads: i64, cache_threshold: i64) -> Self {
        Self {
            n_heads,
            query: nn::linear(
                &vs / "query",
                n_states,
                n_states,
                nn::LinearConfig::default(),
            ),
            key: Cached::new(
                nn::linear(
                    &vs / "key",
                    n_states,
                    n_states,
                    nn::LinearConfig {
                        bias: false,
                        ..Default::default()
                    },
                ),
                cache_threshold,
            ),
            value: Cached::new(
                nn::linear(
                    &vs / "value",
                    n_states,
                    n_states,
                    nn::LinearConfig::default(),
                ),
                cache_threshold,
            ),
            out: nn::linear(&vs / "out", n_states, n_states, nn::LinearConfig::default()),
        }
    }

    pub fn forward_ext(&self, xs: &Tensor, xa: Option<&Tensor>, mask: Option<&Tensor>) -> Tensor {
        let q = self.query.forward(&xs);
        let k = self.key.forward(xa.unwrap_or(&xs));
        let v = self.value.forward(xa.unwrap_or(&xs));

        // tensor_dbg!(xs);
        // tensor_dbg!(q);
        // tensor_dbg!(k);
        // tensor_dbg!(v);

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
        let v = v
            .view([v_size.0, v_size.1, self.n_heads, -1])
            .permute(&[0, 2, 1, 3]);

        // tensor_dbg!(q);
        // tensor_dbg!(k);
        // tensor_dbg!(v);

        let mut qk = q.matmul(&k);
        // tensor_dbg!(qk);

        if let Some(mask) = mask {
            qk += mask.i((..n_ctx, ..n_ctx));
        }

        // tensor_dbg!(qk);

        let w = qk.softmax(-1, q.kind());
        let wv = w.matmul(&v).permute(&[0, 2, 1, 3]).flatten(2, -1);

        // tensor_dbg!(wv);

        self.out.forward(&wv)
    }

    pub fn update_cache(&self, sources_indices: &Tensor) {
        self.value.update_cache(sources_indices);
        self.key.update_cache(sources_indices);
    }

    pub fn clear_cache(&self) {
        self.value.clear_cache();
        self.key.clear_cache();
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
    pub fn new<'a>(
        vs: nn::Path<'a>,
        n_states: i64,
        n_heads: i64,
        cross_attn: bool,
        cache_threshold: i64,
    ) -> Self {
        let n_mlps = n_states * 4;
        Self {
            attn: MultiHeadAttention::new(&vs / "attn", n_states, n_heads, cache_threshold),
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
                    cache_threshold,
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

        // tensor_dbg!(x);

        if let (Some(cross_attn), Some(cross_attn_ln)) = (&self.cross_attn, &self.cross_attn_ln) {
            x += cross_attn.forward_ext(&cross_attn_ln.forward(&x), xa, None);
        }

        // tensor_dbg!(x);

        x += self.mlp.forward(&self.mlp_ln.forward(&x));

        // tensor_dbg!(x);
        x
    }

    pub fn update_cache(&self, sources_indices: &Tensor) {
        self.attn.update_cache(sources_indices);

        if let Some(cross_attn) = &self.cross_attn {
            cross_attn.update_cache(sources_indices);
        }
    }

    pub fn clear_cache(&self) {
        self.attn.clear_cache();

        if let Some(cross_attn) = &self.cross_attn {
            cross_attn.clear_cache();
        }
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
        device: Device,
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
                .map(|i| {
                    ResidualAttentionBlock::new(
                        &vs / "blocks" / i,
                        n_states,
                        n_heads,
                        false,
                        n_ctxs,
                    )
                })
                .collect(),
            position_emb: sinusoids(n_ctxs, n_states, None, device),
            ln_post: nn::layer_norm(
                &vs / "ln_post",
                vec![n_states],
                nn::LayerNormConfig::default(),
            ),
        }
    }

    pub fn update_cache(&self, sources_indices: &Tensor) {
        for block in &self.blocks {
            block.update_cache(sources_indices);
        }
    }

    pub fn clear_cache(&self) {
        for block in &self.blocks {
            block.clear_cache();
        }
    }
}

impl Module for AudioEncoder {
    /// xs: shape = (batch_size, n_mels, n_ctx)
    ///     the mel spectrogram of the audio
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = self.conv1.forward(xs).gelu("none");
        let xs = self.conv2.forward(&xs).gelu("none").permute(&[0, 2, 1]);

        // // tensor_dbg!(xs);

        debug_assert_eq!(
            &xs.size()[1..],
            &self.position_emb.size()[..],
            "incorrect audio shape"
        );
        let mut xs = xs + &self.position_emb;

        for block in &self.blocks {
            // // tensor_dbg!(xs);
            xs = block.forward(&xs);
        }

        // // tensor_dbg!(xs);
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
        device: Device,
    ) -> Self {
        Self {
            token_emb: nn::embedding(
                &vs / "token_embedding",
                n_vocab,
                n_states,
                nn::EmbeddingConfig::default(),
            ),
            position_emb: vs.var(
                "positional_embedding",
                &[n_ctxs, n_states],
                nn::Init::Const(0.0),
            ),
            mask: Tensor::empty(&[n_ctxs, n_ctxs], (Kind::Float, device))
                .fill_(f64::NEG_INFINITY)
                .triu_(1),
            blocks: (0..n_layers)
                .map(|i| {
                    ResidualAttentionBlock::new(&vs / "blocks" / i, n_states, n_heads, true, n_ctxs)
                })
                .collect(),
            ln_pre: nn::layer_norm(&vs / "ln", vec![n_states], nn::LayerNormConfig::default()),
        }
    }

    /// x : shape = (batch_size, <= n_ctx)
    ///     the text tokens
    /// xa: shape = (batch_size, n_mels, n_audio_ctx)
    ///     the encoded audio features to be attended onxs: shape = (batch_size, n_mels, n_ctx)
    pub fn forward_ext(&self, xs: &Tensor, xa: &Tensor, offset: i64) -> Tensor {
        // offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        // x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        // x = x.to(xa.dtype)

        // for block in self.blocks:
        //     x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        // x = self.ln(x)
        // logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()

        // return logits

        let mut x = self.token_emb.forward(xs)
            + self
                .position_emb
                .i(offset..offset + *xs.size().last().unwrap());

        // tensor_dbg!(self.position_emb);
        // tensor_dbg!(x);
        // tensor_dbg!(xa);

        for block in &self.blocks {
            // tensor_dbg!(x);
            x = block.forward_ext(&x, Some(xa), Some(&self.mask));
        }

        // tensor_dbg!(x);
        x = self.ln_pre.forward(&x);

        // tensor_dbg!(x);
        x.matmul(&self.token_emb.ws.transpose(0, 1))
    }

    pub fn update_cache(&self, sources_indices: &Tensor) {
        for block in &self.blocks {
            block.update_cache(sources_indices);
        }
    }

    pub fn clear_cache(&self) {
        for block in &self.blocks {
            block.clear_cache();
        }
    }
}

#[derive(Debug)]
pub struct Whisper {
    device: Device,

    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    pub dims: ModelDims,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ModelDims {
    pub n_vocab: i64,
    pub n_audio_ctxs: i64,
    pub n_audio_states: i64,
    pub n_audio_heads: i64,
    pub n_audio_layers: i64,
    pub n_text_ctxs: i64,
    pub n_text_states: i64,
    pub n_text_heads: i64,
    pub n_text_layers: i64,
}

impl Whisper {
    pub fn new<'a>(vs: nn::Path<'a>, dims: ModelDims) -> Self {
        let device = vs.device();

        Self {
            encoder: AudioEncoder::new(
                &vs / "encoder",
                dims.n_audio_ctxs,
                dims.n_audio_states,
                dims.n_audio_heads,
                dims.n_audio_layers,
                device,
            ),
            decoder: TextDecoder::new(
                &vs / "decoder",
                dims.n_vocab,
                dims.n_text_ctxs,
                dims.n_text_states,
                dims.n_text_heads,
                dims.n_text_layers,
                device,
            ),
            device,
            dims,
        }
    }

    pub fn forward_ext(&self, mel: &Tensor, tokens: &Tensor) -> Tensor {
        self.decoder
            .forward_ext(tokens, &self.encoder.forward(mel), 0)
    }

    pub fn update_cache(&self, sources_indices: &Tensor) {
        self.decoder.update_cache(sources_indices);
        self.encoder.update_cache(sources_indices);
    }

    pub fn clear_cache(&self) {
        self.decoder.clear_cache();
        self.encoder.clear_cache();
    }

    pub fn device(&self) -> Device {
        self.device
    }
}
