use std::{collections::HashMap, ops::Deref};

use once_cell::sync::Lazy;
use tch::Tensor;

pub const LANGUAGES: Lazy<HashMap<&str, &str>> = Lazy::new(|| {
    HashMap::from([
        ("en", "english"),
        ("zh", "chinese"),
        ("de", "german"),
        ("es", "spanish"),
        ("ru", "russian"),
        ("ko", "korean"),
        ("fr", "french"),
        ("ja", "japanese"),
        ("pt", "portuguese"),
        ("tr", "turkish"),
        ("pl", "polish"),
        ("ca", "catalan"),
        ("nl", "dutch"),
        ("ar", "arabic"),
        ("sv", "swedish"),
        ("it", "italian"),
        ("id", "indonesian"),
        ("hi", "hindi"),
        ("fi", "finnish"),
        ("vi", "vietnamese"),
        ("iw", "hebrew"),
        ("uk", "ukrainian"),
        ("el", "greek"),
        ("ms", "malay"),
        ("cs", "czech"),
        ("ro", "romanian"),
        ("da", "danish"),
        ("hu", "hungarian"),
        ("ta", "tamil"),
        ("no", "norwegian"),
        ("th", "thai"),
        ("ur", "urdu"),
        ("hr", "croatian"),
        ("bg", "bulgarian"),
        ("lt", "lithuanian"),
        ("la", "latin"),
        ("mi", "maori"),
        ("ml", "malayalam"),
        ("cy", "welsh"),
        ("sk", "slovak"),
        ("te", "telugu"),
        ("fa", "persian"),
        ("lv", "latvian"),
        ("bn", "bengali"),
        ("sr", "serbian"),
        ("az", "azerbaijani"),
        ("sl", "slovenian"),
        ("kn", "kannada"),
        ("et", "estonian"),
        ("mk", "macedonian"),
        ("br", "breton"),
        ("eu", "basque"),
        ("is", "icelandic"),
        ("hy", "armenian"),
        ("ne", "nepali"),
        ("mn", "mongolian"),
        ("bs", "bosnian"),
        ("kk", "kazakh"),
        ("sq", "albanian"),
        ("sw", "swahili"),
        ("gl", "galician"),
        ("mr", "marathi"),
        ("pa", "punjabi"),
        ("si", "sinhala"),
        ("km", "khmer"),
        ("sn", "shona"),
        ("yo", "yoruba"),
        ("so", "somali"),
        ("af", "afrikaans"),
        ("oc", "occitan"),
        ("ka", "georgian"),
        ("be", "belarusian"),
        ("tg", "tajik"),
        ("sd", "sindhi"),
        ("gu", "gujarati"),
        ("am", "amharic"),
        ("yi", "yiddish"),
        ("lo", "lao"),
        ("uz", "uzbek"),
        ("fo", "faroese"),
        ("ht", "haitian creole"),
        ("ps", "pashto"),
        ("tk", "turkmen"),
        ("nn", "nynorsk"),
        ("mt", "maltese"),
        ("sa", "sanskrit"),
        ("lb", "luxembourgish"),
        ("my", "myanmar"),
        ("bo", "tibetan"),
        ("tl", "tagalog"),
        ("mg", "malagasy"),
        ("as", "assamese"),
        ("tt", "tatar"),
        ("haw", "hawaiian"),
        ("ln", "lingala"),
        ("ha", "hausa"),
        ("ba", "bashkir"),
        ("jw", "javanese"),
        ("su", "sundanese"),
    ])
});

/// A wrapper around HuggingFace tokenizer that allows fast access to common token IDs.
#[derive(Debug)]
pub struct Tokenizer {
    pub tokenizer: tokenizers::Tokenizer,
    /// "start of transcript" token
    pub token_id_sot: u32,
    /// "translate" control token, tells model to translate text
    pub token_id_translate: u32,
    /// "transcribe" control token, tells the model to transcribe text
    pub token_id_transcribe: u32,
    /// "end of transcript" token
    pub token_id_eot: u32,
    /// "no timestamps" control token, tells the model not to generate timestamps
    pub token_id_no_timestamps: u32,
    /// "no speech" token, used to indicate there is no speech here
    pub token_id_no_speech: u32,
    pub token_id_startofprev: u32,
    pub token_id_startoflm: u32,
    /// "timestamp begin" token, a timestamp should come after this
    pub token_id_ts_begin: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Task {
    LanguageId,
    Translate,
    Transcribe,
}

impl Tokenizer {
    /// Simple implementation of tokenizer, only works for english.
    pub fn new(task: Task) -> anyhow::Result<Self> {
        assert_eq!(
            task,
            Task::Transcribe,
            "multilingual tokenizer is not implemented yet so translation is not possible"
        );

        let mut tokenizer = tokenizers::Tokenizer::from_pretrained("gpt2", None).unwrap();
        let special_tokens: Vec<_> = ["<|startoftranscript|>"]
            .iter()
            .map(|s| (*s).to_owned())
            .chain(LANGUAGES.keys().map(|t| format!("<|{t}|>")))
            .chain(
                [
                    "<|translate|>",
                    "<|transcribe|>",
                    "<|startoflm|>",
                    "<|startofprev|>",
                    "<|nospeech|>",
                    "<|notimestamps|>",
                ]
                .map(|s| s.to_owned()),
            )
            .map(|t| tokenizers::AddedToken::from(t, true))
            .collect();

        tokenizer.add_special_tokens(&special_tokens[..]);

        let special_token_ids: Vec<_> = special_tokens
            .iter()
            .map(|tok| tokenizer.token_to_id(&tok.content).unwrap())
            .collect();

        Ok(Tokenizer {
            token_id_sot: tokenizer.token_to_id("<|startoftranscript|>").unwrap(),
            token_id_eot: tokenizer.token_to_id("<|endoftext|>").unwrap(),
            token_id_transcribe: tokenizer.token_to_id("<|transcribe|>").unwrap(),
            token_id_translate: tokenizer.token_to_id("<|translate|>").unwrap(),
            token_id_no_timestamps: tokenizer.token_to_id("<|notimestamps|>").unwrap(),
            token_id_no_speech: tokenizer.token_to_id("<|nospeech|>").unwrap(),
            token_id_startofprev: tokenizer.token_to_id("<|startofprev|>").unwrap(),
            token_id_startoflm: tokenizer.token_to_id("<|startoflm|>").unwrap(),
            // timestamp tokens are not real tokens, instead they are numbers
            // that are outside the range of the tokenizer 
            token_id_ts_begin: *special_token_ids.last().unwrap() + 1,
            tokenizer,
        })
    }

    pub fn sequence_sot(&self) -> Vec<u32> {
        // TODO: add sot sequence for translation
        // vec![self.token_id_sot, self.token_id_transcribe]
        vec![self.token_id_sot]
    }

    pub fn decode(&self, tokens: &Tensor) -> anyhow::Result<String> {
        let token_ids = tokens.iter::<i64>()?.map(|id| id as u32).collect();
        Ok(self.tokenizer.decode(token_ids, false).unwrap())
    }
}

impl Deref for Tokenizer {
    type Target = tokenizers::Tokenizer;

    fn deref(&self) -> &Self::Target {
        &self.tokenizer
    }
}
