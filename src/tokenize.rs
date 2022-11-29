use std::collections::HashMap;

use once_cell::sync::Lazy;
use tokenizers::Tokenizer;

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

/// Simple implementation of tokenizer, only works for english.
pub fn get_tokenizer() -> anyhow::Result<Tokenizer> {
    let mut tok = tokenizers::Tokenizer::from_pretrained("gpt2", None).unwrap();
    let special_tokens: Vec<_> = IntoIterator::into_iter([
        "<|startoftranscript|>",
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ])
    .map(|s| s.to_owned())
    .chain(LANGUAGES.keys().map(|t| format!("<|{t}|>")))
    .map(|t| tokenizers::AddedToken::from(t, true))
    .collect();

    // let bpe_builder = tokenizers::models::bpe::BPE::from_file("gpt2/vocab.json", "gpt2/merges.txt");
    // let bpe = bpe_builder.unk_token("<|endoftext|>".to_owned()).build().unwrap();
    // let mut tok = Tokenizer::new(bpe);

    tok.add_special_tokens(&special_tokens[..]);

    Ok(tok)
}
