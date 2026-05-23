#[derive(Debug, thiserror::Error)]
pub enum BpeError {
    #[error("invalid Python bytes literal: {0}")]
    InvalidByteLiteral(String),
    #[error("unsupported vocabulary file format: {0}")]
    UnsupportedVocabFormat(String),
    #[error("unsupported merges file format: {0}")]
    UnsupportedMergesFormat(String),
}
