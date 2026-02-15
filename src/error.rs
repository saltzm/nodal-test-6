/// Unified error type for the database.
use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// SQL parsing error.
    Parse(String),
    /// Schema / catalog error (e.g. table not found, duplicate table).
    Catalog(String),
    /// Type mismatch or constraint violation.
    Type(String),
    /// Storage / I/O error.
    Storage(String),
    /// Transaction error (e.g. conflict, not in transaction).
    Transaction(String),
    /// Query execution error.
    Execution(String),
    /// Generic internal error.
    Internal(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse(msg) => write!(f, "Parse error: {}", msg),
            Error::Catalog(msg) => write!(f, "Catalog error: {}", msg),
            Error::Type(msg) => write!(f, "Type error: {}", msg),
            Error::Storage(msg) => write!(f, "Storage error: {}", msg),
            Error::Transaction(msg) => write!(f, "Transaction error: {}", msg),
            Error::Execution(msg) => write!(f, "Execution error: {}", msg),
            Error::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Storage(e.to_string())
    }
}

/// Convenience type alias for Results using our Error type.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = Error::Parse("unexpected token".into());
        assert_eq!(format!("{}", e), "Parse error: unexpected token");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Storage(_)));
        assert!(format!("{}", err).contains("file not found"));
    }
}
