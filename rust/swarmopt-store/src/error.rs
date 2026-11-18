use std::fmt;
use std::io;
use std::path::PathBuf;

/// Everything that can go wrong in the store.
///
/// Corruption found while *reading* is normally not an error: readers skip
/// damaged bytes and report what they skipped in their stats. `Corrupt` is
/// returned only where carrying on would be unsafe, such as a damaged
/// segment header on a file a writer is about to append to.
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    /// Bytes that don't match the expected format.
    Corrupt(String),
    /// A record or checkpoint larger than the configured limit.
    TooLarge {
        size: usize,
        limit: usize,
    },
    /// The storage budget is used up and the policy is to refuse new data.
    StorageFull {
        used: u64,
        budget: u64,
    },
    /// Another writer holds the directory lock.
    Locked(PathBuf),
    /// A path that is, or resolves through, a symlink where a real file or
    /// directory is required.
    UnsafePath(PathBuf),
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Corrupt(m) => write!(f, "corrupt data: {m}"),
            Error::TooLarge { size, limit } => {
                write!(f, "payload of {size} bytes exceeds the {limit}-byte limit")
            }
            Error::StorageFull { used, budget } => {
                write!(
                    f,
                    "storage budget exhausted ({used} of {budget} bytes used)"
                )
            }
            Error::Locked(p) => write!(f, "{} is locked by another writer", p.display()),
            Error::UnsafePath(p) => write!(f, "refusing to use symlinked path {}", p.display()),
            Error::InvalidConfig(m) => write!(f, "invalid configuration: {m}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}
