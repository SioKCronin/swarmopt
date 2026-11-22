//! Filesystem helpers: durable renames, owner-only permissions, symlink
//! refusal, and an exclusive writer lock.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};

/// Fail if `path` exists and is a symlink.
pub(crate) fn reject_symlink(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(m) if m.file_type().is_symlink() => Err(Error::UnsafePath(path.to_path_buf())),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Create `dir` (owner-only on Unix) if needed, and make sure it is a real
/// directory rather than a symlink to somewhere else.
pub(crate) fn ensure_dir(dir: &Path) -> Result<()> {
    reject_symlink(dir)?;
    if !dir.exists() {
        let mut b = fs::DirBuilder::new();
        b.recursive(true);
        #[cfg(unix)]
        b.mode(0o700);
        b.create(dir)?;
        if let Some(parent) = dir.parent() {
            if !parent.as_os_str().is_empty() {
                sync_dir(parent)?;
            }
        }
    }
    if !fs::symlink_metadata(dir)?.is_dir() {
        return Err(Error::InvalidConfig(format!(
            "{} is not a directory",
            dir.display()
        )));
    }
    Ok(())
}

fn base_options() -> OpenOptions {
    let mut o = OpenOptions::new();
    #[cfg(unix)]
    o.mode(0o600);
    o
}

/// Open a file, refusing symlinks, and check that the file we got is the
/// one at `path` (so a symlink swapped in between check and open is caught).
pub(crate) fn open_checked(path: &Path, opts: &OpenOptions) -> Result<File> {
    reject_symlink(path)?;
    let f = opts.open(path)?;
    #[cfg(unix)]
    {
        let opened = f.metadata()?;
        let on_disk = fs::symlink_metadata(path)?;
        if on_disk.file_type().is_symlink()
            || opened.dev() != on_disk.dev()
            || opened.ino() != on_disk.ino()
        {
            return Err(Error::UnsafePath(path.to_path_buf()));
        }
    }
    Ok(f)
}

pub(crate) fn open_read(path: &Path) -> Result<File> {
    let mut o = OpenOptions::new();
    o.read(true);
    open_checked(path, &o)
}

pub(crate) fn open_append_rw(path: &Path) -> Result<File> {
    let mut o = base_options();
    o.read(true).write(true);
    open_checked(path, &o)
}

/// Create a new file; fails if it already exists.
pub(crate) fn create_new(path: &Path) -> Result<File> {
    reject_symlink(path)?;
    let mut o = base_options();
    o.read(true).write(true).create_new(true);
    Ok(o.open(path)?)
}

/// fsync a directory so that file creations, renames and deletions in it
/// survive a power cut. A no-op where directories can't be opened.
pub(crate) fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(dir)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Write `data` to `path` atomically: temp file, fsync, rename, fsync dir.
pub(crate) fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
    use std::io::Write;
    reject_symlink(path)?;
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let tmp = tmp_path(path);
    // A leftover temp file from a crash mid-write is garbage; replace it.
    if fs::symlink_metadata(&tmp).is_ok() {
        fs::remove_file(&tmp)?;
    }
    let result = (|| -> Result<()> {
        let mut f = create_new(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)?;
        sync_dir(dir)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

pub(crate) fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .map(|n| n.to_os_string())
        .unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
}

/// An exclusive advisory lock on `<dir>/<name>`, held until dropped. Stops two
/// processes on the same vehicle from interleaving writes to one store.
#[derive(Debug)]
pub(crate) struct DirLock {
    _file: File,
}

impl DirLock {
    pub(crate) fn acquire(dir: &Path, name: &str) -> Result<DirLock> {
        let path = dir.join(name);
        let mut o = base_options();
        o.read(true).write(true).create(true).truncate(false);
        let f = open_checked(&path, &o)?;
        match f.try_lock() {
            Ok(()) => Ok(DirLock { _file: f }),
            Err(fs::TryLockError::WouldBlock) => Err(Error::Locked(dir.to_path_buf())),
            Err(fs::TryLockError::Error(e)) => Err(e.into()),
        }
    }
}
