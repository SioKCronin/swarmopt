//! Two-slot (A/B) checkpoints that survive a power cut at any instant.
//!
//! A save always goes to the slot that does *not* hold the newest valid
//! checkpoint, through a temp file, fsync and atomic rename. So whatever
//! moment power is lost, the previous checkpoint is still intact on disk,
//! and a load falls back to it if the newest one is damaged.
//!
//! ```text
//! <dir>/CKPT.LOCK
//! <dir>/ckpt-a.swc
//! <dir>/ckpt-b.swc
//!
//! slot file (little-endian):
//!   magic "SWOCKP\0\x01" | version u16 | flags u16 | generation u64
//!   | len u32 | reserved u32 | header_crc u32 = crc32c(bytes 0..28)
//!   | payload [len] | body_crc u32 = crc32c(bytes 0..32 ++ payload)
//! ```
//!
//! Generations are a counter, not a timestamp, so ordering never depends on
//! a clock that may be wrong or reset.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::crc::{crc32c, Crc32c};
use crate::error::{Error, Result};
use crate::fsutil::{self, DirLock};

pub const CHECKPOINT_MAGIC: [u8; 8] = *b"SWOCKP\0\x01";
pub const CHECKPOINT_VERSION: u16 = 1;
const HEADER_LEN: usize = 32;
const SLOTS: [&str; 2] = ["ckpt-a.swc", "ckpt-b.swc"];
pub const HARD_MAX_CHECKPOINT: usize = 256 * 1024 * 1024;

fn encode(generation: u64, payload: &[u8]) -> Vec<u8> {
    let len = u32::try_from(payload.len()).expect("size checked by caller");
    let mut h = [0u8; HEADER_LEN];
    h[0..8].copy_from_slice(&CHECKPOINT_MAGIC);
    h[8..10].copy_from_slice(&CHECKPOINT_VERSION.to_le_bytes());
    h[12..20].copy_from_slice(&generation.to_le_bytes());
    h[20..24].copy_from_slice(&len.to_le_bytes());
    let hc = crc32c(&h[0..28]);
    h[28..32].copy_from_slice(&hc.to_le_bytes());
    let mut c = Crc32c::new();
    c.update(&h);
    c.update(payload);
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len() + 4);
    out.extend_from_slice(&h);
    out.extend_from_slice(payload);
    out.extend_from_slice(&c.finish().to_le_bytes());
    out
}

fn decode(buf: &[u8], max_bytes: usize) -> Option<(u64, Vec<u8>)> {
    if buf.len() < HEADER_LEN + 4 || buf[0..8] != CHECKPOINT_MAGIC {
        return None;
    }
    let hc = u32::from_le_bytes(buf[28..32].try_into().ok()?);
    if crc32c(&buf[0..28]) != hc {
        return None;
    }
    if u16::from_le_bytes(buf[8..10].try_into().ok()?) != CHECKPOINT_VERSION {
        return None;
    }
    let generation = u64::from_le_bytes(buf[12..20].try_into().ok()?);
    let len = u32::from_le_bytes(buf[20..24].try_into().ok()?) as usize;
    if len > max_bytes || buf.len() != HEADER_LEN + len + 4 {
        return None;
    }
    let payload = &buf[HEADER_LEN..HEADER_LEN + len];
    let bc = u32::from_le_bytes(buf[HEADER_LEN + len..].try_into().ok()?);
    let mut c = Crc32c::new();
    c.update(&buf[0..HEADER_LEN]);
    c.update(payload);
    if c.finish() != bc {
        return None;
    }
    Some((generation, payload.to_vec()))
}

/// A checkpoint read back from disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loaded {
    pub generation: u64,
    pub payload: Vec<u8>,
    /// The newest slot was damaged and this is the one before it.
    pub fell_back: bool,
    /// Slot files that exist but failed validation.
    pub damaged_slots: u32,
}

struct SlotState {
    valid: Option<(u64, Vec<u8>)>,
    exists: bool,
}

fn read_slot(path: &Path, max_bytes: usize) -> Result<SlotState> {
    if fs::symlink_metadata(path).is_err() {
        return Ok(SlotState {
            valid: None,
            exists: false,
        });
    }
    let mut f = fsutil::open_read(path)?;
    let len = f.metadata()?.len();
    if len > (HEADER_LEN + max_bytes + 4) as u64 {
        return Ok(SlotState {
            valid: None,
            exists: true,
        });
    }
    let mut buf = Vec::with_capacity(len as usize);
    f.read_to_end(&mut buf)?;
    Ok(SlotState {
        valid: decode(&buf, max_bytes),
        exists: true,
    })
}

fn read_slots(dir: &Path, max_bytes: usize) -> Result<[SlotState; 2]> {
    Ok([
        read_slot(&dir.join(SLOTS[0]), max_bytes)?,
        read_slot(&dir.join(SLOTS[1]), max_bytes)?,
    ])
}

/// Index of the slot with the newest valid checkpoint, if any.
fn newest(slots: &[SlotState; 2]) -> Option<usize> {
    match (&slots[0].valid, &slots[1].valid) {
        (Some((a, _)), Some((b, _))) => Some(if b > a { 1 } else { 0 }),
        (Some(_), None) => Some(0),
        (None, Some(_)) => Some(1),
        (None, None) => None,
    }
}

fn load_from(slots: [SlotState; 2]) -> Option<Loaded> {
    let damaged = slots
        .iter()
        .filter(|s| s.exists && s.valid.is_none())
        .count() as u32;
    let i = newest(&slots)?;
    let other = 1 - i;
    // The other slot is damaged and *might* have held something newer.
    let fell_back = slots[other].exists && slots[other].valid.is_none();
    let [a, b] = slots;
    let (generation, payload) = if i == 0 { a.valid? } else { b.valid? };
    Some(Loaded {
        generation,
        payload,
        fell_back,
        damaged_slots: damaged,
    })
}

/// Read the newest valid checkpoint in `dir` without locking or writing.
/// For ground tools and post-mortems.
pub fn read_checkpoint(dir: impl AsRef<Path>, max_bytes: usize) -> Result<Option<Loaded>> {
    let dir = dir.as_ref();
    fsutil::reject_symlink(dir)?;
    if !dir.exists() {
        return Ok(None);
    }
    Ok(load_from(read_slots(
        dir,
        max_bytes.min(HARD_MAX_CHECKPOINT),
    )?))
}

/// Exclusive handle on a checkpoint directory.
#[derive(Debug)]
pub struct CheckpointStore {
    dir: PathBuf,
    max_bytes: usize,
    _lock: DirLock,
}

impl CheckpointStore {
    pub fn open(dir: impl AsRef<Path>, max_bytes: usize) -> Result<Self> {
        if max_bytes == 0 || max_bytes > HARD_MAX_CHECKPOINT {
            return Err(Error::InvalidConfig(
                "max_bytes must be between 1 byte and 256 MiB".into(),
            ));
        }
        let dir = dir.as_ref().to_path_buf();
        fsutil::ensure_dir(&dir)?;
        let lock = DirLock::acquire(&dir, "CKPT.LOCK")?;
        // Clear temp files left by a save that was interrupted.
        for slot in SLOTS {
            let tmp = fsutil::tmp_path(&dir.join(slot));
            if fs::symlink_metadata(&tmp).is_ok() {
                fs::remove_file(&tmp)?;
            }
        }
        Ok(CheckpointStore {
            dir,
            max_bytes,
            _lock: lock,
        })
    }

    /// Save a new checkpoint and return its generation.
    pub fn save(&mut self, payload: &[u8]) -> Result<u64> {
        if payload.len() > self.max_bytes {
            return Err(Error::TooLarge {
                size: payload.len(),
                limit: self.max_bytes,
            });
        }
        let slots = read_slots(&self.dir, self.max_bytes)?;
        let (target, generation) = match newest(&slots) {
            Some(i) => {
                let g = slots[i].valid.as_ref().map(|(g, _)| *g).unwrap_or(0);
                (
                    1 - i,
                    g.checked_add(1)
                        .ok_or_else(|| Error::Corrupt("generation overflow".into()))?,
                )
            }
            None => (0, 1),
        };
        fsutil::atomic_write(&self.dir.join(SLOTS[target]), &encode(generation, payload))?;
        Ok(generation)
    }

    pub fn load(&self) -> Result<Option<Loaded>> {
        Ok(load_from(read_slots(&self.dir, self.max_bytes)?))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_every_bit_flip_detected() {
        let e = encode(5, b"swarm state");
        assert_eq!(decode(&e, 1024), Some((5, b"swarm state".to_vec())));
        for byte in 0..e.len() {
            for bit in 0..8 {
                let mut g = e.clone();
                g[byte] ^= 1 << bit;
                assert!(
                    decode(&g, 1024).is_none(),
                    "flip at {byte}:{bit} not caught"
                );
            }
        }
    }

    #[test]
    fn truncation_detected() {
        let e = encode(1, b"abcdef");
        for n in 0..e.len() {
            assert!(decode(&e[..n], 1024).is_none());
        }
    }
}
