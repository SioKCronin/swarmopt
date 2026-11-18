//! Append-only record log built to survive power loss and bit flips.
//!
//! # Layout
//!
//! ```text
//! <root>/LOG.LOCK                 exclusive writer lock
//! <root>/copy-0/seg-0000000001.swl
//! <root>/copy-0/seg-0000000002.swl
//! <root>/copy-1/...               independent replica (same records)
//! ```
//!
//! Each replica is a complete log in its own directory, so one bad flash
//! block or one corrupted file costs at most one copy of the records in it.
//! Readers merge the copies by sequence number.
//!
//! # Segment format (little-endian)
//!
//! ```text
//! header (36 bytes):
//!   magic "SWOLOG\0\x01" | version u16 | flags u16 | segment_index u32
//!   | first_seq u64 | stream_id u64 | crc32c(previous 32 bytes) u32
//! records, back to back:
//!   sync [D3 5A 7E 91] | len u32 | seq u64 | kind u16 | flags u16
//!   | header_crc u32 = crc32c(len..flags)
//!   | payload [len]
//!   | body_crc u32 = crc32c(len..flags ++ payload)
//! ```
//!
//! The header CRC lets a reader reject a damaged length *before* trusting
//! it, so a flipped bit can't make it allocate or skip gigabytes. The sync
//! marker lets it find the next intact record after a damaged one instead
//! of losing the rest of the file.
//!
//! Records carry a sequence number, never a wall-clock time: an under-ice
//! vehicle may have no trustworthy clock, and ordering must not depend on
//! one. Callers can put their own timestamps in the payload.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::crc::{crc32c, Crc32c};
use crate::error::{Error, Result};
use crate::fsutil::{self, DirLock};

pub const SEGMENT_MAGIC: [u8; 8] = *b"SWOLOG\0\x01";
pub const FORMAT_VERSION: u16 = 1;
pub const SEGMENT_HEADER_LEN: usize = 36;
pub const RECORD_SYNC: [u8; 4] = [0xD3, 0x5A, 0x7E, 0x91];
/// sync + len + seq + kind + flags + header_crc = 24 bytes
pub const RECORD_HEADER_LEN: usize = 4 + 4 + 8 + 2 + 2 + 4;
/// Bytes a record costs on top of its payload.
pub const RECORD_OVERHEAD: usize = RECORD_HEADER_LEN + 4;
/// No configuration may allow records bigger than this.
pub const HARD_MAX_RECORD: usize = 64 * 1024 * 1024;
const MAX_REPLICAS: usize = 8;

/// When appended records are forced to stable storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyncPolicy {
    /// fsync after every record. Slowest, loses nothing that `append` acknowledged.
    Always,
    /// fsync after every N records; a power cut can lose up to N-1 records.
    EveryN(u32),
    /// Only when `sync()` is called (or on rotation and close).
    Manual,
}

/// What to do when the storage budget would be exceeded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FullPolicy {
    /// Refuse the write with `Error::StorageFull`. Nothing is ever deleted
    /// except by an explicit `prune_through` (e.g. after the data was
    /// confirmed received on the surface).
    Reject,
    /// Delete the oldest segments to make room. Use only for data that is
    /// worthless once stale (housekeeping telemetry, not science).
    DropOldest,
}

#[derive(Clone, Debug)]
pub struct LogConfig {
    /// Identifies the mission or vehicle. Opening a log written under a
    /// different id fails, so two missions' data can't be mixed.
    pub stream_id: u64,
    /// Independent copies of every record (1..=8).
    pub replicas: usize,
    /// Start a new segment file once the current one reaches this size.
    pub segment_bytes: u64,
    /// Total bytes allowed *per replica*, across all its segments.
    pub max_total_bytes: u64,
    pub max_record_bytes: usize,
    pub sync: SyncPolicy,
    pub on_full: FullPolicy,
}

impl Default for LogConfig {
    fn default() -> Self {
        LogConfig {
            stream_id: 0,
            replicas: 2,
            segment_bytes: 4 * 1024 * 1024,
            max_total_bytes: 256 * 1024 * 1024,
            max_record_bytes: 1024 * 1024,
            sync: SyncPolicy::Always,
            on_full: FullPolicy::Reject,
        }
    }
}

impl LogConfig {
    fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::InvalidConfig(m.to_string()));
        if self.replicas == 0 || self.replicas > MAX_REPLICAS {
            return bad("replicas must be between 1 and 8");
        }
        if self.max_record_bytes == 0 || self.max_record_bytes > HARD_MAX_RECORD {
            return bad("max_record_bytes must be between 1 byte and 64 MiB");
        }
        let min_segment = (SEGMENT_HEADER_LEN + RECORD_OVERHEAD + self.max_record_bytes) as u64;
        if self.segment_bytes < min_segment {
            return bad("segment_bytes must hold a header and one maximum-size record");
        }
        if self.max_total_bytes < self.segment_bytes {
            return bad("max_total_bytes must be at least segment_bytes");
        }
        if let SyncPolicy::EveryN(0) = self.sync {
            return bad("sync every 0 records is meaningless");
        }
        Ok(())
    }
}

/// A record read back from the log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub seq: u64,
    pub kind: u16,
    pub payload: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Encoding and scanning
// ---------------------------------------------------------------------------

fn encode_segment_header(index: u32, first_seq: u64, stream_id: u64) -> [u8; SEGMENT_HEADER_LEN] {
    let mut h = [0u8; SEGMENT_HEADER_LEN];
    h[0..8].copy_from_slice(&SEGMENT_MAGIC);
    h[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    h[10..12].copy_from_slice(&0u16.to_le_bytes());
    h[12..16].copy_from_slice(&index.to_le_bytes());
    h[16..24].copy_from_slice(&first_seq.to_le_bytes());
    h[24..32].copy_from_slice(&stream_id.to_le_bytes());
    let crc = crc32c(&h[0..32]);
    h[32..36].copy_from_slice(&crc.to_le_bytes());
    h
}

#[derive(Clone, Copy, Debug)]
struct SegmentHeader {
    index: u32,
    first_seq: u64,
    stream_id: u64,
}

fn decode_segment_header(buf: &[u8]) -> Option<SegmentHeader> {
    if buf.len() < SEGMENT_HEADER_LEN || buf[0..8] != SEGMENT_MAGIC {
        return None;
    }
    let crc = u32::from_le_bytes(buf[32..36].try_into().ok()?);
    if crc32c(&buf[0..32]) != crc {
        return None;
    }
    if u16::from_le_bytes(buf[8..10].try_into().ok()?) != FORMAT_VERSION {
        return None;
    }
    Some(SegmentHeader {
        index: u32::from_le_bytes(buf[12..16].try_into().ok()?),
        first_seq: u64::from_le_bytes(buf[16..24].try_into().ok()?),
        stream_id: u64::from_le_bytes(buf[24..32].try_into().ok()?),
    })
}

/// Encode one record frame.
pub fn encode_record(seq: u64, kind: u16, payload: &[u8]) -> Vec<u8> {
    let len = u32::try_from(payload.len()).expect("payload length checked by caller");
    let mut fields = [0u8; 16];
    fields[0..4].copy_from_slice(&len.to_le_bytes());
    fields[4..12].copy_from_slice(&seq.to_le_bytes());
    fields[12..14].copy_from_slice(&kind.to_le_bytes());
    fields[14..16].copy_from_slice(&0u16.to_le_bytes());
    let header_crc = crc32c(&fields);
    let mut body = Crc32c::new();
    body.update(&fields);
    body.update(payload);

    let mut out = Vec::with_capacity(RECORD_OVERHEAD + payload.len());
    out.extend_from_slice(&RECORD_SYNC);
    out.extend_from_slice(&fields);
    out.extend_from_slice(&header_crc.to_le_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&body.finish().to_le_bytes());
    out
}

/// Try to decode a record frame at the start of `buf`.
/// Returns the record and the frame length, or `None` if it isn't intact.
fn decode_record_at(buf: &[u8], max_record: usize) -> Option<(Record, usize)> {
    if buf.len() < RECORD_OVERHEAD || buf[0..4] != RECORD_SYNC {
        return None;
    }
    let fields = &buf[4..20];
    let header_crc = u32::from_le_bytes(buf[20..24].try_into().ok()?);
    if crc32c(fields) != header_crc {
        return None;
    }
    let len = u32::from_le_bytes(fields[0..4].try_into().ok()?) as usize;
    if len > max_record {
        return None;
    }
    let frame = RECORD_OVERHEAD.checked_add(len)?;
    if buf.len() < frame {
        return None;
    }
    let payload = &buf[RECORD_HEADER_LEN..RECORD_HEADER_LEN + len];
    let body_crc = u32::from_le_bytes(buf[RECORD_HEADER_LEN + len..frame].try_into().ok()?);
    let mut c = Crc32c::new();
    c.update(fields);
    c.update(payload);
    if c.finish() != body_crc {
        return None;
    }
    let seq = u64::from_le_bytes(fields[4..12].try_into().ok()?);
    let kind = u16::from_le_bytes(fields[12..14].try_into().ok()?);
    Some((
        Record {
            seq,
            kind,
            payload: payload.to_vec(),
        },
        frame,
    ))
}

struct Scan {
    records: Vec<Record>,
    /// Offset just past the last intact record (or `start` if none).
    end_of_valid: usize,
    /// Bytes that weren't part of any intact record.
    skipped_bytes: u64,
    /// Separate runs of damaged bytes.
    damaged_regions: u64,
}

/// Read every intact record in `buf[start..]`, resynchronising on the sync
/// marker after damage. Never panics, whatever `buf` contains.
fn scan_records(buf: &[u8], start: usize, max_record: usize) -> Scan {
    let mut records = Vec::new();
    let mut pos = start.min(buf.len());
    let mut end_of_valid = pos;
    let mut skipped = 0u64;
    let mut regions = 0u64;
    let mut in_damage = false;
    while pos < buf.len() {
        if let Some((rec, frame)) = decode_record_at(&buf[pos..], max_record) {
            records.push(rec);
            pos += frame;
            end_of_valid = pos;
            in_damage = false;
            continue;
        }
        if !in_damage {
            regions += 1;
            in_damage = true;
        }
        // Jump to the next candidate sync marker.
        let next = buf[pos + 1..]
            .windows(RECORD_SYNC.len())
            .position(|w| w == RECORD_SYNC)
            .map(|i| pos + 1 + i)
            .unwrap_or(buf.len());
        skipped += (next - pos) as u64;
        pos = next;
    }
    Scan {
        records,
        end_of_valid,
        skipped_bytes: skipped,
        damaged_regions: regions,
    }
}

fn segment_name(index: u32) -> String {
    format!("seg-{index:010}.swl")
}

fn parse_segment_name(name: &str) -> Option<u32> {
    name.strip_prefix("seg-")?
        .strip_suffix(".swl")?
        .parse()
        .ok()
}

fn copy_dir(root: &Path, i: usize) -> PathBuf {
    root.join(format!("copy-{i}"))
}

fn existing_copies(root: &Path) -> Result<Vec<(usize, PathBuf)>> {
    let mut out = Vec::new();
    if !root.exists() {
        return Ok(out);
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(i) = name
            .strip_prefix("copy-")
            .and_then(|s| s.parse::<usize>().ok())
        else {
            continue;
        };
        if entry.file_type()?.is_dir() {
            out.push((i, entry.path()));
        }
    }
    out.sort();
    Ok(out)
}

fn list_segments(dir: &Path) -> Result<Vec<(u32, PathBuf)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue; // symlinks and directories are never segments
        }
        if let Some(idx) = entry.file_name().to_str().and_then(parse_segment_name) {
            out.push((idx, entry.path()));
        }
    }
    out.sort();
    Ok(out)
}

fn read_capped(path: &Path, cap: u64) -> Result<Vec<u8>> {
    let mut f = fsutil::open_read(path)?;
    let len = f.metadata()?.len();
    if len > cap {
        return Err(Error::Corrupt(format!(
            "{} is {len} bytes, larger than any segment this log can write",
            path.display()
        )));
    }
    let mut buf = Vec::with_capacity(len as usize);
    f.read_to_end(&mut buf)?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Segment {
    index: u32,
    /// `None` if the header is damaged; such segments are kept for readers
    /// but never pruned on the strength of a sequence number.
    first_seq: Option<u64>,
    path: PathBuf,
    len: u64,
}

/// What opening the log found and repaired.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    /// Torn bytes cut from the end of an active segment (interrupted write).
    pub truncated_bytes: u64,
    /// Segments with damaged headers that were set aside as `*.quarantine`.
    pub quarantined_segments: u64,
    /// Empty segment files left by a crash during creation, removed.
    pub removed_empty_segments: u64,
}

#[derive(Debug)]
struct ReplicaWriter {
    dir: PathBuf,
    segments: Vec<Segment>,
    active: File,
    total_bytes: u64,
    unsynced: u32,
}

impl ReplicaWriter {
    fn open(dir: PathBuf, cfg: &LogConfig, report: &mut RecoveryReport) -> Result<(Self, u64)> {
        fsutil::ensure_dir(&dir)?;
        let seg_cap = cfg
            .segment_bytes
            .saturating_add(cfg.max_record_bytes as u64 + RECORD_OVERHEAD as u64);
        let mut segments = Vec::new();
        let mut max_seq = 0u64;
        // Set when the file we would have appended to was removed or set aside.
        let mut tail_dropped = false;
        let listed = list_segments(&dir)?;
        let n = listed.len();
        for (pos, (index, path)) in listed.into_iter().enumerate() {
            let is_last = pos + 1 == n;
            let buf = read_capped(&path, seg_cap)?;
            match decode_segment_header(&buf) {
                Some(h) => {
                    if h.stream_id != cfg.stream_id {
                        return Err(Error::InvalidConfig(format!(
                            "{} belongs to stream {:#x}, not {:#x}",
                            path.display(),
                            h.stream_id,
                            cfg.stream_id
                        )));
                    }
                    if h.index != index {
                        return Err(Error::Corrupt(format!(
                            "{} has header index {}, expected {index}",
                            path.display(),
                            h.index
                        )));
                    }
                    let scan = scan_records(&buf, SEGMENT_HEADER_LEN, cfg.max_record_bytes);
                    if let Some(m) = scan.records.iter().map(|r| r.seq).max() {
                        max_seq = max_seq.max(m);
                    }
                    max_seq = max_seq.max(h.first_seq.saturating_sub(1));
                    let mut len = buf.len() as u64;
                    if is_last && (scan.end_of_valid as u64) < len {
                        // Everything after the last intact record in the
                        // segment we were appending to is a torn write.
                        let f = fsutil::open_append_rw(&path)?;
                        f.set_len(scan.end_of_valid as u64)?;
                        f.sync_all()?;
                        report.truncated_bytes += len - scan.end_of_valid as u64;
                        len = scan.end_of_valid as u64;
                    }
                    segments.push(Segment {
                        index,
                        first_seq: Some(h.first_seq),
                        path,
                        len,
                    });
                }
                None if is_last
                    && (buf.len() < SEGMENT_HEADER_LEN || buf.iter().all(|&b| b == 0)) =>
                {
                    // Crashed between creating the file and finishing its
                    // header: there can't be any records in it.
                    fs::remove_file(&path)?;
                    fsutil::sync_dir(&dir)?;
                    report.removed_empty_segments += 1;
                    tail_dropped = true;
                }
                None => {
                    // Still learn the highest sequence number in it, so new
                    // records never reuse a number that exists on disk.
                    let scan = scan_records(&buf, 0, cfg.max_record_bytes);
                    if let Some(m) = scan.records.iter().map(|r| r.seq).max() {
                        max_seq = max_seq.max(m);
                    }
                    let mut q = path.clone().into_os_string();
                    q.push(".quarantine");
                    fs::rename(&path, &q)?;
                    fsutil::sync_dir(&dir)?;
                    report.quarantined_segments += 1;
                    tail_dropped |= is_last;
                }
            }
        }
        let total_bytes = segments.iter().map(|s| s.len).sum();
        let w = match segments.last() {
            // Normal restart: keep appending to the segment we were writing.
            Some(last) if !tail_dropped => {
                let mut f = fsutil::open_append_rw(&last.path)?;
                f.seek(SeekFrom::End(0))?;
                ReplicaWriter {
                    dir,
                    segments,
                    active: f,
                    total_bytes,
                    unsynced: 0,
                }
            }
            // After quarantining or removing the tail, start a fresh segment
            // rather than appending to an older one that was already closed.
            Some(last) => {
                let index = last.index.max(Self::highest_index_on_disk(&dir)?) + 1;
                let (seg, f) = Self::create_segment(&dir, index, max_seq + 1, cfg.stream_id)?;
                let mut segments = segments;
                let total = total_bytes + seg.len;
                segments.push(seg);
                ReplicaWriter {
                    dir,
                    segments,
                    active: f,
                    total_bytes: total,
                    unsynced: 0,
                }
            }
            None => {
                let index = Self::highest_index_on_disk(&dir)? + 1;
                let (seg, f) = Self::create_segment(&dir, index, max_seq + 1, cfg.stream_id)?;
                let total = seg.len;
                ReplicaWriter {
                    dir,
                    segments: vec![seg],
                    active: f,
                    total_bytes: total,
                    unsynced: 0,
                }
            }
        };
        Ok((w, max_seq))
    }

    /// Highest segment index used by any file, quarantined ones included,
    /// so a new segment never takes the name of one set aside.
    fn highest_index_on_disk(dir: &Path) -> Result<u32> {
        let mut hi = 0;
        for entry in fs::read_dir(dir)? {
            let name = entry?.file_name();
            let Some(name) = name.to_str() else { continue };
            let base = name.strip_suffix(".quarantine").unwrap_or(name);
            if let Some(i) = parse_segment_name(base) {
                hi = hi.max(i);
            }
        }
        Ok(hi)
    }

    fn create_segment(
        dir: &Path,
        index: u32,
        first_seq: u64,
        stream_id: u64,
    ) -> Result<(Segment, File)> {
        let path = dir.join(segment_name(index));
        let mut f = fsutil::create_new(&path)?;
        f.write_all(&encode_segment_header(index, first_seq, stream_id))?;
        f.sync_all()?;
        fsutil::sync_dir(dir)?;
        Ok((
            Segment {
                index,
                first_seq: Some(first_seq),
                path,
                len: SEGMENT_HEADER_LEN as u64,
            },
            f,
        ))
    }

    fn active_len(&self) -> u64 {
        self.segments.last().map(|s| s.len).unwrap_or(0)
    }

    fn rotate(&mut self, first_seq: u64, cfg: &LogConfig) -> Result<()> {
        self.active.sync_all()?;
        let index = self
            .segments
            .last()
            .map(|s| {
                s.index
                    .checked_add(1)
                    .ok_or_else(|| Error::Corrupt("segment index overflow".into()))
            })
            .transpose()?
            .unwrap_or(1);
        let (seg, f) = Self::create_segment(&self.dir, index, first_seq, cfg.stream_id)?;
        self.total_bytes += seg.len;
        self.segments.push(seg);
        self.active = f;
        self.unsynced = 0;
        Ok(())
    }

    fn make_room(&mut self, need: u64, cfg: &LogConfig) -> Result<()> {
        while self.total_bytes + need > cfg.max_total_bytes {
            if cfg.on_full == FullPolicy::Reject || self.segments.len() <= 1 {
                return Err(Error::StorageFull {
                    used: self.total_bytes,
                    budget: cfg.max_total_bytes,
                });
            }
            let oldest = self.segments.remove(0);
            fs::remove_file(&oldest.path)?;
            fsutil::sync_dir(&self.dir)?;
            self.total_bytes -= oldest.len;
        }
        Ok(())
    }

    fn append(&mut self, seq: u64, frame: &[u8], cfg: &LogConfig) -> Result<()> {
        let flen = frame.len() as u64;
        let has_records = self.active_len() > SEGMENT_HEADER_LEN as u64;
        let rotating = has_records && self.active_len() + flen > cfg.segment_bytes;
        let need = flen
            + if rotating {
                SEGMENT_HEADER_LEN as u64
            } else {
                0
            };
        self.make_room(need, cfg)?;
        if rotating {
            self.rotate(seq, cfg)?;
        }
        let before = self.active_len();
        if let Err(e) = self.active.write_all(frame) {
            // Don't leave half a frame behind for the next append to follow.
            let _ = self.active.set_len(before);
            let _ = self.active.seek(SeekFrom::Start(before));
            return Err(e.into());
        }
        if let Some(s) = self.segments.last_mut() {
            s.len += flen;
        }
        self.total_bytes += flen;
        self.unsynced += 1;
        match cfg.sync {
            SyncPolicy::Always => self.sync()?,
            SyncPolicy::EveryN(n) if self.unsynced >= n => self.sync()?,
            _ => {}
        }
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        if self.unsynced > 0 {
            self.active.sync_data()?;
            self.unsynced = 0;
        }
        Ok(())
    }

    /// Delete segments whose records are all `<= seq`. Returns bytes freed.
    fn prune_through(&mut self, seq: u64) -> Result<u64> {
        let mut freed = 0;
        // A segment's last seq is the next segment's first_seq - 1, so it can
        // only be judged when the next segment's header is intact.
        while self.segments.len() > 1 {
            let next_first = self.segments[1].first_seq;
            match next_first {
                Some(nf) if nf > 0 && nf - 1 <= seq => {
                    let s = self.segments.remove(0);
                    fs::remove_file(&s.path)?;
                    self.total_bytes -= s.len;
                    freed += s.len;
                }
                _ => break,
            }
        }
        if freed > 0 {
            fsutil::sync_dir(&self.dir)?;
        }
        Ok(freed)
    }
}

/// Writer for a replicated record log. One per directory at a time.
#[derive(Debug)]
pub struct LogWriter {
    root: PathBuf,
    cfg: LogConfig,
    replicas: Vec<ReplicaWriter>,
    next_seq: u64,
    degraded_writes: u64,
    recovery: RecoveryReport,
    _lock: DirLock,
}

impl LogWriter {
    pub fn open(root: impl AsRef<Path>, cfg: LogConfig) -> Result<Self> {
        cfg.validate()?;
        let root = root.as_ref().to_path_buf();
        fsutil::ensure_dir(&root)?;
        let lock = DirLock::acquire(&root, "LOG.LOCK")?;
        let existing = existing_copies(&root)?;
        if let Some((i, _)) = existing.iter().find(|(i, _)| *i >= cfg.replicas) {
            return Err(Error::InvalidConfig(format!(
                "log already has copy-{i}; open it with replicas >= {}",
                i + 1
            )));
        }
        let mut report = RecoveryReport::default();
        let mut replicas = Vec::with_capacity(cfg.replicas);
        let mut max_seq = 0;
        for i in 0..cfg.replicas {
            let (w, m) = ReplicaWriter::open(copy_dir(&root, i), &cfg, &mut report)?;
            max_seq = max_seq.max(m);
            replicas.push(w);
        }
        fsutil::sync_dir(&root)?;
        Ok(LogWriter {
            root,
            cfg,
            replicas,
            next_seq: max_seq + 1,
            degraded_writes: 0,
            recovery: report,
            _lock: lock,
        })
    }

    /// Append a record to every replica and return its sequence number.
    ///
    /// Succeeds if at least one replica took the record (and counts the
    /// write as degraded if any failed). Fails only if none did.
    pub fn append(&mut self, kind: u16, payload: &[u8]) -> Result<u64> {
        if payload.len() > self.cfg.max_record_bytes {
            return Err(Error::TooLarge {
                size: payload.len(),
                limit: self.cfg.max_record_bytes,
            });
        }
        let seq = self.next_seq;
        let frame = encode_record(seq, kind, payload);
        let mut first_err = None;
        let mut ok = 0;
        for r in &mut self.replicas {
            match r.append(seq, &frame, &self.cfg) {
                Ok(()) => ok += 1,
                Err(e) => {
                    first_err.get_or_insert(e);
                }
            }
        }
        if ok == 0 {
            return Err(first_err.expect("at least one replica"));
        }
        if first_err.is_some() {
            self.degraded_writes += 1;
        }
        self.next_seq = seq
            .checked_add(1)
            .ok_or_else(|| Error::Corrupt("sequence overflow".into()))?;
        Ok(seq)
    }

    /// Force everything appended so far to stable storage.
    pub fn sync(&mut self) -> Result<()> {
        for r in &mut self.replicas {
            r.sync()?;
        }
        Ok(())
    }

    /// Delete whole segments holding only records `<= seq`, e.g. after the
    /// surface station acknowledged receiving them. Returns bytes freed per replica (max).
    pub fn prune_through(&mut self, seq: u64) -> Result<u64> {
        let mut freed = 0;
        for r in &mut self.replicas {
            freed = freed.max(r.prune_through(seq)?);
        }
        Ok(freed)
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }
    pub fn degraded_writes(&self) -> u64 {
        self.degraded_writes
    }
    pub fn recovery(&self) -> &RecoveryReport {
        &self.recovery
    }
    /// Bytes used by the largest replica.
    pub fn bytes_used(&self) -> u64 {
        self.replicas
            .iter()
            .map(|r| r.total_bytes)
            .max()
            .unwrap_or(0)
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
}

impl Drop for LogWriter {
    fn drop(&mut self) {
        let _ = self.sync();
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// What a read found, so damage is visible rather than silent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReadStats {
    pub replicas: u64,
    pub segments: u64,
    pub unreadable_segments: u64,
    /// Bytes across all replicas that weren't part of an intact record.
    pub skipped_bytes: u64,
    pub damaged_regions: u64,
    /// Records that at least one replica was missing or had damaged, but
    /// that another replica supplied intact.
    pub repaired_from_replica: u64,
    /// Sequence numbers between the first and last record that no replica
    /// had intact (or that were pruned from all of them).
    pub missing_seqs: u64,
    /// Same seq, different contents in different replicas. Should never
    /// happen; if it does, the lowest-numbered replica's copy is returned.
    pub conflicts: u64,
}

#[derive(Clone, Debug)]
pub struct ReadOptions {
    pub max_record_bytes: usize,
    /// Refuse to read any file larger than this.
    pub max_segment_bytes: u64,
    /// Only accept segments written under this stream id, if set.
    pub stream_id: Option<u64>,
}

impl Default for ReadOptions {
    fn default() -> Self {
        ReadOptions {
            max_record_bytes: HARD_MAX_RECORD,
            max_segment_bytes: 1024 * 1024 * 1024,
            stream_id: None,
        }
    }
}

/// Read and merge every replica under `root`. Takes no lock and changes
/// nothing on disk, so it is safe on a live log or a downlinked copy.
pub fn read_log(root: impl AsRef<Path>, opts: &ReadOptions) -> Result<(Vec<Record>, ReadStats)> {
    let root = root.as_ref();
    fsutil::reject_symlink(root)?;
    let copies = existing_copies(root)?;
    let mut stats = ReadStats {
        replicas: copies.len() as u64,
        ..Default::default()
    };
    let mut merged: BTreeMap<u64, (Record, u64)> = BTreeMap::new(); // seq -> (record, copies seen)
    for (_, dir) in &copies {
        if fs::symlink_metadata(dir)?.file_type().is_symlink() {
            stats.unreadable_segments += 1;
            continue;
        }
        for (index, path) in list_segments(dir)? {
            stats.segments += 1;
            let buf = match read_capped(&path, opts.max_segment_bytes) {
                Ok(b) => b,
                Err(_) => {
                    stats.unreadable_segments += 1;
                    continue;
                }
            };
            // A damaged header doesn't stop us reading the records after
            // it: they carry their own sequence numbers and CRCs.
            let start = match decode_segment_header(&buf) {
                Some(h) => {
                    if h.index != index || opts.stream_id.is_some_and(|s| s != h.stream_id) {
                        stats.unreadable_segments += 1;
                        continue;
                    }
                    SEGMENT_HEADER_LEN
                }
                None => {
                    stats.damaged_regions += 1;
                    stats.skipped_bytes += SEGMENT_HEADER_LEN.min(buf.len()) as u64;
                    SEGMENT_HEADER_LEN.min(buf.len())
                }
            };
            let scan = scan_records(&buf, start, opts.max_record_bytes);
            stats.skipped_bytes += scan.skipped_bytes;
            stats.damaged_regions += scan.damaged_regions;
            for rec in scan.records {
                match merged.get_mut(&rec.seq) {
                    Some((existing, seen)) => {
                        *seen += 1;
                        if existing.payload != rec.payload || existing.kind != rec.kind {
                            stats.conflicts += 1;
                        }
                    }
                    None => {
                        merged.insert(rec.seq, (rec, 1));
                    }
                }
            }
        }
    }
    let n_replicas = copies.len() as u64;
    stats.repaired_from_replica = merged
        .values()
        .filter(|(_, seen)| *seen < n_replicas)
        .count() as u64;
    if let (Some(first), Some(last)) = (merged.keys().next(), merged.keys().next_back()) {
        stats.missing_seqs = (last - first + 1) - merged.len() as u64;
    }
    Ok((merged.into_values().map(|(r, _)| r).collect(), stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_roundtrip() {
        let f = encode_record(42, 7, b"hello ganymede");
        let (r, n) = decode_record_at(&f, 1024).unwrap();
        assert_eq!(n, f.len());
        assert_eq!(
            r,
            Record {
                seq: 42,
                kind: 7,
                payload: b"hello ganymede".to_vec()
            }
        );
    }

    #[test]
    fn every_single_bit_flip_is_detected() {
        let f = encode_record(9, 1, b"payload under the ice");
        for byte in 0..f.len() {
            for bit in 0..8 {
                let mut g = f.clone();
                g[byte] ^= 1 << bit;
                assert!(
                    decode_record_at(&g, 1024).is_none(),
                    "flip at {byte}:{bit} not caught"
                );
            }
        }
    }

    #[test]
    fn oversized_length_is_rejected_before_reading_payload() {
        let mut f = encode_record(1, 1, b"x");
        // Forge a huge length *and* a matching header CRC: still refused by the cap.
        f[4..8].copy_from_slice(&(u32::MAX).to_le_bytes());
        let crc = crc32c(&f[4..20]);
        f[20..24].copy_from_slice(&crc.to_le_bytes());
        assert!(decode_record_at(&f, 1024).is_none());
    }

    #[test]
    fn scan_resyncs_after_damage() {
        let mut buf = Vec::new();
        buf.extend(encode_record(1, 0, b"a"));
        buf.extend(b"\x00garbage\xD3\x5A\x7E\x91 more garbage");
        buf.extend(encode_record(2, 0, b"b"));
        let s = scan_records(&buf, 0, 1024);
        assert_eq!(
            s.records.iter().map(|r| r.seq).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(s.damaged_regions, 1);
        assert_eq!(s.end_of_valid, buf.len());
    }

    #[test]
    fn header_roundtrip_and_tamper() {
        let h = encode_segment_header(3, 100, 0xABCD);
        let d = decode_segment_header(&h).unwrap();
        assert_eq!((d.index, d.first_seq, d.stream_id), (3, 100, 0xABCD));
        let mut bad = h;
        bad[20] ^= 0x10;
        assert!(decode_segment_header(&bad).is_none());
    }
}
