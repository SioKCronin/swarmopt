//! Fault injection: power cuts, bit flips, garbage, full disks, races.
//!
//! These model what an under-ice drone's storage actually goes through.
//! No dependencies: a tiny xorshift RNG and a self-cleaning temp dir.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use swarmopt_store::{
    read_checkpoint, read_log, CheckpointStore, Error, FullPolicy, LogConfig, LogWriter,
    ReadOptions, SyncPolicy,
};

// ---------------------------------------------------------------- helpers

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "swarmopt-store-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn small_cfg(replicas: usize) -> LogConfig {
    LogConfig {
        stream_id: 0x006A_7570_6974_6572, // "jupiter"
        replicas,
        segment_bytes: 4096,
        max_total_bytes: 1 << 20,
        max_record_bytes: 512,
        sync: SyncPolicy::Always,
        on_full: FullPolicy::Reject,
    }
}

fn payload(i: u64) -> Vec<u8> {
    format!("{{\"epoch\":{i},\"best_cost\":{}}}", 1.0 / (i as f64 + 1.0)).into_bytes()
}

fn write_n(root: &Path, cfg: &LogConfig, n: u64) {
    let mut w = LogWriter::open(root, cfg.clone()).unwrap();
    for i in 0..n {
        w.append(1, &payload(i)).unwrap();
    }
}

fn segments_of(copy: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(copy)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "swl"))
        .collect();
    v.sort();
    v
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dest = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_tree(&e.path(), &dest);
        } else {
            fs::copy(e.path(), dest).unwrap();
        }
    }
}

fn flip_random_bits(path: &Path, rng: &mut Rng, flips: usize) {
    let mut b = fs::read(path).unwrap();
    for _ in 0..flips {
        let i = rng.below(b.len() as u64) as usize;
        b[i] ^= 1 << rng.below(8);
    }
    fs::write(path, b).unwrap();
}

fn opts() -> ReadOptions {
    ReadOptions {
        max_record_bytes: 512,
        ..Default::default()
    }
}

// ------------------------------------------------------------------ tests

#[test]
fn roundtrip_and_restart_continue_sequence() {
    let d = TempDir::new("rt");
    let cfg = small_cfg(2);
    write_n(d.path(), &cfg, 100);
    {
        let mut w = LogWriter::open(d.path(), cfg.clone()).unwrap();
        assert_eq!(w.next_seq(), 101);
        assert_eq!(w.append(2, b"after restart").unwrap(), 101);
    }
    let (recs, stats) = read_log(d.path(), &opts()).unwrap();
    assert_eq!(recs.len(), 101);
    assert!(recs.iter().zip(1..).all(|(r, s)| r.seq == s));
    assert_eq!(recs[100].payload, b"after restart");
    assert_eq!(stats.skipped_bytes, 0);
    assert_eq!(stats.missing_seqs, 0);
    assert_eq!(stats.repaired_from_replica, 0);
    assert!(stats.segments > 2, "should have rotated: {stats:?}");
}

/// Cut power at every possible byte while the last record is being written.
#[test]
fn power_cut_at_every_byte_of_the_tail() {
    let base = TempDir::new("pc-base");
    let cfg = small_cfg(1);
    write_n(base.path(), &cfg, 10);
    let seg = segments_of(&base.path().join("copy-0")).pop().unwrap();
    let full_len = fs::metadata(&seg).unwrap().len();
    let last_frame = (payload(9).len() + swarmopt_store::log::RECORD_OVERHEAD) as u64;

    for cut in (full_len - last_frame)..=full_len {
        let d = TempDir::new("pc");
        copy_tree(base.path(), d.path());
        let s = segments_of(&d.path().join("copy-0")).pop().unwrap();
        OpenOptions::new()
            .write(true)
            .open(&s)
            .unwrap()
            .set_len(cut)
            .unwrap();

        let mut w = LogWriter::open(d.path(), cfg.clone()).unwrap();
        let survived = if cut == full_len { 10 } else { 9 };
        assert_eq!(w.next_seq(), survived + 1, "cut at {cut}");
        if cut != full_len && cut != full_len - last_frame {
            assert!(w.recovery().truncated_bytes > 0);
        }
        w.append(1, b"resumed").unwrap();
        drop(w);

        let (recs, stats) = read_log(d.path(), &opts()).unwrap();
        assert_eq!(recs.len() as u64, survived + 1, "cut at {cut}");
        assert_eq!(recs.last().unwrap().payload, b"resumed");
        assert_eq!(
            stats.skipped_bytes, 0,
            "torn bytes must be gone after recovery"
        );
        assert_eq!(stats.conflicts, 0);
    }
}

/// Power cut between creating a segment file and writing its header.
#[test]
fn power_cut_during_segment_creation() {
    let d = TempDir::new("pc-create");
    let cfg = small_cfg(1);
    write_n(d.path(), &cfg, 5);
    let copy = d.path().join("copy-0");
    let next = copy.join("seg-0000000002.swl");
    fs::write(&next, b"SWO").unwrap(); // a few header bytes, then nothing
    let mut w = LogWriter::open(d.path(), cfg).unwrap();
    assert_eq!(w.recovery().removed_empty_segments, 1);
    assert_eq!(w.append(1, b"x").unwrap(), 6);
    drop(w);
    let (recs, _) = read_log(d.path(), &opts()).unwrap();
    assert_eq!(recs.len(), 6);
}

/// Radiation: random bit flips in one replica are fully repaired by the other.
#[test]
fn bit_flips_in_one_replica_are_repaired() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for trial in 0..20 {
        let d = TempDir::new("flip1");
        let cfg = small_cfg(2);
        write_n(d.path(), &cfg, 200);
        for seg in segments_of(&d.path().join("copy-0")) {
            flip_random_bits(&seg, &mut rng, 1 + (trial % 5));
        }
        let (recs, stats) = read_log(d.path(), &opts()).unwrap();
        assert_eq!(recs.len(), 200, "trial {trial}: {stats:?}");
        for (r, i) in recs.iter().zip(0u64..) {
            assert_eq!(r.seq, i + 1);
            assert_eq!(r.payload, payload(i), "trial {trial}");
        }
        assert!(stats.repaired_from_replica > 0);
        assert_eq!(stats.conflicts, 0);
    }
}

/// Flips in *both* replicas: every record intact in either copy comes back,
/// and no corrupted record is ever returned as good.
#[test]
fn bit_flips_in_both_replicas_never_yield_bad_data() {
    let mut rng = Rng(42);
    for _ in 0..30 {
        let d = TempDir::new("flip2");
        write_n(d.path(), &small_cfg(2), 150);
        for c in ["copy-0", "copy-1"] {
            for seg in segments_of(&d.path().join(c)) {
                flip_random_bits(&seg, &mut rng, 3);
            }
        }
        let (recs, stats) = read_log(d.path(), &opts()).unwrap();
        for r in &recs {
            assert_eq!(
                r.payload,
                payload(r.seq - 1),
                "a damaged record got through"
            );
        }
        assert_eq!(stats.conflicts, 0);
        assert!(recs.len() >= 140, "lost too much: {}", recs.len());
    }
}

/// Writer reopening a log whose replicas have independent damage keeps going.
#[test]
fn writer_reopens_damaged_log() {
    let mut rng = Rng(7);
    let d = TempDir::new("reopen-damaged");
    let cfg = small_cfg(2);
    write_n(d.path(), &cfg, 120);
    for seg in segments_of(&d.path().join("copy-1")) {
        flip_random_bits(&seg, &mut rng, 4);
    }
    let mut w = LogWriter::open(d.path(), cfg).unwrap();
    assert_eq!(w.next_seq(), 121);
    w.append(1, b"still logging").unwrap();
    drop(w);
    let (recs, _) = read_log(d.path(), &opts()).unwrap();
    assert_eq!(recs.len(), 121);
}

/// A damaged header on the segment being appended to: quarantine it, start a
/// new segment, and never reuse a sequence number.
#[test]
fn damaged_tail_header_is_quarantined() {
    let d = TempDir::new("quarantine");
    let cfg = small_cfg(1);
    write_n(d.path(), &cfg, 60);
    let copy = d.path().join("copy-0");
    let tail = segments_of(&copy).pop().unwrap();
    let mut b = fs::read(&tail).unwrap();
    b[3] ^= 0xFF;
    fs::write(&tail, b).unwrap();
    let mut w = LogWriter::open(d.path(), cfg).unwrap();
    assert_eq!(w.recovery().quarantined_segments, 1);
    assert_eq!(w.next_seq(), 61, "seq must not go backwards");
    w.append(1, b"new").unwrap();
    drop(w);
    assert!(fs::read_dir(&copy).unwrap().any(|e| e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".quarantine")));
}

/// Arbitrary bytes posing as segments: the reader must never panic or
/// allocate wildly, and the writer must fail cleanly or recover.
#[test]
fn garbage_segments_never_panic() {
    let mut rng = Rng(0xDEAD_BEEF);
    for i in 0..200 {
        let d = TempDir::new("garbage");
        let copy = d.path().join("copy-0");
        fs::create_dir_all(&copy).unwrap();
        let len = rng.below(3000) as usize;
        let mut bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        // Sometimes plant real-looking sync markers and huge lengths.
        if i % 3 == 0 && len > 40 {
            let at = rng.below((len - 40) as u64) as usize;
            bytes[at..at + 4].copy_from_slice(&[0xD3, 0x5A, 0x7E, 0x91]);
            bytes[at + 4..at + 8].copy_from_slice(&u32::MAX.to_le_bytes());
        }
        fs::write(copy.join("seg-0000000001.swl"), &bytes).unwrap();
        let _ = read_log(d.path(), &opts()).unwrap();
        let _ = LogWriter::open(d.path(), small_cfg(1));
    }
}

#[test]
fn storage_budget_reject_and_prune() {
    let d = TempDir::new("budget");
    let cfg = LogConfig {
        max_total_bytes: 16 * 1024,
        ..small_cfg(1)
    };
    let mut w = LogWriter::open(d.path(), cfg).unwrap();
    let mut last_ok = 0;
    let err = loop {
        match w.append(1, &[0xAB; 400]) {
            Ok(s) => last_ok = s,
            Err(e) => break e,
        }
    };
    assert!(matches!(err, Error::StorageFull { .. }), "{err}");
    assert!(w.bytes_used() <= 16 * 1024);
    // Surface confirms receipt of everything so far: prune and keep going.
    let freed = w.prune_through(last_ok).unwrap();
    assert!(freed > 0);
    w.append(1, &[0xCD; 400]).unwrap();
    drop(w);
    let (recs, _) = read_log(d.path(), &opts()).unwrap();
    assert!(recs.iter().all(|r| r.seq > last_ok - 20));
    assert_eq!(recs.last().unwrap().payload, vec![0xCD; 400]);
}

#[test]
fn storage_budget_drop_oldest_stays_bounded() {
    let d = TempDir::new("drop");
    let cfg = LogConfig {
        max_total_bytes: 16 * 1024,
        on_full: FullPolicy::DropOldest,
        ..small_cfg(2)
    };
    let mut w = LogWriter::open(d.path(), cfg).unwrap();
    for _ in 0..500 {
        w.append(1, &[7; 300]).unwrap();
        assert!(w.bytes_used() <= 16 * 1024);
    }
    drop(w);
    let (recs, stats) = read_log(d.path(), &opts()).unwrap();
    assert_eq!(recs.last().unwrap().seq, 500);
    assert_eq!(stats.missing_seqs, 0, "kept records must be contiguous");
}

#[test]
fn oversized_record_rejected() {
    let d = TempDir::new("big");
    let mut w = LogWriter::open(d.path(), small_cfg(1)).unwrap();
    assert!(matches!(
        w.append(1, &[0; 513]),
        Err(Error::TooLarge { .. })
    ));
    assert_eq!(w.next_seq(), 1, "a refused record must not consume a seq");
}

#[test]
fn second_writer_is_locked_out() {
    let d = TempDir::new("lock");
    let _w = LogWriter::open(d.path(), small_cfg(1)).unwrap();
    assert!(matches!(
        LogWriter::open(d.path(), small_cfg(1)),
        Err(Error::Locked(_))
    ));
    // Readers don't need the lock.
    read_log(d.path(), &opts()).unwrap();
}

#[test]
fn other_missions_log_is_refused() {
    let d = TempDir::new("stream");
    write_n(d.path(), &small_cfg(1), 3);
    let other = LogConfig {
        stream_id: 1,
        ..small_cfg(1)
    };
    assert!(matches!(
        LogWriter::open(d.path(), other),
        Err(Error::InvalidConfig(_))
    ));
    let filtered = ReadOptions {
        stream_id: Some(1),
        ..opts()
    };
    assert!(read_log(d.path(), &filtered).unwrap().0.is_empty());
}

#[test]
fn fewer_replicas_than_on_disk_is_refused() {
    let d = TempDir::new("replicas");
    write_n(d.path(), &small_cfg(3), 3);
    assert!(matches!(
        LogWriter::open(d.path(), small_cfg(2)),
        Err(Error::InvalidConfig(_))
    ));
}

#[cfg(unix)]
#[test]
fn symlinks_are_refused() {
    use std::os::unix::fs::symlink;
    let d = TempDir::new("symlink");
    let elsewhere = TempDir::new("symlink-target");
    let link = d.path().join("log");
    symlink(elsewhere.path(), &link).unwrap();
    assert!(matches!(
        LogWriter::open(&link, small_cfg(1)),
        Err(Error::UnsafePath(_))
    ));
    assert!(matches!(
        CheckpointStore::open(&link, 1024),
        Err(Error::UnsafePath(_))
    ));

    // A segment replaced by a symlink is not followed by the writer.
    let real = d.path().join("real");
    write_n(&real, &small_cfg(1), 2);
    let seg = segments_of(&real.join("copy-0")).pop().unwrap();
    fs::remove_file(&seg).unwrap();
    let secret = elsewhere.path().join("secret");
    fs::write(&secret, b"not yours").unwrap();
    symlink(&secret, &seg).unwrap();
    let r = LogWriter::open(&real, small_cfg(1));
    assert!(r.is_ok() || matches!(r, Err(Error::UnsafePath(_))));
    assert_eq!(
        fs::read(&secret).unwrap(),
        b"not yours",
        "symlink target was modified"
    );
}

#[cfg(unix)]
#[test]
fn files_are_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let d = TempDir::new("perms");
    let root = d.path().join("log");
    write_n(&root, &small_cfg(1), 1);
    let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(&root), 0o700);
    assert_eq!(mode(&segments_of(&root.join("copy-0"))[0]), 0o600);
}

// ------------------------------------------------------------ checkpoints

#[test]
fn checkpoint_generations_and_fallback() {
    let d = TempDir::new("ckpt");
    let mut s = CheckpointStore::open(d.path(), 1024).unwrap();
    assert!(s.load().unwrap().is_none());
    for g in 1..=5u64 {
        assert_eq!(s.save(format!("state {g}").as_bytes()).unwrap(), g);
    }
    let l = s.load().unwrap().unwrap();
    assert_eq!(
        (l.generation, l.payload.as_slice(), l.fell_back),
        (5, &b"state 5"[..], false)
    );

    // Gen 5 sits in slot a (odd gens -> a). Damage it: fall back to 4.
    let a = d.path().join("ckpt-a.swc");
    let mut b = fs::read(&a).unwrap();
    let mid = b.len() / 2;
    b[mid] ^= 0x04;
    fs::write(&a, b).unwrap();
    let l = s.load().unwrap().unwrap();
    assert_eq!((l.generation, l.fell_back, l.damaged_slots), (4, true, 1));

    // Next save overwrites the damaged slot, not the good one.
    assert_eq!(s.save(b"state 5 again").unwrap(), 5);
    let l = s.load().unwrap().unwrap();
    assert_eq!(
        (l.generation, l.payload.as_slice()),
        (5, &b"state 5 again"[..])
    );
    assert_eq!(
        read_checkpoint(d.path(), 1024).unwrap().unwrap().generation,
        5
    );
}

/// Power cut mid-save: only the temp file is partial; the last good
/// checkpoint is untouched and the temp file is cleaned up on open.
#[test]
fn checkpoint_power_cut_mid_save() {
    let d = TempDir::new("ckpt-cut");
    {
        let mut s = CheckpointStore::open(d.path(), 1024).unwrap();
        s.save(b"one").unwrap();
        s.save(b"two").unwrap();
    }
    // Simulate an interrupted third save into slot a.
    let mut f = fs::File::create(d.path().join("ckpt-a.swc.tmp")).unwrap();
    f.write_all(b"SWOCKP\0\x01 partial").unwrap();
    drop(f);
    let s = CheckpointStore::open(d.path(), 1024).unwrap();
    assert!(!d.path().join("ckpt-a.swc.tmp").exists());
    let l = s.load().unwrap().unwrap();
    assert_eq!(
        (l.generation, l.payload.as_slice(), l.fell_back),
        (2, &b"two"[..], false)
    );
}

#[test]
fn checkpoint_both_slots_damaged_is_none() {
    let d = TempDir::new("ckpt-both");
    let mut s = CheckpointStore::open(d.path(), 1024).unwrap();
    s.save(b"a").unwrap();
    s.save(b"b").unwrap();
    for n in ["ckpt-a.swc", "ckpt-b.swc"] {
        fs::write(d.path().join(n), b"\0\0\0\0").unwrap();
    }
    assert!(s.load().unwrap().is_none());
    assert_eq!(s.save(b"fresh").unwrap(), 1);
}

#[test]
fn checkpoint_size_and_lock() {
    let d = TempDir::new("ckpt-lock");
    let mut s = CheckpointStore::open(d.path(), 8).unwrap();
    assert!(matches!(s.save(&[0; 9]), Err(Error::TooLarge { .. })));
    assert!(matches!(
        CheckpointStore::open(d.path(), 8),
        Err(Error::Locked(_))
    ));
    // A log and a checkpoint store can share a directory.
    LogWriter::open(d.path(), small_cfg(1)).unwrap();
}
