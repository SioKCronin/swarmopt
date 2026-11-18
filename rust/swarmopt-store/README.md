# swarmopt-store

Crash-safe, corruption-tolerant persistence for SwarmOpt, written in Rust.

The motivating case is a swarm of research drones under the ice shell of
Ganymede. Those drones:

- lose power without warning,
- sit inside Jupiter's radiation belts, so stored bits flip,
- have no wall clock they can trust,
- have a fixed amount of storage, and
- reach the surface only now and then, to send data up.

The store is built for exactly those conditions, and it works the same way
on a laptop. It is the host half of the Rust split: anything with `std` and
a filesystem. The bare-metal onboard core is
[`prototypes/deep_space`](../../prototypes/deep_space/), and the boundary is
[`docs/RUST_ARCHITECTURE.md`](../../docs/RUST_ARCHITECTURE.md).

| Condition | What the store does |
| --- | --- |
| Power cut mid-write | fsync per policy; torn tails cut on restart; checkpoints use A/B slots with atomic rename |
| Bit flips (SEUs) | CRC-32C on every record header, record body, segment header and checkpoint; damaged bytes are never returned as data |
| Damage mid-file | Readers skip ahead to the next sync marker instead of losing the rest of the file |
| Bad flash block | Each record is written to N independent replica directories (`replicas`, default 2), and readers merge them |
| Corrupted length fields | Header CRC checked *before* the length is used; hard caps on record and file sizes |
| No trusted clock | Ordering by sequence number and checkpoint generation only |
| Fixed storage | Byte budget per replica; `reject` (default) or `drop_oldest`; `prune_through(seq)` after downlink |
| Two writers | Exclusive lock file per store |
| Hostile paths | Symlinks refused; files `0600`, directories `0700` |
| Mixed missions | `stream_id` in every segment header; a writer refuses another stream's log |

The core is `std`-only, with no dependencies. PyO3 is compiled in only
with the `python` feature.

## Python

```bash
pip install maturin
cd rust/swarmopt-store && maturin build --release && pip install target/wheels/*.whl
```

```python
import swarmopt_store as st

with st.LogWriter("mission/log", stream_id=0x4A55, replicas=2) as log:
    seq = log.append(1, b'{"epoch": 10, "best_cost": 0.02}')

records, stats = st.read_log("mission/log")   # [(seq, kind, bytes)], {...}

ckpt = st.CheckpointStore("mission/ckpt")
gen = ckpt.save(b"...state...")
latest = ckpt.load()   # {"generation", "payload", "fell_back", "damaged_slots"} or None
```

SwarmOpt itself uses this through `swarmopt.persist`, which encodes payloads
as strict JSON (never pickle) and adds `Swarm` checkpoint/resume.

## On-disk format (version 1, little-endian)

### Log segment `copy-<i>/seg-<index>.swl`

```
header, 36 bytes:
  magic "SWOLOG\0\x01" | version u16 | flags u16 | segment_index u32
  | first_seq u64 | stream_id u64 | crc32c(bytes 0..32) u32
record, 28 bytes + payload:
  sync D3 5A 7E 91 | len u32 | seq u64 | kind u16 | flags u16
  | header_crc u32 = crc32c(len..flags)
  | payload [len]
  | body_crc u32 = crc32c(len..flags ++ payload)
```

### Checkpoint slot `ckpt-a.swc`, `ckpt-b.swc`

```
magic "SWOCKP\0\x01" | version u16 | flags u16 | generation u64
| len u32 | reserved u32 | header_crc u32 = crc32c(bytes 0..28)
| payload [len] | body_crc u32 = crc32c(bytes 0..32 ++ payload)
```

## Tests

```bash
cargo test                          # unit + fault injection
cargo clippy --all-targets --features python -- -D warnings
```

`tests/fault_injection.rs` covers:

- a power cut at every byte of a record, and during segment creation,
- random bit flips in one replica (always fully repaired) and in both
  replicas (a damaged record is never returned),
- garbage files posing as segments (no panics, no runaway allocation),
- quarantine of damaged headers, storage budgets, locking, stream
  mismatch, symlink refusal and permissions.

## Known limits

- CRCs *detect* corruption; only replication *repairs* it. A record
  damaged in every replica is lost, though it is reported in `missing_seqs`
  and never silently returned. Forward error correction (e.g. Reed-Solomon
  per segment) is the next step for single-copy storage.
- There is no scrubber yet that rewrites a damaged replica from a good one.
  Reads repair in memory only.
- Payloads are not encrypted or authenticated. CRC protects against
  accidents, not an adversary who can rewrite the file. If the threat model
  includes tampering (e.g. on the downlink), add an HMAC or signature at the
  transport or payload layer.
- Locking is advisory (`flock`). It stops cooperating processes, not a
  hostile process with the same permissions.
