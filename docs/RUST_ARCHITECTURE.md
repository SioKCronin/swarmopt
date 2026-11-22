# Rust architecture

SwarmOpt has two Rust crates. They answer the same unattended-hardware
problem at two different layers, and they stay separate.

| | Onboard research core | Persistence |
| --- | --- | --- |
| Path | `prototypes/deep_space` | `rust/persist` |
| Package | `swarmopt-deep-space` | `swarmopt-persist` |
| Job | Decide the next observation under a hard energy reserve | Keep run logs and checkpoints across power cuts and bit flips |
| Runtime | `no_std`, no allocator, `forbid(unsafe_code)` | `std` and `std::fs`. PyO3 only behind the `python` feature |
| Dependencies | none | none in the core |
| Checksum | CRC-32 (IEEE), computed bitwise | CRC-32C (Castagnoli), table-driven |
| Durable state | one fixed-size snapshot, two slots in a byte array | variable-length log plus A/B checkpoint files |
| Who calls it | the mission sim, and `compare_python.py` offline | `swarmopt.persist` and the Python extension |
| CI | `.github/workflows/rust-prototype.yml` | `.github/workflows/persist.yml` |
| Briefing | [`deep_space_mission.md`](deep_space_mission.md) | [`SECURITY_AUDIT.md`](SECURITY_AUDIT.md) |

Python SwarmOpt stays the place for algorithm research, benchmarks, and
reference behaviour. Neither crate replaces it.

## What each crate owns

The onboard core proposes a plan and a separate safety gate approves or
rejects it. Its snapshot and its 32-byte peer envelopes are fixed layouts
inside that crate. A platform adapter would copy those bytes to NVM or the
radio. The sim in `sim.rs` stands in for that adapter. This crate is a
research prototype, not flight software.

`swarmopt-persist` is what a process with a filesystem uses: a laptop, a
ground station, or a payload computer running an OS. Appends are checksummed
and fsynced, checkpoints use two files and an atomic rename, and readers
merge independent replicas. Checkpoint payloads are opaque bytes. SwarmOpt
writes strict JSON and never pickle. The crate does not know about
particles, PSO, or spacecraft snapshots.

## What stays on its own side

The crates do not depend on each other.

- The onboard core cannot link `swarmopt-persist`. That crate needs `std`, a
  filesystem, and, for Python, an FFI boundary. The core is built for
  bare-metal targets (`thumbv7em-none-eabihf`, `riscv32imac-unknown-none-elf`).
- `swarmopt-persist` cannot link the onboard core. A host checkpoint is not
  an onboard snapshot, and a log record is not a peer envelope.
- The checksums stay different. IEEE CRC-32 matches the fixed onboard
  layouts and needs no table. CRC-32C is the storage checksum, with published
  test vectors and stronger burst detection on variable-length records.
  There is no shared decoder, so one polynomial would not make the formats
  interoperable.
- They are not one Cargo workspace. Workspace profiles apply only at the
  workspace root. The onboard core aborts on panic so embedded and desktop
  runs match. `swarmopt-persist` unwinds so the Python extension can turn a panic
  into an exception. One profile would break one of those.

A later downlink adapter can append a copy of an onboard snapshot or
envelope into a `swarmopt-persist` log as an opaque payload. That adapter
does not exist yet. It would live outside both crates.

## Build

```bash
cargo test --release --manifest-path prototypes/deep_space/Cargo.toml
cargo test --locked --release --manifest-path rust/persist/Cargo.toml
```

`swarmopt-persist`'s release profile keeps overflow checks on. The onboard core's
dev and release profiles abort on panic. Run each crate with its own manifest
so those profiles stay in effect.
