//! Persistence for SwarmOpt on hardware nobody can reach.
//!
//! Motivating case: a swarm of research drones under the ice shell of
//! Ganymede. They lose power without warning, sit in Jupiter's radiation
//! belts (so stored bits flip), have no trustworthy wall clock, have a
//! fixed storage budget, and talk to the surface only now and then. So the
//! crate is built around four rules:
//!
//! 1. **Never trust a byte you didn't check.** Every record and checkpoint
//!    carries CRC-32C; lengths are checked before they are used, so damaged
//!    bytes can't cause huge allocations or out-of-bounds reads.
//! 2. **A power cut at any instant loses at most what wasn't acknowledged.**
//!    Appends are fsynced per the sync policy; checkpoints use A/B slots
//!    with atomic rename; torn tails are cut on restart.
//! 3. **Damage is contained and reported, not fatal.** Readers resync past
//!    damaged bytes, merge independent replicas, and report what they
//!    skipped or repaired.
//! 4. **Bounded and clock-free.** A fixed byte budget per replica, explicit
//!    pruning after downlink, and ordering by sequence number only.
//!
//! The core uses only `std` and has no dependencies. `swarmopt-persist` is
//! the host persistence crate: a laptop, a ground station, or a payload
//! computer with a filesystem.
//! The `no_std` onboard research core is a separate crate
//! (`prototypes/deep_space`). See `docs/RUST_ARCHITECTURE.md`.

pub mod checkpoint;
pub mod crc;
pub mod error;
mod fsutil;
pub mod log;

#[cfg(feature = "python")]
mod python;

pub use checkpoint::{read_checkpoint, CheckpointStore, Loaded};
pub use error::{Error, Result};
pub use log::{
    read_log, FullPolicy, LogConfig, LogWriter, ReadOptions, ReadStats, Record, RecoveryReport,
    SyncPolicy,
};
