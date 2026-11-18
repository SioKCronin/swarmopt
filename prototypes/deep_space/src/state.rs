//! Persistent spacecraft state with A/B slots.
//!
//! A snapshot is written alternately to two slots. On restart, the valid slot
//! with the higher generation wins; a torn or corrupted write to one slot
//! falls back to the other. This is the standard double-buffer pattern for
//! non-volatile storage that can be interrupted mid-write.

use crate::envelope::{crc32, PeerWindow, ReplayGuard, MAX_PEERS};

pub const SNAPSHOT_VERSION: u8 = 1;
const PEER_REC: usize = 2 + 1 + 1 + 4 + 8; // id, active, seen_any, highest, bitmap
const BODY_LEN: usize = 1 + 1 + 2 + 4 + 4 + 8 + 8 + 4 + 4 + MAX_PEERS * PEER_REC;
pub const SNAPSHOT_LEN: usize = BODY_LEN + 4;

/// Sequence numbers reserved across a restart. Any messages sent after the
/// last snapshot used numbers below `next_seq + SEQ_RESTART_GAP`, so resuming
/// above the gap guarantees peers never see a reused sequence number.
pub const SEQ_RESTART_GAP: u32 = 4096;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snapshot {
    pub generation: u32,
    pub spacecraft_id: u16,
    pub tick: u32,
    pub known_observed: u64,
    pub rng_state: u64,
    pub next_seq: u32,
    pub energy: f32,
    pub replay: ReplayGuard,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotError {
    ChecksumMismatch,
    UnsupportedVersion,
    Malformed,
}

impl Snapshot {
    pub fn encode(&self) -> [u8; SNAPSHOT_LEN] {
        let mut b = [0u8; SNAPSHOT_LEN];
        let mut o = 0;
        let mut put = |bytes: &[u8]| {
            b[o..o + bytes.len()].copy_from_slice(bytes);
            o += bytes.len();
        };
        put(&[SNAPSHOT_VERSION, 0]);
        put(&self.spacecraft_id.to_le_bytes());
        put(&self.generation.to_le_bytes());
        put(&self.tick.to_le_bytes());
        put(&self.known_observed.to_le_bytes());
        put(&self.rng_state.to_le_bytes());
        put(&self.next_seq.to_le_bytes());
        put(&self.energy.to_bits().to_le_bytes());
        for w in &self.replay.peers {
            put(&w.peer.to_le_bytes());
            put(&[u8::from(w.active), u8::from(w.seen_any)]);
            put(&w.highest.to_le_bytes());
            put(&w.bitmap.to_le_bytes());
        }
        let c = crc32(&b[..BODY_LEN]);
        b[BODY_LEN..].copy_from_slice(&c.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8; SNAPSHOT_LEN]) -> Result<Snapshot, SnapshotError> {
        let stored = u32::from_le_bytes([
            b[BODY_LEN],
            b[BODY_LEN + 1],
            b[BODY_LEN + 2],
            b[BODY_LEN + 3],
        ]);
        if crc32(&b[..BODY_LEN]) != stored {
            return Err(SnapshotError::ChecksumMismatch);
        }
        if b[0] != SNAPSHOT_VERSION {
            return Err(SnapshotError::UnsupportedVersion);
        }
        if b[1] != 0 {
            return Err(SnapshotError::Malformed);
        }
        let mut o = 2;
        let mut take = |n: usize| {
            let s = &b[o..o + n];
            o += n;
            s
        };
        let u16_ = |s: &[u8]| u16::from_le_bytes([s[0], s[1]]);
        let u32_ = |s: &[u8]| u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
        let u64_ = |s: &[u8]| {
            let mut a = [0u8; 8];
            a.copy_from_slice(s);
            u64::from_le_bytes(a)
        };
        let spacecraft_id = u16_(take(2));
        let generation = u32_(take(4));
        let tick = u32_(take(4));
        let known_observed = u64_(take(8));
        let rng_state = u64_(take(8));
        let next_seq = u32_(take(4));
        let energy = f32::from_bits(u32_(take(4)));
        if !energy.is_finite() {
            return Err(SnapshotError::Malformed);
        }
        let mut replay = ReplayGuard::new(&[]);
        for w in replay.peers.iter_mut() {
            let peer = u16_(take(2));
            let flags = take(2);
            if flags[0] > 1 || flags[1] > 1 {
                return Err(SnapshotError::Malformed);
            }
            let highest = u32_(take(4));
            let bitmap = u64_(take(8));
            *w = PeerWindow {
                peer,
                active: flags[0] == 1,
                seen_any: flags[1] == 1,
                highest,
                bitmap,
            };
        }
        Ok(Snapshot {
            generation,
            spacecraft_id,
            tick,
            known_observed,
            rng_state,
            next_seq,
            energy,
            replay,
        })
    }
}

/// Two-slot non-volatile store (simulated in RAM here).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AbStore {
    pub slots: [[u8; SNAPSHOT_LEN]; 2],
    next_slot: usize,
}

impl AbStore {
    pub const fn new() -> Self {
        Self {
            slots: [[0u8; SNAPSHOT_LEN]; 2],
            next_slot: 0,
        }
    }

    pub fn write(&mut self, s: &Snapshot) {
        self.slots[self.next_slot] = s.encode();
        self.next_slot ^= 1;
    }

    /// Most recent valid snapshot, or `None` if both slots are unreadable.
    pub fn restore(&self) -> Option<Snapshot> {
        let a = Snapshot::decode(&self.slots[0]).ok();
        let b = Snapshot::decode(&self.slots[1]).ok();
        match (a, b) {
            (Some(x), Some(y)) => Some(if y.generation > x.generation { y } else { x }),
            (Some(x), None) | (None, Some(x)) => Some(x),
            (None, None) => None,
        }
    }

    /// Fault injection: flip one bit in the most recently written slot.
    pub fn corrupt_latest(&mut self, byte: usize) {
        let latest = self.next_slot ^ 1;
        let i = byte % SNAPSHOT_LEN;
        self.slots[latest][i] ^= 0x01;
    }
}

impl Default for AbStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(generation: u32) -> Snapshot {
        let mut replay = ReplayGuard::new(&[2, 3]);
        replay.accept(2, 5).unwrap();
        Snapshot {
            generation,
            spacecraft_id: 1,
            tick: generation * 10,
            known_observed: 0b1011,
            rng_state: 0xDEAD_BEEF,
            next_seq: 77,
            energy: 42.5,
            replay,
        }
    }

    #[test]
    fn roundtrip() {
        let s = snap(3);
        assert_eq!(Snapshot::decode(&s.encode()), Ok(s));
    }

    #[test]
    fn restore_picks_newest_and_falls_back_on_corruption() {
        let mut st = AbStore::new();
        assert_eq!(st.restore(), None);
        st.write(&snap(1));
        st.write(&snap(2));
        assert_eq!(st.restore().unwrap().generation, 2);
        st.corrupt_latest(9);
        assert_eq!(st.restore().unwrap().generation, 1);
    }
}
