//! Versioned inter-spacecraft message envelopes and replay protection.
//!
//! Wire format (fixed 32 bytes, little-endian):
//!
//! | bytes  | field                                   |
//! |--------|-----------------------------------------|
//! | 0      | magic `0xD5`                            |
//! | 1      | format version (`1`)                    |
//! | 2      | kind (`1` = claim, `2` = heartbeat)     |
//! | 3      | reserved, must be 0                     |
//! | 4..6   | sender spacecraft id (`u16`)            |
//! | 6..10  | sequence number (`u32`)                 |
//! | 10..14 | sender tick (`u32`)                     |
//! | 14..18 | target id (`u32`)                       |
//! | 18..22 | science value (`f32` bits, finite)      |
//! | 22..28 | reserved, must be 0                     |
//! | 28..32 | CRC-32 (IEEE) over bytes 0..28          |
//!
//! Decoding is strict: anything that is not exactly a well-formed, supported
//! envelope is rejected with a specific reason. There is no "best effort"
//! parse of a partially valid message.

pub const ENVELOPE_LEN: usize = 32;
pub const MAGIC: u8 = 0xD5;
pub const VERSION: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// "I have observed this target" — peers stop considering it.
    Claim,
    Heartbeat,
}

impl Kind {
    fn to_byte(self) -> u8 {
        match self {
            Kind::Claim => 1,
            Kind::Heartbeat => 2,
        }
    }
    fn from_byte(b: u8) -> Option<Kind> {
        match b {
            1 => Some(Kind::Claim),
            2 => Some(Kind::Heartbeat),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Envelope {
    pub kind: Kind,
    pub sender: u16,
    pub seq: u32,
    pub tick: u32,
    pub target: u32,
    pub value: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    BadLength,
    BadMagic,
    UnsupportedVersion,
    UnknownKind,
    ReservedNonZero,
    NonFiniteValue,
    ChecksumMismatch,
}

/// CRC-32 (IEEE 802.3, reflected, poly 0xEDB88320). Bitwise, table-free.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

impl Envelope {
    pub fn encode(&self) -> [u8; ENVELOPE_LEN] {
        let mut b = [0u8; ENVELOPE_LEN];
        b[0] = MAGIC;
        b[1] = VERSION;
        b[2] = self.kind.to_byte();
        b[4..6].copy_from_slice(&self.sender.to_le_bytes());
        b[6..10].copy_from_slice(&self.seq.to_le_bytes());
        b[10..14].copy_from_slice(&self.tick.to_le_bytes());
        b[14..18].copy_from_slice(&self.target.to_le_bytes());
        b[18..22].copy_from_slice(&self.value.to_bits().to_le_bytes());
        let c = crc32(&b[..28]);
        b[28..32].copy_from_slice(&c.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Result<Envelope, DecodeError> {
        if b.len() != ENVELOPE_LEN {
            return Err(DecodeError::BadLength);
        }
        // Integrity first: a corrupted header should report as corruption,
        // not as an unsupported version.
        let stored = u32::from_le_bytes([b[28], b[29], b[30], b[31]]);
        if crc32(&b[..28]) != stored {
            return Err(DecodeError::ChecksumMismatch);
        }
        if b[0] != MAGIC {
            return Err(DecodeError::BadMagic);
        }
        if b[1] != VERSION {
            return Err(DecodeError::UnsupportedVersion);
        }
        let kind = Kind::from_byte(b[2]).ok_or(DecodeError::UnknownKind)?;
        if b[3] != 0 || b[22..28].iter().any(|&x| x != 0) {
            return Err(DecodeError::ReservedNonZero);
        }
        let value = f32::from_bits(u32::from_le_bytes([b[18], b[19], b[20], b[21]]));
        if !value.is_finite() {
            return Err(DecodeError::NonFiniteValue);
        }
        Ok(Envelope {
            kind,
            sender: u16::from_le_bytes([b[4], b[5]]),
            seq: u32::from_le_bytes([b[6], b[7], b[8], b[9]]),
            tick: u32::from_le_bytes([b[10], b[11], b[12], b[13]]),
            target: u32::from_le_bytes([b[14], b[15], b[16], b[17]]),
            value,
        })
    }
}

pub const MAX_PEERS: usize = 8;
/// Out-of-order acceptance window (sequence numbers), as in IPsec anti-replay.
pub const REPLAY_WINDOW: u32 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayError {
    UnknownPeer,
    Duplicate,
    TooOld,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeerWindow {
    pub peer: u16,
    pub active: bool,
    /// Highest sequence number accepted so far (valid when `seen_any`).
    pub highest: u32,
    pub seen_any: bool,
    /// Bit `k` set means `highest - k` was accepted.
    pub bitmap: u64,
}

impl PeerWindow {
    const EMPTY: PeerWindow = PeerWindow {
        peer: 0,
        active: false,
        highest: 0,
        seen_any: false,
        bitmap: 0,
    };
}

/// Per-peer sliding-window replay guard. Accepts delayed, out-of-order
/// messages within [`REPLAY_WINDOW`] while rejecting exact duplicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayGuard {
    pub peers: [PeerWindow; MAX_PEERS],
}

impl ReplayGuard {
    pub fn new(peer_ids: &[u16]) -> Self {
        let mut g = ReplayGuard {
            peers: [PeerWindow::EMPTY; MAX_PEERS],
        };
        for (slot, id) in g.peers.iter_mut().zip(peer_ids.iter()) {
            slot.peer = *id;
            slot.active = true;
        }
        g
    }

    pub fn accept(&mut self, peer: u16, seq: u32) -> Result<(), ReplayError> {
        let w = self
            .peers
            .iter_mut()
            .find(|w| w.active && w.peer == peer)
            .ok_or(ReplayError::UnknownPeer)?;
        if !w.seen_any {
            w.seen_any = true;
            w.highest = seq;
            w.bitmap = 1;
            return Ok(());
        }
        if seq > w.highest {
            let shift = seq - w.highest;
            w.bitmap = if shift >= 64 { 0 } else { w.bitmap << shift };
            w.bitmap |= 1;
            w.highest = seq;
            return Ok(());
        }
        let back = w.highest - seq;
        if back >= REPLAY_WINDOW {
            return Err(ReplayError::TooOld);
        }
        let bit = 1u64 << back;
        if w.bitmap & bit != 0 {
            return Err(ReplayError::Duplicate);
        }
        w.bitmap |= bit;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Envelope {
        Envelope {
            kind: Kind::Claim,
            sender: 2,
            seq: 41,
            tick: 1000,
            target: 17,
            value: 3.5,
        }
    }

    #[test]
    fn roundtrip() {
        let e = sample();
        assert_eq!(Envelope::decode(&e.encode()), Ok(e));
    }

    #[test]
    fn crc32_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn every_single_bit_flip_is_rejected() {
        let b = sample().encode();
        for i in 0..ENVELOPE_LEN * 8 {
            let mut c = b;
            c[i / 8] ^= 1 << (i % 8);
            assert!(Envelope::decode(&c).is_err(), "bit {i} flip accepted");
        }
    }

    #[test]
    fn strict_rejections() {
        assert_eq!(Envelope::decode(&[0u8; 31]), Err(DecodeError::BadLength));

        let reseal = |mut b: [u8; ENVELOPE_LEN]| {
            let c = crc32(&b[..28]);
            b[28..32].copy_from_slice(&c.to_le_bytes());
            b
        };
        let mut b = sample().encode();
        b[1] = 2;
        assert_eq!(
            Envelope::decode(&reseal(b)),
            Err(DecodeError::UnsupportedVersion)
        );
        let mut b = sample().encode();
        b[2] = 9;
        assert_eq!(Envelope::decode(&reseal(b)), Err(DecodeError::UnknownKind));
        let mut b = sample().encode();
        b[25] = 1;
        assert_eq!(
            Envelope::decode(&reseal(b)),
            Err(DecodeError::ReservedNonZero)
        );
        let mut e = sample();
        e.value = f32::NAN;
        assert_eq!(
            Envelope::decode(&e.encode()),
            Err(DecodeError::NonFiniteValue)
        );
    }

    #[test]
    fn replay_window() {
        let mut g = ReplayGuard::new(&[1, 2]);
        assert_eq!(g.accept(3, 0), Err(ReplayError::UnknownPeer));
        assert_eq!(g.accept(1, 10), Ok(()));
        assert_eq!(g.accept(1, 10), Err(ReplayError::Duplicate));
        assert_eq!(g.accept(1, 12), Ok(()));
        assert_eq!(g.accept(1, 11), Ok(())); // delayed, out of order
        assert_eq!(g.accept(1, 11), Err(ReplayError::Duplicate));
        assert_eq!(g.accept(1, 200), Ok(()));
        assert_eq!(g.accept(1, 12), Err(ReplayError::TooOld));
        // Peers are independent.
        assert_eq!(g.accept(2, 10), Ok(()));
    }
}
