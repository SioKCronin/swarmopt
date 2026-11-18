//! CRC-32C (Castagnoli), table-driven, computed at compile time.
//!
//! CRC-32C detects all burst errors up to 32 bits and every odd number of
//! bit flips, which is the failure mode we care about most: single-event
//! upsets from radiation flipping bits in storage or RAM.

const POLY: u32 = 0x82F6_3B78; // reflected Castagnoli polynomial

const fn make_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ POLY
            } else {
                crc >> 1
            };
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static TABLE: [u32; 256] = make_table();

/// Incremental CRC-32C.
#[derive(Clone, Copy, Debug)]
pub struct Crc32c(u32);

impl Default for Crc32c {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32c {
    pub const fn new() -> Self {
        Crc32c(0xFFFF_FFFF)
    }

    pub fn update(&mut self, data: &[u8]) {
        let mut crc = self.0;
        for &b in data {
            crc = TABLE[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
        self.0 = crc;
    }

    pub fn finish(self) -> u32 {
        self.0 ^ 0xFFFF_FFFF
    }
}

/// One-shot CRC-32C of `data`.
pub fn crc32c(data: &[u8]) -> u32 {
    let mut c = Crc32c::new();
    c.update(data);
    c.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        // RFC 3720 (iSCSI) test vectors.
        assert_eq!(crc32c(b""), 0);
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(&[0u8; 32]), 0x8A91_36AA);
        assert_eq!(crc32c(&[0xFFu8; 32]), 0x62A8_AB43);
    }

    #[test]
    fn incremental_matches_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 31 % 251) as u8).collect();
        let mut c = Crc32c::new();
        for chunk in data.chunks(7) {
            c.update(chunk);
        }
        assert_eq!(c.finish(), crc32c(&data));
    }
}
