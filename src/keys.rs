use blake2::{Blake2b, Digest, digest::consts::U32};

use crate::siphash::SipKeys;

type Blake2b256 = Blake2b<U32>;

/// Grin derives the four SipHash keys from BLAKE2b-256(pre_pow || nonce_be).
pub fn derive_keys(pre_pow: &[u8], nonce: u64) -> SipKeys {
    let mut hash = Blake2b256::new();
    hash.update(pre_pow);
    hash.update(nonce.to_be_bytes());
    let digest = hash.finalize();
    SipKeys {
        k0: u64::from_le_bytes(digest[0..8].try_into().expect("fixed slice")),
        k1: u64::from_le_bytes(digest[8..16].try_into().expect("fixed slice")),
        k2: u64::from_le_bytes(digest[16..24].try_into().expect("fixed slice")),
        k3: u64::from_le_bytes(digest[24..32].try_into().expect("fixed slice")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_native_range_harness() {
        let keys = derive_keys(&[0], 0);
        assert_eq!(keys.k0, 0xf2f4_1b02_a8b7_8751);
        assert_eq!(keys.k1, 0xe1ec_cf54_3aea_04c0);
        assert_eq!(keys.k2, 0x6323_4d62_c711_4f75);
        assert_eq!(keys.k3, 0x1e44_d1dd_4fcf_f4c7);
    }
}
