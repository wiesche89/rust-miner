use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SipKeys {
    pub k0: u64,
    pub k1: u64,
    pub k2: u64,
    pub k3: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuSipKeys {
    pub words: [u32; 8],
}

impl From<SipKeys> for GpuSipKeys {
    fn from(keys: SipKeys) -> Self {
        let split = |value: u64| [value as u32, (value >> 32) as u32];
        let a = split(keys.k0);
        let b = split(keys.k1);
        let c = split(keys.k2);
        let d = split(keys.k3);
        Self {
            words: [a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]],
        }
    }
}

#[inline]
fn sip_round(v: &mut [u64; 4]) {
    v[0] = v[0].wrapping_add(v[1]);
    v[2] = v[2].wrapping_add(v[3]);
    v[1] = v[1].rotate_left(13) ^ v[0];
    v[3] = v[3].rotate_left(16) ^ v[2];
    v[0] = v[0].rotate_left(32);
    v[2] = v[2].wrapping_add(v[1]);
    v[0] = v[0].wrapping_add(v[3]);
    v[1] = v[1].rotate_left(17) ^ v[2];
    v[3] = v[3].rotate_left(21) ^ v[0];
    v[2] = v[2].rotate_left(32);
}

#[inline]
pub fn siphash24(keys: SipKeys, nonce: u64) -> u64 {
    let mut v = [keys.k0, keys.k1, keys.k2, keys.k3 ^ nonce];
    sip_round(&mut v);
    sip_round(&mut v);
    v[0] ^= nonce;
    v[2] ^= 0xff;
    sip_round(&mut v);
    sip_round(&mut v);
    sip_round(&mut v);
    sip_round(&mut v);
    v[0] ^ v[1] ^ v[2] ^ v[3]
}

#[inline]
pub fn endpoint(keys: SipKeys, edge_bits: u8, edge: u64, side: u8) -> u32 {
    let mask = if edge_bits == 32 {
        u32::MAX as u64
    } else {
        (1_u64 << edge_bits) - 1
    };
    (siphash24(keys, edge.wrapping_mul(2) + u64::from(side & 1)) & mask) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimized_endpoint_is_masked_siphash() {
        let keys = SipKeys {
            k0: 1,
            k1: 2,
            k2: 3,
            k3: 4,
        };
        assert_eq!(
            endpoint(keys, 20, 7, 1),
            (siphash24(keys, 15) & ((1 << 20) - 1)) as u32
        );
    }
}
