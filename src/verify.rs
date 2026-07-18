use std::collections::HashMap;

use blake2::{Blake2b, Digest, digest::consts::U32};
use thiserror::Error;

use crate::siphash::{SipKeys, endpoint};

type Blake2b256 = Blake2b<U32>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Proof {
    pub nonces: Vec<u64>,
}

impl Proof {
    pub fn sorted(mut nonces: Vec<u64>) -> Self {
        nonces.sort_unstable();
        Self { nonces }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum VerifyError {
    #[error("edge_bits must be in 1..=32")]
    BadEdgeBits,
    #[error("proof has {actual} edges, expected {expected}")]
    BadProofSize { expected: usize, actual: usize },
    #[error("proof nonce is outside the graph")]
    Range,
    #[error("proof nonces are not strictly sorted and distinct")]
    NotSortedDistinct,
    #[error("an endpoint does not have exactly one complementary mate")]
    EndpointDegree,
    #[error("proof is not one exact cycle")]
    NotSingleCycle,
}

/// Exact Cuckatoo verifier ported from m1rsi_verify42_scaffold_cycle.
pub fn verify_cycle(
    keys: SipKeys,
    edge_bits: u8,
    cycle_length: usize,
    proof: &Proof,
) -> Result<(), VerifyError> {
    if !(1..=32).contains(&edge_bits) {
        return Err(VerifyError::BadEdgeBits);
    }
    if proof.nonces.len() != cycle_length {
        return Err(VerifyError::BadProofSize {
            expected: cycle_length,
            actual: proof.nonces.len(),
        });
    }
    let limit = 1_u64 << edge_bits;
    for (index, nonce) in proof.nonces.iter().copied().enumerate() {
        if nonce >= limit {
            return Err(VerifyError::Range);
        }
        if index > 0 && proof.nonces[index - 1] >= nonce {
            return Err(VerifyError::NotSortedDistinct);
        }
    }

    let endpoints: Vec<[u32; 2]> = proof
        .nonces
        .iter()
        .map(|&edge| {
            [
                endpoint(keys, edge_bits, edge, 0),
                endpoint(keys, edge_bits, edge, 1),
            ]
        })
        .collect();
    let mut mate = vec![[usize::MAX; 2]; cycle_length];

    for i in 0..cycle_length {
        for side in 0..2 {
            let wanted = endpoints[i][side] ^ 1;
            let mut found = None;
            for (j, candidate) in endpoints.iter().enumerate() {
                if i != j && candidate[side] == wanted && found.replace(j).is_some() {
                    return Err(VerifyError::EndpointDegree);
                }
            }
            mate[i][side] = found.ok_or(VerifyError::EndpointDegree)?;
        }
    }

    let mut visited = vec![false; cycle_length];
    let mut index = 0;
    let mut side = 0;
    for _ in 0..cycle_length {
        if visited[index] {
            return Err(VerifyError::NotSingleCycle);
        }
        visited[index] = true;
        index = mate[index][side];
        side ^= 1;
    }
    if index != 0 || side != 0 || visited.iter().any(|seen| !seen) {
        return Err(VerifyError::NotSingleCycle);
    }
    Ok(())
}

pub fn graph_weight(height: u64, edge_bits: u8) -> u64 {
    const YEAR_HEIGHT: u64 = 524_160;
    const WEEK_HEIGHT: u64 = 10_080;
    const BASE_EDGE_BITS: u8 = 24;
    if edge_bits < BASE_EDGE_BITS {
        return 0;
    }
    let mut adjusted = u64::from(edge_bits);
    if edge_bits == 31 && height >= YEAR_HEIGHT {
        adjusted = adjusted.saturating_sub(1 + (height - YEAR_HEIGHT) / WEEK_HEIGHT);
    }
    (2_u64 << (edge_bits - BASE_EDGE_BITS)) * adjusted
}

pub fn proof_hash(proof: &Proof, edge_bits: u8) -> [u8; 32] {
    let bit_len = proof.nonces.len() * usize::from(edge_bits);
    let mut packed = vec![0_u8; bit_len.div_ceil(8)];
    for (i, nonce) in proof.nonces.iter().copied().enumerate() {
        let start = i * usize::from(edge_bits);
        for bit in 0..edge_bits {
            if (nonce >> bit) & 1 != 0 {
                let position = start + usize::from(bit);
                packed[position / 8] |= 1 << (position % 8);
            }
        }
    }
    Blake2b256::digest(packed).into()
}

pub fn proof_difficulty(proof: &Proof, edge_bits: u8, height: u64) -> u64 {
    let hash = proof_hash(proof, edge_bits);
    let denominator = u64::from_be_bytes(hash[0..8].try_into().expect("fixed slice")).max(1);
    let numerator = u128::from(graph_weight(height, edge_bits)) << 64;
    (numerator / u128::from(denominator)).min(u128::from(u64::MAX)) as u64
}

/// Builds endpoint lookup tables shared by CPU and GPU survivor search.
pub(crate) type EndpointLookup = HashMap<u32, Vec<usize>>;
pub(crate) type EndpointIndex = [EndpointLookup; 2];

pub(crate) fn endpoint_index(
    keys: SipKeys,
    edge_bits: u8,
    edges: &[u64],
) -> (Vec<[u32; 2]>, EndpointIndex) {
    let endpoints: Vec<_> = edges
        .iter()
        .map(|&edge| {
            [
                endpoint(keys, edge_bits, edge, 0),
                endpoint(keys, edge_bits, edge, 1),
            ]
        })
        .collect();
    let mut index = [HashMap::new(), HashMap::new()];
    for (edge_index, nodes) in endpoints.iter().enumerate() {
        for side in 0..2 {
            index[side]
                .entry(nodes[side])
                .or_insert_with(Vec::new)
                .push(edge_index);
        }
    }
    (endpoints, index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::derive_keys;

    const NONCE_45: [u64; 42] = [
        113834002, 175464874, 214250696, 260483081, 279835969, 298338403, 319521892, 416749607,
        478585563, 496642789, 543585876, 553445468, 611349361, 786110984, 851260400, 875914375,
        1185481902, 1210858959, 1489536276, 1528899199, 1646662212, 1691906112, 1897278521,
        1906375855, 2396865210, 2657593731, 2814677804, 2844237190, 2951901179, 3078404053,
        3206629131, 3219415784, 3222789694, 3256659650, 3518924379, 3634906144, 3778533919,
        3871344428, 3884396807, 3916620735, 3996475271, 4266550641,
    ];
    const NONCE_74: [u64; 42] = [
        214230679, 223185705, 244015894, 468650174, 667618686, 727065687, 761796471, 892321202,
        998276703, 1063542673, 1067574420, 1090012996, 1297340770, 1316216345, 1483296014,
        1571974979, 1726910084, 1745427983, 1757737325, 1965603751, 2086566586, 2171184816,
        2351364403, 2573372611, 2605077018, 2614064261, 2903644688, 3014168874, 3070395929,
        3154431639, 3227693524, 3298173101, 3379248045, 3390587362, 3427810150, 3434087339,
        3476427957, 3479202442, 3541814557, 3770567624, 4045656206, 4132800603,
    ];

    #[test]
    fn verifies_both_known_c32_gate_proofs() {
        for (nonce, proof) in [(45, NONCE_45), (74, NONCE_74)] {
            verify_cycle(
                derive_keys(&[0], nonce),
                32,
                42,
                &Proof {
                    nonces: proof.to_vec(),
                },
            )
            .unwrap();
        }
    }

    #[test]
    fn rejects_mutated_gate_proof() {
        let mut proof = NONCE_45;
        proof[0] += 1;
        assert!(
            verify_cycle(
                derive_keys(&[0], 45),
                32,
                42,
                &Proof {
                    nonces: proof.to_vec()
                },
            )
            .is_err()
        );
    }
}
