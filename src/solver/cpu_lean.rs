use std::{
    collections::HashSet,
    sync::atomic::{AtomicBool, Ordering},
};

use crate::{
    siphash::endpoint,
    solver::{
        BackendCapabilities, GraphParams, SolveError, SolveOutcome, SolveRequest, Solver,
        validate_request as validate_work_request,
    },
    verify::{Proof, endpoint_index, verify_cycle},
};

const DEFAULT_MAX_EDGE_BITS: u8 = 28;
const MAX_SEARCH_BRANCHES: u64 = 100_000_000;

pub struct CpuLeanSolver {
    max_edge_bits: u8,
}

impl Default for CpuLeanSolver {
    fn default() -> Self {
        Self {
            max_edge_bits: DEFAULT_MAX_EDGE_BITS,
        }
    }
}

impl CpuLeanSolver {
    pub fn with_max_edge_bits(max_edge_bits: u8) -> Self {
        Self { max_edge_bits }
    }
}

impl Solver for CpuLeanSolver {
    fn name(&self) -> &'static str {
        "cpu-lean"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            min_edge_bits: 1,
            max_edge_bits: self.max_edge_bits,
            cycle_length: 42,
        }
    }

    fn solve(
        &mut self,
        request: SolveRequest,
        cancel: &AtomicBool,
    ) -> Result<SolveOutcome, SolveError> {
        validate_work_request(&request, self.capabilities())?;
        let params = request.graph_params();
        let Some(survivors) = trim_survivors_cancellable(params, self.max_edge_bits, cancel)?
        else {
            return Ok(SolveOutcome::Cancelled);
        };
        if cancel.load(Ordering::Relaxed) {
            return Ok(SolveOutcome::Cancelled);
        }
        match find_cycle(params, &survivors) {
            Ok(Some(proof)) => Ok(SolveOutcome::Proof(proof)),
            Ok(None) => Ok(SolveOutcome::NoCycle),
            Err(SolveError::SearchLimit(reason)) => Ok(SolveOutcome::Inconclusive(reason)),
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
pub(crate) fn trim_survivors(
    request: GraphParams,
    max_edge_bits: u8,
) -> Result<Vec<u64>, SolveError> {
    trim_survivors_cancellable(request, max_edge_bits, &crate::solver::NEVER_CANCEL)?.ok_or_else(
        || SolveError::Unsupported("non-cancellable trim was unexpectedly cancelled".into()),
    )
}

fn trim_survivors_cancellable(
    request: GraphParams,
    max_edge_bits: u8,
    cancel: &AtomicBool,
) -> Result<Option<Vec<u64>>, SolveError> {
    if request.edge_bits > max_edge_bits {
        return Err(SolveError::Unsupported(format!(
            "CPU backend is capped at edge_bits={max_edge_bits}; use --backend wgpu for C32"
        )));
    }
    let edge_count = 1_u64 << request.edge_bits;
    let word_count = usize::try_from(edge_count.div_ceil(64))
        .map_err(|_| SolveError::Unsupported("graph does not fit this CPU address space".into()))?;
    let mut alive = vec![u64::MAX; word_count];
    if edge_count < 64 {
        alive[0] = (1_u64 << edge_count) - 1;
    }
    let mut occupied = vec![0_u64; word_count];
    for round in 0..request.rounds {
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
        occupied.fill(0);
        let side = (round & 1) as u8;
        for edge in 0..edge_count {
            if alive[edge as usize / 64] & (1_u64 << (edge % 64)) == 0 {
                continue;
            }
            let node = endpoint(request.keys, request.edge_bits, edge, side) as usize;
            occupied[node / 64] |= 1_u64 << (node % 64);
        }
        let mut killed = 0_usize;
        for edge in 0..edge_count {
            let word = edge as usize / 64;
            let bit = 1_u64 << (edge % 64);
            if alive[word] & bit == 0 {
                continue;
            }
            let mate = (endpoint(request.keys, request.edge_bits, edge, side) ^ 1) as usize;
            if occupied[mate / 64] & (1_u64 << (mate % 64)) == 0 {
                alive[word] &= !bit;
                killed += 1;
            }
        }
        if killed == 0 {
            break;
        }
    }

    let mut survivors = Vec::new();
    for (word_index, mut word) in alive.into_iter().enumerate() {
        while word != 0 {
            let bit = word.trailing_zeros();
            let edge = word_index as u64 * 64 + u64::from(bit);
            if edge < edge_count {
                survivors.push(edge);
            }
            word &= word - 1;
        }
    }
    Ok(Some(survivors))
}

pub(crate) fn find_cycle(
    request: GraphParams,
    survivors: &[u64],
) -> Result<Option<Proof>, SolveError> {
    if survivors.len() < request.cycle_length {
        return Ok(None);
    }
    let (nodes, index) = endpoint_index(request.keys, request.edge_bits, survivors);
    let mut used = HashSet::with_capacity(request.cycle_length);
    let mut path = Vec::with_capacity(request.cycle_length);

    struct Search<'a> {
        request: GraphParams,
        survivors: &'a [u64],
        nodes: &'a [[u32; 2]],
        index: &'a crate::verify::EndpointIndex,
    }

    fn dfs(
        search: &Search<'_>,
        start: usize,
        current: usize,
        side: usize,
        used: &mut HashSet<usize>,
        path: &mut Vec<usize>,
        branches: &mut u64,
    ) -> Result<Option<Proof>, ()> {
        if path.len() == search.request.cycle_length {
            let closes = search.index[side]
                .get(&(search.nodes[current][side] ^ 1))
                .is_some_and(|candidates| candidates.contains(&start));
            if !closes {
                return Ok(None);
            }
            let proof = Proof::sorted(path.iter().map(|&i| search.survivors[i]).collect());
            return Ok(verify_cycle(
                search.request.keys,
                search.request.edge_bits,
                search.request.cycle_length,
                &proof,
            )
            .is_ok()
            .then_some(proof));
        }

        let wanted = search.nodes[current][side] ^ 1;
        for &next in search.index[side].get(&wanted).into_iter().flatten() {
            *branches += 1;
            if *branches > MAX_SEARCH_BRANCHES {
                return Err(());
            }
            if used.insert(next) {
                path.push(next);
                if let Some(proof) = dfs(search, start, next, side ^ 1, used, path, branches)? {
                    return Ok(Some(proof));
                }
                path.pop();
                used.remove(&next);
            }
        }
        Ok(None)
    }

    let search = Search {
        request,
        survivors,
        nodes: &nodes,
        index: &index,
    };
    let mut branches = 0;
    for start in 0..survivors.len() {
        used.clear();
        path.clear();
        used.insert(start);
        path.push(start);
        let result = dfs(
            &search,
            start,
            start,
            0,
            &mut used,
            &mut path,
            &mut branches,
        )
        .map_err(|()| {
            SolveError::SearchLimit(format!(
                "more than {MAX_SEARCH_BRANCHES} branches in generic cycle search"
            ))
        })?;
        if let Some(proof) = result {
            return Ok(Some(proof));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::NEVER_CANCEL;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn small_graph_proofs_verify() {
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce: 0,
            live_work: false,
            edge_bits: 12,
            cycle_length: 4,
            rounds: 20,
        };
        let params = request.graph_params();
        let result = CpuLeanSolver::default()
            .solve(request, &NEVER_CANCEL)
            .unwrap();
        if let SolveOutcome::Proof(proof) = result {
            verify_cycle(params.keys, params.edge_bits, params.cycle_length, &proof).unwrap();
        }
    }

    #[test]
    fn rejects_edge_bits_before_allocating() {
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce: 0,
            live_work: false,
            edge_bits: 0,
            cycle_length: 42,
            rounds: 160,
        };
        assert!(matches!(
            CpuLeanSolver::default().solve(request, &NEVER_CANCEL),
            Err(SolveError::InvalidConfig(_))
        ));
    }

    #[test]
    fn pre_cancel_returns_cancelled() {
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce: 0,
            live_work: false,
            edge_bits: 12,
            cycle_length: 4,
            rounds: 20,
        };
        let cancel = AtomicBool::new(true);
        assert!(matches!(
            CpuLeanSolver::default().solve(request, &cancel).unwrap(),
            SolveOutcome::Cancelled
        ));
    }
}
