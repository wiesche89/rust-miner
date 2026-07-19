use std::collections::VecDeque;

use crate::{
    siphash::endpoint,
    solver::{GraphParams, SolveError, d2::radix_order},
};

/// Removes edges without a live mate until the bipartite core is stable.
pub(crate) fn peel_two_core(
    request: GraphParams,
    survivors: &[u64],
) -> Result<Vec<u64>, SolveError> {
    if survivors.len() > u32::MAX as usize {
        return Err(SolveError::Unsupported(
            "2-core survivor index exceeds u32".into(),
        ));
    }
    if survivors.is_empty() {
        return Ok(Vec::new());
    }

    let mut edge_groups = vec![[0_u32; 2]; survivors.len()];
    let mut group_keys = Vec::<u32>::with_capacity(survivors.len());
    let mut side_offsets = [vec![0_u32], vec![0_u32]];
    let mut side_edges = [
        Vec::<u32>::with_capacity(survivors.len()),
        Vec::<u32>::with_capacity(survivors.len()),
    ];
    let mut side_group_starts = [0_usize; 3];
    let endpoint_pairs: Vec<[u32; 2]> = survivors
        .iter()
        .map(|&edge| {
            [
                endpoint(request.keys, request.edge_bits, edge, 0),
                endpoint(request.keys, request.edge_bits, edge, 1),
            ]
        })
        .collect();

    // Radix order makes each endpoint group contiguous.
    for side in [0_usize, 1] {
        let keys: Vec<u32> = endpoint_pairs.iter().map(|pair| pair[side]).collect();
        let order = radix_order(&keys);
        let mut begin = 0;
        while begin < order.len() {
            let key = keys[order[begin] as usize];
            let mut end = begin + 1;
            while end < order.len() && keys[order[end] as usize] == key {
                end += 1;
            }
            let group = group_keys.len() as u32;
            group_keys.push(key);
            for &edge in &order[begin..end] {
                edge_groups[edge as usize][side] = group;
                side_edges[side].push(edge);
            }
            side_offsets[side].push(side_edges[side].len() as u32);
            begin = end;
        }
        side_group_starts[side + 1] = group_keys.len();
    }

    let mut mate_group = vec![None; group_keys.len()];
    for side in 0..2 {
        let start = side_group_starts[side];
        let end = side_group_starts[side + 1];
        for group in start..end {
            let neighbor = if group_keys[group].is_multiple_of(2) {
                (group + 1 < end).then_some(group + 1)
            } else {
                group.checked_sub(1).filter(|&index| index >= start)
            };
            mate_group[group] = neighbor
                .filter(|&index| group_keys[index] == group_keys[group] ^ 1)
                .map(|index| index as u32);
        }
    }

    let mut live_count = Vec::with_capacity(group_keys.len());
    for offsets in &side_offsets {
        live_count.extend(offsets.windows(2).map(|range| range[1] - range[0]));
    }
    let mut alive = vec![true; survivors.len()];
    let mut queued = vec![false; group_keys.len()];
    let mut queue = VecDeque::new();
    for group in 0..group_keys.len() {
        if mate_group[group].is_none_or(|mate| live_count[mate as usize] == 0) {
            queued[group] = true;
            queue.push_back(group as u32);
        }
    }

    while let Some(group) = queue.pop_front() {
        let group = group as usize;
        let side = usize::from(group >= side_group_starts[1]);
        let local_group = group - side_group_starts[side];
        let begin = side_offsets[side][local_group] as usize;
        let end = side_offsets[side][local_group + 1] as usize;
        for &edge in &side_edges[side][begin..end] {
            let edge = edge as usize;
            if !alive[edge] {
                continue;
            }
            alive[edge] = false;
            for &incident in &edge_groups[edge] {
                let incident = incident as usize;
                live_count[incident] -= 1;
                if live_count[incident] == 0
                    && let Some(mate) = mate_group[incident]
                {
                    let mate = mate as usize;
                    if !queued[mate] {
                        queued[mate] = true;
                        queue.push_back(mate as u32);
                    }
                }
            }
        }
    }

    Ok(survivors
        .iter()
        .zip(alive)
        .filter_map(|(&edge, alive)| alive.then_some(edge))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{keys::derive_keys, solver::cpu_lean::trim_survivors};

    #[test]
    fn peel_reaches_a_fixed_core() {
        let request = GraphParams {
            keys: derive_keys(&[0], 7),
            edge_bits: 12,
            cycle_length: 4,
            rounds: 4,
        };
        let survivors = trim_survivors(request, 12).unwrap();
        let peeled = peel_two_core(request, &survivors).unwrap();
        assert!(peeled.len() <= survivors.len());
        assert_eq!(peel_two_core(request, &peeled).unwrap(), peeled);

        let converged = trim_survivors(
            GraphParams {
                rounds: 200,
                ..request
            },
            12,
        )
        .unwrap();
        assert_eq!(peeled, converged);
    }
}
