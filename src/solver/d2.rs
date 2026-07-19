use crate::{
    siphash::endpoint,
    solver::{GraphParams, SolveError, cpu_lean::find_cycle},
    verify::{Proof, verify_cycle},
};

const ARC_CAP: usize = 16 * 1024 * 1024;
const MAX_SEARCH_BRANCHES: u64 = 100_000_000;

#[derive(Clone, Copy)]
struct Arc {
    dst: u32,
    mid: u32,
}

struct DfsContext<'a> {
    request: GraphParams,
    survivors: &'a [u64],
    offsets: &'a [u32],
    arcs: &'a [Arc],
    start: u32,
}

struct DfsState<'a> {
    path: &'a mut [u32; 21],
    mids: &'a mut [u32; 21],
    on_path: &'a mut [bool],
    branches: &'a mut u64,
}

/// Finds an exact cycle in the trimmed graph.
pub(crate) fn find_cycle_d2(
    request: GraphParams,
    survivors: &[u64],
) -> Result<Option<Proof>, SolveError> {
    if survivors.len() < request.cycle_length {
        return Ok(None);
    }
    if request.cycle_length != 42 {
        return find_cycle(request, survivors);
    }
    if survivors.len() > u32::MAX as usize {
        return Err(SolveError::Unsupported(
            "D2 survivor index exceeds u32".into(),
        ));
    }

    let mut key_u = Vec::with_capacity(survivors.len());
    let mut key_v = Vec::with_capacity(survivors.len());
    for &edge in survivors {
        key_u.push(endpoint(request.keys, request.edge_bits, edge, 0));
        key_v.push(endpoint(request.keys, request.edge_bits, edge, 1));
    }
    let order_u = radix_order(&key_u);
    let order_v = radix_order(&key_v);

    let mut offsets = Vec::with_capacity(survivors.len() + 1);
    let mut arcs = Vec::with_capacity(survivors.len().min(ARC_CAP));
    for src in 0..survivors.len() {
        offsets.push(arcs.len() as u32);
        let (u_begin, u_end) = equal_range(&order_u, &key_u, key_u[src] ^ 1);
        for &mid in &order_u[u_begin..u_end] {
            if mid as usize == src {
                continue;
            }
            let (v_begin, v_end) = equal_range(&order_v, &key_v, key_v[mid as usize] ^ 1);
            for &dst in &order_v[v_begin..v_end] {
                if dst as usize == src {
                    continue;
                }
                if arcs.len() == ARC_CAP {
                    // Overflow makes the verdict unknowable, never a negative proof.
                    return Err(SolveError::SearchLimit(format!(
                        "D2 adjacency exceeded {ARC_CAP} arcs"
                    )));
                }
                arcs.push(Arc { dst, mid });
            }
        }
    }
    offsets.push(arcs.len() as u32);

    let mut path = [0_u32; 21];
    let mut mids = [0_u32; 21];
    let mut on_path = vec![false; survivors.len()];
    let mut branches = 0;
    for start in 0..survivors.len() as u32 {
        if offsets[start as usize] == offsets[start as usize + 1] {
            continue;
        }
        path[0] = start;
        on_path[start as usize] = true;
        let context = DfsContext {
            request,
            survivors,
            offsets: &offsets,
            arcs: &arcs,
            start,
        };
        let mut state = DfsState {
            path: &mut path,
            mids: &mut mids,
            on_path: &mut on_path,
            branches: &mut branches,
        };
        if let Some(proof) = dfs(&context, start, 0, &mut state)? {
            return Ok(Some(proof));
        }
        on_path[start as usize] = false;
    }
    Ok(None)
}

fn dfs(
    context: &DfsContext<'_>,
    current: u32,
    depth: usize,
    state: &mut DfsState<'_>,
) -> Result<Option<Proof>, SolveError> {
    let begin = context.offsets[current as usize] as usize;
    let end = context.offsets[current as usize + 1] as usize;
    for arc in &context.arcs[begin..end] {
        *state.branches += 1;
        if *state.branches > MAX_SEARCH_BRANCHES {
            return Err(SolveError::SearchLimit(format!(
                "more than {MAX_SEARCH_BRANCHES} branches in D2 search"
            )));
        }
        let steps = depth + 1;
        if arc.dst == context.start {
            if steps == 21 {
                state.mids[depth] = arc.mid;
                let mut nonces = Vec::with_capacity(42);
                for i in 0..21 {
                    nonces.push(context.survivors[state.path[i] as usize]);
                    nonces.push(context.survivors[state.mids[i] as usize]);
                }
                let proof = Proof::sorted(nonces);
                if verify_cycle(context.request.keys, context.request.edge_bits, 42, &proof).is_ok()
                {
                    return Ok(Some(proof));
                }
            }
            continue;
        }
        if steps == 21 || state.on_path[arc.dst as usize] {
            continue;
        }
        state.mids[depth] = arc.mid;
        state.path[steps] = arc.dst;
        state.on_path[arc.dst as usize] = true;
        if let Some(proof) = dfs(context, arc.dst, steps, state)? {
            return Ok(Some(proof));
        }
        state.on_path[arc.dst as usize] = false;
    }
    Ok(None)
}

pub(crate) fn radix_order(keys: &[u32]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..keys.len() as u32).collect();
    let mut scratch = vec![0_u32; keys.len()];
    let mut counts = vec![0_usize; 1 << 16];
    for shift in [0, 16] {
        counts.fill(0);
        for &index in &order {
            counts[((keys[index as usize] >> shift) & 0xffff) as usize] += 1;
        }
        let mut position = 0;
        for count in &mut counts {
            let next = position + *count;
            *count = position;
            position = next;
        }
        for &index in &order {
            let bucket = ((keys[index as usize] >> shift) & 0xffff) as usize;
            scratch[counts[bucket]] = index;
            counts[bucket] += 1;
        }
        std::mem::swap(&mut order, &mut scratch);
    }
    order
}

fn equal_range(order: &[u32], keys: &[u32], wanted: u32) -> (usize, usize) {
    let lower = order.partition_point(|&index| keys[index as usize] < wanted);
    let upper = order[lower..].partition_point(|&index| keys[index as usize] == wanted) + lower;
    (lower, upper)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix_matches_sort() {
        let keys = [7, 1, u32::MAX, 7, 0, 65_537, 1];
        let order = radix_order(&keys);
        let sorted: Vec<_> = order.iter().map(|&i| keys[i as usize]).collect();
        let mut expected = keys.to_vec();
        expected.sort_unstable();
        assert_eq!(sorted, expected);
        assert_eq!(equal_range(&order, &keys, 7), (3, 5));
    }
}
