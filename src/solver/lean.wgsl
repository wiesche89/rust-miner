// Slean implementation based on the Cuckatoo reference algorithm.

struct U64 {
    lo: u32,
    hi: u32,
}

struct SipState {
    v0: U64,
    v1: U64,
    v2: U64,
    v3: U64,
}

struct Params {
    key_words_a: vec4<u32>,
    key_words_b: vec4<u32>,
    edge_bits: u32,
    side: u32,
    edge_count_lo: u32,
    word_count: u32,
    node_mask: u32,
    chunk_base: u32,
    chunk_count: u32,
    bucket_config: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> edges: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> nodes: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> bucket_scratch: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> bucket_scratch_second: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> slean_dead_scratch: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> bucket_scratch_third: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> bucket_scratch_fourth: array<atomic<u32>>;
@group(1) @binding(0) var<storage, read_write> fine_counts: array<atomic<u32>>;
@group(2) @binding(0) var<storage, read> fine_offsets: array<u32>;
@group(2) @binding(1) var<storage, read_write> fine_cursors: array<atomic<u32>>;
@group(2) @binding(2) var<storage, read_write> fine_arena: array<u32>;
@group(3) @binding(0) var<storage, read> fine_next_offsets: array<u32>;
@group(3) @binding(1) var<storage, read_write> fine_next_cursors: array<atomic<u32>>;
@group(3) @binding(2) var<storage, read_write> fine_next_arena: array<u32>;
var<workgroup> survivor_count_scratch: array<u32, 256>;
var<workgroup> fine_seen: array<atomic<u32>, 4096>;
// One workgroup owns a complete 32 KiB endpoint bucket.
var<workgroup> slean_seen: array<atomic<u32>, 8192>;
// A 14-bit prefix leaves a 32 KiB endpoint bitmap at C32.
var<workgroup> coarse_stage_counts: array<atomic<u32>, 64>;
var<workgroup> coarse_stage_bases: array<u32, 64>;
fn add64(a: U64, b: U64) -> U64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    return U64(lo, a.hi + b.hi + carry);
}

fn xor64(a: U64, b: U64) -> U64 {
    return U64(a.lo ^ b.lo, a.hi ^ b.hi);
}

fn rotl64(a: U64, amount: u32) -> U64 {
    if amount == 32u {
        return U64(a.hi, a.lo);
    }
    if amount < 32u {
        return U64(
            (a.lo << amount) | (a.hi >> (32u - amount)),
            (a.hi << amount) | (a.lo >> (32u - amount)),
        );
    }
    let shift = amount - 32u;
    return U64(
        (a.hi << shift) | (a.lo >> (32u - shift)),
        (a.lo << shift) | (a.hi >> (32u - shift)),
    );
}

fn sip_round(input: SipState) -> SipState {
    var v = input;
    v.v0 = add64(v.v0, v.v1);
    v.v2 = add64(v.v2, v.v3);
    v.v1 = xor64(rotl64(v.v1, 13u), v.v0);
    v.v3 = xor64(rotl64(v.v3, 16u), v.v2);
    v.v0 = rotl64(v.v0, 32u);
    v.v2 = add64(v.v2, v.v1);
    v.v0 = add64(v.v0, v.v3);
    v.v1 = xor64(rotl64(v.v1, 17u), v.v2);
    v.v3 = xor64(rotl64(v.v3, 21u), v.v0);
    v.v2 = rotl64(v.v2, 32u);
    return v;
}

// Endpoint generation only needs the low 32 hash bits.
fn sip_final_low(v: SipState) -> u32 {
    let b0 = add64(v.v0, v.v1);
    let b2 = add64(v.v2, v.v3);
    let c1 = xor64(rotl64(v.v1, 13u), b0);
    let c3 = xor64(rotl64(v.v3, 16u), b2);
    let d2 = add64(b2, c1);
    return rotl64(c1, 17u).lo ^ rotl64(c3, 21u).lo ^ d2.lo ^ d2.hi;
}

// SIPHASH_ENDPOINT_BEGIN
fn endpoint_for_side(edge: u32, side: u32) -> u32 {
    let nonce = U64((edge << 1u) | side, edge >> 31u);
    var v = SipState(
        U64(params.key_words_a.x, params.key_words_a.y),
        U64(params.key_words_a.z, params.key_words_a.w),
        U64(params.key_words_b.x, params.key_words_b.y),
        xor64(U64(params.key_words_b.z, params.key_words_b.w), nonce),
    );
    v = sip_round(v);
    v = sip_round(v);
    v.v0 = xor64(v.v0, nonce);
    v.v2 = xor64(v.v2, U64(255u, 0u));
    v = sip_round(v);
    v = sip_round(v);
    v = sip_round(v);
    var result = sip_final_low(v);
    if params.edge_bits < 32u {
        result &= (1u << params.edge_bits) - 1u;
    }
    return result;
}
// SIPHASH_ENDPOINT_END

fn endpoint(edge: u32) -> u32 {
    return endpoint_for_side(edge, params.side);
}

@compute @workgroup_size(256)
fn init_edges(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let stride = groups.x * 256u;
    var word = gid.x;
    loop {
        if word >= params.word_count { break; }
        var value = 0xffffffffu;
        let remaining = params.edge_count_lo & 31u;
        if params.edge_count_lo != 0u && word + 1u == params.word_count && remaining != 0u {
            value = (1u << remaining) - 1u;
        }
        atomicStore(&edges[word], value);
        word += stride;
    }
}

@compute @workgroup_size(256)
fn clear_nodes(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let stride = groups.x * 256u;
    var word = gid.x;
    loop {
        if word >= params.word_count { break; }
        atomicStore(&nodes[word], 0u);
        word += stride;
    }
}

// DIAGNOSTIC_KERNEL_BEGIN
// Measures endpoint hashing without graph work.
@compute @workgroup_size(256)
fn siphash_only(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    if gid.x >= params.word_count { return; }
    let stride = groups.x * 256u;
    var accumulator = gid.x;
    var word = gid.x;
    loop {
        if word >= params.word_count { break; }
        var bit = 0u;
        while bit < 32u {
            let edge = word * 32u + bit;
            if params.edge_count_lo == 0u || edge < params.edge_count_lo {
                accumulator ^= endpoint(edge);
            }
            bit += 1u;
        }
        word += stride;
    }
    atomicStore(&edges[gid.x], accumulator);
}
// DIAGNOSTIC_KERNEL_END

// Counts live edges into bucket_scratch[0].
@compute @workgroup_size(256)
fn count_alive_edges(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = groups.x * 256u;
    var word = gid.x;
    var local_count = 0u;
    loop {
        if word >= params.word_count { break; }
        local_count += countOneBits(atomicLoad(&edges[word]));
        word += stride;
    }
    survivor_count_scratch[lid.x] = local_count;
    workgroupBarrier();
    if lid.x == 0u {
        var group_count = 0u;
        var lane = 0u;
        while lane < 256u {
            group_count += survivor_count_scratch[lane];
            lane += 1u;
        }
        atomicAdd(&bucket_scratch[0], group_count);
    }
}

const FINE_BUCKETS: u32 = 32768u;
const FINE_BUCKET_BITS: u32 = 15u;

fn fine_bucket(node: u32) -> u32 {
    return node >> (params.edge_bits - FINE_BUCKET_BITS);
}

fn fine_z(node: u32) -> u32 {
    return node & ((1u << (params.edge_bits - FINE_BUCKET_BITS)) - 1u);
}

@compute @workgroup_size(256)
fn fine_histogram_alive(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let stride = groups.x * 256u;
    var word = gid.x;
    loop {
        if word >= params.word_count { break; }
        var bits = atomicLoad(&edges[word]);
        while bits != 0u {
            let bit = firstTrailingBit(bits);
            let edge = word * 32u + bit;
            let node = endpoint(edge);
            atomicAdd(&fine_counts[fine_bucket(node)], 1u);
            bits &= bits - 1u;
        }
        word += stride;
    }
}

@compute @workgroup_size(256)
fn fine_scatter_alive(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let stride = groups.x * 256u;
    var word = gid.x;
    loop {
        if word >= params.word_count { break; }
        var bits = atomicLoad(&edges[word]);
        while bits != 0u {
            let bit = firstTrailingBit(bits);
            let edge = word * 32u + bit;
            let node = endpoint(edge);
            let bucket = fine_bucket(node);
            let slot = atomicAdd(&fine_cursors[bucket], 1u);
            fine_arena[fine_offsets[bucket] + slot] = edge;
            bits &= bits - 1u;
        }
        word += stride;
    }
}

fn fine_scatter_alive_half(gid: vec3<u32>, groups: vec3<u32>, upper: bool) {
    let stride = groups.x * 256u;
    var word = gid.x;
    loop {
        if word >= params.word_count { break; }
        var bits = atomicLoad(&edges[word]);
        while bits != 0u {
            let bit = firstTrailingBit(bits);
            let edge = word * 32u + bit;
            let node = endpoint(edge);
            let global_bucket = fine_bucket(node);
            let in_half = select((global_bucket < FINE_BUCKETS / 2u),
                                 (global_bucket >= FINE_BUCKETS / 2u), upper);
            if in_half {
                let bucket = global_bucket & (FINE_BUCKETS / 2u - 1u);
                let slot = atomicAdd(&fine_cursors[bucket], 1u);
                fine_arena[fine_offsets[bucket] + slot] = edge;
            }
            bits &= bits - 1u;
        }
        word += stride;
    }
}

@compute @workgroup_size(256)
fn fine_scatter_alive_low(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    fine_scatter_alive_half(gid, groups, false);
}

@compute @workgroup_size(256)
fn fine_scatter_alive_high(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    fine_scatter_alive_half(gid, groups, true);
}

@compute @workgroup_size(256)
fn fine_verify_arena(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let bucket = group.x;
    if bucket >= FINE_BUCKETS { return; }
    let start = fine_offsets[bucket];
    let end = start + atomicLoad(&fine_cursors[bucket]);
    var index = start + lid.x;
    loop {
        if index >= end { break; }
        let edge = fine_arena[index];
        let node = endpoint(edge);
        if fine_bucket(node) != bucket {
            atomicAdd(&bucket_scratch[0], 1u);
        }
        index += 256u;
    }
}

fn fine_seen_words() -> u32 {
    return 1u << (params.edge_bits - FINE_BUCKET_BITS - 5u);
}

@compute @workgroup_size(256)
fn fine_trim_count(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let bucket = group.x;
    if bucket >= FINE_BUCKETS { return; }
    let start = fine_offsets[bucket];
    let end = start + atomicLoad(&fine_cursors[bucket]);
    var seen_word = lid.x;
    loop {
        if seen_word >= fine_seen_words() { break; }
        atomicStore(&fine_seen[seen_word], 0u);
        seen_word += 256u;
    }
    workgroupBarrier();
    var index = start + lid.x;
    loop {
        if index >= end { break; }
        let edge = fine_arena[index];
        let z = fine_z(endpoint(edge));
        atomicOr(&fine_seen[z >> 5u], 1u << (z & 31u));
        index += 256u;
    }
    workgroupBarrier();
    index = start + lid.x;
    loop {
        if index >= end { break; }
        let edge = fine_arena[index];
        let z = fine_z(endpoint(edge));
        let mate = z ^ 1u;
        if (atomicLoad(&fine_seen[mate >> 5u]) & (1u << (mate & 31u))) != 0u {
            let other = endpoint_for_side(edge, params.side ^ 1u);
            atomicAdd(&fine_counts[fine_bucket(other)], 1u);
        }
        index += 256u;
    }
}

@compute @workgroup_size(256)
fn fine_trim_scatter(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let bucket = group.x;
    if bucket >= FINE_BUCKETS { return; }
    let start = fine_offsets[bucket];
    let end = start + atomicLoad(&fine_cursors[bucket]);
    var seen_word = lid.x;
    loop {
        if seen_word >= fine_seen_words() { break; }
        atomicStore(&fine_seen[seen_word], 0u);
        seen_word += 256u;
    }
    workgroupBarrier();
    var index = start + lid.x;
    loop {
        if index >= end { break; }
        let edge = fine_arena[index];
        let z = fine_z(endpoint(edge));
        atomicOr(&fine_seen[z >> 5u], 1u << (z & 31u));
        index += 256u;
    }
    workgroupBarrier();
    index = start + lid.x;
    loop {
        if index >= end { break; }
        let edge = fine_arena[index];
        let z = fine_z(endpoint(edge));
        let mate = z ^ 1u;
        if (atomicLoad(&fine_seen[mate >> 5u]) & (1u << (mate & 31u))) != 0u {
            let other = endpoint_for_side(edge, params.side ^ 1u);
            let next_bucket = fine_bucket(other);
            let slot = atomicAdd(&fine_next_cursors[next_bucket], 1u);
            fine_next_arena[fine_next_offsets[next_bucket] + slot] = edge;
        }
        index += 256u;
    }
}

// One-pass fine trim. Overflow triggers the exact host fallback.
@compute @workgroup_size(256)
fn fine_trim_fixed(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let bucket = group.x;
    if bucket >= FINE_BUCKETS { return; }
    let start = fine_offsets[bucket];
    let end = start + atomicLoad(&fine_cursors[bucket]);
    var seen_word = lid.x;
    loop {
        if seen_word >= fine_seen_words() { break; }
        atomicStore(&fine_seen[seen_word], 0u);
        seen_word += 256u;
    }
    workgroupBarrier();
    var index = start + lid.x;
    loop {
        if index >= end { break; }
        let edge = fine_arena[index];
        let z = fine_z(endpoint(edge));
        atomicOr(&fine_seen[z >> 5u], 1u << (z & 31u));
        index += 256u;
    }
    workgroupBarrier();
    let output_capacity = fine_next_offsets[1u] - fine_next_offsets[0u];
    index = start + lid.x;
    loop {
        if index >= end { break; }
        let edge = fine_arena[index];
        let z = fine_z(endpoint(edge));
        let mate = z ^ 1u;
        if (atomicLoad(&fine_seen[mate >> 5u]) & (1u << (mate & 31u))) != 0u {
            let other = endpoint_for_side(edge, params.side ^ 1u);
            let next_bucket = fine_bucket(other);
            let slot = atomicAdd(&fine_counts[next_bucket], 1u);
            if slot < output_capacity {
                fine_next_arena[fine_next_offsets[next_bucket] + slot] = edge;
            } else {
                atomicAdd(&bucket_scratch[0], 1u);
            }
        }
        index += 256u;
    }
}

// DIAGNOSTIC_KERNEL_BEGIN
@compute @workgroup_size(256)
fn fine_emit_bitmap(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let bucket = group.x;
    if bucket >= FINE_BUCKETS { return; }
    let start = fine_offsets[bucket];
    let end = fine_offsets[bucket + 1u];
    var index = start + lid.x;
    loop {
        if index >= end { break; }
        let edge = fine_arena[index];
        atomicOr(&nodes[edge >> 5u], 1u << (edge & 31u));
        index += 256u;
    }
}

@compute @workgroup_size(256)
fn compare_edge_node_bitmaps(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let stride = groups.x * 256u;
    var word = gid.x;
    loop {
        if word >= params.word_count { break; }
        if atomicLoad(&edges[word]) != atomicLoad(&nodes[word]) {
            atomicAdd(&bucket_scratch[0], 1u);
        }
        word += stride;
    }
}
// DIAGNOSTIC_KERNEL_END

// DIAGNOSTIC_KERNEL_BEGIN
// Round-zero marking diagnostic.
@compute @workgroup_size(256)
fn dense_mark_nodes(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let stride = groups.x * 256u;
    var word = gid.x;
    loop {
        if word >= params.word_count { break; }
        var bit = 0u;
        while bit < 32u {
            let edge = word * 32u + bit;
            if params.edge_count_lo == 0u || edge < params.edge_count_lo {
                let node = endpoint(edge);
                atomicOr(&nodes[node >> 5u], 1u << (node & 31u));
            }
            bit += 1u;
        }
        word += stride;
    }
}
// DIAGNOSTIC_KERNEL_END

// Chunked round-zero bucketing diagnostic.
const DIAGNOSTIC_BUCKET_HEADER_WORDS: u32 = 256u;
const DIAGNOSTIC_BUCKET_OVERFLOW_WORD: u32 = 128u;
const DIAGNOSTIC_BUCKET_MARGIN: u32 = 1u << 16u;

fn bucket_config_count() -> u32 {
    return params.bucket_config >> 16u;
}

fn bucket_config_capacity() -> u32 {
    let buckets = bucket_config_count();
    return (params.chunk_count + buckets - 1u) / buckets
        + DIAGNOSTIC_BUCKET_MARGIN;
}

// DIAGNOSTIC_KERNEL_BEGIN
@compute @workgroup_size(256)
fn bucket_scatter_nodes(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let stride = groups.x * 256u;
    var local_edge = gid.x;
    loop {
        if local_edge >= params.chunk_count { break; }
        let edge = params.chunk_base + local_edge;
        let node = endpoint(edge);
        let bucket_count = bucket_config_count();
        let bucket_bits = countOneBits(bucket_count - 1u);
        let bucket = node >> (params.edge_bits - bucket_bits);
        let bucket_capacity = bucket_config_capacity();
        let slot = atomicAdd(&bucket_scratch[bucket], 1u);
        if slot < bucket_capacity {
            let output = DIAGNOSTIC_BUCKET_HEADER_WORDS
                + bucket * bucket_capacity + slot;
            atomicStore(&bucket_scratch[output], node);
        } else {
            atomicAdd(&bucket_scratch[DIAGNOSTIC_BUCKET_OVERFLOW_WORD], 1u);
        }
        local_edge += stride;
    }
}
// DIAGNOSTIC_KERNEL_END

// Reserve dense-round bucket space once per workgroup.
@compute @workgroup_size(256)
fn bucket_scatter_dense_nodes_staged(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = groups.x * 256u;
    let iterations = (params.chunk_count + stride - 1u) / stride;
    let bucket_count = bucket_config_count();
    let bucket_bits = countOneBits(bucket_count - 1u);
    let capacity = bucket_config_capacity();
    var iteration = 0u;
    loop {
        if iteration >= iterations { break; }
        if lid.x < bucket_count {
            atomicStore(&coarse_stage_counts[lid.x], 0u);
        }
        workgroupBarrier();
        let local_edge = group.x * 256u + lid.x + iteration * stride;
        let live = local_edge < params.chunk_count;
        var node = 0u;
        var bucket = 0u;
        var local_slot = 0u;
        if live {
            node = endpoint(params.chunk_base + local_edge);
            bucket = node >> (params.edge_bits - bucket_bits);
            local_slot = atomicAdd(&coarse_stage_counts[bucket], 1u);
        }
        workgroupBarrier();
        if lid.x < bucket_count {
            let count = atomicLoad(&coarse_stage_counts[lid.x]);
            coarse_stage_bases[lid.x] = atomicAdd(&bucket_scratch[lid.x], count);
        }
        workgroupBarrier();
        if live {
            let slot = coarse_stage_bases[bucket] + local_slot;
            if slot < capacity {
                let output = DIAGNOSTIC_BUCKET_HEADER_WORDS + bucket * capacity + slot;
                atomicStore(&bucket_scratch[output], node);
            } else {
                atomicAdd(&bucket_scratch[DIAGNOSTIC_BUCKET_OVERFLOW_WORD], 1u);
            }
        }
        workgroupBarrier();
        iteration += 1u;
    }
}

@compute @workgroup_size(256)
fn bucket_mark_nodes(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let bucket = params.bucket_config & 0xffffu;
    let bucket_capacity = bucket_config_capacity();
    let count = min(atomicLoad(&bucket_scratch[bucket]), bucket_capacity);
    let start = DIAGNOSTIC_BUCKET_HEADER_WORDS + bucket * bucket_capacity;
    let stride = groups.x * 256u;
    var slot = gid.x;
    loop {
        if slot >= count { break; }
        let node = atomicLoad(&bucket_scratch[start + slot]);
        atomicOr(&nodes[node >> 5u], 1u << (node & 31u));
        slot += stride;
    }
}

// Scatter live edges from the selected range.
@compute @workgroup_size(256)
fn bucket_scatter_alive_nodes(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let first_word = params.chunk_base >> 5u;
    let chunk_words = (params.chunk_count + 31u) >> 5u;
    let stride = groups.x * 256u;
    var local_word = gid.x;
    loop {
        if local_word >= chunk_words { break; }
        let word = first_word + local_word;
        var bits = atomicLoad(&edges[word]);
        while bits != 0u {
            let bit = firstTrailingBit(bits);
            let edge = word * 32u + bit;
            if edge >= params.chunk_base
                && edge - params.chunk_base < params.chunk_count {
                let node = endpoint(edge);
                let bucket_count = bucket_config_count();
                let bucket_bits = countOneBits(bucket_count - 1u);
                let bucket = node >> (params.edge_bits - bucket_bits);
                let bucket_capacity = bucket_config_capacity();
                let slot = atomicAdd(&bucket_scratch[bucket], 1u);
                if slot < bucket_capacity {
                    let output = DIAGNOSTIC_BUCKET_HEADER_WORDS
                        + bucket * bucket_capacity + slot;
                    atomicStore(&bucket_scratch[output], node);
                } else {
                    atomicAdd(&bucket_scratch[DIAGNOSTIC_BUCKET_OVERFLOW_WORD], 1u);
                }
            }
            bits &= bits - 1u;
        }
        local_word += stride;
    }
}

// Re-enumerate one chunk after marking and retain its edge indexes.
@compute @workgroup_size(256)
fn bucket_scatter_alive_pairs(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let first_word = params.chunk_base >> 5u;
    let chunk_words = (params.chunk_count + 31u) >> 5u;
    let stride = groups.x * 256u;
    var local_word = gid.x;
    loop {
        if local_word >= chunk_words { break; }
        let word = first_word + local_word;
        var bits = atomicLoad(&edges[word]);
        while bits != 0u {
            let bit = firstTrailingBit(bits);
            let edge = word * 32u + bit;
            if edge >= params.chunk_base
                && edge - params.chunk_base < params.chunk_count {
                let node = endpoint(edge);
                let bucket_count = bucket_config_count();
                let bucket_bits = countOneBits(bucket_count - 1u);
                let bucket = node >> (params.edge_bits - bucket_bits);
                let bucket_capacity = bucket_config_capacity();
                let slot = atomicAdd(&bucket_scratch[bucket], 1u);
                if slot < bucket_capacity {
                    let output = DIAGNOSTIC_BUCKET_HEADER_WORDS
                        + 2u * (bucket * bucket_capacity + slot);
                    atomicStore(&bucket_scratch[output], node);
                    atomicStore(&bucket_scratch[output + 1u], edge);
                } else {
                    atomicAdd(&bucket_scratch[DIAGNOSTIC_BUCKET_OVERFLOW_WORD], 1u);
                }
            }
            bits &= bits - 1u;
        }
        local_word += stride;
    }
}

@compute @workgroup_size(256)
fn bucket_scatter_dense_pairs_staged(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = groups.x * 256u;
    let iterations = (params.chunk_count + stride - 1u) / stride;
    let bucket_count = bucket_config_count();
    let bucket_bits = countOneBits(bucket_count - 1u);
    let capacity = bucket_config_capacity();
    var iteration = 0u;
    loop {
        if iteration >= iterations { break; }
        if lid.x < bucket_count {
            atomicStore(&coarse_stage_counts[lid.x], 0u);
        }
        workgroupBarrier();
        let local_edge = group.x * 256u + lid.x + iteration * stride;
        let live = local_edge < params.chunk_count;
        var edge = 0u;
        var node = 0u;
        var bucket = 0u;
        var local_slot = 0u;
        if live {
            edge = params.chunk_base + local_edge;
            node = endpoint(edge);
            bucket = node >> (params.edge_bits - bucket_bits);
            local_slot = atomicAdd(&coarse_stage_counts[bucket], 1u);
        }
        workgroupBarrier();
        if lid.x < bucket_count {
            let count = atomicLoad(&coarse_stage_counts[lid.x]);
            coarse_stage_bases[lid.x] = atomicAdd(&bucket_scratch[lid.x], count);
        }
        workgroupBarrier();
        if live {
            let slot = coarse_stage_bases[bucket] + local_slot;
            if slot < capacity {
                let output = DIAGNOSTIC_BUCKET_HEADER_WORDS + 2u * (bucket * capacity + slot);
                atomicStore(&bucket_scratch[output], node);
                atomicStore(&bucket_scratch[output + 1u], edge);
            } else {
                atomicAdd(&bucket_scratch[DIAGNOSTIC_BUCKET_OVERFLOW_WORD], 1u);
            }
        }
        workgroupBarrier();
        iteration += 1u;
    }
}

@compute @workgroup_size(256)
fn bucket_trim_pairs(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    // Overflow leaves the edge bitmap untouched for the exact fallback.
    if atomicLoad(&bucket_scratch[DIAGNOSTIC_BUCKET_OVERFLOW_WORD]) != 0u {
        return;
    }
    let bucket = params.bucket_config & 0xffffu;
    let bucket_capacity = bucket_config_capacity();
    let count = min(atomicLoad(&bucket_scratch[bucket]), bucket_capacity);
    let start = DIAGNOSTIC_BUCKET_HEADER_WORDS + 2u * bucket * bucket_capacity;
    let stride = groups.x * 256u;
    var slot = gid.x;
    loop {
        if slot >= count { break; }
        let input = start + 2u * slot;
        let node = atomicLoad(&bucket_scratch[input]);
        let edge = atomicLoad(&bucket_scratch[input + 1u]);
        let mate = node ^ 1u;
        if (atomicLoad(&nodes[mate >> 5u]) & (1u << (mate & 31u))) == 0u {
            atomicAnd(&edges[edge >> 5u], ~(1u << (edge & 31u)));
        }
        slot += stride;
    }
}

// Slean seed and trim kernels.
fn slean_header_words() -> u32 {
    return bucket_config_count() / 4u + 1u;
}

fn slean_overflow_word() -> u32 {
    return bucket_config_count() / 4u;
}

fn slean_shard_width() -> u32 {
    return bucket_config_count() / 4u;
}

fn slean_count_add(shard: u32, index: u32, value: u32) -> u32 {
    if shard == 0u { return atomicAdd(&bucket_scratch[index], value); }
    if shard == 1u { return atomicAdd(&bucket_scratch_second[index], value); }
    if shard == 2u { return atomicAdd(&bucket_scratch_third[index], value); }
    return atomicAdd(&bucket_scratch_fourth[index], value);
}

fn slean_load(shard: u32, index: u32) -> u32 {
    if shard == 0u { return atomicLoad(&bucket_scratch[index]); }
    if shard == 1u { return atomicLoad(&bucket_scratch_second[index]); }
    if shard == 2u { return atomicLoad(&bucket_scratch_third[index]); }
    return atomicLoad(&bucket_scratch_fourth[index]);
}

fn slean_store(shard: u32, index: u32, value: u32) {
    if shard == 0u { atomicStore(&bucket_scratch[index], value); }
    else if shard == 1u { atomicStore(&bucket_scratch_second[index], value); }
    else if shard == 2u { atomicStore(&bucket_scratch_third[index], value); }
    else { atomicStore(&bucket_scratch_fourth[index], value); }
}

fn slean_has_overflow() -> bool {
    let word = slean_overflow_word();
    return slean_load(0u, word) != 0u || slean_load(1u, word) != 0u
        || slean_load(2u, word) != 0u || slean_load(3u, word) != 0u;
}

fn slean_capacity() -> u32 {
    // The host uses the same 5% margin and 64-edge reserve in slean_sizing().
    // The low config bit is reserved for the overflow regression test.
    if (params.bucket_config & 1u) != 0u {
        return 1u;
    }
    let base = (params.chunk_count + bucket_config_count() - 1u)
        / bucket_config_count();
    return base + (base + 19u) / 20u + 64u;
}

fn slean_bucket(node: u32) -> u32 {
    let bits = countOneBits(bucket_config_count() - 1u);
    return node >> (params.edge_bits - bits);
}

fn slean_bucket_words() -> u32 {
    return params.word_count / bucket_config_count();
}

fn slean_dead_bucket_count() -> u32 {
    return ((params.chunk_count + 31u) >> 5u) / slean_bucket_words();
}

fn slean_dead_header_words() -> u32 {
    return slean_dead_bucket_count() + 1u;
}

fn slean_dead_overflow_word() -> u32 {
    return slean_dead_bucket_count();
}

fn slean_dead_capacity() -> u32 {
    let base = (params.chunk_count + slean_dead_bucket_count() - 1u)
        / slean_dead_bucket_count();
    return (base * 45u + 99u) / 100u + 64u;
}

fn slean_scatter_edge(edge: u32) {
    let node = endpoint(edge);
    let bucket = slean_bucket(node);
    let width = slean_shard_width();
    let shard = min(bucket / width, 3u);
    let local_bucket = bucket - shard * width;
    let slot = slean_count_add(shard, local_bucket, 1u);
    let capacity = slean_capacity();
    if slot < capacity {
        let output = slean_header_words() + local_bucket * capacity + slot;
        slean_store(shard, output, edge);
    } else {
        _ = slean_count_add(shard, slean_overflow_word(), 1u);
    }
}

@compute @workgroup_size(256)
fn slean_scatter_dense(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let stride = groups.x * 256u;
    var local_edge = gid.x;
    loop {
        if local_edge >= params.chunk_count { break; }
        slean_scatter_edge(params.chunk_base + local_edge);
        local_edge += stride;
    }
}

@compute @workgroup_size(256)
fn slean_scatter_alive(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let first_word = params.chunk_base >> 5u;
    let chunk_words = (params.chunk_count + 31u) >> 5u;
    let stride = groups.x * 256u;
    var local_word = gid.x;
    loop {
        if local_word >= chunk_words { break; }
        let word = first_word + local_word;
        var bits = atomicLoad(&edges[word]);
        while bits != 0u {
            let bit = firstTrailingBit(bits);
            let edge = word * 32u + bit;
            if edge >= params.chunk_base
                && edge - params.chunk_base < params.chunk_count {
                slean_scatter_edge(edge);
            }
            bits &= bits - 1u;
        }
        local_word += stride;
    }
}

@compute @workgroup_size(256)
fn slean_mark_buckets(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    let bucket = group.x;
    if bucket >= bucket_config_count() { return; }
    let words = slean_bucket_words();
    var word = lid.x;
    loop {
        if word >= words { break; }
        atomicStore(&slean_seen[word], 0u);
        word += 256u;
    }
    workgroupBarrier();
    let capacity = slean_capacity();
    let width = slean_shard_width();
    let shard = min(bucket / width, 3u);
    let local_bucket = bucket - shard * width;
    let count = min(slean_load(shard, local_bucket), capacity);
    let start = slean_header_words() + local_bucket * capacity;
    var slot = lid.x;
    loop {
        if slot >= count { break; }
        let edge = slean_load(shard, start + slot);
        let node = endpoint(edge);
        let z = node & (words * 32u - 1u);
        atomicOr(&slean_seen[z >> 5u], 1u << (z & 31u));
        slot += 256u;
    }
    workgroupBarrier();
    word = lid.x;
    let global_start = bucket * words;
    loop {
        if word >= words { break; }
        let marked = atomicLoad(&slean_seen[word]);
        if marked != 0u {
            // Parts are ordered, so an ordinary load/store preserves old marks.
            let index = global_start + word;
            atomicStore(&nodes[index], atomicLoad(&nodes[index]) | marked);
        }
        word += 256u;
    }
}

@compute @workgroup_size(256)
fn slean_trim_buckets(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    if slean_has_overflow() { return; }
    let bucket = group.x;
    if bucket >= bucket_config_count() { return; }
    let words = slean_bucket_words();
    let global_start = bucket * words;
    var word = lid.x;
    loop {
        if word >= words { break; }
        atomicStore(&slean_seen[word], atomicLoad(&nodes[global_start + word]));
        word += 256u;
    }
    workgroupBarrier();
    let capacity = slean_capacity();
    let width = slean_shard_width();
    let shard = min(bucket / width, 3u);
    let local_bucket = bucket - shard * width;
    let count = min(slean_load(shard, local_bucket), capacity);
    let start = slean_header_words() + local_bucket * capacity;
    var slot = lid.x;
    loop {
        if slot >= count { break; }
        let edge = slean_load(shard, start + slot);
        let node = endpoint(edge);
        let mate = (node ^ 1u) & (words * 32u - 1u);
        if (atomicLoad(&slean_seen[mate >> 5u]) & (1u << (mate & 31u))) == 0u {
            let local_edge = edge - params.chunk_base;
            let dead_bucket = local_edge / (words * 32u);
            let dead_slot = atomicAdd(&slean_dead_scratch[dead_bucket], 1u);
            let dead_capacity = slean_dead_capacity();
            if dead_slot < dead_capacity {
                atomicStore(
                    &slean_dead_scratch[
                        slean_dead_header_words() + dead_bucket * dead_capacity + dead_slot
                    ],
                    edge,
                );
            } else {
                atomicAdd(&slean_dead_scratch[slean_dead_overflow_word()], 1u);
            }
        }
        slot += 256u;
    }
}

// Merge and trim the final resident part in one pass.
@compute @workgroup_size(256)
fn slean_mark_and_trim_final_part(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    if slean_has_overflow() { return; }
    let bucket = group.x;
    if bucket >= bucket_config_count() { return; }
    let words = slean_bucket_words();
    let global_start = bucket * words;
    var word = lid.x;
    loop {
        if word >= words { break; }
        atomicStore(&slean_seen[word], atomicLoad(&nodes[global_start + word]));
        word += 256u;
    }

    let capacity = slean_capacity();
    let width = slean_shard_width();
    let shard = min(bucket / width, 3u);
    let local_bucket = bucket - shard * width;
    let count = min(slean_load(shard, local_bucket), capacity);
    let start = slean_header_words() + local_bucket * capacity;
    workgroupBarrier();

    var slot = lid.x;
    loop {
        if slot >= count { break; }
        let edge = slean_load(shard, start + slot);
        let node = endpoint(edge);
        let z = node & (words * 32u - 1u);
        atomicOr(&slean_seen[z >> 5u], 1u << (z & 31u));
        slot += 256u;
    }
    workgroupBarrier();

    word = lid.x;
    loop {
        if word >= words { break; }
        atomicStore(&nodes[global_start + word], atomicLoad(&slean_seen[word]));
        word += 256u;
    }
    workgroupBarrier();

    slot = lid.x;
    loop {
        if slot >= count { break; }
        let edge = slean_load(shard, start + slot);
        let node = endpoint(edge);
        let mate = (node ^ 1u) & (words * 32u - 1u);
        if (atomicLoad(&slean_seen[mate >> 5u]) & (1u << (mate & 31u))) == 0u {
            let local_edge = edge - params.chunk_base;
            let dead_bucket = local_edge / (words * 32u);
            let dead_slot = atomicAdd(&slean_dead_scratch[dead_bucket], 1u);
            let dead_capacity = slean_dead_capacity();
            if dead_slot < dead_capacity {
                atomicStore(
                    &slean_dead_scratch[
                        slean_dead_header_words() + dead_bucket * dead_capacity + dead_slot
                    ],
                    edge,
                );
            } else {
                atomicAdd(&slean_dead_scratch[slean_dead_overflow_word()], 1u);
            }
        }
        slot += 256u;
    }
}

@compute @workgroup_size(256)
fn slean_apply_deaths(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    if slean_has_overflow()
        || atomicLoad(&slean_dead_scratch[slean_dead_overflow_word()]) != 0u { return; }
    let bucket = group.x;
    if bucket >= slean_dead_bucket_count() { return; }
    let words = slean_bucket_words();
    let first_word = (params.chunk_base >> 5u) + bucket * words;
    var word = lid.x;
    loop {
        if word >= words { break; }
        atomicStore(&slean_seen[word], atomicLoad(&edges[first_word + word]));
        word += 256u;
    }
    workgroupBarrier();
    let capacity = slean_dead_capacity();
    let count = min(atomicLoad(&slean_dead_scratch[bucket]), capacity);
    let start = slean_dead_header_words() + bucket * capacity;
    var slot = lid.x;
    loop {
        if slot >= count { break; }
        let edge = atomicLoad(&slean_dead_scratch[start + slot]);
        let local_bit = edge - params.chunk_base - bucket * words * 32u;
        atomicAnd(&slean_seen[local_bit >> 5u], ~(1u << (local_bit & 31u)));
        slot += 256u;
    }
    workgroupBarrier();
    word = lid.x;
    loop {
        if word >= words { break; }
        atomicStore(&edges[first_word + word], atomicLoad(&slean_seen[word]));
        word += 256u;
    }
}

@compute @workgroup_size(256)
fn mark_nodes(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let stride = groups.x * 256u;
    var word = gid.x;
    loop {
        if word >= params.word_count { break; }
        var bits = atomicLoad(&edges[word]);
        while bits != 0u {
            let bit = firstTrailingBit(bits);
            let edge = word * 32u + bit;
            let node = endpoint(edge) & params.node_mask;
            atomicOr(&nodes[node >> 5u], 1u << (node & 31u));
            bits &= bits - 1u;
        }
        word += stride;
    }
}

@compute @workgroup_size(256)
fn trim_edges(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let stride = groups.x * 256u;
    var word = gid.x;
    loop {
        if word >= params.word_count { break; }
        var alive_bits = atomicLoad(&edges[word]);
        var inspect = alive_bits;
        while inspect != 0u {
            let bit = firstTrailingBit(inspect);
            let edge = word * 32u + bit;
            let mate = (endpoint(edge) ^ 1u) & params.node_mask;
            if (atomicLoad(&nodes[mate >> 5u]) & (1u << (mate & 31u))) == 0u {
                alive_bits &= ~(1u << bit);
            }
            inspect &= inspect - 1u;
        }
        atomicStore(&edges[word], alive_bits);
        word += stride;
    }
}
