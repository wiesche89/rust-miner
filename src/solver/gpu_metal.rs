//! macOS GPU backend boundary.
//!
//! The public solver type is isolated from the portable wgpu backend so both
//! implementations share the CLI and coordinator without sharing GPU state.

use std::sync::atomic::{AtomicBool, Ordering};

use super::{
    BackendCapabilities, SolveError, SolveOutcome, SolveRequest, Solver, d2::find_cycle_d2,
    peel::peel_two_core, validate_request,
};
use crate::solver::gpu_wgpu::GpuWgpuConfig;

#[cfg(target_os = "macos")]
mod native {
    use std::{
        ffi::c_void,
        ptr::NonNull,
        sync::atomic::{AtomicBool, Ordering},
        time::Instant,
    };

    use objc2::{rc::Retained, runtime::ProtocolObject};
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
    };

    use super::SolveError;
    use crate::siphash::{SipKeys, endpoint};

    #[link(name = "CoreGraphics", kind = "framework")]
    unsafe extern "C" {}

    const PROBE_SOURCE: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void grin_miner_native_probe(
            device uint *values [[buffer(0)]],
            uint gid [[thread_position_in_grid]])
        {
            values[gid] = values[gid] * 1664525u + 1013904223u + gid;
        }

        struct EndpointParams {
            ulong k0;
            ulong k1;
            ulong k2;
            ulong k3;
            uint edge_bits;
            uint side;
            uint edge_base;
            uint edge_count;
            uint capacity;
            uint destination_capacity;
        };

        struct U64Words { uint lo; uint hi; };
        struct SipState32 { U64Words v0; U64Words v1; U64Words v2; U64Words v3; };

        inline U64Words words(const ulong value) {
            return U64Words{uint(value), uint(value >> 32u)};
        }

        inline U64Words add_words(const U64Words a, const U64Words b) {
            const uint lo = a.lo + b.lo;
            return U64Words{lo, a.hi + b.hi + uint(lo < a.lo)};
        }

        inline U64Words xor_words(const U64Words a, const U64Words b) {
            return U64Words{a.lo ^ b.lo, a.hi ^ b.hi};
        }

        inline U64Words rotl_words(const U64Words a, const uint amount) {
            if (amount == 32u) return U64Words{a.hi, a.lo};
            if (amount < 32u) {
                return U64Words{
                    (a.lo << amount) | (a.hi >> (32u - amount)),
                    (a.hi << amount) | (a.lo >> (32u - amount))};
            }
            const uint shift = amount - 32u;
            return U64Words{
                (a.hi << shift) | (a.lo >> (32u - shift)),
                (a.lo << shift) | (a.hi >> (32u - shift))};
        }

        inline SipState32 sip_round32(SipState32 v) {
            v.v0 = add_words(v.v0, v.v1);
            v.v2 = add_words(v.v2, v.v3);
            v.v1 = xor_words(rotl_words(v.v1, 13u), v.v0);
            v.v3 = xor_words(rotl_words(v.v3, 16u), v.v2);
            v.v0 = rotl_words(v.v0, 32u);
            v.v2 = add_words(v.v2, v.v1);
            v.v0 = add_words(v.v0, v.v3);
            v.v1 = xor_words(rotl_words(v.v1, 17u), v.v2);
            v.v3 = xor_words(rotl_words(v.v3, 21u), v.v0);
            v.v2 = rotl_words(v.v2, 32u);
            return v;
        }

        inline uint sip_final_low(const SipState32 v) {
            const U64Words b0 = add_words(v.v0, v.v1);
            const U64Words b2 = add_words(v.v2, v.v3);
            const U64Words c1 = xor_words(rotl_words(v.v1, 13u), b0);
            const U64Words c3 = xor_words(rotl_words(v.v3, 16u), b2);
            const U64Words d2 = add_words(b2, c1);
            return rotl_words(c1, 17u).lo ^ rotl_words(c3, 21u).lo ^ d2.lo ^ d2.hi;
        }

        inline uint cuckatoo_endpoint_side(const uint edge,
                                           constant EndpointParams &params,
                                           const uint side) {
            const U64Words nonce = U64Words{(edge << 1u) | (side & 1u), edge >> 31u};
            SipState32 v = SipState32{
                words(params.k0), words(params.k1), words(params.k2),
                xor_words(words(params.k3), nonce)};
            v = sip_round32(v);
            v = sip_round32(v);
            v.v0 = xor_words(v.v0, nonce);
            v.v2 = xor_words(v.v2, U64Words{255u, 0u});
            v = sip_round32(v);
            v = sip_round32(v);
            v = sip_round32(v);
            uint result = sip_final_low(v);
            if (params.edge_bits < 32u) {
                result &= (1u << params.edge_bits) - 1u;
            }
            return result;
        }

        inline uint cuckatoo_endpoint(const uint edge,
                                      constant EndpointParams &params) {
            return cuckatoo_endpoint_side(edge, params, params.side);
        }

        kernel void grin_miner_endpoints(
            device uint *nodes [[buffer(0)]],
            constant EndpointParams &params [[buffer(1)]],
            uint gid [[thread_position_in_grid]])
        {
            if (gid >= params.edge_count) return;
            nodes[gid] = cuckatoo_endpoint(params.edge_base + gid, params);
        }

        kernel void grin_miner_clear_nodes(
            device atomic_uint *nodes [[buffer(0)]],
            constant EndpointParams &params [[buffer(1)]],
            uint gid [[thread_position_in_grid]])
        {
            const uint word_count = 1u << (params.edge_bits - 5u);
            if (gid < word_count) {
                atomic_store_explicit(&nodes[gid], 0u, memory_order_relaxed);
            }
        }

        kernel void grin_miner_clear_bucket_counts(
            device atomic_uint *counts [[buffer(0)]],
            constant EndpointParams &params [[buffer(1)]],
            uint gid [[thread_position_in_grid]])
        {
            const uint bucket_shift = params.edge_bits > 18u ? 18u : params.edge_bits;
            const uint bucket_count = 1u << (params.edge_bits - bucket_shift);
            if (gid < bucket_count) {
                atomic_store_explicit(&counts[gid], 0u, memory_order_relaxed);
            }
        }

        kernel void grin_miner_clear_dead_counts(
            device atomic_uint *counts [[buffer(0)]],
            constant EndpointParams &params [[buffer(1)]],
            uint gid [[thread_position_in_grid]])
        {
            const uint count = max(1u, params.edge_count >> 18u);
            if (gid < count) {
                atomic_store_explicit(&counts[gid], 0u, memory_order_relaxed);
            }
        }

        kernel void grin_miner_bucket_seed(
            device uint *arena [[buffer(0)]],
            device atomic_uint *counts [[buffer(1)]],
            device atomic_uint *overflow [[buffer(2)]],
            constant EndpointParams &params [[buffer(3)]],
            uint gid [[thread_position_in_grid]])
        {
            if (gid >= params.edge_count) return;
            const uint edge = params.edge_base + gid;
            const uint node = cuckatoo_endpoint(edge, params);
            // edge_base is aligned to the part size; edge_count and the
            // endpoint prefix determine the same 1.05x capacity as the host.
            const uint bucket_shift = params.edge_bits > 18u ? 18u : params.edge_bits;
            const uint bucket = node >> bucket_shift;
            const uint capacity = params.capacity;
            const uint slot = atomic_fetch_add_explicit(
                &counts[bucket], 1u, memory_order_relaxed);
            if (slot < capacity) {
                arena[bucket * capacity + slot] = edge;
            } else {
                atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
            }
        }

        kernel void grin_miner_bucket_seed_alive_words(
            device uint *arena [[buffer(0)]],
            device atomic_uint *counts [[buffer(1)]],
            device atomic_uint *overflow [[buffer(2)]],
            constant EndpointParams &params [[buffer(3)]],
            device const uint *edges [[buffer(4)]],
            uint gid [[thread_position_in_grid]])
        {
            const uint word_count = (params.edge_count + 31u) >> 5u;
            const uint first_word = gid;
            if (first_word >= word_count) return;
            const uint bucket_shift = params.edge_bits > 18u ? 18u : params.edge_bits;
            for (uint word_offset = 0u; word_offset < 1u; ++word_offset) {
                const uint local_word = first_word + word_offset;
                if (local_word >= word_count) break;
                const uint first_edge = params.edge_base + local_word * 32u;
                uint alive = edges[first_edge >> 5u];
                while (alive != 0u) {
                    const uint bit = ctz(alive);
                    const uint edge = first_edge + bit;
                    if (edge - params.edge_base < params.edge_count) {
                        const uint node = cuckatoo_endpoint(edge, params);
                        const uint bucket = node >> bucket_shift;
                        const uint slot = atomic_fetch_add_explicit(
                            &counts[bucket], 1u, memory_order_relaxed);
                        if (slot < params.capacity) {
                            arena[bucket * params.capacity + slot] = edge;
                        } else {
                            atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
                        }
                    }
                    alive &= alive - 1u;
                }
            }
        }

        kernel void grin_miner_bucket_mark(
            device const uint *arena [[buffer(0)]],
            device const uint *counts [[buffer(1)]],
            device atomic_uint *nodes [[buffer(2)]],
            constant EndpointParams &params [[buffer(3)]],
            uint lid [[thread_position_in_threadgroup]],
            uint bucket [[threadgroup_position_in_grid]],
            uint group_size [[threads_per_threadgroup]])
        {
            threadgroup atomic_uint seen[8192];
            for (uint word = lid; word < 8192u; word += group_size) {
                atomic_store_explicit(&seen[word], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint bucket_shift = params.edge_bits > 18u ? 18u : params.edge_bits;
            const uint capacity = params.capacity;
            const uint count = min(counts[bucket], capacity);
            for (uint slot = lid; slot < count; slot += group_size) {
                const uint edge = arena[bucket * capacity + slot];
                const uint node = cuckatoo_endpoint(edge, params);
                const uint suffix = node & ((1u << bucket_shift) - 1u);
                atomic_fetch_or_explicit(
                    &seen[suffix >> 5u], 1u << (suffix & 31u),
                    memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint words_per_bucket = 1u << (bucket_shift - 5u);
            for (uint word = lid; word < words_per_bucket; word += group_size) {
                const uint value = atomic_load_explicit(&seen[word], memory_order_relaxed);
                atomic_fetch_or_explicit(
                    &nodes[bucket * words_per_bucket + word], value,
                    memory_order_relaxed);
            }
        }

        kernel void grin_miner_bucket_trim_to_bitmap(
            device const uint *arena [[buffer(0)]],
            device const uint *counts [[buffer(1)]],
            device const uint *nodes [[buffer(2)]],
            device atomic_uint *survivors [[buffer(3)]],
            constant EndpointParams &params [[buffer(4)]],
            uint lid [[thread_position_in_threadgroup]],
            uint bucket [[threadgroup_position_in_grid]],
            uint group_size [[threads_per_threadgroup]])
        {
            threadgroup uint seen[8192];
            const uint bucket_shift = params.edge_bits > 18u ? 18u : params.edge_bits;
            const uint words_per_bucket = 1u << (bucket_shift - 5u);
            for (uint word = lid; word < words_per_bucket; word += group_size) {
                seen[word] = nodes[bucket * words_per_bucket + word];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint capacity = params.capacity;
            const uint count = min(counts[bucket], capacity);
            for (uint slot = lid; slot < count; slot += group_size) {
                const uint edge = arena[bucket * capacity + slot];
                const uint node = cuckatoo_endpoint(edge, params);
                const uint mate = (node ^ 1u) & ((1u << bucket_shift) - 1u);
                if ((seen[mate >> 5u] & (1u << (mate & 31u))) != 0u) {
                    atomic_fetch_or_explicit(
                        &survivors[edge >> 5u], 1u << (edge & 31u),
                        memory_order_relaxed);
                }
            }
        }

        // Flamel steps four/six: collect dead edges by contiguous edge range.
        // The following apply kernel can therefore update the global bitmap
        // with sequential stores instead of one random atomic per dead edge.
        kernel void grin_miner_bucket_collect_dead(
            device const uint *arena [[buffer(0)]],
            device const uint *counts [[buffer(1)]],
            device const uint *nodes [[buffer(2)]],
            device uint *dead_arena [[buffer(3)]],
            device atomic_uint *dead_counts [[buffer(4)]],
            device atomic_uint *overflow [[buffer(5)]],
            constant EndpointParams &params [[buffer(6)]],
            uint lid [[thread_position_in_threadgroup]],
            uint bucket [[threadgroup_position_in_grid]],
            uint group_size [[threads_per_threadgroup]])
        {
            threadgroup uint seen[8192];
            const uint bucket_shift = params.edge_bits > 18u ? 18u : params.edge_bits;
            const uint words_per_bucket = 1u << (bucket_shift - 5u);
            for (uint word = lid; word < words_per_bucket; word += group_size) {
                seen[word] = nodes[bucket * words_per_bucket + word];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint count = min(counts[bucket], params.capacity);
            for (uint slot = lid; slot < count; slot += group_size) {
                const uint edge = arena[bucket * params.capacity + slot];
                const uint node = cuckatoo_endpoint(edge, params);
                const uint mate = (node ^ 1u) & ((1u << bucket_shift) - 1u);
                if ((seen[mate >> 5u] & (1u << (mate & 31u))) == 0u) {
                    const uint dead_bucket = (edge - params.edge_base) >> 18u;
                    const uint dead_slot = atomic_fetch_add_explicit(
                        &dead_counts[dead_bucket], 1u, memory_order_relaxed);
                    if (dead_slot < params.destination_capacity) {
                        dead_arena[dead_bucket * params.destination_capacity + dead_slot] = edge;
                    } else {
                        atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
                    }
                }
            }
        }

        // Flamel step five: one workgroup owns one 2^18-edge range. Deaths
        // are applied in threadgroup memory and the 32 KiB result is copied
        // back with coalesced, non-atomic writes.
        kernel void grin_miner_bucket_apply_dead(
            device const uint *dead_arena [[buffer(0)]],
            device const uint *dead_counts [[buffer(1)]],
            device uint *edges [[buffer(2)]],
            constant EndpointParams &params [[buffer(3)]],
            uint lid [[thread_position_in_threadgroup]],
            uint bucket [[threadgroup_position_in_grid]],
            uint group_size [[threads_per_threadgroup]])
        {
            threadgroup atomic_uint alive[8192];
            const uint base_word = (params.edge_base >> 5u) + bucket * 8192u;
            for (uint word = lid; word < 8192u; word += group_size) {
                atomic_store_explicit(&alive[word], edges[base_word + word], memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint count = min(dead_counts[bucket], params.destination_capacity);
            for (uint slot = lid; slot < count; slot += group_size) {
                const uint edge = dead_arena[bucket * params.destination_capacity + slot];
                const uint suffix = (edge - params.edge_base) & ((1u << 18u) - 1u);
                atomic_fetch_and_explicit(
                    &alive[suffix >> 5u], ~(1u << (suffix & 31u)),
                    memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint word = lid; word < 8192u; word += group_size) {
                edges[base_word + word] = atomic_load_explicit(
                    &alive[word], memory_order_relaxed);
            }
        }

        kernel void grin_miner_bucket_trim_to_buckets_fused(
            device const uint *source_arena [[buffer(0)]],
            device const uint *source_counts [[buffer(1)]],
            device uint *destination_arena [[buffer(2)]],
            device atomic_uint *destination_counts [[buffer(3)]],
            device atomic_uint *overflow [[buffer(4)]],
            constant EndpointParams &params [[buffer(5)]],
            uint lid [[thread_position_in_threadgroup]],
            uint bucket [[threadgroup_position_in_grid]],
            uint group_size [[threads_per_threadgroup]])
        {
            threadgroup atomic_uint seen[8192];
            const uint bucket_shift = params.edge_bits > 18u ? 18u : params.edge_bits;
            const uint words_per_bucket = 1u << (bucket_shift - 5u);
            for (uint word = lid; word < words_per_bucket; word += group_size) {
                atomic_store_explicit(&seen[word], 0u, memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint capacity = params.capacity;
            const uint count = min(source_counts[bucket], capacity);
            for (uint slot = lid; slot < count; slot += group_size) {
                const uint edge = source_arena[bucket * capacity + slot];
                const uint node = cuckatoo_endpoint(edge, params);
                const uint suffix = node & ((1u << bucket_shift) - 1u);
                atomic_fetch_or_explicit(
                    &seen[suffix >> 5u], 1u << (suffix & 31u),
                    memory_order_relaxed);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint slot = lid; slot < count; slot += group_size) {
                const uint edge = source_arena[bucket * capacity + slot];
                const uint node = cuckatoo_endpoint(edge, params);
                const uint mate = (node ^ 1u) & ((1u << bucket_shift) - 1u);
                if ((atomic_load_explicit(
                        &seen[mate >> 5u], memory_order_relaxed) &
                        (1u << (mate & 31u))) != 0u) {
                    const uint other = cuckatoo_endpoint_side(
                        edge, params, params.side ^ 1u);
                    const uint destination_bucket = other >> bucket_shift;
                    const uint destination_slot = atomic_fetch_add_explicit(
                        &destination_counts[destination_bucket], 1u,
                        memory_order_relaxed);
                    if (destination_slot < params.destination_capacity) {
                        destination_arena[
                            destination_bucket * params.destination_capacity + destination_slot] = edge;
                    } else {
                        atomic_fetch_add_explicit(
                            overflow, 1u, memory_order_relaxed);
                    }
                }
            }
        }
    "#;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct EndpointParams {
        k0: u64,
        k1: u64,
        k2: u64,
        k3: u64,
        edge_bits: u32,
        side: u32,
        edge_base: u32,
        edge_count: u32,
        capacity: u32,
        destination_capacity: u32,
    }

    struct NativeSeedPart {
        arena: Retained<ProtocolObject<dyn MTLBuffer>>,
        counts: Retained<ProtocolObject<dyn MTLBuffer>>,
        params: EndpointParams,
    }

    struct BucketMarkCheck<'a> {
        arena: &'a ProtocolObject<dyn MTLBuffer>,
        counts: &'a ProtocolObject<dyn MTLBuffer>,
        params: EndpointParams,
        keys: SipKeys,
        edge_bits: u8,
        side: u8,
        edge_base: u32,
        edge_count: usize,
        buckets: usize,
    }

    struct BitmapContinuation<'a> {
        keys: SipKeys,
        edge_bits: u8,
        start_round: u32,
        rounds: u32,
        parts: u32,
        edges: Retained<ProtocolObject<dyn MTLBuffer>>,
        nodes: Retained<ProtocolObject<dyn MTLBuffer>>,
        cancel: &'a AtomicBool,
        started: Instant,
    }

    /// Long-lived native objects. Keeping the device and queue alive also
    /// gives the future C32 path a stable owner for sharded arena buffers.
    pub(super) struct NativeMetalContext {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        probe_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        endpoint_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        clear_nodes_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        clear_counts_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        clear_dead_counts_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        bucket_seed_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        bucket_seed_alive_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        bucket_mark_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        bucket_trim_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        bucket_collect_dead_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        bucket_apply_dead_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        bucket_ping_pong_fused_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    }

    impl NativeMetalContext {
        pub(super) fn new() -> Result<Self, SolveError> {
            let device = objc2_metal::MTLCreateSystemDefaultDevice()
                .ok_or_else(|| SolveError::Gpu("Metal returned no default device".into()))?;
            let queue = device
                .newCommandQueue()
                .ok_or_else(|| SolveError::Gpu("Metal command queue creation failed".into()))?;
            let library = compile_library(&device, PROBE_SOURCE)?;
            let function = library
                .newFunctionWithName(&NSString::from_str("grin_miner_native_probe"))
                .ok_or_else(|| SolveError::Gpu("native probe function is missing".into()))?;
            let probe_pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| {
                    SolveError::Gpu(format!("Metal pipeline compile failed: {error}"))
                })?;
            let endpoint_function = library
                .newFunctionWithName(&NSString::from_str("grin_miner_endpoints"))
                .ok_or_else(|| SolveError::Gpu("native endpoint function is missing".into()))?;
            let endpoint_pipeline = device
                .newComputePipelineStateWithFunction_error(&endpoint_function)
                .map_err(|error| {
                    SolveError::Gpu(format!("Metal endpoint pipeline compile failed: {error}"))
                })?;
            let clear_nodes_function = library
                .newFunctionWithName(&NSString::from_str("grin_miner_clear_nodes"))
                .ok_or_else(|| SolveError::Gpu("native clear-nodes function is missing".into()))?;
            let clear_nodes_pipeline = device
                .newComputePipelineStateWithFunction_error(&clear_nodes_function)
                .map_err(|error| {
                    SolveError::Gpu(format!(
                        "Metal clear-nodes pipeline compile failed: {error}"
                    ))
                })?;
            let clear_counts_function = library
                .newFunctionWithName(&NSString::from_str("grin_miner_clear_bucket_counts"))
                .ok_or_else(|| SolveError::Gpu("native clear-counts function is missing".into()))?;
            let clear_counts_pipeline = device
                .newComputePipelineStateWithFunction_error(&clear_counts_function)
                .map_err(|error| {
                    SolveError::Gpu(format!(
                        "Metal clear-counts pipeline compile failed: {error}"
                    ))
                })?;
            let clear_dead_counts_function = library
                .newFunctionWithName(&NSString::from_str("grin_miner_clear_dead_counts"))
                .ok_or_else(|| {
                    SolveError::Gpu("native clear-dead-counts function is missing".into())
                })?;
            let clear_dead_counts_pipeline = device
                .newComputePipelineStateWithFunction_error(&clear_dead_counts_function)
                .map_err(|error| {
                    SolveError::Gpu(format!(
                        "Metal clear-dead-counts pipeline compile failed: {error}"
                    ))
                })?;
            let bucket_seed_function = library
                .newFunctionWithName(&NSString::from_str("grin_miner_bucket_seed"))
                .ok_or_else(|| SolveError::Gpu("native bucket seed function is missing".into()))?;
            let bucket_seed_pipeline = device
                .newComputePipelineStateWithFunction_error(&bucket_seed_function)
                .map_err(|error| {
                    SolveError::Gpu(format!(
                        "Metal bucket seed pipeline compile failed: {error}"
                    ))
                })?;
            let bucket_seed_alive_function = library
                .newFunctionWithName(&NSString::from_str("grin_miner_bucket_seed_alive_words"))
                .ok_or_else(|| {
                    SolveError::Gpu("native alive bucket seed function is missing".into())
                })?;
            let bucket_seed_alive_pipeline = device
                .newComputePipelineStateWithFunction_error(&bucket_seed_alive_function)
                .map_err(|error| {
                    SolveError::Gpu(format!(
                        "Metal alive bucket seed pipeline compile failed: {error}"
                    ))
                })?;
            let bucket_mark_function = library
                .newFunctionWithName(&NSString::from_str("grin_miner_bucket_mark"))
                .ok_or_else(|| SolveError::Gpu("native bucket mark function is missing".into()))?;
            let bucket_mark_pipeline = device
                .newComputePipelineStateWithFunction_error(&bucket_mark_function)
                .map_err(|error| {
                    SolveError::Gpu(format!(
                        "Metal bucket mark pipeline compile failed: {error}"
                    ))
                })?;
            let bucket_trim_function = library
                .newFunctionWithName(&NSString::from_str("grin_miner_bucket_trim_to_bitmap"))
                .ok_or_else(|| SolveError::Gpu("native bucket trim function is missing".into()))?;
            let bucket_trim_pipeline = device
                .newComputePipelineStateWithFunction_error(&bucket_trim_function)
                .map_err(|error| {
                    SolveError::Gpu(format!(
                        "Metal bucket trim pipeline compile failed: {error}"
                    ))
                })?;
            let bucket_collect_dead_function = library
                .newFunctionWithName(&NSString::from_str("grin_miner_bucket_collect_dead"))
                .ok_or_else(|| SolveError::Gpu("native collect-dead function is missing".into()))?;
            let bucket_collect_dead_pipeline = device
                .newComputePipelineStateWithFunction_error(&bucket_collect_dead_function)
                .map_err(|error| {
                    SolveError::Gpu(format!(
                        "Metal collect-dead pipeline compile failed: {error}"
                    ))
                })?;
            let bucket_apply_dead_function = library
                .newFunctionWithName(&NSString::from_str("grin_miner_bucket_apply_dead"))
                .ok_or_else(|| SolveError::Gpu("native apply-dead function is missing".into()))?;
            let bucket_apply_dead_pipeline = device
                .newComputePipelineStateWithFunction_error(&bucket_apply_dead_function)
                .map_err(|error| {
                    SolveError::Gpu(format!("Metal apply-dead pipeline compile failed: {error}"))
                })?;
            let bucket_ping_pong_fused_function = library
                .newFunctionWithName(&NSString::from_str(
                    "grin_miner_bucket_trim_to_buckets_fused",
                ))
                .ok_or_else(|| {
                    SolveError::Gpu("native fused bucket ping-pong function is missing".into())
                })?;
            let bucket_ping_pong_fused_pipeline = device
                .newComputePipelineStateWithFunction_error(&bucket_ping_pong_fused_function)
                .map_err(|error| {
                    SolveError::Gpu(format!(
                        "Metal fused bucket ping-pong pipeline compile failed: {error}"
                    ))
                })?;

            let context = Self {
                device,
                queue,
                probe_pipeline,
                endpoint_pipeline,
                clear_nodes_pipeline,
                clear_counts_pipeline,
                clear_dead_counts_pipeline,
                bucket_seed_pipeline,
                bucket_seed_alive_pipeline,
                bucket_mark_pipeline,
                bucket_trim_pipeline,
                bucket_collect_dead_pipeline,
                bucket_apply_dead_pipeline,
                bucket_ping_pong_fused_pipeline,
            };
            context.verify_dispatch()?;
            context.verify_endpoints()?;
            context.verify_bucket_seed()?;
            context.verify_full_part_round()?;
            Ok(context)
        }

        fn verify_dispatch(&self) -> Result<(), SolveError> {
            const COUNT: usize = 256;
            let byte_len = COUNT * size_of::<u32>();
            let buffer = self
                .device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| SolveError::Gpu("native probe buffer allocation failed".into()))?;

            // Shared storage is CPU coherent on Apple silicon. Initialize it
            // before encoding, retain the buffer through completion, then
            // validate every GPU-written word.
            let words = unsafe {
                std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u32>(), COUNT)
            };
            for (index, word) in words.iter_mut().enumerate() {
                *word = index as u32;
            }

            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| SolveError::Gpu("native probe command buffer failed".into()))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| SolveError::Gpu("native probe encoder failed".into()))?;
            encoder.setComputePipelineState(&self.probe_pipeline);
            unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer), 0, 0) };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: COUNT,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if let Some(error) = command_buffer.error() {
                return Err(SolveError::Gpu(format!(
                    "native probe command failed: {error}"
                )));
            }

            for (index, &actual) in words.iter().enumerate() {
                let input = index as u32;
                let expected = input
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223)
                    .wrapping_add(input);
                if actual != expected {
                    return Err(SolveError::Gpu(format!(
                        "native probe mismatch at {index}: {actual:#x} != {expected:#x}"
                    )));
                }
            }
            Ok(())
        }

        fn verify_endpoints(&self) -> Result<(), SolveError> {
            const EDGE_BITS: u8 = 12;
            const EDGE_BASE: u32 = 997;
            const COUNT: usize = 4096;
            let keys = SipKeys {
                k0: 0xf2f4_1b02_a8b7_8751,
                k1: 0xe1ec_cf54_3aea_04c0,
                k2: 0x6323_4d62_c711_4f75,
                k3: 0x1e44_d1dd_4fcf_f4c7,
            };
            for side in 0..=1 {
                let actual = self.generate_endpoints(keys, EDGE_BITS, side, EDGE_BASE, COUNT)?;
                for (offset, &node) in actual.iter().enumerate() {
                    let edge = EDGE_BASE.wrapping_add(offset as u32);
                    let expected = endpoint(keys, EDGE_BITS, u64::from(edge), side as u8);
                    if node != expected {
                        return Err(SolveError::Gpu(format!(
                            "native endpoint mismatch side={side} edge={edge}: {node:#x} != {expected:#x}"
                        )));
                    }
                }
            }
            Ok(())
        }

        fn generate_endpoints(
            &self,
            keys: SipKeys,
            edge_bits: u8,
            side: u32,
            edge_base: u32,
            count: usize,
        ) -> Result<Vec<u32>, SolveError> {
            let byte_len = count
                .checked_mul(size_of::<u32>())
                .ok_or_else(|| SolveError::Gpu("native endpoint buffer size overflow".into()))?;
            let output = self
                .device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| {
                    SolveError::Gpu("native endpoint buffer allocation failed".into())
                })?;
            let params = EndpointParams {
                k0: keys.k0,
                k1: keys.k1,
                k2: keys.k2,
                k3: keys.k3,
                edge_bits: u32::from(edge_bits),
                side,
                edge_base,
                edge_count: count
                    .try_into()
                    .map_err(|_| SolveError::Gpu("native endpoint dispatch exceeds u32".into()))?,
                capacity: 0,
                destination_capacity: 0,
            };

            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| SolveError::Gpu("native endpoint command buffer failed".into()))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| SolveError::Gpu("native endpoint encoder failed".into()))?;
            encoder.setComputePipelineState(&self.endpoint_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&output), 0, 0);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&params).cast::<c_void>(),
                    size_of::<EndpointParams>(),
                    1,
                );
            }
            const THREADS: usize = 256;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: count.div_ceil(THREADS),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if let Some(error) = command_buffer.error() {
                return Err(SolveError::Gpu(format!(
                    "native endpoint command failed: {error}"
                )));
            }
            let words = unsafe {
                std::slice::from_raw_parts(output.contents().as_ptr().cast::<u32>(), count)
            };
            Ok(words.to_vec())
        }

        fn seed_part(
            &self,
            keys: SipKeys,
            edge_bits: u8,
            side: u32,
            part: u32,
            parts: u32,
        ) -> Result<NativeSeedPart, SolveError> {
            let local_bits = u32::from(edge_bits).min(18);
            let buckets = 1usize << (u32::from(edge_bits) - local_bits);
            let part_edges = (1usize << edge_bits) / parts as usize;
            let capacity = ((part_edges / buckets) * 105 / 100) & !3;
            let destination_capacity = ((part_edges / buckets) * 73 / 100) & !3;
            let params = EndpointParams {
                k0: keys.k0,
                k1: keys.k1,
                k2: keys.k2,
                k3: keys.k3,
                edge_bits: u32::from(edge_bits),
                side,
                edge_base: part * part_edges as u32,
                edge_count: part_edges as u32,
                capacity: capacity as u32,
                destination_capacity: destination_capacity as u32,
            };
            let arena = self.private_buffer(buckets * capacity * size_of::<u32>(), "arena")?;
            let counts = self.shared_buffer(buckets * size_of::<u32>(), "counts")?;
            let overflow = self.shared_buffer(size_of::<u32>(), "overflow")?;
            unsafe {
                counts.contents().write_bytes(0, buckets * size_of::<u32>());
                overflow.contents().write_bytes(0, size_of::<u32>());
            }
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| SolveError::Gpu("native seed command buffer failed".into()))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| SolveError::Gpu("native seed encoder failed".into()))?;
            encoder.setComputePipelineState(&self.bucket_seed_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&arena), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&counts), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&overflow), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&params).cast::<c_void>(),
                    size_of::<EndpointParams>(),
                    3,
                );
            }
            const THREADS: usize = 256;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: part_edges.div_ceil(THREADS),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if let Some(error) = command_buffer.error() {
                return Err(SolveError::Gpu(format!(
                    "native seed command failed: {error}"
                )));
            }
            let overflow_count = unsafe { *overflow.contents().as_ptr().cast::<u32>() };
            if overflow_count != 0 {
                return Err(SolveError::Gpu(format!(
                    "native seed part {part} overflowed by {overflow_count} edges"
                )));
            }
            Ok(NativeSeedPart {
                arena,
                counts,
                params,
            })
        }

        fn verify_bucket_seed(&self) -> Result<(), SolveError> {
            const EDGE_BITS: u8 = 20;
            const PARTS: u32 = 4;
            const PART: u32 = 2;
            const SIDE: u32 = 1;
            const LOCAL_BITS: u32 = 18;
            const BUCKETS: usize = 1 << (EDGE_BITS as u32 - LOCAL_BITS);
            const PART_EDGES: usize = (1usize << EDGE_BITS) / PARTS as usize;
            const CAPACITY: usize = ((PART_EDGES / BUCKETS) * 105 / 100) & !3;
            let edge_base = PART * PART_EDGES as u32;
            let keys = SipKeys {
                k0: 0xf2f4_1b02_a8b7_8751,
                k1: 0xe1ec_cf54_3aea_04c0,
                k2: 0x6323_4d62_c711_4f75,
                k3: 0x1e44_d1dd_4fcf_f4c7,
            };
            let params = EndpointParams {
                k0: keys.k0,
                k1: keys.k1,
                k2: keys.k2,
                k3: keys.k3,
                edge_bits: u32::from(EDGE_BITS),
                side: SIDE,
                edge_base,
                edge_count: PART_EDGES as u32,
                capacity: CAPACITY as u32,
                destination_capacity: CAPACITY as u32,
            };

            let arena = self.shared_buffer(BUCKETS * CAPACITY * size_of::<u32>(), "arena")?;
            let counts = self.shared_buffer(BUCKETS * size_of::<u32>(), "counts")?;
            let overflow = self.shared_buffer(size_of::<u32>(), "overflow")?;
            unsafe {
                counts.contents().write_bytes(0, BUCKETS * size_of::<u32>());
                overflow.contents().write_bytes(0, size_of::<u32>());
            }
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| SolveError::Gpu("native bucket command buffer failed".into()))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| SolveError::Gpu("native bucket encoder failed".into()))?;
            encoder.setComputePipelineState(&self.bucket_seed_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&arena), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&counts), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&overflow), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&params).cast::<c_void>(),
                    size_of::<EndpointParams>(),
                    3,
                );
            }
            const THREADS: usize = 256;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: PART_EDGES.div_ceil(THREADS),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if let Some(error) = command_buffer.error() {
                return Err(SolveError::Gpu(format!(
                    "native bucket command failed: {error}"
                )));
            }
            let overflow_count = unsafe { *overflow.contents().as_ptr().cast::<u32>() };
            if overflow_count != 0 {
                return Err(SolveError::Gpu(format!(
                    "native bucket oracle overflowed by {overflow_count} edges"
                )));
            }

            let gpu_counts = unsafe {
                std::slice::from_raw_parts(counts.contents().as_ptr().cast::<u32>(), BUCKETS)
            };
            let gpu_arena = unsafe {
                std::slice::from_raw_parts(
                    arena.contents().as_ptr().cast::<u32>(),
                    BUCKETS * CAPACITY,
                )
            };
            let mut cpu = vec![Vec::<u32>::new(); BUCKETS];
            for edge in edge_base..edge_base + PART_EDGES as u32 {
                let node = endpoint(keys, EDGE_BITS, u64::from(edge), SIDE as u8);
                cpu[(node >> LOCAL_BITS) as usize].push(edge);
            }
            for bucket in 0..BUCKETS {
                let count = gpu_counts[bucket] as usize;
                if count > CAPACITY || count != cpu[bucket].len() {
                    return Err(SolveError::Gpu(format!(
                        "native bucket {bucket} count mismatch: {count} != {}",
                        cpu[bucket].len()
                    )));
                }
                let mut actual = gpu_arena[bucket * CAPACITY..bucket * CAPACITY + count].to_vec();
                actual.sort_unstable();
                if actual != cpu[bucket] {
                    return Err(SolveError::Gpu(format!(
                        "native bucket {bucket} payload differs from CPU oracle"
                    )));
                }
            }
            self.verify_bucket_marks(BucketMarkCheck {
                arena: &arena,
                counts: &counts,
                params,
                keys,
                edge_bits: EDGE_BITS,
                side: SIDE as u8,
                edge_base,
                edge_count: PART_EDGES,
                buckets: BUCKETS,
            })?;
            Ok(())
        }

        fn mark_part(
            &self,
            seed: &NativeSeedPart,
            nodes: &ProtocolObject<dyn MTLBuffer>,
            buckets: usize,
        ) -> Result<(), SolveError> {
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| SolveError::Gpu("native mark command buffer failed".into()))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| SolveError::Gpu("native mark encoder failed".into()))?;
            encoder.setComputePipelineState(&self.bucket_mark_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&seed.arena), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&seed.counts), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(nodes), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&seed.params).cast::<c_void>(),
                    size_of::<EndpointParams>(),
                    3,
                );
            }
            const THREADS: usize = 256;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: buckets,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if let Some(error) = command_buffer.error() {
                return Err(SolveError::Gpu(format!(
                    "native mark command failed: {error}"
                )));
            }
            Ok(())
        }

        fn verify_bucket_marks(&self, check: BucketMarkCheck<'_>) -> Result<(), SolveError> {
            let BucketMarkCheck {
                arena,
                counts,
                params,
                keys,
                edge_bits,
                side,
                edge_base,
                edge_count,
                buckets,
            } = check;
            let word_count = (1usize << edge_bits) / 32;
            let nodes = self.shared_buffer(word_count * size_of::<u32>(), "node bitmap")?;
            unsafe {
                nodes
                    .contents()
                    .write_bytes(0, word_count * size_of::<u32>());
            }
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| SolveError::Gpu("native mark command buffer failed".into()))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| SolveError::Gpu("native mark encoder failed".into()))?;
            encoder.setComputePipelineState(&self.bucket_mark_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(arena), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(counts), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&nodes), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&params).cast::<c_void>(),
                    size_of::<EndpointParams>(),
                    3,
                );
            }
            const THREADS: usize = 256;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: buckets,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if let Some(error) = command_buffer.error() {
                return Err(SolveError::Gpu(format!(
                    "native mark command failed: {error}"
                )));
            }

            let actual = unsafe {
                std::slice::from_raw_parts(nodes.contents().as_ptr().cast::<u32>(), word_count)
            };
            let mut expected = vec![0u32; word_count];
            for edge in edge_base..edge_base + edge_count as u32 {
                let node = endpoint(keys, edge_bits, u64::from(edge), side);
                expected[node as usize >> 5] |= 1 << (node & 31);
            }
            if actual != expected {
                let mismatch = actual
                    .iter()
                    .zip(&expected)
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or(0);
                return Err(SolveError::Gpu(format!(
                    "native bucket mark differs at word {mismatch}: {:#x} != {:#x}",
                    actual[mismatch], expected[mismatch]
                )));
            }
            Ok(())
        }

        fn trim_part_to_bitmap(
            &self,
            seed: &NativeSeedPart,
            nodes: &ProtocolObject<dyn MTLBuffer>,
            survivors: &ProtocolObject<dyn MTLBuffer>,
            buckets: usize,
        ) -> Result<(), SolveError> {
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| SolveError::Gpu("native trim command buffer failed".into()))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| SolveError::Gpu("native trim encoder failed".into()))?;
            encoder.setComputePipelineState(&self.bucket_trim_pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&seed.arena), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&seed.counts), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(nodes), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(survivors), 0, 3);
                encoder.setBytes_length_atIndex(
                    NonNull::from(&seed.params).cast::<c_void>(),
                    size_of::<EndpointParams>(),
                    4,
                );
            }
            const THREADS: usize = 256;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: buckets,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if let Some(error) = command_buffer.error() {
                return Err(SolveError::Gpu(format!(
                    "native trim command failed: {error}"
                )));
            }
            Ok(())
        }

        fn verify_full_part_round(&self) -> Result<(), SolveError> {
            const EDGE_BITS: u8 = 20;
            const PARTS: u32 = 4;
            const SIDE: u32 = 0;
            const BUCKETS: usize = 1 << (EDGE_BITS as u32 - 18);
            const EDGE_COUNT: usize = 1 << EDGE_BITS;
            const WORD_COUNT: usize = EDGE_COUNT / 32;
            let keys = SipKeys {
                k0: 0xf2f4_1b02_a8b7_8751,
                k1: 0xe1ec_cf54_3aea_04c0,
                k2: 0x6323_4d62_c711_4f75,
                k3: 0x1e44_d1dd_4fcf_f4c7,
            };
            let nodes = self.shared_buffer(WORD_COUNT * size_of::<u32>(), "round nodes")?;
            let survivors = self.shared_buffer(WORD_COUNT * size_of::<u32>(), "round survivors")?;
            unsafe {
                nodes
                    .contents()
                    .write_bytes(0, WORD_COUNT * size_of::<u32>());
                survivors
                    .contents()
                    .write_bytes(0, WORD_COUNT * size_of::<u32>());
            }

            let mut seeds = Vec::with_capacity(PARTS as usize);
            for part in 0..PARTS {
                let seed = self.seed_part(keys, EDGE_BITS, SIDE, part, PARTS)?;
                self.mark_part(&seed, &nodes, BUCKETS)?;
                seeds.push(seed);
            }
            for seed in &seeds {
                self.trim_part_to_bitmap(seed, &nodes, &survivors, BUCKETS)?;
            }

            let actual = unsafe {
                std::slice::from_raw_parts(survivors.contents().as_ptr().cast::<u32>(), WORD_COUNT)
            };
            let mut occupied = vec![0u32; WORD_COUNT];
            for edge in 0..EDGE_COUNT as u32 {
                let node = endpoint(keys, EDGE_BITS, u64::from(edge), SIDE as u8);
                occupied[node as usize >> 5] |= 1 << (node & 31);
            }
            let mut expected = vec![0u32; WORD_COUNT];
            for edge in 0..EDGE_COUNT as u32 {
                let node = endpoint(keys, EDGE_BITS, u64::from(edge), SIDE as u8);
                let mate = node ^ 1;
                if occupied[mate as usize >> 5] & (1 << (mate & 31)) != 0 {
                    expected[edge as usize >> 5] |= 1 << (edge & 31);
                }
            }
            if actual != expected {
                let mismatch = actual
                    .iter()
                    .zip(&expected)
                    .position(|(actual, expected)| actual != expected)
                    .unwrap_or(0);
                return Err(SolveError::Gpu(format!(
                    "native four-part trim differs at word {mismatch}: {:#x} != {:#x}",
                    actual[mismatch], expected[mismatch]
                )));
            }
            Ok(())
        }

        fn empty_part_like(
            &self,
            source: &NativeSeedPart,
            buckets: usize,
        ) -> Result<NativeSeedPart, SolveError> {
            let capacity = source.params.destination_capacity as usize;
            let mut params = source.params;
            params.side ^= 1;
            params.capacity = source.params.destination_capacity;
            params.destination_capacity = source.params.destination_capacity;
            Ok(NativeSeedPart {
                arena: self.private_buffer(
                    buckets * capacity * size_of::<u32>(),
                    "hybrid destination arena",
                )?,
                counts: self
                    .shared_buffer(buckets * size_of::<u32>(), "hybrid destination counts")?,
                params,
            })
        }

        pub(super) fn trim_survivors(
            &self,
            keys: SipKeys,
            edge_bits: u8,
            rounds: u32,
            parts: u32,
            cancel: &AtomicBool,
        ) -> Result<Option<Vec<u64>>, SolveError> {
            self.trim_survivors_low_memory(keys, edge_bits, rounds, parts, cancel)
        }

        fn trim_survivors_low_memory(
            &self,
            keys: SipKeys,
            edge_bits: u8,
            rounds: u32,
            parts: u32,
            cancel: &AtomicBool,
        ) -> Result<Option<Vec<u64>>, SolveError> {
            if !(18..=32).contains(&edge_bits) || parts == 0 || !parts.is_power_of_two() {
                return Err(SolveError::InvalidConfig(
                    "native Metal requires edge_bits 18..=32 and power-of-two parts".into(),
                ));
            }
            let buckets = 1usize << (u32::from(edge_bits) - 18);
            let edge_count = 1usize << edge_bits;
            let word_count = edge_count / 32;
            let part_edges = edge_count / parts as usize;
            let capacity = ((part_edges / buckets) * 105 / 100) & !3;
            let edges = self.shared_buffer(word_count * size_of::<u32>(), "edge bitmap")?;
            let nodes = self.shared_buffer(word_count * size_of::<u32>(), "node bitmap")?;
            unsafe {
                edges
                    .contents()
                    .write_bytes(0xff, word_count * size_of::<u32>());
            }
            let params = EndpointParams {
                k0: keys.k0,
                k1: keys.k1,
                k2: keys.k2,
                k3: keys.k3,
                edge_bits: u32::from(edge_bits),
                side: 0,
                edge_base: 0,
                edge_count: part_edges as u32,
                capacity: capacity as u32,
                destination_capacity: 0,
            };
            let mut scratch = NativeSeedPart {
                arena: self.private_buffer(
                    buckets * capacity * size_of::<u32>(),
                    "low-memory source arena",
                )?,
                counts: self
                    .shared_buffer(buckets * size_of::<u32>(), "low-memory source counts")?,
                params,
            };
            // Same 32 KiB edge-range buckets and 0.38 capacity used by the
            // reference slean StepFour/StepFive path. Only one edge part is
            // resident at a time, so C32/parts=4 needs about 1.52 GiB here.
            let dead_bucket_count = (part_edges >> 18).max(1);
            let dead_capacity = (((1usize << 18) * 38 / 100) / parts as usize) * parts as usize;
            scratch.params.destination_capacity = dead_capacity as u32;
            let dead_arena = self.private_buffer(
                dead_bucket_count * dead_capacity * size_of::<u32>(),
                "compact dead-edge arena",
            )?;
            let dead_counts = self.shared_buffer(
                dead_bucket_count * size_of::<u32>(),
                "compact dead-edge counts",
            )?;
            let overflow = self.shared_buffer(size_of::<u32>(), "round overflow")?;
            let started = Instant::now();
            // Dense bitmap rounds are cheap enough while the population falls
            // rapidly. Afterwards a compact global ping-pong arena avoids
            // hashing the complete 2^32 edge range twice per round.
            // Flamel switches representation after the first three dense
            // rounds. At that point two globally bucketed survivor arenas
            // fit comfortably on the 16 GiB target and every later round is
            // proportional to the live population instead of 2^edge_bits.
            let dense_rounds = rounds.min(3);
            for round in 0..dense_rounds {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(None);
                }
                scratch.params.side = round & 1;
                unsafe { overflow.contents().write_bytes(0, size_of::<u32>()) };
                let command_buffer = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| SolveError::Gpu("native round command buffer failed".into()))?;
                let encoder = command_buffer
                    .computeCommandEncoder()
                    .ok_or_else(|| SolveError::Gpu("native round compute encoder failed".into()))?;
                const THREADS: usize = 256;
                encoder.setComputePipelineState(&self.clear_nodes_pipeline);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&nodes), 0, 0);
                    encoder.setBytes_length_atIndex(
                        NonNull::from(&scratch.params).cast::<c_void>(),
                        size_of::<EndpointParams>(),
                        1,
                    );
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: word_count.div_ceil(THREADS),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: THREADS,
                        height: 1,
                        depth: 1,
                    },
                );

                for trim_phase in [false, true] {
                    for part in 0..parts {
                        scratch.params.edge_base = part * scratch.params.edge_count;
                        encoder.setComputePipelineState(&self.clear_counts_pipeline);
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(&scratch.counts), 0, 0);
                            encoder.setBytes_length_atIndex(
                                NonNull::from(&scratch.params).cast::<c_void>(),
                                size_of::<EndpointParams>(),
                                1,
                            );
                        }
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: buckets.div_ceil(THREADS),
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: THREADS,
                                height: 1,
                                depth: 1,
                            },
                        );

                        encoder.setComputePipelineState(&self.bucket_seed_alive_pipeline);
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(&scratch.arena), 0, 0);
                            encoder.setBuffer_offset_atIndex(Some(&scratch.counts), 0, 1);
                            encoder.setBuffer_offset_atIndex(Some(&overflow), 0, 2);
                            encoder.setBytes_length_atIndex(
                                NonNull::from(&scratch.params).cast::<c_void>(),
                                size_of::<EndpointParams>(),
                                3,
                            );
                            encoder.setBuffer_offset_atIndex(Some(&edges), 0, 4);
                        }
                        encoder.dispatchThreadgroups_threadsPerThreadgroup(
                            MTLSize {
                                width: part_edges.div_ceil(32).div_ceil(THREADS),
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: THREADS,
                                height: 1,
                                depth: 1,
                            },
                        );

                        if trim_phase {
                            encoder.setComputePipelineState(&self.clear_dead_counts_pipeline);
                            unsafe {
                                encoder.setBuffer_offset_atIndex(Some(&dead_counts), 0, 0);
                                encoder.setBytes_length_atIndex(
                                    NonNull::from(&scratch.params).cast::<c_void>(),
                                    size_of::<EndpointParams>(),
                                    1,
                                );
                            }
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                                MTLSize {
                                    width: dead_bucket_count.div_ceil(THREADS),
                                    height: 1,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: THREADS,
                                    height: 1,
                                    depth: 1,
                                },
                            );

                            encoder.setComputePipelineState(&self.bucket_collect_dead_pipeline);
                            unsafe {
                                encoder.setBuffer_offset_atIndex(Some(&scratch.arena), 0, 0);
                                encoder.setBuffer_offset_atIndex(Some(&scratch.counts), 0, 1);
                                encoder.setBuffer_offset_atIndex(Some(&nodes), 0, 2);
                                encoder.setBuffer_offset_atIndex(Some(&dead_arena), 0, 3);
                                encoder.setBuffer_offset_atIndex(Some(&dead_counts), 0, 4);
                                encoder.setBuffer_offset_atIndex(Some(&overflow), 0, 5);
                                encoder.setBytes_length_atIndex(
                                    NonNull::from(&scratch.params).cast::<c_void>(),
                                    size_of::<EndpointParams>(),
                                    6,
                                );
                            }
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                                MTLSize {
                                    width: buckets,
                                    height: 1,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: THREADS,
                                    height: 1,
                                    depth: 1,
                                },
                            );

                            encoder.setComputePipelineState(&self.bucket_apply_dead_pipeline);
                            unsafe {
                                encoder.setBuffer_offset_atIndex(Some(&dead_arena), 0, 0);
                                encoder.setBuffer_offset_atIndex(Some(&dead_counts), 0, 1);
                                encoder.setBuffer_offset_atIndex(Some(&edges), 0, 2);
                                encoder.setBytes_length_atIndex(
                                    NonNull::from(&scratch.params).cast::<c_void>(),
                                    size_of::<EndpointParams>(),
                                    3,
                                );
                            }
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                                MTLSize {
                                    width: dead_bucket_count,
                                    height: 1,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: THREADS,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                        } else {
                            encoder.setComputePipelineState(&self.bucket_mark_pipeline);
                            unsafe {
                                encoder.setBuffer_offset_atIndex(Some(&scratch.arena), 0, 0);
                                encoder.setBuffer_offset_atIndex(Some(&scratch.counts), 0, 1);
                                encoder.setBuffer_offset_atIndex(Some(&nodes), 0, 2);
                                encoder.setBytes_length_atIndex(
                                    NonNull::from(&scratch.params).cast::<c_void>(),
                                    size_of::<EndpointParams>(),
                                    3,
                                );
                            }
                            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                                MTLSize {
                                    width: buckets,
                                    height: 1,
                                    depth: 1,
                                },
                                MTLSize {
                                    width: THREADS,
                                    height: 1,
                                    depth: 1,
                                },
                            );
                        }
                    }
                }
                encoder.endEncoding();
                command_buffer.commit();
                command_buffer.waitUntilCompleted();
                if let Some(error) = command_buffer.error() {
                    return Err(SolveError::Gpu(format!(
                        "native round command failed: {error}"
                    )));
                }
                let overflow_count = unsafe { *overflow.contents().as_ptr().cast::<u32>() };
                if overflow_count != 0 {
                    return Err(SolveError::Gpu(format!(
                        "native round {} bucket overflowed by {overflow_count} edges",
                        round + 1
                    )));
                }
                if round < 8 || (round + 1).is_multiple_of(16) {
                    eprintln!(
                        "C{edge_bits} native-low-memory round={} elapsed={:.3}s",
                        round + 1,
                        started.elapsed().as_secs_f64()
                    );
                }
            }
            if rounds > dense_rounds {
                drop(scratch);
                drop(dead_arena);
                drop(dead_counts);
                drop(overflow);
                return self.continue_from_bitmap(BitmapContinuation {
                    keys,
                    edge_bits,
                    start_round: dense_rounds,
                    rounds,
                    parts,
                    edges,
                    nodes,
                    cancel,
                    started,
                });
            }
            let bitmap = unsafe {
                std::slice::from_raw_parts(edges.contents().as_ptr().cast::<u32>(), word_count)
            };
            let mut result = Vec::new();
            for (word_index, &value) in bitmap.iter().enumerate() {
                let mut bits = value;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    result.push((word_index * 32 + bit) as u64);
                    bits &= bits - 1;
                }
            }
            eprintln!(
                "C{edge_bits} native-low-memory rounds={rounds} survivors={} trim={:.3}s",
                result.len(),
                started.elapsed().as_secs_f64()
            );
            Ok(Some(result))
        }

        fn continue_from_bitmap(
            &self,
            continuation: BitmapContinuation<'_>,
        ) -> Result<Option<Vec<u64>>, SolveError> {
            let BitmapContinuation {
                keys,
                edge_bits,
                start_round,
                rounds,
                parts,
                edges,
                nodes,
                cancel,
                started,
            } = continuation;
            let buckets = 1usize << (u32::from(edge_bits) - 18);
            let edge_count = 1usize << edge_bits;
            let word_count = edge_count / 32;
            let part_edges = edge_count / parts as usize;
            let bitmap = unsafe {
                std::slice::from_raw_parts(edges.contents().as_ptr().cast::<u32>(), word_count)
            };
            let alive = bitmap
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum::<usize>();
            // Uniform hashing makes +15% and 1024 entries per bucket a very
            // conservative exact capacity while keeping both arenas compact.
            let capacity = (alive.div_ceil(buckets) * 115 / 100 + 1024 + 3) & !3;
            eprintln!(
                "C{edge_bits} native transition round={start_round} alive={alive} capacity={capacity}"
            );

            let source_arena =
                self.private_buffer(buckets * capacity * size_of::<u32>(), "hybrid source arena")?;
            let source_counts =
                self.shared_buffer(buckets * size_of::<u32>(), "hybrid source counts")?;
            let overflow = self.shared_buffer(size_of::<u32>(), "hybrid seed overflow")?;
            unsafe {
                source_counts
                    .contents()
                    .write_bytes(0, buckets * size_of::<u32>());
                overflow.contents().write_bytes(0, size_of::<u32>());
            }
            let mut params = EndpointParams {
                k0: keys.k0,
                k1: keys.k1,
                k2: keys.k2,
                k3: keys.k3,
                edge_bits: u32::from(edge_bits),
                side: start_round & 1,
                edge_base: 0,
                edge_count: part_edges as u32,
                capacity: capacity as u32,
                destination_capacity: capacity as u32,
            };
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| SolveError::Gpu("hybrid seed command buffer failed".into()))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| SolveError::Gpu("hybrid seed encoder failed".into()))?;
            const THREADS: usize = 256;
            encoder.setComputePipelineState(&self.bucket_seed_alive_pipeline);
            for part in 0..parts {
                params.edge_base = part * params.edge_count;
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&source_arena), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&source_counts), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&overflow), 0, 2);
                    encoder.setBytes_length_atIndex(
                        NonNull::from(&params).cast::<c_void>(),
                        size_of::<EndpointParams>(),
                        3,
                    );
                    encoder.setBuffer_offset_atIndex(Some(&edges), 0, 4);
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: part_edges.div_ceil(32).div_ceil(THREADS),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: THREADS,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if let Some(error) = command_buffer.error() {
                return Err(SolveError::Gpu(format!("hybrid seed failed: {error}")));
            }
            let overflow_count = unsafe { *overflow.contents().as_ptr().cast::<u32>() };
            if overflow_count != 0 {
                return Err(SolveError::Gpu(format!(
                    "hybrid seed overflowed by {overflow_count} edges"
                )));
            }

            params.edge_base = 0;
            let source = NativeSeedPart {
                arena: source_arena,
                counts: source_counts,
                params,
            };
            let destination = self.empty_part_like(&source, buckets)?;
            let survivors =
                self.shared_buffer(word_count * size_of::<u32>(), "hybrid survivors")?;
            unsafe {
                survivors
                    .contents()
                    .write_bytes(0, word_count * size_of::<u32>());
                overflow.contents().write_bytes(0, size_of::<u32>());
            }
            let command_buffer = self
                .queue
                .commandBuffer()
                .ok_or_else(|| SolveError::Gpu("hybrid loop command buffer failed".into()))?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| SolveError::Gpu("hybrid loop encoder failed".into()))?;
            for round in start_round..rounds {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(None);
                }
                let even = (round - start_round).is_multiple_of(2);
                let (from, to) = if even {
                    (&source, &destination)
                } else {
                    (&destination, &source)
                };

                if round + 1 == rounds {
                    encoder.setComputePipelineState(&self.clear_nodes_pipeline);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(&nodes), 0, 0);
                        encoder.setBytes_length_atIndex(
                            NonNull::from(&from.params).cast::<c_void>(),
                            size_of::<EndpointParams>(),
                            1,
                        );
                    }
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: word_count.div_ceil(THREADS),
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: THREADS,
                            height: 1,
                            depth: 1,
                        },
                    );

                    encoder.setComputePipelineState(&self.bucket_mark_pipeline);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(&from.arena), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(&from.counts), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(&nodes), 0, 2);
                        encoder.setBytes_length_atIndex(
                            NonNull::from(&from.params).cast::<c_void>(),
                            size_of::<EndpointParams>(),
                            3,
                        );
                    }
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: buckets,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: THREADS,
                            height: 1,
                            depth: 1,
                        },
                    );

                    unsafe {
                        encoder.setComputePipelineState(&self.bucket_trim_pipeline);
                        encoder.setBuffer_offset_atIndex(Some(&from.arena), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(&from.counts), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(&nodes), 0, 2);
                        encoder.setBuffer_offset_atIndex(Some(&survivors), 0, 3);
                        encoder.setBytes_length_atIndex(
                            NonNull::from(&from.params).cast::<c_void>(),
                            size_of::<EndpointParams>(),
                            4,
                        );
                    }
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: buckets,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: THREADS,
                            height: 1,
                            depth: 1,
                        },
                    );
                    break;
                }

                encoder.setComputePipelineState(&self.clear_counts_pipeline);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&to.counts), 0, 0);
                    encoder.setBytes_length_atIndex(
                        NonNull::from(&from.params).cast::<c_void>(),
                        size_of::<EndpointParams>(),
                        1,
                    );
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: buckets.div_ceil(THREADS),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: THREADS,
                        height: 1,
                        depth: 1,
                    },
                );

                encoder.setComputePipelineState(&self.bucket_ping_pong_fused_pipeline);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&from.arena), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&from.counts), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&to.arena), 0, 2);
                    encoder.setBuffer_offset_atIndex(Some(&to.counts), 0, 3);
                    encoder.setBuffer_offset_atIndex(Some(&overflow), 0, 4);
                    encoder.setBytes_length_atIndex(
                        NonNull::from(&from.params).cast::<c_void>(),
                        size_of::<EndpointParams>(),
                        5,
                    );
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: buckets,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: THREADS,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
            if let Some(error) = command_buffer.error() {
                return Err(SolveError::Gpu(format!("hybrid loop failed: {error}")));
            }
            let overflow_count = unsafe { *overflow.contents().as_ptr().cast::<u32>() };
            if overflow_count != 0 {
                return Err(SolveError::Gpu(format!(
                    "hybrid loop overflowed by {overflow_count} edges"
                )));
            }
            let bitmap = unsafe {
                std::slice::from_raw_parts(survivors.contents().as_ptr().cast::<u32>(), word_count)
            };
            let mut result = Vec::new();
            for (word_index, &value) in bitmap.iter().enumerate() {
                let mut bits = value;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    result.push((word_index * 32 + bit) as u64);
                    bits &= bits - 1;
                }
            }
            let trim_seconds = started.elapsed().as_secs_f64();
            eprintln!(
                "C{edge_bits} native-hybrid rounds={rounds} survivors={} trim={trim_seconds:.3}s rate={:.4} G/s",
                result.len(),
                1.0 / trim_seconds
            );
            Ok(Some(result))
        }

        fn shared_buffer(
            &self,
            byte_len: usize,
            label: &str,
        ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, SolveError> {
            self.device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| SolveError::Gpu(format!("native {label} allocation failed")))
        }

        fn private_buffer(
            &self,
            byte_len: usize,
            label: &str,
        ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, SolveError> {
            self.device
                .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModePrivate)
                .ok_or_else(|| SolveError::Gpu(format!("native private {label} allocation failed")))
        }
    }

    fn compile_library(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, SolveError> {
        device
            .newLibraryWithSource_options_error(&NSString::from_str(source), None)
            .map_err(|error| SolveError::Gpu(format!("Metal shader compile failed: {error}")))
    }

    #[cfg(test)]
    mod tests {
        use std::sync::atomic::AtomicBool;

        use super::NativeMetalContext;
        use crate::{
            keys::derive_keys,
            solver::{GraphParams, cpu_lean::trim_survivors},
        };

        #[test]
        fn native_multi_round_ping_pong_matches_cpu() {
            let context = NativeMetalContext::new().unwrap();
            let params = GraphParams {
                keys: derive_keys(&[0], 9),
                edge_bits: 18,
                cycle_length: 42,
                rounds: 12,
            };
            let cancel = AtomicBool::new(false);
            let native = context
                .trim_survivors(params.keys, params.edge_bits, params.rounds, 4, &cancel)
                .unwrap()
                .unwrap();
            let cpu = trim_survivors(params, params.edge_bits).unwrap();
            assert_eq!(native, cpu);
        }
    }
}

pub struct GpuMetalSolver {
    #[cfg(target_os = "macos")]
    _native: native::NativeMetalContext,
    config: GpuWgpuConfig,
}

impl GpuMetalSolver {
    #[cfg(target_os = "macos")]
    pub fn new(config: GpuWgpuConfig) -> Result<Self, SolveError> {
        if config.trimming == crate::solver::TrimmingMode::Lean {
            return Err(SolveError::Unsupported(
                "native Metal supports slean trimming; use --backend gpu for lean".into(),
            ));
        }
        Ok(Self {
            _native: native::NativeMetalContext::new()?,
            config,
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn new(_config: GpuWgpuConfig) -> Result<Self, SolveError> {
        Err(SolveError::Unsupported(
            "the Metal backend is available only on macOS; use --backend auto or gpu".into(),
        ))
    }

    pub const fn is_native() -> bool {
        cfg!(target_os = "macos")
    }
}

impl Solver for GpuMetalSolver {
    fn name(&self) -> &'static str {
        "metal-native"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            min_edge_bits: 18,
            max_edge_bits: 32,
            cycle_length: 42,
        }
    }

    fn solve(
        &mut self,
        request: SolveRequest,
        cancel: &AtomicBool,
    ) -> Result<SolveOutcome, SolveError> {
        #[cfg(target_os = "macos")]
        {
            validate_request(&request, self.capabilities())?;
            let graph = request.graph_params();
            let solve_started = std::time::Instant::now();
            let Some(survivors) = self._native.trim_survivors(
                graph.keys,
                graph.edge_bits,
                graph.rounds,
                self.config.slean_parts,
                cancel,
            )?
            else {
                return Ok(SolveOutcome::Cancelled);
            };
            let before_peel = survivors.len();
            let peel_started = std::time::Instant::now();
            let survivors = peel_two_core(graph, &survivors)?;
            let peel_elapsed = peel_started.elapsed();
            if cancel.load(Ordering::Relaxed) {
                return Ok(SolveOutcome::Cancelled);
            }
            let search_started = std::time::Instant::now();
            let proof = find_cycle_d2(graph, &survivors);
            let search_elapsed = search_started.elapsed();
            let solve_elapsed = solve_started.elapsed().as_secs_f64();
            eprintln!(
                "C{} native-metal survivors={}->{} peel={:.3}s d2={:.3}s total={solve_elapsed:.3}s rate={:.4} G/s",
                graph.edge_bits,
                before_peel,
                survivors.len(),
                peel_elapsed.as_secs_f64(),
                search_elapsed.as_secs_f64(),
                1.0 / solve_elapsed
            );
            match proof {
                Ok(Some(proof)) => Ok(SolveOutcome::Proof(proof)),
                Ok(None) => Ok(SolveOutcome::NoCycle),
                Err(SolveError::SearchLimit(reason)) => Ok(SolveOutcome::Inconclusive(reason)),
                Err(error) => Err(error),
            }
        }
        #[cfg(not(target_os = "macos"))]
        Err(SolveError::Unsupported(
            "the native Metal backend is available only on macOS".into(),
        ))
    }
}
