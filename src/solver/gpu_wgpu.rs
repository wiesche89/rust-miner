use std::{
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::{
    siphash::GpuSipKeys,
    solver::{
        BackendCapabilities, GraphParams, SolveError, SolveOutcome, SolveRequest, Solver,
        TrimmingMode, d2::find_cycle_d2, peel::peel_two_core, validate_request,
    },
};

#[path = "gpu_diagnostics.rs"]
mod gpu_diagnostics;
#[path = "gpu_fine.rs"]
mod gpu_fine;

use self::gpu_diagnostics::{
    BUCKET_HEADER_WORDS as DIAGNOSTIC_BUCKET_HEADER_WORDS,
    BUCKET_MARGIN as DIAGNOSTIC_BUCKET_MARGIN,
    BUCKET_OVERFLOW_WORD as DIAGNOSTIC_BUCKET_OVERFLOW_WORD, Diagnostics,
};

const WORKGROUP_SIZE: u32 = 256;
const DISPATCH_GROUPS: u32 = 4096;
const ROUNDS_PER_SUBMISSION: u32 = 16;
const EARLY_VERDICT_ROUND: u32 = 128;
const BUCKET_CHUNK_EDGES: u64 = 1 << 28;
const BUCKETED_MARK_BUCKETS: u32 = 64;
const BUCKETED_MARK_ROUNDS: u32 = 4;
const BUCKETED_MARK_MIN_EDGE_BITS: u8 = 24;
const FINE_BUCKETS: u32 = 32_768;
const FINE_FIXED_MIN_MARGIN: u64 = 64;
const SLEAN_PARTS_TWO_AVAILABLE_MEMORY: u64 = 18 * 1024 * 1024 * 1024;

const NATIVE_INT64_ENDPOINT: &str = r#"
struct NativeSipState {
    v0: u64,
    v1: u64,
    v2: u64,
    v3: u64,
}

fn native_rotl(value: u64, amount: u32) -> u64 {
    return (value << amount) | (value >> (64u - amount));
}

fn native_sip_round(input: NativeSipState) -> NativeSipState {
    var v = input;
    v.v0 += v.v1;
    v.v2 += v.v3;
    v.v1 = native_rotl(v.v1, 13u) ^ v.v0;
    v.v3 = native_rotl(v.v3, 16u) ^ v.v2;
    v.v0 = native_rotl(v.v0, 32u);
    v.v2 += v.v1;
    v.v0 += v.v3;
    v.v1 = native_rotl(v.v1, 17u) ^ v.v2;
    v.v3 = native_rotl(v.v3, 21u) ^ v.v0;
    v.v2 = native_rotl(v.v2, 32u);
    return v;
}

fn native_word(lo: u32, hi: u32) -> u64 {
    return u64(lo) | (u64(hi) << 32u);
}

fn endpoint_for_side(edge: u32, side: u32) -> u32 {
    let nonce = (u64(edge) << 1u) | u64(side);
    var v = NativeSipState(
        native_word(params.key_words_a.x, params.key_words_a.y),
        native_word(params.key_words_a.z, params.key_words_a.w),
        native_word(params.key_words_b.x, params.key_words_b.y),
        native_word(params.key_words_b.z, params.key_words_b.w) ^ nonce,
    );
    v = native_sip_round(v);
    v = native_sip_round(v);
    v.v0 ^= nonce;
    v.v2 ^= 255lu;
    v = native_sip_round(v);
    v = native_sip_round(v);
    v = native_sip_round(v);
    v = native_sip_round(v);
    var result = u32(v.v0 ^ v.v1 ^ v.v2 ^ v.v3);
    if params.edge_bits < 32u {
        result &= (1u << params.edge_bits) - 1u;
    }
    return result;
}
"#;

fn shader_source(native_int64: bool) -> Result<String, SolveError> {
    let source = include_str!("lean.wgsl").to_owned();
    if !native_int64 {
        return Ok(source);
    }
    let begin_marker = "// SIPHASH_ENDPOINT_BEGIN";
    let end_marker = "// SIPHASH_ENDPOINT_END";
    let begin = source
        .find(begin_marker)
        .ok_or_else(|| SolveError::Gpu("missing SipHash begin marker".into()))?;
    let end = source
        .find(end_marker)
        .ok_or_else(|| SolveError::Gpu("missing SipHash end marker".into()))?;
    let mut native = String::with_capacity(source.len() + NATIVE_INT64_ENDPOINT.len());
    native.push_str(&source[..begin + begin_marker.len()]);
    native.push('\n');
    native.push_str(NATIVE_INT64_ENDPOINT);
    native.push_str(&source[end..]);
    Ok(native)
}

#[cfg(target_os = "linux")]
fn available_memory_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    kib.checked_mul(1024)
}

#[cfg(target_os = "macos")]
fn available_memory_bytes() -> Option<u64> {
    let output = std::process::Command::new("/usr/bin/vm_stat")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = std::str::from_utf8(&output.stdout).ok()?;
    let page_size = text
        .lines()
        .next()?
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    let available_pages = text.lines().skip(1).filter_map(|line| {
        let (name, value) = line.split_once(':')?;
        matches!(
            name,
            "Pages free" | "Pages inactive" | "Pages speculative" | "Pages purgeable"
        )
        .then(|| value.trim().trim_end_matches('.').parse::<u64>().ok())?
    });
    available_pages.sum::<u64>().checked_mul(page_size)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn available_memory_bytes() -> Option<u64> {
    None
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    key_words: [u32; 8],
    edge_bits: u32,
    side: u32,
    edge_count_lo: u32,
    word_count: u32,
    node_mask: u32,
    diagnostic_chunk_base: u32,
    diagnostic_chunk_count: u32,
    diagnostic_bucket: u32,
}

const _: () = {
    assert!(std::mem::size_of::<Params>() == 64);
    assert!(std::mem::offset_of!(Params, edge_bits) == 32);
    assert!(std::mem::offset_of!(Params, node_mask) == 48);
};

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    module: wgpu::ShaderModule,
    limits: wgpu::Limits,
    adapter_name: String,
    native_int64: bool,
}

impl GpuContext {
    fn submit(&self, encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        self.queue.submit(Some(encoder.finish()))
    }

    fn wait(&self, submission: wgpu::SubmissionIndex, operation: &str) -> Result<(), SolveError> {
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(300)),
            })
            .map_err(|error| SolveError::Gpu(format!("waiting for {operation}: {error}")))?;
        Ok(())
    }

    fn run(&self, encoder: wgpu::CommandEncoder, operation: &str) -> Result<(), SolveError> {
        self.wait(self.submit(encoder), operation)
    }

    fn read_u32(
        &self,
        source: &wgpu::Buffer,
        byte_len: u64,
        label: &'static str,
    ) -> Result<Vec<u32>, SolveError> {
        self.read_u32_at(source, 0, byte_len, label)
    }

    fn read_u32_at(
        &self,
        source: &wgpu::Buffer,
        source_offset: u64,
        byte_len: u64,
        label: &'static str,
    ) -> Result<Vec<u32>, SolveError> {
        self.read_buffer_ranges(&[(source, source_offset, byte_len)], label)
    }

    fn read_u32_offsets(
        &self,
        sources: &[(&wgpu::Buffer, u64)],
        label: &'static str,
    ) -> Result<Vec<u32>, SolveError> {
        let ranges: Vec<_> = sources
            .iter()
            .map(|(source, offset)| (*source, *offset, 4))
            .collect();
        self.read_buffer_ranges(&ranges, label)
    }

    fn read_buffer_ranges(
        &self,
        ranges: &[(&wgpu::Buffer, u64, u64)],
        label: &'static str,
    ) -> Result<Vec<u32>, SolveError> {
        let byte_len = ranges.iter().map(|(_, _, len)| len).sum();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        let mut target_offset = 0;
        for (source, source_offset, len) in ranges {
            encoder.copy_buffer_to_buffer(source, *source_offset, &staging, target_offset, *len);
            target_offset += len;
        }
        let submission = self.submit(encoder);
        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.wait(submission, label)?;
        receiver
            .recv()
            .map_err(|error| SolveError::Gpu(format!("map callback for {label}: {error}")))?
            .map_err(|error| SolveError::Gpu(format!("mapping {label}: {error}")))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| SolveError::Gpu(format!("accessing {label}: {error}")))?;
        Ok(bytemuck::cast_slice::<u8, u32>(&mapped).to_vec())
    }
}

/// Portable bitmap-based lean trimmer. It needs one edge bitmap and one node
/// bitmap (about 1 GiB total for C32), then searches the compact survivors on CPU.
pub struct GpuWgpuSolver {
    context: GpuContext,
    pipelines: TrimPipelines,
    config: GpuWgpuConfig,
    bucketed_rounds: AtomicU32,
    bucket_trim_rounds: AtomicU32,
    bucket_fallbacks: AtomicU32,
    fine_rounds: AtomicU32,
    fine_mismatched_words: AtomicU32,
    force_slean_overflow: AtomicBool,
    #[cfg(test)]
    verify_slean_seed: AtomicBool,
    #[cfg(test)]
    slean_seed_checks: AtomicU32,
}

#[derive(Debug, Clone, Copy)]
pub struct GpuWgpuConfig {
    pub trimming: TrimmingMode,
    pub slean_parts: u32,
    pub local_ram_kib: u32,
}

impl Default for GpuWgpuConfig {
    fn default() -> Self {
        Self {
            trimming: TrimmingMode::Lean,
            slean_parts: 4,
            local_ram_kib: 32,
        }
    }
}

#[derive(Clone, Copy)]
struct TrimOptions {
    bucketed_mark: bool,
    slean: bool,
    fine_end: Option<u32>,
}

#[derive(Clone, Copy)]
struct ProductionModes {
    fine: bool,
    bucketed_mark: bool,
    slean: bool,
}

#[derive(Clone, Copy)]
struct GraphSize {
    edge_count: u64,
    word_count: u64,
    bitmap_bytes: u64,
}

#[derive(Clone, Copy)]
struct SleanSizing {
    bucket_count: u64,
    parts: u64,
    part_edges: u64,
    capacity: u64,
    scratch_bytes: u64,
    dead_bucket_count: u64,
    dead_scratch_bytes: u64,
}

impl GraphSize {
    fn new(context: &GpuContext, request: GraphParams) -> Result<Self, SolveError> {
        if request.edge_bits == 0 || request.edge_bits > 32 {
            return Err(SolveError::InvalidConfig(
                "edge_bits must be in 1..=32".into(),
            ));
        }
        if request.cycle_length == 0 || !request.cycle_length.is_multiple_of(2) {
            return Err(SolveError::InvalidConfig(
                "cycle_length must be positive and even".into(),
            ));
        }
        let edge_count = 1_u64 << request.edge_bits;
        let word_count = edge_count.div_ceil(32);
        let bitmap_bytes = word_count * 4;
        if bitmap_bytes > context.limits.max_buffer_size
            || bitmap_bytes > context.limits.max_storage_buffer_binding_size
        {
            return Err(SolveError::Unsupported(format!(
                "{} exposes max storage buffer {} MiB, but C{} needs {} MiB per bitmap",
                context.adapter_name,
                context.limits.max_storage_buffer_binding_size / (1024 * 1024),
                request.edge_bits,
                bitmap_bytes / (1024 * 1024),
            )));
        }
        Ok(Self {
            edge_count,
            word_count,
            bitmap_bytes,
        })
    }
}

enum TrimOutcome {
    Survivors(Vec<u64>),
    Cancelled,
    Diagnostic(&'static str),
}

struct BucketRound0Resources<'a> {
    edges: &'a wgpu::Buffer,
    nodes: &'a wgpu::Buffer,
    scratch: &'a wgpu::Buffer,
    layout: &'a wgpu::BindGroupLayout,
    dense_mark_pipeline: &'a wgpu::ComputePipeline,
    scatter_pipeline: &'a wgpu::ComputePipeline,
    bucket_mark_pipeline: &'a wgpu::ComputePipeline,
    baseline_bind_group: &'a wgpu::BindGroup,
    groups_for_words: u32,
}

struct BucketRoundResources<'a> {
    nodes: &'a wgpu::Buffer,
    scratch: &'a wgpu::Buffer,
    scatter: &'a [wgpu::BindGroup],
    mark: &'a [wgpu::BindGroup],
    lean: &'a wgpu::BindGroup,
    groups_for_words: u32,
}

struct FineCsrResources<'a> {
    scratch: &'a wgpu::Buffer,
    lean_bind_group: &'a wgpu::BindGroup,
    count_layout: &'a wgpu::BindGroupLayout,
    arena_layout: &'a wgpu::BindGroupLayout,
    histogram_pipeline: &'a wgpu::ComputePipeline,
    scatter_pipeline: &'a wgpu::ComputePipeline,
    verify_pipeline: &'a wgpu::ComputePipeline,
    groups_for_words: u32,
}

#[cfg(test)]
struct SleanSeedCheck<'a> {
    request: GraphParams,
    side: u32,
    part_base: u64,
    part_count: u64,
    bucket_count: u64,
    capacity: u64,
    scratch_bytes: u64,
    arenas: [&'a wgpu::Buffer; 4],
}

struct FineTrimResources<'a> {
    current_lean_bind_group: &'a wgpu::BindGroup,
    next_lean_bind_group: &'a wgpu::BindGroup,
    count_layout: &'a wgpu::BindGroupLayout,
    arena_layout: &'a wgpu::BindGroupLayout,
    count_pipeline: &'a wgpu::ComputePipeline,
    scatter_pipeline: &'a wgpu::ComputePipeline,
    fixed_pipeline: &'a wgpu::ComputePipeline,
    verify_pipeline: &'a wgpu::ComputePipeline,
    scratch: &'a wgpu::Buffer,
    validate_output: bool,
}

struct FineLoopResources<'a> {
    bind_groups: &'a [wgpu::BindGroup],
    scratch: &'a wgpu::Buffer,
    end_round: u32,
    production: bool,
    validate_output: bool,
}

struct FineDiagnosticResources<'a> {
    nodes: &'a wgpu::Buffer,
    scratch: &'a wgpu::Buffer,
    bind_groups: &'a [wgpu::BindGroup],
    completed_round: u32,
    end_round: u32,
    input_survivors: u64,
    seed_histogram: Duration,
    seed_scatter: Duration,
    fine_count: Duration,
    fine_scatter: Duration,
    fine_wall: Duration,
    groups_for_words: u32,
}

struct FineTransitionResources<'a> {
    buffers: &'a TrimBuffers,
    bindings: &'a CoreBindings,
    completed_round: u32,
    end_round: u32,
    production: bool,
    sharded_seed: bool,
    validate_output: bool,
    groups_for_words: u32,
}

struct FineTransitionSeed<'a> {
    arena: FineCsrArena,
    survivors: u64,
    loop_start: u32,
    scratch: &'a wgpu::Buffer,
    bind_groups: &'a [wgpu::BindGroup],
}

struct SleanDenseResources<'a> {
    edges: &'a wgpu::Buffer,
    nodes: &'a wgpu::Buffer,
    backup: &'a wgpu::Buffer,
    arenas: [&'a wgpu::Buffer; 4],
    dead_arena: &'a wgpu::Buffer,
    part_bind_groups: &'a [wgpu::BindGroup],
    lean_bind_groups: &'a [wgpu::BindGroup],
    sizing: SleanSizing,
    bitmap_bytes: u64,
    groups_for_words: u32,
    measure_phases: bool,
}

struct LeanRoundFallback<'a> {
    backup: &'a wgpu::Buffer,
    edges: &'a wgpu::Buffer,
    bitmap_bytes: u64,
    groups_for_words: u32,
    bind_groups: &'a [wgpu::BindGroup],
    label: &'static str,
    wait: bool,
}

struct TrimRun<'a> {
    solver: &'a GpuWgpuSolver,
    request: GraphParams,
    rounds: u32,
    live_work: bool,
    cancel: &'a AtomicBool,
    diagnostics: Diagnostics,
    production_fine: bool,
    production_slean: bool,
    production_bucketed_mark: bool,
    bucket_rounds: u32,
    fine_transition_round: u32,
    fine_end_round: u32,
    sharded_fine_seed: bool,
    graph: GraphSize,
    buffers: &'a TrimBuffers,
    core: &'a CoreBindings,
    bucket: &'a BucketBindings,
    slean: &'a SleanBindings,
    slean_sizing: SleanSizing,
}

struct PreparedTrim {
    diagnostics: Diagnostics,
    rounds: u32,
    production_fine: bool,
    production_slean: bool,
    production_bucketed_mark: bool,
    graph: GraphSize,
    slean_sizing: SleanSizing,
    buffers: TrimBuffers,
    core: CoreBindings,
    bucket: BucketBindings,
    slean: SleanBindings,
    groups_for_words: u32,
}

struct FineShardSeedResources<'a> {
    current_lean_bind_group: &'a wgpu::BindGroup,
    count_layout: &'a wgpu::BindGroupLayout,
    arena_layout: &'a wgpu::BindGroupLayout,
    histogram_pipeline: &'a wgpu::ComputePipeline,
    scatter_low_pipeline: &'a wgpu::ComputePipeline,
    scatter_high_pipeline: &'a wgpu::ComputePipeline,
    trim_count_pipeline: &'a wgpu::ComputePipeline,
    trim_scatter_pipeline: &'a wgpu::ComputePipeline,
    groups_for_words: u32,
}

struct FineCsrArena {
    counts: Vec<u32>,
    survivor_count: u32,
    offsets_buffer: wgpu::Buffer,
    arena: wgpu::Buffer,
    histogram_elapsed: Duration,
    scatter_elapsed: Duration,
}

struct FineSeedShard {
    counts: Vec<u32>,
    offsets_buffer: wgpu::Buffer,
    cursors_buffer: wgpu::Buffer,
    arena: wgpu::Buffer,
}

struct SleanBufferSizes {
    arena: u64,
    dead: u64,
}

struct TrimBuffers {
    edges: wgpu::Buffer,
    nodes: wgpu::Buffer,
    edge_backup: Option<wgpu::Buffer>,
    bucket_scratch: wgpu::Buffer,
    slean_second: wgpu::Buffer,
    slean_third: wgpu::Buffer,
    slean_fourth: wgpu::Buffer,
    slean_dead: wgpu::Buffer,
    fine_scratch: wgpu::Buffer,
    fine_dummy: wgpu::Buffer,
}

impl TrimBuffers {
    fn new(
        context: &GpuContext,
        bitmap_bytes: u64,
        bucket_scratch_bytes: u64,
        slean: Option<SleanBufferSizes>,
    ) -> Self {
        let storage = |label, size, usage| {
            context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let storage_copy = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let slean_arena = slean.as_ref().map_or(4, |sizes| sizes.arena);
        let slean_dead = slean.as_ref().map_or(4, |sizes| sizes.dead);
        Self {
            edges: storage("alive-edge-bitmap", bitmap_bytes, storage_copy),
            nodes: storage(
                "occupied-node-bitmap",
                bitmap_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            edge_backup: slean.as_ref().map(|_| {
                storage(
                    "slean exact-fallback edge backup",
                    bitmap_bytes,
                    wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                )
            }),
            bucket_scratch: storage("bucketed-mark-scratch", bucket_scratch_bytes, storage_copy),
            slean_second: storage("slean edge arena second half", slean_arena, storage_copy),
            slean_third: storage("slean edge arena third quarter", slean_arena, storage_copy),
            slean_fourth: storage("slean edge arena fourth quarter", slean_arena, storage_copy),
            slean_dead: storage("slean dead-edge range arena", slean_dead, storage_copy),
            fine_scratch: storage("fine-status-scratch", 4, storage_copy),
            fine_dummy: storage(
                "fine-unused-storage-binding",
                4,
                wgpu::BufferUsages::STORAGE,
            ),
        }
    }
}

struct CoreBindings {
    lean: Vec<wgpu::BindGroup>,
    fine_seed: Vec<wgpu::BindGroup>,
    fine_trim: Vec<wgpu::BindGroup>,
}

impl CoreBindings {
    fn new(
        context: &GpuContext,
        pipelines: &TrimPipelines,
        buffers: &TrimBuffers,
        params: &[Params; 2],
    ) -> Self {
        let uniforms: Vec<_> = params
            .iter()
            .enumerate()
            .map(|(side, params)| {
                context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(if side == 0 { "params-u" } else { "params-v" }),
                        contents: bytemuck::bytes_of(params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    })
            })
            .collect();
        let lean = uniforms
            .iter()
            .enumerate()
            .map(|(side, uniform)| {
                lean_bind_group(
                    context,
                    &pipelines.lean_layout,
                    if side == 0 { "lean-u" } else { "lean-v" },
                    uniform,
                    &buffers.edges,
                    &buffers.nodes,
                    &buffers.bucket_scratch,
                )
            })
            .collect();
        let fine_seed = uniforms
            .iter()
            .enumerate()
            .map(|(side, uniform)| {
                lean_bind_group(
                    context,
                    &pipelines.lean_layout,
                    if side == 0 {
                        "fine-seed-u"
                    } else {
                        "fine-seed-v"
                    },
                    uniform,
                    &buffers.edges,
                    &buffers.fine_dummy,
                    &buffers.fine_scratch,
                )
            })
            .collect();
        let fine_trim = uniforms
            .iter()
            .enumerate()
            .map(|(side, uniform)| {
                lean_bind_group(
                    context,
                    &pipelines.lean_layout,
                    if side == 0 {
                        "fine-trim-u"
                    } else {
                        "fine-trim-v"
                    },
                    uniform,
                    &buffers.fine_dummy,
                    &buffers.fine_dummy,
                    &buffers.fine_scratch,
                )
            })
            .collect();
        Self {
            lean,
            fine_seed,
            fine_trim,
        }
    }
}

struct BucketBindings {
    scatter: Vec<Vec<wgpu::BindGroup>>,
    mark: Vec<Vec<wgpu::BindGroup>>,
}

struct SleanBindings {
    parts: Vec<Vec<wgpu::BindGroup>>,
}

impl SleanBindings {
    fn new(
        solver: &GpuWgpuSolver,
        buffers: &TrimBuffers,
        params: &[Params; 2],
        enabled: bool,
        sizing: &SleanSizing,
        edge_count: u64,
    ) -> Self {
        if !enabled {
            return Self {
                parts: vec![Vec::new(), Vec::new()],
            };
        }
        let context = &solver.context;
        let part_uniforms: Vec<Vec<_>> = params
            .iter()
            .map(|base_params| {
                (0..sizing.parts)
                    .map(|part| {
                        let base = part * sizing.part_edges;
                        let mut params = *base_params;
                        params.diagnostic_chunk_base = base as u32;
                        params.diagnostic_chunk_count =
                            sizing.part_edges.min(edge_count - base) as u32;
                        params.diagnostic_bucket = (sizing.bucket_count as u32) << 16
                            | u32::from(solver.force_slean_overflow.load(Ordering::Relaxed));
                        uniform_buffer(context, "slean part params", &params)
                    })
                    .collect()
            })
            .collect();
        let parts = part_uniforms
            .iter()
            .map(|side| {
                side.iter()
                    .map(|uniform| slean_bind_group(solver, buffers, uniform))
                    .collect()
            })
            .collect();
        Self { parts }
    }
}

fn uniform_buffer(context: &GpuContext, label: &str, params: &Params) -> wgpu::Buffer {
    context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(params),
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

fn slean_bind_group(
    solver: &GpuWgpuSolver,
    buffers: &TrimBuffers,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let resources = [
        uniform,
        &buffers.edges,
        &buffers.nodes,
        &buffers.bucket_scratch,
        &buffers.slean_second,
        &buffers.slean_dead,
        &buffers.slean_third,
        &buffers.slean_fourth,
    ];
    let entries: Vec<_> = resources
        .into_iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    solver
        .context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slean part"),
            layout: &solver.pipelines.slean_layout,
            entries: &entries,
        })
}

impl BucketBindings {
    fn new(
        context: &GpuContext,
        pipelines: &TrimPipelines,
        buffers: &TrimBuffers,
        params: &[Params; 2],
        enabled: bool,
        edge_count: u64,
        chunk_edges: u64,
    ) -> Self {
        if !enabled {
            return Self {
                scatter: vec![Vec::new(), Vec::new()],
                mark: vec![Vec::new(), Vec::new()],
            };
        }
        let make_uniform = |label, params: &Params| {
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::bytes_of(params),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        };
        let scatter = params
            .iter()
            .map(|base_params| {
                (0..edge_count.div_ceil(chunk_edges))
                    .map(|chunk| {
                        let base = chunk * chunk_edges;
                        let count = chunk_edges.min(edge_count - base);
                        let mut params = *base_params;
                        params.diagnostic_chunk_base = base as u32;
                        params.diagnostic_chunk_count = count as u32;
                        params.diagnostic_bucket = BUCKETED_MARK_BUCKETS << 16;
                        make_uniform("bucketed alive scatter params", &params)
                    })
                    .map(|uniform| {
                        bucket_bind_group(
                            context,
                            pipelines,
                            buffers,
                            "bucketed alive scatter",
                            &uniform,
                        )
                    })
                    .collect()
            })
            .collect();
        let mark = params
            .iter()
            .map(|base_params| {
                (0..BUCKETED_MARK_BUCKETS)
                    .map(|bucket| {
                        let mut params = *base_params;
                        params.diagnostic_chunk_count = chunk_edges as u32;
                        params.diagnostic_bucket = (BUCKETED_MARK_BUCKETS << 16) | bucket;
                        make_uniform("bucketed mark params", &params)
                    })
                    .map(|uniform| {
                        bucket_bind_group(
                            context,
                            pipelines,
                            buffers,
                            "bucketed node mark",
                            &uniform,
                        )
                    })
                    .collect()
            })
            .collect();
        Self { scatter, mark }
    }
}

fn bucket_bind_group(
    context: &GpuContext,
    pipelines: &TrimPipelines,
    buffers: &TrimBuffers,
    label: &str,
    uniform: &wgpu::Buffer,
) -> wgpu::BindGroup {
    lean_bind_group(
        context,
        &pipelines.lean_layout,
        label,
        uniform,
        &buffers.edges,
        &buffers.nodes,
        &buffers.bucket_scratch,
    )
}

fn lean_bind_group(
    context: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    uniform: &wgpu::Buffer,
    first: &wgpu::Buffer,
    second: &wgpu::Buffer,
    scratch: &wgpu::Buffer,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: first.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: second.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch.as_entire_binding(),
                },
            ],
        })
}

fn single_storage_bind_group(
    context: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
}

fn fine_arena_bind_group(
    context: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    first: &wgpu::Buffer,
    second: &wgpu::Buffer,
    arena: &wgpu::Buffer,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: first.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: second.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: arena.as_entire_binding(),
                },
            ],
        })
}

struct TrimPipelines {
    lean_layout: wgpu::BindGroupLayout,
    slean_layout: wgpu::BindGroupLayout,
    fine_count_layout: wgpu::BindGroupLayout,
    fine_arena_layout: wgpu::BindGroupLayout,
    init: wgpu::ComputePipeline,
    clear: wgpu::ComputePipeline,
    siphash_only: wgpu::ComputePipeline,
    count_alive: wgpu::ComputePipeline,
    dense_mark: wgpu::ComputePipeline,
    bucket_scatter: wgpu::ComputePipeline,
    bucket_scatter_dense_staged: wgpu::ComputePipeline,
    bucket_scatter_alive: wgpu::ComputePipeline,
    bucket_scatter_alive_pairs: wgpu::ComputePipeline,
    bucket_scatter_dense_pairs_staged: wgpu::ComputePipeline,
    bucket_mark: wgpu::ComputePipeline,
    bucket_trim_pairs: wgpu::ComputePipeline,
    slean_scatter_dense: wgpu::ComputePipeline,
    slean_scatter_alive: wgpu::ComputePipeline,
    slean_mark: wgpu::ComputePipeline,
    slean_trim: wgpu::ComputePipeline,
    slean_mark_and_trim_final: wgpu::ComputePipeline,
    slean_apply_deaths: wgpu::ComputePipeline,
    mark: wgpu::ComputePipeline,
    trim: wgpu::ComputePipeline,
    fine_histogram: wgpu::ComputePipeline,
    fine_scatter: wgpu::ComputePipeline,
    fine_scatter_low: wgpu::ComputePipeline,
    fine_scatter_high: wgpu::ComputePipeline,
    fine_verify: wgpu::ComputePipeline,
    fine_trim_count: wgpu::ComputePipeline,
    fine_trim_scatter: wgpu::ComputePipeline,
    fine_trim_fixed: wgpu::ComputePipeline,
    fine_emit_bitmap: wgpu::ComputePipeline,
    compare_bitmaps: wgpu::ComputePipeline,
}

impl TrimPipelines {
    fn new(context: &GpuContext) -> Self {
        let lean_layout = bind_group_layout(&context.device, "lean-bindings", 4, true, false);
        let slean_layout = bind_group_layout(&context.device, "slean bindings", 8, true, false);
        let fine_count_layout =
            bind_group_layout(&context.device, "fine CSR count layout", 1, false, false);
        let fine_arena_layout =
            bind_group_layout(&context.device, "fine CSR arena layout", 3, false, true);

        let lean = [&lean_layout];
        let slean = [&slean_layout];
        let fine_count = [&lean_layout, &fine_count_layout];
        let fine_arena = [&lean_layout, &fine_count_layout, &fine_arena_layout];
        let fine_ping_pong = [
            &lean_layout,
            &fine_count_layout,
            &fine_arena_layout,
            &fine_arena_layout,
        ];
        let make =
            |entry, layouts: &[&wgpu::BindGroupLayout]| compute_pipeline(context, entry, layouts);

        Self {
            init: make("init_edges", &lean),
            clear: make("clear_nodes", &lean),
            siphash_only: make("siphash_only", &lean),
            count_alive: make("count_alive_edges", &lean),
            dense_mark: make("dense_mark_nodes", &lean),
            bucket_scatter: make("bucket_scatter_nodes", &lean),
            bucket_scatter_dense_staged: make("bucket_scatter_dense_nodes_staged", &lean),
            bucket_scatter_alive: make("bucket_scatter_alive_nodes", &lean),
            bucket_scatter_alive_pairs: make("bucket_scatter_alive_pairs", &lean),
            bucket_scatter_dense_pairs_staged: make("bucket_scatter_dense_pairs_staged", &lean),
            bucket_mark: make("bucket_mark_nodes", &lean),
            bucket_trim_pairs: make("bucket_trim_pairs", &lean),
            slean_scatter_dense: make("slean_scatter_dense", &slean),
            slean_scatter_alive: make("slean_scatter_alive", &slean),
            slean_mark: make("slean_mark_buckets", &slean),
            slean_trim: make("slean_trim_buckets", &slean),
            slean_mark_and_trim_final: make("slean_mark_and_trim_final_part", &slean),
            slean_apply_deaths: make("slean_apply_deaths", &slean),
            mark: make("mark_nodes", &lean),
            trim: make("trim_edges", &lean),
            fine_histogram: make("fine_histogram_alive", &fine_count),
            fine_scatter: make("fine_scatter_alive", &fine_arena),
            fine_scatter_low: make("fine_scatter_alive_low", &fine_arena),
            fine_scatter_high: make("fine_scatter_alive_high", &fine_arena),
            fine_verify: make("fine_verify_arena", &fine_arena),
            fine_trim_count: make("fine_trim_count", &fine_arena),
            fine_trim_scatter: make("fine_trim_scatter", &fine_ping_pong),
            fine_trim_fixed: make("fine_trim_fixed", &fine_ping_pong),
            fine_emit_bitmap: make("fine_emit_bitmap", &fine_arena),
            compare_bitmaps: make("compare_edge_node_bitmaps", &lean),
            lean_layout,
            slean_layout,
            fine_count_layout,
            fine_arena_layout,
        }
    }
}

fn bind_group_layout(
    device: &wgpu::Device,
    label: &'static str,
    bindings: u32,
    uniform_first: bool,
    first_read_only: bool,
) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = (0..bindings)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 0 && uniform_first {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: first_read_only && binding == 0,
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn compute_pipeline(
    context: &GpuContext,
    entry_point: &'static str,
    layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::ComputePipeline {
    let layouts: Vec<_> = layouts.iter().copied().map(Some).collect();
    let layout = context
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry_point),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
    context
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&layout),
            module: &context.module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
}

impl PreparedTrim {
    fn run_diagnostic(
        &self,
        solver: &GpuWgpuSolver,
        request: GraphParams,
    ) -> Result<Option<TrimOutcome>, SolveError> {
        if self.diagnostics.siphash_only {
            return solver
                .run_siphash_diagnostic(
                    request,
                    self.graph.edge_count,
                    self.groups_for_words,
                    &self.core.lean[0],
                )
                .map(Some);
        }
        if self.diagnostics.bucket_round0 {
            return solver
                .run_bucket_round0_diagnostic(
                    request,
                    self.diagnostics.bucket_count,
                    BucketRound0Resources {
                        edges: &self.buffers.edges,
                        nodes: &self.buffers.nodes,
                        scratch: &self.buffers.bucket_scratch,
                        layout: &solver.pipelines.lean_layout,
                        dense_mark_pipeline: &solver.pipelines.dense_mark,
                        scatter_pipeline: &solver.pipelines.bucket_scatter,
                        bucket_mark_pipeline: &solver.pipelines.bucket_mark,
                        baseline_bind_group: &self.core.lean[0],
                        groups_for_words: self.groups_for_words,
                    },
                )
                .map(Some);
        }
        Ok(None)
    }

    fn execute(
        &self,
        solver: &GpuWgpuSolver,
        request: GraphParams,
        live_work: bool,
        cancel: &AtomicBool,
    ) -> Result<TrimOutcome, SolveError> {
        if self.diagnostics.fine_csr || self.production_fine {
            solver.fine_rounds.store(0, Ordering::Relaxed);
            solver.fine_mismatched_words.store(0, Ordering::Relaxed);
        }
        if let Some(bits) = self.diagnostics.mask_bits {
            eprintln!(
                "WARNING: diagnostic node mask C{bits} is active; results are not valid proofs"
            );
        }
        if let Some(outcome) = self.run_diagnostic(solver, request)? {
            return Ok(outcome);
        }
        solver.initialize_edges(&self.core.lean[0], self.groups_for_words);
        let fine_transition_round = if self.production_fine {
            if self.production_slean { 3 } else { 2 }
        } else {
            6
        };
        let bucket_rounds = if self.production_fine {
            if self.production_slean {
                3
            } else {
                fine_transition_round
            }
        } else {
            BUCKETED_MARK_ROUNDS
        };
        TrimRun {
            solver,
            request,
            rounds: self.rounds,
            live_work,
            cancel,
            diagnostics: self.diagnostics,
            production_fine: self.production_fine,
            production_slean: self.production_slean,
            production_bucketed_mark: self.production_bucketed_mark,
            bucket_rounds,
            fine_transition_round,
            fine_end_round: if self.production_fine {
                self.rounds
            } else {
                self.diagnostics.fine_end_round
            },
            sharded_fine_seed: self.production_fine && fine_transition_round == 2,
            graph: self.graph,
            buffers: &self.buffers,
            core: &self.core,
            bucket: &self.bucket,
            slean: &self.slean,
            slean_sizing: self.slean_sizing,
        }
        .run()
    }
}

impl TrimRun<'_> {
    fn report_round(&self, completed_round: u32) -> Result<Option<Vec<u64>>, SolveError> {
        if self.diagnostics.survivor_counts && matches!(completed_round, 1 | 2 | 3 | 4 | 6 | 8) {
            let survivors = self.solver.count_alive_edges(
                &self.buffers.bucket_scratch,
                &self.core.lean[0],
                &self.solver.pipelines.count_alive,
                self.groups_for_words(),
            )?;
            let one_arena_gib = survivors as f64 * 4.0 / 1024_f64.powi(3);
            eprintln!(
                "C{} survivor-count round={} survivors={} one_u32_arena={:.3}GiB two_u32_arenas={:.3}GiB",
                self.request.edge_bits,
                completed_round,
                survivors,
                one_arena_gib,
                one_arena_gib * 2.0,
            );
        }
        if (self.diagnostics.fine_csr || self.production_fine)
            && completed_round == self.fine_transition_round
            && completed_round < self.fine_end_round
        {
            return self.solver.run_fine_transition(
                self.request,
                FineTransitionResources {
                    buffers: self.buffers,
                    bindings: self.core,
                    completed_round,
                    end_round: self.fine_end_round,
                    production: self.production_fine,
                    sharded_seed: self.sharded_fine_seed,
                    validate_output: self.diagnostics.fine_csr,
                    groups_for_words: self.groups_for_words(),
                },
                self.cancel,
            );
        }
        Ok(None)
    }

    fn run_dense_round(&self, round: u32) -> Result<bool, SolveError> {
        let side = (round & 1) as usize;
        self.solver.run_slean_dense_round(
            self.request,
            round,
            SleanDenseResources {
                edges: &self.buffers.edges,
                nodes: &self.buffers.nodes,
                backup: self
                    .buffers
                    .edge_backup
                    .as_ref()
                    .expect("production slean has an edge backup"),
                arenas: [
                    &self.buffers.bucket_scratch,
                    &self.buffers.slean_second,
                    &self.buffers.slean_third,
                    &self.buffers.slean_fourth,
                ],
                dead_arena: &self.buffers.slean_dead,
                part_bind_groups: &self.slean.parts[side],
                lean_bind_groups: &self.core.lean,
                sizing: self.slean_sizing,
                bitmap_bytes: self.graph.bitmap_bytes,
                groups_for_words: self.groups_for_words(),
                measure_phases: self.diagnostics.slean_phases,
            },
            self.cancel,
        )
    }

    fn run_bucket_round(&self, round: u32) -> Result<bool, SolveError> {
        self.solver.bucketed_rounds.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let side = (round & 1) as usize;
        let resources = BucketRoundResources {
            nodes: &self.buffers.nodes,
            scratch: &self.buffers.bucket_scratch,
            scatter: &self.bucket.scatter[side],
            mark: &self.bucket.mark[side],
            lean: &self.core.lean[side],
            groups_for_words: self.groups_for_words(),
        };
        let (completed, mark_elapsed) = self.solver.mark_bucket_round(
            round,
            &resources,
            self.cancel,
            self.diagnostics.early_phases,
        )?;
        if !completed {
            return Ok(false);
        }
        let Some(overflow) = self
            .solver
            .trim_bucket_round(round, &resources, self.cancel)?
        else {
            return Ok(false);
        };
        eprintln!(
            "C{} bucketed-mark-trim round={} buckets={} overflow={} fallback={} time={:.3}s",
            self.request.edge_bits,
            round,
            BUCKETED_MARK_BUCKETS,
            overflow,
            overflow != 0,
            started.elapsed().as_secs_f64(),
        );
        if let Some(mark_elapsed) = mark_elapsed {
            let total = started.elapsed();
            eprintln!(
                "C{} early-phase round={} mark={:.3}s trim={:.3}s total={:.3}s",
                self.request.edge_bits,
                round,
                mark_elapsed.as_secs_f64(),
                total.saturating_sub(mark_elapsed).as_secs_f64(),
                total.as_secs_f64(),
            );
        }
        Ok(true)
    }

    fn run(&self) -> Result<TrimOutcome, SolveError> {
        let mut round = 0;
        while round < self.rounds {
            if self.cancel.load(Ordering::Relaxed) {
                return Ok(TrimOutcome::Cancelled);
            }
            if self.production_slean && round < self.bucket_rounds {
                if !self.run_dense_round(round)? {
                    return Ok(TrimOutcome::Cancelled);
                }
                round += 1;
            } else if self.production_bucketed_mark
                && !self.production_slean
                && round < self.bucket_rounds
            {
                if !self.run_bucket_round(round)? {
                    return Ok(TrimOutcome::Cancelled);
                }
                round += 1;
            } else {
                let one_at_a_time = self.diagnostics.per_round
                    || self.diagnostics.survivor_counts
                    || self.diagnostics.fine_csr
                    || self.live_work;
                round = self.solver.run_lean_batch(
                    self.request,
                    round,
                    self.rounds,
                    one_at_a_time,
                    self.groups_for_words(),
                    &self.core.lean,
                )?;
            }
            if self.cancel.load(Ordering::Relaxed) {
                return Ok(TrimOutcome::Cancelled);
            }
            if let Some(survivors) = self.report_round(round)? {
                return Ok(TrimOutcome::Survivors(survivors));
            }
        }
        if self.diagnostics.survivor_counts {
            return Ok(TrimOutcome::Diagnostic(
                "round-4/6/8 survivor-count diagnostic complete",
            ));
        }
        if self.diagnostics.fine_csr {
            return Ok(TrimOutcome::Diagnostic(
                "round-6 compact fine-CSR diagnostic complete",
            ));
        }
        if self.diagnostics.mask_bits.is_some() {
            return Ok(TrimOutcome::Diagnostic(
                "node-mask locality diagnostic complete",
            ));
        }
        Ok(TrimOutcome::Survivors(self.solver.read_bitmap_survivors(
            &self.buffers.edges,
            self.graph.edge_count,
            self.graph.bitmap_bytes,
        )?))
    }

    fn groups_for_words(&self) -> u32 {
        self.graph
            .word_count
            .div_ceil(u64::from(WORKGROUP_SIZE))
            .min(u64::from(DISPATCH_GROUPS)) as u32
    }
}

impl GpuWgpuSolver {
    pub fn new() -> Result<Self, SolveError> {
        Self::new_with_config(GpuWgpuConfig::default())
    }

    pub fn new_with_config(config: GpuWgpuConfig) -> Result<Self, SolveError> {
        if config.slean_parts < 2 || !config.slean_parts.is_power_of_two() {
            return Err(SolveError::InvalidConfig(
                "slean_parts must be a power of two and at least 2".into(),
            ));
        }
        if config.local_ram_kib < 2 || !config.local_ram_kib.is_power_of_two() {
            return Err(SolveError::InvalidConfig(
                "local_ram_kib must be a power of two and at least 2".into(),
            ));
        }
        pollster::block_on(Self::new_async(config))
    }

    async fn new_async(config: GpuWgpuConfig) -> Result<Self, SolveError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|error| SolveError::Gpu(format!("requesting adapter: {error}")))?;
        let limits = adapter.limits();
        let adapter_name = adapter.get_info().name;
        let native_int64 = adapter.features().contains(wgpu::Features::SHADER_INT64)
            && std::env::var_os("GRIN_MINER_DISABLE_NATIVE_INT64").is_none();
        let required_features = if native_int64 {
            wgpu::Features::SHADER_INT64
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("cuckatoo-lean-device"),
                required_features,
                required_limits: limits.clone(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| SolveError::Gpu(format!("requesting device: {error}")))?;
        let shader = shader_source(native_int64)?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cuckatoo-lean-wgsl"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        let context = GpuContext {
            device,
            queue,
            module,
            limits,
            adapter_name,
            native_int64,
        };
        let pipelines = TrimPipelines::new(&context);
        Ok(Self {
            context,
            pipelines,
            config,
            bucketed_rounds: AtomicU32::new(0),
            bucket_trim_rounds: AtomicU32::new(0),
            bucket_fallbacks: AtomicU32::new(0),
            fine_rounds: AtomicU32::new(0),
            fine_mismatched_words: AtomicU32::new(0),
            force_slean_overflow: AtomicBool::new(false),
            #[cfg(test)]
            verify_slean_seed: AtomicBool::new(false),
            #[cfg(test)]
            slean_seed_checks: AtomicU32::new(0),
        })
    }

    fn selected_slean_parts(&self, enabled: bool) -> u64 {
        let mut parts = u64::from(self.config.slean_parts);
        if enabled
            && self.config.trimming == TrimmingMode::Auto
            && let Some(available) = available_memory_bytes()
        {
            parts = if available >= SLEAN_PARTS_TWO_AVAILABLE_MEMORY {
                2
            } else {
                4
            };
            eprintln!(
                "slean auto memory={:.1}GiB selected_parts={parts}",
                available as f64 / 1024_f64.powi(3)
            );
        }
        if enabled
            && !cfg!(test)
            && parts == 2
            && available_memory_bytes()
                .is_none_or(|available| available < SLEAN_PARTS_TWO_AVAILABLE_MEMORY)
        {
            eprintln!("slean parts increased 2->4: parts=2 needs at least 18 GiB available memory");
            parts = 4;
        }
        parts
    }

    fn slean_sizing(
        &self,
        enabled: bool,
        edge_count: u64,
        bitmap_bytes: u64,
    ) -> Result<SleanSizing, SolveError> {
        let adapter_local_bytes = u64::from(self.context.limits.max_compute_workgroup_storage_size);
        let requested_local_bytes = u64::from(self.config.local_ram_kib) * 1024;
        let local_bytes = requested_local_bytes
            .min(adapter_local_bytes)
            .min(bitmap_bytes);
        if enabled && requested_local_bytes > adapter_local_bytes {
            eprintln!(
                "slean local RAM request {} KiB reduced to adapter limit {} KiB",
                self.config.local_ram_kib,
                adapter_local_bytes / 1024
            );
        }
        let bucket_count = bitmap_bytes.div_ceil(local_bytes);
        if enabled && (!bucket_count.is_power_of_two() || bucket_count > u64::from(u16::MAX)) {
            return Err(SolveError::Unsupported(format!(
                "slean requires a power-of-two bucket count fitting u16, got {bucket_count}"
            )));
        }

        let mut parts = self.selected_slean_parts(enabled);
        let (part_edges, capacity, scratch_bytes, dead_bucket_count, early_dead_bytes) = loop {
            if parts > edge_count {
                return Err(SolveError::Unsupported(
                    "slean cannot fit one part in the adapter's storage binding limit".into(),
                ));
            }
            let part_edges = edge_count.div_ceil(parts);
            let base = part_edges.div_ceil(bucket_count);
            let capacity = if self.force_slean_overflow.load(Ordering::Relaxed) {
                1
            } else {
                base + base.div_ceil(20) + 64
            };
            let buckets_per_buffer = bucket_count / 4;
            let bytes = (buckets_per_buffer + 1 + buckets_per_buffer * capacity) * 4;
            let dead_bucket_count = part_edges.div_ceil(8).div_ceil(local_bytes).max(1);
            let dead_base = part_edges.div_ceil(dead_bucket_count);
            let dead_capacity = (dead_base * 45).div_ceil(100) + 64;
            let dead_bytes = (dead_bucket_count + 1 + dead_bucket_count * dead_capacity) * 4;
            if !enabled
                || (bytes <= self.context.limits.max_buffer_size
                    && bytes <= self.context.limits.max_storage_buffer_binding_size
                    && dead_bytes <= self.context.limits.max_buffer_size
                    && dead_bytes <= self.context.limits.max_storage_buffer_binding_size)
            {
                break (part_edges, capacity, bytes, dead_bucket_count, dead_bytes);
            }
            parts *= 2;
        };
        if enabled && parts != u64::from(self.config.slean_parts) {
            eprintln!(
                "slean parts increased {}->{} to fit a {:.3} GiB storage binding",
                self.config.slean_parts,
                parts,
                scratch_bytes as f64 / 1024_f64.powi(3),
            );
        }
        Ok(SleanSizing {
            bucket_count,
            parts,
            part_edges,
            capacity,
            scratch_bytes,
            dead_bucket_count,
            dead_scratch_bytes: early_dead_bytes,
        })
    }

    fn count_alive_edges(
        &self,
        scratch: &wgpu::Buffer,
        bind_group: &wgpu::BindGroup,
        pipeline: &wgpu::ComputePipeline,
        groups_for_words: u32,
    ) -> Result<u64, SolveError> {
        let staging = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("alive-edge-count-readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("count alive edges"),
                });
        encoder.clear_buffer(scratch, 0, Some(4));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("alive-edge popcount"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups_for_words, 1, 1);
        }
        encoder.copy_buffer_to_buffer(scratch, 0, &staging, 0, 4);
        let submission = self.context.submit(encoder);
        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.context.wait(submission, "edge count")?;
        receiver
            .recv()
            .map_err(|error| SolveError::Gpu(format!("map callback: {error}")))?
            .map_err(|error| SolveError::Gpu(format!("mapping edge count: {error}")))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| SolveError::Gpu(format!("accessing edge count: {error}")))?;
        Ok(u64::from(bytemuck::cast_slice::<u8, u32>(&mapped)[0]))
    }

    #[cfg(test)]
    fn verify_slean_seed_part(&self, check: SleanSeedCheck<'_>) -> Result<(), SolveError> {
        let SleanSeedCheck {
            request,
            side,
            part_base,
            part_count,
            bucket_count,
            capacity,
            scratch_bytes,
            arenas,
        } = check;
        let width = bucket_count / 4;
        let header = width + 1;
        let arena_words = arenas
            .iter()
            .enumerate()
            .map(|(index, arena)| {
                let label = match index {
                    0 => "slean seed arena 0",
                    1 => "slean seed arena 1",
                    2 => "slean seed arena 2",
                    _ => "slean seed arena 3",
                };
                self.context.read_u32(arena, scratch_bytes, label)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if arena_words.iter().any(|words| words[width as usize] != 0) {
            return Err(SolveError::Gpu("slean seed oracle overflowed".into()));
        }

        let mut actual = vec![Vec::<u32>::new(); bucket_count as usize];
        for bucket in 0..bucket_count {
            let shard = bucket / width;
            let local = bucket - shard * width;
            let words = &arena_words[shard as usize];
            let count = u64::from(words[local as usize]);
            let start = header + local * capacity;
            actual[bucket as usize]
                .extend_from_slice(&words[start as usize..(start + count.min(capacity)) as usize]);
        }

        let bucket_shift = request.edge_bits - bucket_count.ilog2() as u8;
        let mut expected = vec![Vec::<u32>::new(); bucket_count as usize];
        for edge in part_base..part_base + part_count {
            let node = crate::siphash::endpoint(request.keys, request.edge_bits, edge, side as u8);
            expected[(node >> bucket_shift) as usize].push(edge as u32);
        }
        for (actual_bucket, expected_bucket) in actual.iter_mut().zip(expected.iter_mut()) {
            actual_bucket.sort_unstable();
            expected_bucket.sort_unstable();
        }
        if actual != expected {
            return Err(SolveError::Gpu(
                "slean GPU seed buckets differ from the CPU oracle".into(),
            ));
        }
        self.slean_seed_checks.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn initialize_edges(&self, bind_group: &wgpu::BindGroup, groups_for_words: u32) {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("lean-init"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("init edge bitmap"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.init);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups_for_words, 1, 1);
        }
        self.context.submit(encoder);
    }

    fn read_bitmap_survivors(
        &self,
        edges: &wgpu::Buffer,
        edge_count: u64,
        bitmap_bytes: u64,
    ) -> Result<Vec<u64>, SolveError> {
        let words = self
            .context
            .read_u32(edges, bitmap_bytes, "edge-bitmap-readback")?;
        let mut survivors = Vec::new();
        for (word_index, word) in words.into_iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros();
                let edge = word_index as u64 * 32 + u64::from(bit);
                if edge < edge_count {
                    survivors.push(edge);
                }
                bits &= bits - 1;
            }
        }
        Ok(survivors)
    }

    fn mark_bucket_round(
        &self,
        round: u32,
        resources: &BucketRoundResources<'_>,
        cancel: &AtomicBool,
        measure: bool,
    ) -> Result<(bool, Option<Duration>), SolveError> {
        let started = Instant::now();
        let mut clear =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("bucketed early-round clear"),
                });
        clear.clear_buffer(resources.nodes, 0, None);
        clear.clear_buffer(
            resources.scratch,
            0,
            Some(DIAGNOSTIC_BUCKET_HEADER_WORDS * 4),
        );
        self.context.submit(clear);
        let mut last_submission = None;
        for scatter in resources.scatter {
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("bucketed early-round chunk"),
                    });
            encoder.clear_buffer(
                resources.scratch,
                0,
                Some(u64::from(BUCKETED_MARK_BUCKETS) * 4),
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bucketed alive-edge scatter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(if round == 0 {
                    &self.pipelines.bucket_scatter_dense_staged
                } else {
                    &self.pipelines.bucket_scatter_alive
                });
                pass.set_bind_group(0, scatter, &[]);
                pass.dispatch_workgroups(DISPATCH_GROUPS, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bucketed node mark sequence"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.bucket_mark);
                for mark in resources.mark {
                    pass.set_bind_group(0, mark, &[]);
                    pass.dispatch_workgroups(DISPATCH_GROUPS, 1, 1);
                }
            }
            last_submission = Some(self.context.submit(encoder));
            if cancel.load(Ordering::Relaxed) {
                return Ok((false, None));
            }
        }
        if measure {
            if let Some(submission) = last_submission {
                self.context.wait(submission, "bucketed mark diagnostic")?;
            }
            Ok((true, Some(started.elapsed())))
        } else {
            Ok((true, None))
        }
    }

    fn trim_bucket_round(
        &self,
        round: u32,
        resources: &BucketRoundResources<'_>,
        cancel: &AtomicBool,
    ) -> Result<Option<u32>, SolveError> {
        for scatter in resources.scatter {
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("bucketed early-round pair trim chunk"),
                    });
            encoder.clear_buffer(
                resources.scratch,
                0,
                Some(u64::from(BUCKETED_MARK_BUCKETS) * 4),
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bucketed pair scatter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(if round == 0 {
                    &self.pipelines.bucket_scatter_dense_pairs_staged
                } else {
                    &self.pipelines.bucket_scatter_alive_pairs
                });
                pass.set_bind_group(0, scatter, &[]);
                pass.dispatch_workgroups(DISPATCH_GROUPS, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cache-local bucket trim sequence"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipelines.bucket_trim_pairs);
                for mark in resources.mark {
                    pass.set_bind_group(0, mark, &[]);
                    pass.dispatch_workgroups(DISPATCH_GROUPS, 1, 1);
                }
            }
            self.context.submit(encoder);
            if cancel.load(Ordering::Relaxed) {
                return Ok(None);
            }
        }
        let overflow = self.context.read_u32_at(
            resources.scratch,
            DIAGNOSTIC_BUCKET_OVERFLOW_WORD * 4,
            4,
            "bucketed round overflow",
        )?[0];
        if overflow != 0 {
            self.bucket_fallbacks.fetch_add(1, Ordering::Relaxed);
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("bucketed early-round direct fallback"),
                    });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bucketed mark direct fallback"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, resources.lean, &[]);
            for pipeline in [
                &self.pipelines.clear,
                &self.pipelines.mark,
                &self.pipelines.trim,
            ] {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(resources.groups_for_words, 1, 1);
            }
            drop(pass);
            self.context.run(encoder, "direct trim fallback")?;
        } else {
            self.bucket_trim_rounds.fetch_add(1, Ordering::Relaxed);
        }
        Ok(Some(overflow))
    }

    fn run_lean_batch(
        &self,
        request: GraphParams,
        first_round: u32,
        rounds: u32,
        one_round_at_a_time: bool,
        groups_for_words: u32,
        bind_groups: &[wgpu::BindGroup],
    ) -> Result<u32, SolveError> {
        let started = Instant::now();
        let batch_size = if one_round_at_a_time {
            1
        } else {
            ROUNDS_PER_SUBMISSION
        };
        let batch_end = (first_round + batch_size).min(rounds);
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("lean-round-batch"),
                });
        self.encode_lean_rounds(
            &mut encoder,
            first_round..batch_end,
            groups_for_words,
            bind_groups,
        );
        self.context.run(encoder, "trim batch")?;
        eprintln!(
            "C{} batch={}-{} time={:.3}s",
            request.edge_bits,
            first_round,
            batch_end - 1,
            started.elapsed().as_secs_f64()
        );
        Ok(batch_end)
    }

    fn run_slean_mark_phase(
        &self,
        _request: GraphParams,
        round: u32,
        resources: &SleanDenseResources<'_>,
        started: Instant,
        cancel: &AtomicBool,
    ) -> Result<(bool, Option<Duration>), SolveError> {
        let sizing = resources.sizing;
        let counts_bytes = sizing.bucket_count / 4 * 4;
        let header_bytes = counts_bytes + 4;
        let dead_header_bytes = (sizing.dead_bucket_count + 1) * 4;
        let scatter = if round == 0 {
            &self.pipelines.slean_scatter_dense
        } else {
            &self.pipelines.slean_scatter_alive
        };
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("slean batched seed and mark"),
                });
        encoder.clear_buffer(resources.nodes, 0, None);
        for arena in resources.arenas {
            encoder.clear_buffer(arena, 0, Some(header_bytes));
        }
        encoder.clear_buffer(resources.dead_arena, 0, Some(dead_header_bytes));
        for (part_index, bind_group) in resources.part_bind_groups.iter().enumerate() {
            for arena in resources.arenas {
                encoder.clear_buffer(arena, 0, Some(counts_bytes));
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("slean part seed and mark"),
                timestamp_writes: None,
            });
            pass.set_pipeline(scatter);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(DISPATCH_GROUPS, 1, 1);
            if part_index + 1 != resources.part_bind_groups.len() {
                pass.set_pipeline(&self.pipelines.slean_mark);
                pass.dispatch_workgroups(sizing.bucket_count as u32, 1, 1);
            }
            drop(pass);
            if cancel.load(Ordering::Relaxed) {
                return Ok((false, None));
            }
        }
        let submission = self.context.submit(encoder);
        #[cfg(test)]
        if round == 0 && self.verify_slean_seed.load(Ordering::Relaxed) {
            let last_part = sizing.parts - 1;
            let edge_count = 1_u64 << _request.edge_bits;
            let part_base = last_part * sizing.part_edges;
            self.verify_slean_seed_part(SleanSeedCheck {
                request: _request,
                side: round & 1,
                part_base,
                part_count: sizing.part_edges.min(edge_count - part_base),
                bucket_count: sizing.bucket_count,
                capacity: sizing.capacity,
                scratch_bytes: sizing.scratch_bytes,
                arenas: resources.arenas,
            })?;
        }
        let elapsed = if resources.measure_phases {
            self.context.wait(submission, "slean mark diagnostic")?;
            Some(started.elapsed())
        } else {
            None
        };
        Ok((true, elapsed))
    }

    fn run_slean_trim_phase(
        &self,
        round: u32,
        resources: &SleanDenseResources<'_>,
        cancel: &AtomicBool,
    ) -> Result<Option<u32>, SolveError> {
        let sizing = resources.sizing;
        let counts_bytes = sizing.bucket_count / 4 * 4;
        let dead_counts_bytes = sizing.dead_bucket_count * 4;
        let scatter = if round == 0 {
            &self.pipelines.slean_scatter_dense
        } else {
            &self.pipelines.slean_scatter_alive
        };
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("slean batched backup and trim"),
                });
        encoder.copy_buffer_to_buffer(
            resources.edges,
            0,
            resources.backup,
            0,
            resources.bitmap_bytes,
        );
        let (last_part, earlier_parts) = resources
            .part_bind_groups
            .split_last()
            .expect("production slean has at least one part");
        encoder.clear_buffer(resources.dead_arena, 0, Some(dead_counts_bytes));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("slean resident final-part trim"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, last_part, &[]);
            pass.set_pipeline(&self.pipelines.slean_mark_and_trim_final);
            pass.dispatch_workgroups(sizing.bucket_count as u32, 1, 1);
            pass.set_pipeline(&self.pipelines.slean_apply_deaths);
            pass.dispatch_workgroups(sizing.dead_bucket_count as u32, 1, 1);
        }
        for bind_group in earlier_parts {
            for arena in resources.arenas {
                encoder.clear_buffer(arena, 0, Some(counts_bytes));
            }
            encoder.clear_buffer(resources.dead_arena, 0, Some(dead_counts_bytes));
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("slean part seed and trim"),
                timestamp_writes: None,
            });
            pass.set_pipeline(scatter);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(DISPATCH_GROUPS, 1, 1);
            pass.set_pipeline(&self.pipelines.slean_trim);
            pass.dispatch_workgroups(sizing.bucket_count as u32, 1, 1);
            pass.set_pipeline(&self.pipelines.slean_apply_deaths);
            pass.dispatch_workgroups(sizing.dead_bucket_count as u32, 1, 1);
            drop(pass);
            if cancel.load(Ordering::Relaxed) {
                return Ok(None);
            }
        }
        self.context.submit(encoder);
        let values = self.context.read_u32_offsets(
            &[
                (resources.arenas[0], counts_bytes),
                (resources.arenas[1], counts_bytes),
                (resources.arenas[2], counts_bytes),
                (resources.arenas[3], counts_bytes),
                (resources.dead_arena, dead_counts_bytes),
            ],
            "slean combined overflow readback",
        )?;
        Ok(Some(values.into_iter().fold(0_u32, u32::saturating_add)))
    }

    fn finish_bucket_round(
        &self,
        round: u32,
        overflow: u32,
        resources: LeanRoundFallback<'_>,
    ) -> Result<(), SolveError> {
        if overflow == 0 {
            self.bucket_trim_rounds.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        self.bucket_fallbacks.fetch_add(1, Ordering::Relaxed);
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(resources.label),
                });
        encoder.copy_buffer_to_buffer(
            resources.backup,
            0,
            resources.edges,
            0,
            resources.bitmap_bytes,
        );
        self.encode_lean_rounds(
            &mut encoder,
            round..round + 1,
            resources.groups_for_words,
            resources.bind_groups,
        );
        if resources.wait {
            self.context.run(encoder, "slean fallback")
        } else {
            self.context.submit(encoder);
            Ok(())
        }
    }

    fn run_slean_dense_round(
        &self,
        request: GraphParams,
        round: u32,
        resources: SleanDenseResources<'_>,
        cancel: &AtomicBool,
    ) -> Result<bool, SolveError> {
        self.bucketed_rounds.fetch_add(1, Ordering::Relaxed);
        let started = Instant::now();
        let (completed, mark_elapsed) =
            self.run_slean_mark_phase(request, round, &resources, started, cancel)?;
        if !completed {
            return Ok(false);
        }
        let Some(overflow) = self.run_slean_trim_phase(round, &resources, cancel)? else {
            return Ok(false);
        };
        self.finish_bucket_round(
            round,
            overflow,
            LeanRoundFallback {
                backup: resources.backup,
                edges: resources.edges,
                bitmap_bytes: resources.bitmap_bytes,
                groups_for_words: resources.groups_for_words,
                bind_groups: resources.lean_bind_groups,
                label: "slean exact lean fallback",
                wait: true,
            },
        )?;
        let sizing = resources.sizing;
        eprintln!(
            "C{} slean round={} parts={} buckets={} capacity={} arenas=4x{:.3}GiB dead_arena={:.3}GiB overflow={} fallback={} total={:.3}s",
            request.edge_bits,
            round,
            sizing.parts,
            sizing.bucket_count,
            sizing.capacity,
            sizing.scratch_bytes as f64 / 1024_f64.powi(3),
            sizing.dead_scratch_bytes as f64 / 1024_f64.powi(3),
            overflow,
            overflow != 0,
            started.elapsed().as_secs_f64(),
        );
        if let Some(mark_elapsed) = mark_elapsed {
            let total = started.elapsed();
            eprintln!(
                "C{} slean-phase round={} seed+mark={:.3}s backup+trim+apply={:.3}s",
                request.edge_bits,
                round,
                mark_elapsed.as_secs_f64(),
                total.saturating_sub(mark_elapsed).as_secs_f64(),
            );
        }
        Ok(true)
    }
    fn build_fine_transition_seed<'a>(
        &self,
        resources: &FineTransitionResources<'a>,
    ) -> Result<FineTransitionSeed<'a>, SolveError> {
        let buffers = resources.buffers;
        if resources.production {
            buffers.bucket_scratch.destroy();
            buffers.slean_second.destroy();
            buffers.slean_third.destroy();
            buffers.slean_fourth.destroy();
            buffers.slean_dead.destroy();
            buffers.nodes.destroy();
            if let Some(backup) = &buffers.edge_backup {
                backup.destroy();
            }
        }
        let scratch = if resources.production {
            &buffers.fine_scratch
        } else {
            &buffers.bucket_scratch
        };
        let seed_bind_groups = if resources.production {
            &resources.bindings.fine_seed
        } else {
            &resources.bindings.lean
        };
        let survivors = self.count_alive_edges(
            scratch,
            &seed_bind_groups[0],
            &self.pipelines.count_alive,
            resources.groups_for_words,
        )?;
        let side = (resources.completed_round & 1) as usize;
        let (arena, loop_start) = if resources.sharded_seed {
            let arena = self.build_sharded_fine_seed_and_trim(
                survivors,
                FineShardSeedResources {
                    current_lean_bind_group: &seed_bind_groups[side],
                    count_layout: &self.pipelines.fine_count_layout,
                    arena_layout: &self.pipelines.fine_arena_layout,
                    histogram_pipeline: &self.pipelines.fine_histogram,
                    scatter_low_pipeline: &self.pipelines.fine_scatter_low,
                    scatter_high_pipeline: &self.pipelines.fine_scatter_high,
                    trim_count_pipeline: &self.pipelines.fine_trim_count,
                    trim_scatter_pipeline: &self.pipelines.fine_trim_scatter,
                    groups_for_words: resources.groups_for_words,
                },
            )?;
            self.fine_rounds.fetch_add(1, Ordering::Relaxed);
            (arena, resources.completed_round + 1)
        } else {
            (
                self.build_fine_csr(
                    survivors,
                    FineCsrResources {
                        scratch,
                        lean_bind_group: &seed_bind_groups[side],
                        count_layout: &self.pipelines.fine_count_layout,
                        arena_layout: &self.pipelines.fine_arena_layout,
                        histogram_pipeline: &self.pipelines.fine_histogram,
                        scatter_pipeline: &self.pipelines.fine_scatter,
                        verify_pipeline: &self.pipelines.fine_verify,
                        groups_for_words: resources.groups_for_words,
                    },
                )?,
                resources.completed_round,
            )
        };
        if resources.production {
            buffers.edges.destroy();
        }
        let bind_groups = if resources.production {
            &resources.bindings.fine_trim
        } else {
            &resources.bindings.lean
        };
        Ok(FineTransitionSeed {
            arena,
            survivors,
            loop_start,
            scratch,
            bind_groups,
        })
    }

    fn run_fine_transition(
        &self,
        request: GraphParams,
        resources: FineTransitionResources<'_>,
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<u64>>, SolveError> {
        let seed = self.build_fine_transition_seed(&resources)?;
        let nonempty = seed
            .arena
            .counts
            .iter()
            .filter(|&&count| count != 0)
            .count();
        let max_bucket = seed.arena.counts.iter().copied().max().unwrap_or(0);
        eprintln!(
            "C{} fine-csr round={} survivors={} buckets={} nonempty={} max_bucket={} final_offset={} arena={:.3}GiB histogram={:.3}s scatter={:.3}s",
            request.edge_bits,
            resources.completed_round,
            seed.survivors,
            FINE_BUCKETS,
            nonempty,
            max_bucket,
            seed.arena.survivor_count,
            seed.arena.arena.size() as f64 / 1024_f64.powi(3),
            seed.arena.histogram_elapsed.as_secs_f64(),
            seed.arena.scatter_elapsed.as_secs_f64(),
        );
        let seed_histogram = seed.arena.histogram_elapsed;
        let seed_scatter = seed.arena.scatter_elapsed;
        let started = Instant::now();
        let Some((arena, fine_count, fine_scatter)) = self.run_fine_rounds(
            request,
            seed.arena,
            seed.loop_start,
            FineLoopResources {
                bind_groups: seed.bind_groups,
                scratch: seed.scratch,
                end_round: resources.end_round,
                production: resources.production,
                validate_output: resources.validate_output,
            },
            cancel,
        )?
        else {
            return Ok(None);
        };
        let fine_wall = started.elapsed();
        if resources.production {
            let survivors = self.read_fine_survivors(&arena)?;
            eprintln!(
                "C{} production-fine rounds={}-{} input={} output={} seed={:.3}s count={:.3}s scatter={:.3}s fine={:.3}s",
                request.edge_bits,
                resources.completed_round,
                resources.end_round - 1,
                seed.survivors,
                arena.survivor_count,
                (seed_histogram + seed_scatter).as_secs_f64(),
                fine_count.as_secs_f64(),
                fine_scatter.as_secs_f64(),
                fine_wall.as_secs_f64(),
            );
            return Ok(Some(survivors));
        }
        self.verify_fine_against_direct(
            request,
            &arena,
            FineDiagnosticResources {
                nodes: &resources.buffers.nodes,
                scratch: seed.scratch,
                bind_groups: seed.bind_groups,
                completed_round: resources.completed_round,
                end_round: resources.end_round,
                input_survivors: seed.survivors,
                seed_histogram,
                seed_scatter,
                fine_count,
                fine_scatter,
                fine_wall,
                groups_for_words: resources.groups_for_words,
            },
        )?;
        Ok(None)
    }
    fn encode_lean_rounds(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rounds: std::ops::Range<u32>,
        groups_for_words: u32,
        bind_groups: &[wgpu::BindGroup],
    ) {
        for round in rounds {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lean trim round"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bind_groups[(round & 1) as usize], &[]);
            pass.set_pipeline(&self.pipelines.clear);
            pass.dispatch_workgroups(groups_for_words, 1, 1);
            pass.set_pipeline(&self.pipelines.mark);
            pass.dispatch_workgroups(groups_for_words, 1, 1);
            pass.set_pipeline(&self.pipelines.trim);
            pass.dispatch_workgroups(groups_for_words, 1, 1);
        }
    }

    fn trim(
        &self,
        request: GraphParams,
        rounds: u32,
        live_work: bool,
        cancel: &AtomicBool,
    ) -> Result<TrimOutcome, SolveError> {
        let use_slean = matches!(
            self.config.trimming,
            TrimmingMode::Slean | TrimmingMode::Auto
        );
        self.trim_with_mark_mode(
            request,
            rounds,
            live_work,
            cancel,
            TrimOptions {
                bucketed_mark: true,
                slean: use_slean,
                fine_end: None,
            },
        )
    }

    fn bucket_scratch_size(
        &self,
        request: GraphParams,
        graph: GraphSize,
        diagnostics: Diagnostics,
        modes: ProductionModes,
        slean_sizing: SleanSizing,
    ) -> Result<(u64, u64), SolveError> {
        let bucket_count = if diagnostics.bucket_round0 {
            diagnostics.bucket_count
        } else {
            BUCKETED_MARK_BUCKETS
        };
        let chunk_edges = graph.edge_count.min(BUCKET_CHUNK_EDGES);
        let capacity = chunk_edges.div_ceil(u64::from(bucket_count)) + DIAGNOSTIC_BUCKET_MARGIN;
        let payload_words = u64::from(bucket_count) * capacity;
        let required_bytes = (DIAGNOSTIC_BUCKET_HEADER_WORDS
            + if modes.bucketed_mark {
                2 * payload_words
            } else {
                payload_words
            })
            * 4;
        let scratch_bytes = if modes.slean {
            slean_sizing.scratch_bytes
        } else if diagnostics.bucket_round0 || modes.bucketed_mark {
            required_bytes
        } else {
            4
        };
        if scratch_bytes > self.context.limits.max_buffer_size
            || scratch_bytes > self.context.limits.max_storage_buffer_binding_size
        {
            return Err(SolveError::Unsupported(format!(
                "{} exposes max storage buffer {} MiB, but bucketed C{} marking needs {} MiB scratch",
                self.context.adapter_name,
                self.context.limits.max_storage_buffer_binding_size / (1024 * 1024),
                request.edge_bits,
                scratch_bytes / (1024 * 1024),
            )));
        }
        Ok((scratch_bytes, chunk_edges))
    }

    fn prepare_trim(
        &self,
        request: GraphParams,
        requested_rounds: u32,
        options: TrimOptions,
    ) -> Result<PreparedTrim, SolveError> {
        let diagnostics =
            Diagnostics::from_env(request.edge_bits, requested_rounds, options.fine_end)?;
        let graph = GraphSize::new(&self.context, request)?;
        let production_fine = options.bucketed_mark
            && request.edge_bits >= BUCKETED_MARK_MIN_EDGE_BITS
            && diagnostics.rounds > BUCKETED_MARK_ROUNDS
            && !diagnostics.result_only();
        let production_bucketed_mark = options.bucketed_mark
            && request.edge_bits >= BUCKETED_MARK_MIN_EDGE_BITS
            && !diagnostics.has_exclusive();
        let production_slean =
            options.slean && production_bucketed_mark && !diagnostics.result_only();
        let modes = ProductionModes {
            fine: production_fine,
            bucketed_mark: production_bucketed_mark,
            slean: production_slean,
        };
        let slean_sizing =
            self.slean_sizing(production_slean, graph.edge_count, graph.bitmap_bytes)?;
        let (bucket_scratch_bytes, bucket_chunk_edges) =
            self.bucket_scratch_size(request, graph, diagnostics, modes, slean_sizing)?;
        let params_for_side = |side: u32| Params {
            key_words: GpuSipKeys::from(request.keys).words,
            edge_bits: u32::from(request.edge_bits),
            side,
            edge_count_lo: graph.edge_count as u32,
            word_count: graph.word_count as u32,
            node_mask: diagnostics.node_mask(request.edge_bits),
            diagnostic_chunk_base: 0,
            diagnostic_chunk_count: 0,
            diagnostic_bucket: 0,
        };
        let params = [params_for_side(0), params_for_side(1)];
        let buffers = TrimBuffers::new(
            &self.context,
            graph.bitmap_bytes,
            bucket_scratch_bytes,
            production_slean.then_some(SleanBufferSizes {
                arena: slean_sizing.scratch_bytes,
                dead: slean_sizing.dead_scratch_bytes,
            }),
        );
        let core = CoreBindings::new(&self.context, &self.pipelines, &buffers, &params);
        let bucket = BucketBindings::new(
            &self.context,
            &self.pipelines,
            &buffers,
            &params,
            production_bucketed_mark,
            graph.edge_count,
            bucket_chunk_edges,
        );
        let slean = SleanBindings::new(
            self,
            &buffers,
            &params,
            production_slean,
            &slean_sizing,
            graph.edge_count,
        );
        Ok(PreparedTrim {
            diagnostics,
            rounds: diagnostics.rounds,
            production_fine: modes.fine,
            production_slean: modes.slean,
            production_bucketed_mark: modes.bucketed_mark,
            graph,
            slean_sizing,
            buffers,
            core,
            bucket,
            slean,
            groups_for_words: graph
                .word_count
                .div_ceil(u64::from(WORKGROUP_SIZE))
                .min(u64::from(DISPATCH_GROUPS)) as u32,
        })
    }

    fn trim_with_mark_mode(
        &self,
        request: GraphParams,
        rounds: u32,
        live_work: bool,
        cancel: &AtomicBool,
        options: TrimOptions,
    ) -> Result<TrimOutcome, SolveError> {
        if options.bucketed_mark {
            self.bucketed_rounds.store(0, Ordering::Relaxed);
            self.bucket_trim_rounds.store(0, Ordering::Relaxed);
            self.bucket_fallbacks.store(0, Ordering::Relaxed);
        }
        let prepared = self.prepare_trim(request, rounds, options)?;
        prepared.execute(self, request, live_work, cancel)
    }

    fn verdict_at_round(
        &self,
        request: GraphParams,
        rounds: u32,
        live_work: bool,
        cancel: &AtomicBool,
    ) -> Result<SolveOutcome, SolveError> {
        let trim_start = Instant::now();
        let survivors = match self.trim(request, rounds, live_work, cancel)? {
            TrimOutcome::Survivors(survivors) => survivors,
            TrimOutcome::Cancelled => return Ok(SolveOutcome::Cancelled),
            TrimOutcome::Diagnostic(message) => {
                return Ok(SolveOutcome::Inconclusive(message.into()));
            }
        };
        let trim_elapsed = trim_start.elapsed();
        let peel_start = Instant::now();
        let survivors_before_peel = survivors.len();
        let survivors = peel_two_core(request, &survivors)?;
        let peel_elapsed = peel_start.elapsed();
        let search_start = Instant::now();
        if cancel.load(Ordering::Relaxed) {
            return Ok(SolveOutcome::Cancelled);
        }
        let proof = find_cycle_d2(request, &survivors);
        eprintln!(
            "C{} round={} survivors={}->{} trim={:.3}s peel={:.3}s d2={:.3}s",
            request.edge_bits,
            rounds,
            survivors_before_peel,
            survivors.len(),
            trim_elapsed.as_secs_f64(),
            peel_elapsed.as_secs_f64(),
            search_start.elapsed().as_secs_f64(),
        );
        match proof {
            Ok(Some(proof)) => Ok(SolveOutcome::Proof(proof)),
            Ok(None) => Ok(SolveOutcome::NoCycle),
            Err(SolveError::SearchLimit(reason)) => Ok(SolveOutcome::Inconclusive(reason)),
            Err(error) => Err(error),
        }
    }
}

impl Solver for GpuWgpuSolver {
    fn name(&self) -> &'static str {
        "gpu-wgpu"
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            min_edge_bits: 1,
            max_edge_bits: 32,
            cycle_length: 42,
        }
    }

    fn solve(
        &mut self,
        request: SolveRequest,
        cancel: &AtomicBool,
    ) -> Result<SolveOutcome, SolveError> {
        validate_request(&request, self.capabilities())?;
        let live_work = request.job.is_some();
        let result_only_diagnostic = Diagnostics::result_only_from_env();
        let params = request.graph_params();
        let early_round = params.rounds.min(EARLY_VERDICT_ROUND);
        match self.verdict_at_round(params, early_round, live_work, cancel)? {
            SolveOutcome::Inconclusive(reason)
                if !result_only_diagnostic && params.rounds > early_round =>
            {
                eprintln!(
                    "round-{early_round} verdict inconclusive ({reason}); \
                     falling back to full round-{} trim",
                    params.rounds
                );
                // Resource-limit fallback is deliberately rare. Rebuilding the
                // bitmap keeps this implementation simple and correctness-first;
                // a future persistent trim session can resume the same bitmap.
                self.verdict_at_round(params, params.rounds, live_work, cancel)
            }
            outcome => Ok(outcome),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        solver::{NEVER_CANCEL, cpu_lean::trim_survivors},
        verify::verify_cycle,
    };

    fn gpu_solver() -> Option<GpuWgpuSolver> {
        match GpuWgpuSolver::new() {
            Ok(solver) => Some(solver),
            Err(error) if std::env::var_os("GRIN_MINER_REQUIRE_GPU_TESTS").is_some() => {
                panic!("GPU adapter is required in this CI job: {error}")
            }
            Err(_) => None,
        }
    }

    fn trim_options(bucketed_mark: bool, slean: bool, fine_end: Option<u32>) -> TrimOptions {
        TrimOptions {
            bucketed_mark,
            slean,
            fine_end,
        }
    }

    #[test]
    fn wgsl_parses_without_a_gpu_adapter() {
        naga::front::wgsl::parse_str(include_str!("lean.wgsl")).expect("valid lean WGSL module");
        let native = shader_source(true).expect("native-int64 WGSL source");
        naga::front::wgsl::parse_str(&native).expect("valid native-int64 WGSL module");
    }

    #[test]
    fn reports_gpu_limits_if_adapter_exists() {
        let Some(solver) = gpu_solver() else {
            return;
        };
        let limits = &solver.context.limits;
        eprintln!(
            "gpu={} native_int64={} max_buffer={}MiB max_storage_binding={}MiB storage_buffers_per_stage={} bind_groups={} workgroup_storage={}KiB",
            solver.context.adapter_name,
            solver.context.native_int64,
            limits.max_buffer_size / (1024 * 1024),
            limits.max_storage_buffer_binding_size / (1024 * 1024),
            limits.max_storage_buffers_per_shader_stage,
            limits.max_bind_groups,
            limits.max_compute_workgroup_storage_size / 1024,
        );
    }

    #[test]
    fn gpu_small_graph_is_safe_if_adapter_exists() {
        let Some(mut solver) = gpu_solver() else {
            return;
        };
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce: 0,
            job: None,
            edge_bits: 12,
            cycle_length: 4,
            rounds: 20,
        };
        let params = request.graph_params();
        if let SolveOutcome::Proof(proof) = solver.solve(request, &NEVER_CANCEL).unwrap() {
            verify_cycle(params.keys, params.edge_bits, params.cycle_length, &proof).unwrap();
        }
    }

    #[test]
    fn gpu_survivors_match_cpu_reference() {
        let Some(solver) = gpu_solver() else {
            return;
        };
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce: 7,
            job: None,
            edge_bits: 12,
            cycle_length: 4,
            rounds: 8,
        };
        let params = request.graph_params();
        let TrimOutcome::Survivors(gpu) = solver
            .trim(params, params.rounds, false, &NEVER_CANCEL)
            .unwrap()
        else {
            panic!("ordinary GPU trim returned no survivors");
        };
        let cpu = trim_survivors(params, 24).unwrap();
        assert_eq!(gpu, cpu);
    }

    #[test]
    fn gpu_bucketed_mark_and_trim_match_direct_at_c24() {
        let Some(solver) = gpu_solver() else {
            return;
        };
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce: 7,
            job: None,
            edge_bits: 24,
            cycle_length: 42,
            rounds: BUCKETED_MARK_ROUNDS,
        };
        let params = request.graph_params();
        let TrimOutcome::Survivors(bucketed) = solver
            .trim_with_mark_mode(
                params,
                BUCKETED_MARK_ROUNDS,
                false,
                &NEVER_CANCEL,
                trim_options(true, false, None),
            )
            .unwrap()
        else {
            panic!("bucketed C24 trim returned no survivors");
        };
        assert_eq!(
            solver.bucketed_rounds.load(Ordering::Relaxed),
            BUCKETED_MARK_ROUNDS
        );
        assert_eq!(
            solver.bucket_trim_rounds.load(Ordering::Relaxed),
            BUCKETED_MARK_ROUNDS
        );
        assert_eq!(solver.bucket_fallbacks.load(Ordering::Relaxed), 0);
        let TrimOutcome::Survivors(direct) = solver
            .trim_with_mark_mode(
                params,
                BUCKETED_MARK_ROUNDS,
                false,
                &NEVER_CANCEL,
                trim_options(false, false, None),
            )
            .unwrap()
        else {
            panic!("direct C24 trim returned no survivors");
        };
        assert_eq!(bucketed, direct);
    }

    #[test]
    fn gpu_slean_parts_match_direct_at_c24() {
        let Ok(solver) = GpuWgpuSolver::new_with_config(GpuWgpuConfig {
            trimming: TrimmingMode::Slean,
            // Exercises the four-way arena sharding that makes C32 parts=2
            // fit below wgpu's per-binding limit.
            slean_parts: 2,
            local_ram_kib: 32,
        }) else {
            if std::env::var_os("GRIN_MINER_REQUIRE_GPU_TESTS").is_some() {
                panic!("GPU adapter is required for the slean oracle");
            }
            return;
        };
        solver.verify_slean_seed.store(true, Ordering::Relaxed);
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce: 7,
            job: None,
            edge_bits: 24,
            cycle_length: 42,
            rounds: BUCKETED_MARK_ROUNDS,
        };
        let params = request.graph_params();
        let TrimOutcome::Survivors(slean) = solver
            .trim_with_mark_mode(
                params,
                BUCKETED_MARK_ROUNDS,
                false,
                &NEVER_CANCEL,
                trim_options(true, true, None),
            )
            .unwrap()
        else {
            panic!("slean C24 trim returned no survivors");
        };
        assert_eq!(
            solver.bucketed_rounds.load(Ordering::Relaxed),
            BUCKETED_MARK_ROUNDS
        );
        assert_eq!(
            solver.bucket_trim_rounds.load(Ordering::Relaxed),
            BUCKETED_MARK_ROUNDS
        );
        assert_eq!(solver.bucket_fallbacks.load(Ordering::Relaxed), 0);
        assert_eq!(solver.slean_seed_checks.load(Ordering::Relaxed), 1);
        let TrimOutcome::Survivors(direct) = solver
            .trim_with_mark_mode(
                params,
                BUCKETED_MARK_ROUNDS,
                false,
                &NEVER_CANCEL,
                trim_options(false, false, None),
            )
            .unwrap()
        else {
            panic!("direct C24 trim returned no survivors");
        };
        assert!(!slean.windows(2).any(|pair| pair[0] == pair[1]));
        assert_eq!(slean, direct);
    }

    #[test]
    fn gpu_slean_overflow_restores_and_falls_back_exactly_at_c24() {
        let Ok(solver) = GpuWgpuSolver::new_with_config(GpuWgpuConfig {
            trimming: TrimmingMode::Slean,
            slean_parts: 4,
            local_ram_kib: 32,
        }) else {
            if std::env::var_os("GRIN_MINER_REQUIRE_GPU_TESTS").is_some() {
                panic!("GPU adapter is required for the slean fallback oracle");
            }
            return;
        };
        solver.force_slean_overflow.store(true, Ordering::Relaxed);
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce: 7,
            job: None,
            edge_bits: 24,
            cycle_length: 42,
            rounds: 1,
        };
        let params = request.graph_params();
        let TrimOutcome::Survivors(mut fallback) = solver
            .trim_with_mark_mode(
                params,
                1,
                false,
                &NEVER_CANCEL,
                trim_options(true, true, None),
            )
            .unwrap()
        else {
            panic!("slean overflow fallback returned no survivors");
        };
        assert_eq!(solver.bucket_fallbacks.load(Ordering::Relaxed), 1);
        let TrimOutcome::Survivors(mut direct) = solver
            .trim_with_mark_mode(
                params,
                1,
                false,
                &NEVER_CANCEL,
                trim_options(false, false, None),
            )
            .unwrap()
        else {
            panic!("direct C24 fallback oracle returned no survivors");
        };
        fallback.sort_unstable();
        direct.sort_unstable();
        assert_eq!(fallback, direct);
    }

    #[test]
    fn gpu_full_fine_loop_matches_direct_at_c24() {
        let Some(solver) = gpu_solver() else {
            return;
        };
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce: 7,
            job: None,
            edge_bits: 24,
            cycle_length: 42,
            rounds: 6,
        };
        let params = request.graph_params();
        let TrimOutcome::Diagnostic(_) = solver
            .trim_with_mark_mode(
                params,
                6,
                false,
                &NEVER_CANCEL,
                trim_options(true, false, Some(64)),
            )
            .unwrap()
        else {
            panic!("full C24 fine-loop comparison did not run as a diagnostic");
        };
        assert_eq!(solver.fine_rounds.load(Ordering::Relaxed), 58);
        assert_eq!(solver.fine_mismatched_words.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn gpu_production_sharded_fine_from_round_two_matches_direct_at_c24() {
        let Some(solver) = gpu_solver() else {
            return;
        };
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce: 7,
            job: None,
            edge_bits: 24,
            cycle_length: 42,
            rounds: 64,
        };
        let params = request.graph_params();
        let TrimOutcome::Survivors(mut fine) = solver
            .trim_with_mark_mode(
                params,
                64,
                false,
                &NEVER_CANCEL,
                trim_options(true, false, None),
            )
            .unwrap()
        else {
            panic!("production C24 fine trim returned no survivors");
        };
        assert_eq!(solver.fine_rounds.load(Ordering::Relaxed), 62);
        let TrimOutcome::Survivors(mut direct) = solver
            .trim_with_mark_mode(
                params,
                64,
                false,
                &NEVER_CANCEL,
                trim_options(false, false, None),
            )
            .unwrap()
        else {
            panic!("direct C24 trim returned no survivors");
        };
        fine.sort_unstable();
        direct.sort_unstable();
        assert!(!fine.windows(2).any(|pair| pair[0] == pair[1]));
        assert_eq!(fine, direct);
    }

    #[test]
    fn gpu_slean_to_fine_matches_direct_at_c24() {
        let Ok(solver) = GpuWgpuSolver::new_with_config(GpuWgpuConfig {
            trimming: TrimmingMode::Slean,
            slean_parts: 4,
            local_ram_kib: 32,
        }) else {
            if std::env::var_os("GRIN_MINER_REQUIRE_GPU_TESTS").is_some() {
                panic!("GPU adapter is required for the slean/fine oracle");
            }
            return;
        };
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce: 7,
            job: None,
            edge_bits: 24,
            cycle_length: 42,
            rounds: 64,
        };
        let params = request.graph_params();
        let TrimOutcome::Survivors(mut slean) = solver
            .trim_with_mark_mode(
                params,
                64,
                false,
                &NEVER_CANCEL,
                trim_options(true, true, None),
            )
            .unwrap()
        else {
            panic!("production C24 slean/fine trim returned no survivors");
        };
        assert_eq!(solver.bucketed_rounds.load(Ordering::Relaxed), 3);
        assert_eq!(solver.bucket_trim_rounds.load(Ordering::Relaxed), 3);
        assert_eq!(solver.bucket_fallbacks.load(Ordering::Relaxed), 0);
        assert_eq!(solver.fine_rounds.load(Ordering::Relaxed), 61);
        let TrimOutcome::Survivors(mut direct) = solver
            .trim_with_mark_mode(
                params,
                64,
                false,
                &NEVER_CANCEL,
                trim_options(false, false, None),
            )
            .unwrap()
        else {
            panic!("direct C24 trim returned no survivors");
        };
        slean.sort_unstable();
        direct.sort_unstable();
        assert!(!slean.windows(2).any(|pair| pair[0] == pair[1]));
        assert_eq!(slean, direct);
    }
}
