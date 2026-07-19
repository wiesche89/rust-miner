use super::*;

impl Diagnostics {
    pub fn result_only_from_env() -> bool {
        if !cfg!(any(test, feature = "diagnostics")) {
            return false;
        }
        [
            NODE_MASK_BITS,
            SIPHASH_ONLY,
            BUCKET_ROUND0,
            SURVIVOR_COUNTS,
            FINE_CSR,
        ]
        .into_iter()
        .any(env_flag)
    }

    pub fn from_env(
        edge_bits: u8,
        rounds: u32,
        fine_end_override: Option<u32>,
    ) -> Result<Self, SolveError> {
        if !cfg!(any(test, feature = "diagnostics")) {
            return Ok(Self {
                per_round: false,
                siphash_only: false,
                bucket_round0: false,
                survivor_counts: false,
                early_phases: false,
                slean_phases: false,
                fine_csr: false,
                fine_end_round: fine_end_override.unwrap_or(64),
                bucket_count: 16,
                mask_bits: None,
                rounds,
            });
        }
        let siphash_only = env_flag(SIPHASH_ONLY);
        let bucket_round0 = env_flag(BUCKET_ROUND0);
        let survivor_counts = env_flag(SURVIVOR_COUNTS);
        let fine_csr = fine_end_override.is_some() || env_flag(FINE_CSR);
        let fine_end_round = fine_end_override
            .or(parse_env(FINE_END_ROUND)?)
            .unwrap_or(64);
        let bucket_count = parse_env(BUCKETS)?.unwrap_or(16);
        let mask_bits = parse_env(NODE_MASK_BITS)?;

        let diagnostics = Self {
            per_round: env_flag(PER_ROUND),
            siphash_only,
            bucket_round0,
            survivor_counts,
            early_phases: env_flag(EARLY_PHASES),
            slean_phases: env_flag(SLEAN_PHASES),
            fine_csr,
            fine_end_round,
            bucket_count,
            mask_bits,
            rounds: if survivor_counts {
                8
            } else if fine_csr {
                6
            } else {
                rounds
            },
        };
        diagnostics.validate(edge_bits, rounds)?;
        Ok(diagnostics)
    }

    pub fn result_only(&self) -> bool {
        self.siphash_only
            || self.bucket_round0
            || self.survivor_counts
            || self.fine_csr
            || self.mask_bits.is_some()
    }

    pub fn has_exclusive(&self) -> bool {
        self.exclusive_count() != 0
    }

    pub fn node_mask(&self, edge_bits: u8) -> u32 {
        match self.mask_bits.unwrap_or(edge_bits) {
            32 => u32::MAX,
            bits => (1_u32 << bits) - 1,
        }
    }

    fn exclusive_count(&self) -> u8 {
        u8::from(self.siphash_only)
            + u8::from(self.bucket_round0)
            + u8::from(self.mask_bits.is_some())
    }

    fn validate(&self, edge_bits: u8, requested_rounds: u32) -> Result<(), SolveError> {
        let exclusive = self.exclusive_count();
        if exclusive > 1 {
            return invalid(
                "SIPHASH_ONLY, BUCKET_ROUND0, and NODE_MASK_BITS diagnostics cannot be combined",
            );
        }
        if self.survivor_counts && exclusive != 0 {
            return invalid(
                "SURVIVOR_COUNTS cannot be combined with SIPHASH_ONLY, BUCKET_ROUND0, or NODE_MASK_BITS",
            );
        }
        if self.fine_csr && (exclusive != 0 || self.survivor_counts) {
            return invalid("FINE_CSR cannot be combined with another result-only GPU diagnostic");
        }
        if !self.fine_csr && env_flag(FINE_END_ROUND) {
            return invalid(
                "GRIN_MINER_DIAGNOSTIC_FINE_END_ROUND requires GRIN_MINER_DIAGNOSTIC_FINE_CSR",
            );
        }
        if self.fine_csr && !(7..=64).contains(&self.fine_end_round) {
            return invalid("GRIN_MINER_DIAGNOSTIC_FINE_END_ROUND must be in 7..=64");
        }
        if self.survivor_counts && requested_rounds < 8 {
            return invalid("GRIN_MINER_DIAGNOSTIC_SURVIVOR_COUNTS requires at least 8 rounds");
        }
        if self.fine_csr && requested_rounds < 6 {
            return invalid("GRIN_MINER_DIAGNOSTIC_FINE_CSR requires at least 6 rounds");
        }
        if self.bucket_round0 && edge_bits != 32 {
            return invalid("GRIN_MINER_DIAGNOSTIC_BUCKET_ROUND0 currently requires edge_bits=32");
        }
        if !self.bucket_round0 && env_flag(BUCKETS) {
            return invalid(
                "GRIN_MINER_DIAGNOSTIC_BUCKETS requires GRIN_MINER_DIAGNOSTIC_BUCKET_ROUND0",
            );
        }
        if self.bucket_round0 && !matches!(self.bucket_count, 16 | 32 | 64 | 128) {
            return invalid("GRIN_MINER_DIAGNOSTIC_BUCKETS must be one of 16, 32, 64, or 128");
        }
        if self
            .mask_bits
            .is_some_and(|bits| bits == 0 || bits > edge_bits)
        {
            return invalid(format!(
                "GRIN_MINER_DIAGNOSTIC_NODE_MASK_BITS must be in 1..={edge_bits}"
            ));
        }
        Ok(())
    }
}

fn env_flag(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

fn parse_env<T>(name: &str) -> Result<Option<T>, SolveError>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    std::env::var(name)
        .ok()
        .map(|value| {
            value.parse().map_err(|error| {
                SolveError::InvalidConfig(format!("{name} must be an integer: {error}"))
            })
        })
        .transpose()
}

fn invalid<T>(message: impl Into<String>) -> Result<T, SolveError> {
    Err(SolveError::InvalidConfig(message.into()))
}

struct BucketDiagnosticTimes {
    clear: Duration,
    scatter: Duration,
    mark: Duration,
    total: Duration,
}

impl GpuWgpuSolver {
    fn time_direct_round0_mark(
        &self,
        resources: &BucketRound0Resources<'_>,
    ) -> Result<Duration, SolveError> {
        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("round-0-direct-mark-diagnostic"),
                });
        encoder.clear_buffer(resources.nodes, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("direct dense mark"),
            timestamp_writes: None,
        });
        pass.set_pipeline(resources.dense_mark_pipeline);
        pass.set_bind_group(0, resources.baseline_bind_group, &[]);
        pass.dispatch_workgroups(resources.groups_for_words, 1, 1);
        drop(pass);
        self.context
            .run(encoder, "direct round-0 mark diagnostic")?;
        Ok(started.elapsed())
    }

    fn bucket_diagnostic_bind_groups(
        &self,
        request: GraphParams,
        bucket_count: u32,
        resources: &BucketRound0Resources<'_>,
    ) -> (Vec<wgpu::BindGroup>, Vec<wgpu::BindGroup>) {
        let packed_bucket = |bucket: u32| (bucket_count << 16) | bucket;
        let make_params = |chunk_base: u32, chunk_count: u32, bucket: u32| Params {
            key_words: GpuSipKeys::from(request.keys).words,
            edge_bits: 32,
            side: 0,
            edge_count_lo: 0,
            word_count: (1_u64 << 27) as u32,
            node_mask: u32::MAX,
            chunk_base,
            chunk_count,
            bucket_config: packed_bucket(bucket),
        };
        let make_bind_group = |label: &str, params: Params| {
            let uniform =
                self.context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(label),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            self.context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: resources.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: resources.edges.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: resources.nodes.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: resources.scratch.as_entire_binding(),
                        },
                    ],
                })
        };
        let chunk_count = (1_u64 << 32).div_ceil(BUCKET_CHUNK_EDGES);
        let scatter = (0..chunk_count)
            .map(|chunk| {
                let base = chunk * BUCKET_CHUNK_EDGES;
                let count = BUCKET_CHUNK_EDGES.min((1_u64 << 32) - base);
                make_bind_group(
                    "round-0 bucket scatter",
                    make_params(base as u32, count as u32, 0),
                )
            })
            .collect();
        let mark = (0..bucket_count)
            .map(|bucket| {
                make_bind_group(
                    "round-0 bucket mark",
                    make_params(0, BUCKET_CHUNK_EDGES as u32, bucket),
                )
            })
            .collect();
        (scatter, mark)
    }

    fn run_bucket_diagnostic_chunks(
        &self,
        bucket_count: u32,
        scatter_bind_groups: &[wgpu::BindGroup],
        bucket_bind_groups: &[wgpu::BindGroup],
        resources: &BucketRound0Resources<'_>,
    ) -> Result<BucketDiagnosticTimes, SolveError> {
        let started = Instant::now();
        let clear_started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("round-0 bucket initial clear"),
                });
        encoder.clear_buffer(resources.nodes, 0, None);
        encoder.clear_buffer(
            resources.scratch,
            0,
            Some(DIAGNOSTIC_BUCKET_HEADER_WORDS * 4),
        );
        self.context.run(encoder, "round-0 bucket initial clear")?;
        let clear = clear_started.elapsed();
        let mut scatter = Duration::ZERO;
        let mut mark = Duration::ZERO;
        for (chunk, bind_group) in scatter_bind_groups.iter().enumerate() {
            let last_scatter =
                self.run_bucket_diagnostic_scatter(bucket_count, bind_group, resources)?;
            let last_mark = self.run_bucket_diagnostic_mark(bucket_bind_groups, resources)?;
            scatter += last_scatter;
            mark += last_mark;
            if (chunk + 1) % 8 == 0 || chunk + 1 == scatter_bind_groups.len() {
                eprintln!(
                    "C32 round0-bucket buckets={bucket_count} chunks={}/{} last_scatter={:.3}s last_mark={:.3}s elapsed={:.3}s",
                    chunk + 1,
                    scatter_bind_groups.len(),
                    last_scatter.as_secs_f64(),
                    last_mark.as_secs_f64(),
                    started.elapsed().as_secs_f64()
                );
            }
        }
        Ok(BucketDiagnosticTimes {
            clear,
            scatter,
            mark,
            total: started.elapsed(),
        })
    }

    fn run_bucket_diagnostic_scatter(
        &self,
        bucket_count: u32,
        bind_group: &wgpu::BindGroup,
        resources: &BucketRound0Resources<'_>,
    ) -> Result<Duration, SolveError> {
        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("round-0 bucket scatter chunk"),
                });
        encoder.clear_buffer(resources.scratch, 0, Some(u64::from(bucket_count) * 4));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("round-0 bucket scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(resources.scatter_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(DISPATCH_GROUPS, 1, 1);
        drop(pass);
        self.context.run(encoder, "round-0 bucket scatter chunk")?;
        Ok(started.elapsed())
    }

    fn run_bucket_diagnostic_mark(
        &self,
        bind_groups: &[wgpu::BindGroup],
        resources: &BucketRound0Resources<'_>,
    ) -> Result<Duration, SolveError> {
        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("round-0 bucket mark chunk"),
                });
        for bind_group in bind_groups {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("round-0 bucket-local mark"),
                timestamp_writes: None,
            });
            pass.set_pipeline(resources.bucket_mark_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(DISPATCH_GROUPS, 1, 1);
        }
        self.context.run(encoder, "round-0 bucket mark chunk")?;
        Ok(started.elapsed())
    }

    pub(super) fn run_bucket_round0_diagnostic(
        &self,
        request: GraphParams,
        bucket_count: u32,
        resources: BucketRound0Resources<'_>,
    ) -> Result<TrimOutcome, SolveError> {
        let bucket_capacity =
            BUCKET_CHUNK_EDGES.div_ceil(u64::from(bucket_count)) + DIAGNOSTIC_BUCKET_MARGIN;
        let bucket_words =
            DIAGNOSTIC_BUCKET_HEADER_WORDS + u64::from(bucket_count) * bucket_capacity;
        if bucket_words * 4 > resources.scratch.size() {
            return Err(SolveError::Unsupported(format!(
                "round-0 bucket diagnostic needs {} MiB of scratch, but only {} MiB are available",
                bucket_words * 4 / (1024 * 1024),
                resources.scratch.size() / (1024 * 1024)
            )));
        }
        eprintln!(
            "WARNING: chunked round-0 bucketing diagnostic is active with {bucket_count} buckets; it measures marking only and cannot produce a proof"
        );
        let direct = self.time_direct_round0_mark(&resources)?;
        eprintln!("C32 round0-mark direct={:.3}s", direct.as_secs_f64());
        let (scatter, mark) = self.bucket_diagnostic_bind_groups(request, bucket_count, &resources);
        let times = self.run_bucket_diagnostic_chunks(bucket_count, &scatter, &mark, &resources)?;
        let overflow = self.context.read_buffer_ranges(
            &[(resources.scratch, DIAGNOSTIC_BUCKET_OVERFLOW_WORD * 4, 4)],
            "round-0 bucket overflow readback",
        )?[0];
        eprintln!(
            "C32 round0-mark buckets={bucket_count} clear={:.3}s scatter={:.3}s mark={:.3}s bucketed={:.3}s direct={:.3}s speedup={:.2}x overflow={overflow}",
            times.clear.as_secs_f64(),
            times.scatter.as_secs_f64(),
            times.mark.as_secs_f64(),
            times.total.as_secs_f64(),
            direct.as_secs_f64(),
            direct.as_secs_f64() / times.total.as_secs_f64(),
        );
        Ok(TrimOutcome::Diagnostic(
            "chunked round-0 bucket marking diagnostic complete",
        ))
    }

    pub(super) fn run_siphash_diagnostic(
        &self,
        request: GraphParams,
        edge_count: u64,
        groups_for_words: u32,
        bind_group: &wgpu::BindGroup,
    ) -> Result<TrimOutcome, SolveError> {
        eprintln!("WARNING: SipHash-only GPU diagnostic is active; no graph or proof is produced");
        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("siphash-only-diagnostic"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("siphash-only diagnostic"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.siphash_only);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups_for_words, 1, 1);
        }
        self.context.run(encoder, "SipHash diagnostic")?;
        eprintln!(
            "C{} siphash-only mode={} hashes={} time={:.3}s",
            request.edge_bits,
            if self.context.native_int64 {
                "native-u64"
            } else {
                "emulated-2xu32"
            },
            edge_count,
            started.elapsed().as_secs_f64()
        );
        Ok(TrimOutcome::Diagnostic("SipHash-only diagnostic complete"))
    }

    fn run_direct_fine_comparison(
        &self,
        resources: &FineDiagnosticResources<'_>,
    ) -> Result<(usize, u64, Duration), SolveError> {
        let started = Instant::now();
        let mut round = resources.completed_round;
        while round < resources.end_round {
            let batch_end = (round + ROUNDS_PER_SUBMISSION).min(resources.end_round);
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("fine diagnostic direct comparison batch"),
                    });
            self.encode_lean_rounds(
                &mut encoder,
                round..batch_end,
                resources.groups_for_words,
                resources.bind_groups,
            );
            round = batch_end;
            self.context.run(encoder, "direct fine comparison")?;
        }
        let final_side = (resources.end_round & 1) as usize;
        let survivors = self.count_alive_edges(
            resources.scratch,
            &resources.bind_groups[final_side],
            &self.pipelines.count_alive,
            resources.groups_for_words,
        )?;
        Ok((final_side, survivors, started.elapsed()))
    }

    fn fine_comparison_bind_groups(
        &self,
        arena: &FineCsrArena,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let counts = self
            .context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fine comparison output counts"),
                contents: bytemuck::cast_slice(&arena.counts),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let count_bind_group = single_storage_bind_group(
            &self.context,
            &self.pipelines.fine_count_layout,
            "fine comparison count binding",
            &counts,
        );
        let dummy = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine comparison dummy cursor"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output_bind_group = fine_arena_bind_group(
            &self.context,
            &self.pipelines.fine_arena_layout,
            "fine comparison output arena",
            &arena.offsets_buffer,
            &dummy,
            &arena.arena,
        );
        (count_bind_group, output_bind_group)
    }

    fn compare_fine_bitmap(
        &self,
        arena: &FineCsrArena,
        final_side: usize,
        resources: &FineDiagnosticResources<'_>,
    ) -> Result<u32, SolveError> {
        let (counts, output) = self.fine_comparison_bind_groups(arena);
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fine versus direct bitmap comparison"),
                });
        encoder.clear_buffer(resources.nodes, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fine output bitmap emission"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.fine_emit_bitmap);
        pass.set_bind_group(0, &resources.bind_groups[final_side], &[]);
        pass.set_bind_group(1, &counts, &[]);
        pass.set_bind_group(2, &output, &[]);
        pass.dispatch_workgroups(FINE_BUCKETS, 1, 1);
        drop(pass);
        encoder.clear_buffer(resources.scratch, 0, Some(4));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fine versus direct bitmap comparison"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipelines.compare_bitmaps);
        pass.set_bind_group(0, &resources.bind_groups[final_side], &[]);
        pass.dispatch_workgroups(resources.groups_for_words, 1, 1);
        drop(pass);
        self.context.run(encoder, "fine bitmap comparison")?;
        Ok(self
            .context
            .read_u32(resources.scratch, 4, "fine bitmap comparison readback")?[0])
    }

    pub(super) fn verify_fine_against_direct(
        &self,
        request: GraphParams,
        arena: &FineCsrArena,
        resources: FineDiagnosticResources<'_>,
    ) -> Result<(), SolveError> {
        let (final_side, direct_survivors, direct_wall) =
            self.run_direct_fine_comparison(&resources)?;
        let mismatched_words = self.compare_fine_bitmap(arena, final_side, &resources)?;
        self.fine_mismatched_words
            .store(mismatched_words, Ordering::Relaxed);
        let fine_survivors = u64::from(arena.survivor_count);
        if fine_survivors != direct_survivors || mismatched_words != 0 {
            return Err(SolveError::Gpu(format!(
                "fine/direct mismatch: fine={fine_survivors} direct={direct_survivors} mismatched_words={mismatched_words}"
            )));
        }
        eprintln!(
            "C{} fine-loop rounds={}-{} input={} output={} arena={:.3}GiB seed_histogram={:.3}s seed_scatter={:.3}s fine_count={:.3}s fine_scatter={:.3}s fine_wall={:.3}s direct_wall={:.3}s direct_match=true mismatched_words=0",
            request.edge_bits,
            resources.completed_round,
            resources.end_round - 1,
            resources.input_survivors,
            fine_survivors,
            arena.arena.size() as f64 / 1024_f64.powi(3),
            resources.seed_histogram.as_secs_f64(),
            resources.seed_scatter.as_secs_f64(),
            resources.fine_count.as_secs_f64(),
            resources.fine_scatter.as_secs_f64(),
            resources.fine_wall.as_secs_f64(),
            direct_wall.as_secs_f64(),
        );
        Ok(())
    }
}
