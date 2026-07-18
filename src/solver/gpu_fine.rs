use super::*;

struct FineHistogram {
    counts: Vec<u32>,
    _buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    elapsed: Duration,
}

struct FineStorage {
    offsets: wgpu::Buffer,
    cursors: wgpu::Buffer,
    arena: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct FineFixedStorage {
    offsets: wgpu::Buffer,
    counts: wgpu::Buffer,
    count_bind_group: wgpu::BindGroup,
    input_bind_group: wgpu::BindGroup,
    output_bind_group: wgpu::BindGroup,
    arena: wgpu::Buffer,
}

impl GpuWgpuSolver {
    pub(super) fn read_fine_survivors(&self, arena: &FineCsrArena) -> Result<Vec<u64>, SolveError> {
        let survivor_count = u64::from(arena.survivor_count);
        if survivor_count == 0 {
            return Ok(Vec::new());
        }
        let bytes = survivor_count * 4;
        let staging = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine survivor readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fine survivor readback"),
                });
        encoder.copy_buffer_to_buffer(&arena.arena, 0, &staging, 0, bytes);
        let submission = self.context.submit(encoder);
        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.context.wait(submission, "fine readback")?;
        receiver
            .recv()
            .map_err(|error| SolveError::Gpu(format!("fine map callback: {error}")))?
            .map_err(|error| SolveError::Gpu(format!("mapping fine survivors: {error}")))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| SolveError::Gpu(format!("accessing fine survivors: {error}")))?;
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let survivors = words.iter().map(|&edge| u64::from(edge)).collect();
        drop(mapped);
        staging.unmap();
        Ok(survivors)
    }

    fn fine_histogram(
        &self,
        survivor_count: u64,
        resources: &FineCsrResources<'_>,
    ) -> Result<FineHistogram, SolveError> {
        let table_bytes = u64::from(FINE_BUCKETS) * 4;
        let buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine CSR counts"),
            size: table_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = single_storage_bind_group(
            &self.context,
            resources.count_layout,
            "fine CSR count binding",
            &buffer,
        );
        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fine CSR histogram"),
                });
        encoder.clear_buffer(&buffer, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fine CSR histogram"),
            timestamp_writes: None,
        });
        pass.set_pipeline(resources.histogram_pipeline);
        pass.set_bind_group(0, resources.lean_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.dispatch_workgroups(resources.groups_for_words, 1, 1);
        drop(pass);
        self.context.run(encoder, "fine histogram")?;
        let counts = self
            .context
            .read_u32(&buffer, table_bytes, "fine CSR counts readback")?;
        let counted: u64 = counts.iter().map(|&count| u64::from(count)).sum();
        if counted != survivor_count {
            return Err(SolveError::Gpu(format!(
                "fine histogram counted {counted} edges, expected {survivor_count}"
            )));
        }
        Ok(FineHistogram {
            counts,
            _buffer: buffer,
            bind_group,
            elapsed: started.elapsed(),
        })
    }

    fn allocate_fine_storage(
        &self,
        survivor_count: u64,
        counts: &[u32],
        layout: &wgpu::BindGroupLayout,
    ) -> Result<FineStorage, SolveError> {
        let mut offsets = Vec::with_capacity(FINE_BUCKETS as usize + 1);
        offsets.push(0_u32);
        for &count in counts {
            let next = u64::from(*offsets.last().expect("offset zero exists")) + u64::from(count);
            offsets.push(u32::try_from(next).map_err(|_| {
                SolveError::Unsupported("fine CSR offset exceeds the u32 arena index".into())
            })?);
        }
        let offsets = self
            .context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fine CSR offsets"),
                contents: bytemuck::cast_slice(&offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let table_bytes = u64::from(FINE_BUCKETS) * 4;
        let cursors = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine CSR cursors"),
            size: table_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let arena_bytes = (survivor_count * 4).max(4);
        if arena_bytes > self.context.limits.max_buffer_size
            || arena_bytes > self.context.limits.max_storage_buffer_binding_size
        {
            return Err(SolveError::Unsupported(format!(
                "fine CSR arena needs {:.3} GiB, adapter binding limit is {:.3} GiB",
                arena_bytes as f64 / 1024_f64.powi(3),
                self.context.limits.max_storage_buffer_binding_size as f64 / 1024_f64.powi(3),
            )));
        }
        let arena = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine CSR compact arena"),
            size: arena_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = fine_arena_bind_group(
            &self.context,
            layout,
            "fine CSR arena binding",
            &offsets,
            &cursors,
            &arena,
        );
        Ok(FineStorage {
            offsets,
            cursors,
            arena,
            bind_group,
        })
    }

    fn scatter_fine_csr(
        &self,
        histogram: &FineHistogram,
        storage: &FineStorage,
        resources: &FineCsrResources<'_>,
    ) -> Result<Duration, SolveError> {
        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fine CSR scatter"),
                });
        encoder.clear_buffer(&storage.cursors, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fine CSR scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(resources.scatter_pipeline);
        pass.set_bind_group(0, resources.lean_bind_group, &[]);
        pass.set_bind_group(1, &histogram.bind_group, &[]);
        pass.set_bind_group(2, &storage.bind_group, &[]);
        pass.dispatch_workgroups(resources.groups_for_words, 1, 1);
        drop(pass);
        self.context.run(encoder, "fine scatter")?;
        let cursors = self.context.read_u32(
            &storage.cursors,
            u64::from(FINE_BUCKETS) * 4,
            "fine CSR cursors readback",
        )?;
        if cursors != histogram.counts {
            return Err(SolveError::Gpu(
                "fine CSR scatter cursor totals differ from histogram counts".into(),
            ));
        }
        Ok(started.elapsed())
    }

    fn verify_fine_csr(
        &self,
        histogram: &FineHistogram,
        storage: &FineStorage,
        resources: &FineCsrResources<'_>,
    ) -> Result<(), SolveError> {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fine CSR verification"),
                });
        encoder.clear_buffer(resources.scratch, 0, Some(4));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fine CSR verification"),
            timestamp_writes: None,
        });
        pass.set_pipeline(resources.verify_pipeline);
        pass.set_bind_group(0, resources.lean_bind_group, &[]);
        pass.set_bind_group(1, &histogram.bind_group, &[]);
        pass.set_bind_group(2, &storage.bind_group, &[]);
        pass.dispatch_workgroups(FINE_BUCKETS, 1, 1);
        drop(pass);
        self.context.run(encoder, "fine verification")?;
        let errors =
            self.context
                .read_u32(resources.scratch, 4, "fine CSR verification readback")?[0];
        if errors != 0 {
            return Err(SolveError::Gpu(format!(
                "fine CSR verification found {errors} invalid arena entries"
            )));
        }
        Ok(())
    }

    pub(super) fn build_fine_csr(
        &self,
        survivor_count: u64,
        resources: FineCsrResources<'_>,
    ) -> Result<FineCsrArena, SolveError> {
        let histogram = self.fine_histogram(survivor_count, &resources)?;
        let storage =
            self.allocate_fine_storage(survivor_count, &histogram.counts, resources.arena_layout)?;
        let scatter_elapsed = self.scatter_fine_csr(&histogram, &storage, &resources)?;
        self.verify_fine_csr(&histogram, &storage, &resources)?;
        Ok(FineCsrArena {
            counts: histogram.counts,
            survivor_count: survivor_count as u32,
            offsets_buffer: storage.offsets,
            arena: storage.arena,
            histogram_elapsed: histogram.elapsed,
            scatter_elapsed,
        })
    }
    fn fine_trim_histogram(
        &self,
        input: &FineCsrArena,
        resources: &FineTrimResources<'_>,
    ) -> Result<(FineHistogram, wgpu::BindGroup), SolveError> {
        let table_bytes = u64::from(FINE_BUCKETS) * 4;
        let buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine trim output counts"),
            size: table_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = single_storage_bind_group(
            &self.context,
            resources.count_layout,
            "fine trim count binding",
            &buffer,
        );
        let input_counts =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("fine trim input counts"),
                    contents: bytemuck::cast_slice(&input.counts),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let input_bind_group = fine_arena_bind_group(
            &self.context,
            resources.arena_layout,
            "fine trim input arena",
            &input.offsets_buffer,
            &input_counts,
            &input.arena,
        );
        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fine trim count"),
                });
        encoder.clear_buffer(&buffer, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fine trim count"),
            timestamp_writes: None,
        });
        pass.set_pipeline(resources.count_pipeline);
        pass.set_bind_group(0, resources.current_lean_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.set_bind_group(2, &input_bind_group, &[]);
        pass.dispatch_workgroups(FINE_BUCKETS, 1, 1);
        drop(pass);
        self.context.run(encoder, "fine trim count")?;
        let counts = self
            .context
            .read_u32(&buffer, table_bytes, "fine trim counts readback")?;
        Ok((
            FineHistogram {
                counts,
                _buffer: buffer,
                bind_group,
                elapsed: started.elapsed(),
            },
            input_bind_group,
        ))
    }

    fn allocate_fine_trim_storage(
        &self,
        counts: &[u32],
        layout: &wgpu::BindGroupLayout,
    ) -> Result<(FineStorage, u64), SolveError> {
        let mut offsets = Vec::with_capacity(FINE_BUCKETS as usize + 1);
        offsets.push(0_u32);
        for &count in counts {
            let next = u64::from(*offsets.last().expect("offset zero exists")) + u64::from(count);
            offsets.push(u32::try_from(next).map_err(|_| {
                SolveError::Unsupported("fine trim output exceeds u32 arena indexing".into())
            })?);
        }
        let survivor_count = u64::from(*offsets.last().expect("final offset exists"));
        let offsets = self
            .context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fine trim output offsets"),
                contents: bytemuck::cast_slice(&offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let cursors = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine trim output cursors"),
            size: u64::from(FINE_BUCKETS) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let arena_bytes = (survivor_count * 4).max(4);
        if arena_bytes > self.context.limits.max_storage_buffer_binding_size {
            return Err(SolveError::Unsupported(format!(
                "fine trim output needs {:.3} GiB, above the adapter binding limit",
                arena_bytes as f64 / 1024_f64.powi(3)
            )));
        }
        let arena = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine trim output arena"),
            size: arena_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = fine_arena_bind_group(
            &self.context,
            layout,
            "fine trim output arena",
            &offsets,
            &cursors,
            &arena,
        );
        Ok((
            FineStorage {
                offsets,
                cursors,
                arena,
                bind_group,
            },
            survivor_count,
        ))
    }

    fn scatter_fine_trim(
        &self,
        histogram: &FineHistogram,
        input_bind_group: &wgpu::BindGroup,
        output: &FineStorage,
        resources: &FineTrimResources<'_>,
    ) -> Result<Duration, SolveError> {
        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fine trim scatter"),
                });
        encoder.clear_buffer(&output.cursors, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fine trim scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(resources.scatter_pipeline);
        pass.set_bind_group(0, resources.current_lean_bind_group, &[]);
        pass.set_bind_group(1, &histogram.bind_group, &[]);
        pass.set_bind_group(2, input_bind_group, &[]);
        pass.set_bind_group(3, &output.bind_group, &[]);
        pass.dispatch_workgroups(FINE_BUCKETS, 1, 1);
        drop(pass);
        self.context.run(encoder, "fine trim scatter")?;
        Ok(started.elapsed())
    }

    fn verify_fine_trim(
        &self,
        histogram: &FineHistogram,
        output: &FineStorage,
        resources: &FineTrimResources<'_>,
    ) -> Result<(), SolveError> {
        let cursors = self.context.read_u32(
            &output.cursors,
            u64::from(FINE_BUCKETS) * 4,
            "fine trim cursors readback",
        )?;
        if cursors != histogram.counts {
            return Err(SolveError::Gpu(
                "fine trim output cursors differ from counted survivors".into(),
            ));
        }
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fine trim output verification"),
                });
        encoder.clear_buffer(resources.scratch, 0, Some(4));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fine trim output verification"),
            timestamp_writes: None,
        });
        pass.set_pipeline(resources.verify_pipeline);
        pass.set_bind_group(0, resources.next_lean_bind_group, &[]);
        pass.set_bind_group(1, &histogram.bind_group, &[]);
        pass.set_bind_group(2, &output.bind_group, &[]);
        pass.dispatch_workgroups(FINE_BUCKETS, 1, 1);
        drop(pass);
        self.context.run(encoder, "fine trim verify")?;
        let errors =
            self.context
                .read_u32(resources.scratch, 4, "fine trim verification readback")?[0];
        if errors != 0 {
            return Err(SolveError::Gpu(format!(
                "fine trim output verification found {errors} invalid entries"
            )));
        }
        Ok(())
    }

    pub(super) fn trim_fine_csr_once(
        &self,
        input: &FineCsrArena,
        resources: FineTrimResources<'_>,
    ) -> Result<FineCsrArena, SolveError> {
        let (histogram, input_bind_group) = self.fine_trim_histogram(input, &resources)?;
        let (storage, survivor_count) =
            self.allocate_fine_trim_storage(&histogram.counts, resources.arena_layout)?;
        let scatter_elapsed =
            self.scatter_fine_trim(&histogram, &input_bind_group, &storage, &resources)?;
        if resources.validate_output {
            self.verify_fine_trim(&histogram, &storage, &resources)?;
        }
        Ok(FineCsrArena {
            counts: histogram.counts,
            survivor_count: survivor_count as u32,
            offsets_buffer: storage.offsets,
            arena: storage.arena,
            histogram_elapsed: histogram.elapsed,
            scatter_elapsed,
        })
    }
    pub(super) fn run_fine_rounds(
        &self,
        request: GraphParams,
        mut arena: FineCsrArena,
        start_round: u32,
        resources: FineLoopResources<'_>,
        cancel: &AtomicBool,
    ) -> Result<Option<(FineCsrArena, Duration, Duration)>, SolveError> {
        let mut count_total = Duration::ZERO;
        let mut scatter_total = Duration::ZERO;
        for round in start_round..resources.end_round {
            if cancel.load(Ordering::Relaxed) {
                return Ok(None);
            }
            let side = (round & 1) as usize;
            let trim_resources = |active_side: usize| FineTrimResources {
                current_lean_bind_group: &resources.bind_groups[active_side],
                next_lean_bind_group: &resources.bind_groups[active_side ^ 1],
                count_layout: &self.pipelines.fine_count_layout,
                arena_layout: &self.pipelines.fine_arena_layout,
                count_pipeline: &self.pipelines.fine_trim_count,
                scatter_pipeline: &self.pipelines.fine_trim_scatter,
                fixed_pipeline: &self.pipelines.fine_trim_fixed,
                verify_pipeline: &self.pipelines.fine_verify,
                scratch: resources.scratch,
                validate_output: resources.validate_output,
            };
            let input_count = arena.survivor_count;
            let next = if resources.production && round + 1 < resources.end_round {
                match self.trim_fine_fixed_once(&arena, trim_resources(side))? {
                    Some(arena) => arena,
                    None => {
                        eprintln!(
                            "C{} fine-fixed round={} overflow/capacity fallback",
                            request.edge_bits, round
                        );
                        self.trim_fine_csr_once(&arena, trim_resources(side))?
                    }
                }
            } else {
                self.trim_fine_csr_once(&arena, trim_resources(side))?
            };
            count_total += next.histogram_elapsed;
            scatter_total += next.scatter_elapsed;
            self.fine_rounds.fetch_add(1, Ordering::Relaxed);
            if round < 9 || (round + 1).is_multiple_of(8) {
                eprintln!(
                    "C{} fine-round={} side={} input={} output={} count={:.3}s scatter={:.3}s",
                    request.edge_bits,
                    round,
                    side,
                    input_count,
                    next.survivor_count,
                    next.histogram_elapsed.as_secs_f64(),
                    next.scatter_elapsed.as_secs_f64(),
                );
            }
            arena = next;
        }
        Ok(Some((arena, count_total, scatter_total)))
    }

    fn prepare_fine_fixed_storage(
        &self,
        input: &FineCsrArena,
        resources: &FineTrimResources<'_>,
    ) -> Result<Option<FineFixedStorage>, SolveError> {
        let capacity =
            u64::from(input.counts.iter().copied().max().unwrap_or(0)) + FINE_FIXED_MIN_MARGIN;
        let arena_slots = capacity * u64::from(FINE_BUCKETS);
        let arena_bytes = (arena_slots * 4).max(4);
        if arena_slots > u64::from(u32::MAX)
            || arena_bytes > self.context.limits.max_buffer_size
            || arena_bytes > self.context.limits.max_storage_buffer_binding_size
        {
            return Ok(None);
        }
        let offsets: Vec<_> = (0..=FINE_BUCKETS)
            .map(|bucket| (u64::from(bucket) * capacity) as u32)
            .collect();
        let offsets = self
            .context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fine fixed offsets"),
                contents: bytemuck::cast_slice(&offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let input_counts =
            self.context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("fine fixed input counts"),
                    contents: bytemuck::cast_slice(&input.counts),
                    usage: wgpu::BufferUsages::STORAGE,
                });
        let counts = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine fixed output counts"),
            size: u64::from(FINE_BUCKETS) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let count_bind_group = single_storage_bind_group(
            &self.context,
            resources.count_layout,
            "fine fixed output count binding",
            &counts,
        );
        let input_bind_group = fine_arena_bind_group(
            &self.context,
            resources.arena_layout,
            "fine fixed input arena",
            &input.offsets_buffer,
            &input_counts,
            &input.arena,
        );
        let dummy = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine fixed output dummy cursor"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let arena = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine fixed output arena"),
            size: arena_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let output_bind_group = fine_arena_bind_group(
            &self.context,
            resources.arena_layout,
            "fine fixed output arena",
            &offsets,
            &dummy,
            &arena,
        );
        Ok(Some(FineFixedStorage {
            offsets,
            counts,
            count_bind_group,
            input_bind_group,
            output_bind_group,
            arena,
        }))
    }

    pub(super) fn trim_fine_fixed_once(
        &self,
        input: &FineCsrArena,
        resources: FineTrimResources<'_>,
    ) -> Result<Option<FineCsrArena>, SolveError> {
        let Some(storage) = self.prepare_fine_fixed_storage(input, &resources)? else {
            return Ok(None);
        };
        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fine fixed trim"),
                });
        encoder.clear_buffer(&storage.counts, 0, None);
        encoder.clear_buffer(resources.scratch, 0, Some(4));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fine fixed trim"),
            timestamp_writes: None,
        });
        pass.set_pipeline(resources.fixed_pipeline);
        pass.set_bind_group(0, resources.current_lean_bind_group, &[]);
        pass.set_bind_group(1, &storage.count_bind_group, &[]);
        pass.set_bind_group(2, &storage.input_bind_group, &[]);
        pass.set_bind_group(3, &storage.output_bind_group, &[]);
        pass.dispatch_workgroups(FINE_BUCKETS, 1, 1);
        drop(pass);
        self.context.submit(encoder);
        let words = self.context.read_buffer_ranges(
            &[
                (resources.scratch, 0, 4),
                (&storage.counts, 0, u64::from(FINE_BUCKETS) * 4),
            ],
            "fine fixed combined readback",
        )?;
        if words[0] != 0 {
            return Ok(None);
        }
        let counts = words[1..].to_vec();
        let survivor_count: u64 = counts.iter().map(|&count| u64::from(count)).sum();
        if survivor_count > u64::from(input.survivor_count) {
            return Err(SolveError::Gpu(
                "fine fixed trim increased the survivor population".into(),
            ));
        }
        Ok(Some(FineCsrArena {
            counts,
            survivor_count: survivor_count as u32,
            offsets_buffer: storage.offsets,
            arena: storage.arena,
            histogram_elapsed: Duration::ZERO,
            scatter_elapsed: started.elapsed(),
        }))
    }
    fn sharded_seed_histogram(
        &self,
        survivor_count: u64,
        resources: &FineShardSeedResources<'_>,
    ) -> Result<FineHistogram, SolveError> {
        let table_bytes = u64::from(FINE_BUCKETS) * 4;
        let buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine sharded seed counts"),
            size: table_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = single_storage_bind_group(
            &self.context,
            resources.count_layout,
            "fine sharded seed count binding",
            &buffer,
        );
        let started = Instant::now();
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fine sharded seed histogram"),
                });
        encoder.clear_buffer(&buffer, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fine sharded seed histogram"),
            timestamp_writes: None,
        });
        pass.set_pipeline(resources.histogram_pipeline);
        pass.set_bind_group(0, resources.current_lean_bind_group, &[]);
        pass.set_bind_group(1, &bind_group, &[]);
        pass.dispatch_workgroups(resources.groups_for_words, 1, 1);
        drop(pass);
        self.context.run(encoder, "sharded histogram")?;
        let counts = self
            .context
            .read_u32(&buffer, table_bytes, "fine sharded seed counts")?;
        let counted: u64 = counts.iter().map(|&count| u64::from(count)).sum();
        if counted != survivor_count {
            return Err(SolveError::Gpu(format!(
                "sharded fine histogram counted {counted}, expected {survivor_count}"
            )));
        }
        Ok(FineHistogram {
            counts,
            _buffer: buffer,
            bind_group,
            elapsed: started.elapsed(),
        })
    }

    fn allocate_fine_seed_shards(&self, counts: &[u32]) -> Result<Vec<FineSeedShard>, SolveError> {
        let half_buckets = FINE_BUCKETS / 2;
        let half_table_bytes = u64::from(half_buckets) * 4;
        (0..2_usize)
            .map(|half| {
                let begin = half * half_buckets as usize;
                let shard_counts = counts[begin..begin + half_buckets as usize].to_vec();
                let mut offsets = Vec::with_capacity(half_buckets as usize + 1);
                offsets.push(0_u32);
                for &count in &shard_counts {
                    let next =
                        u64::from(*offsets.last().expect("shard offset zero")) + u64::from(count);
                    offsets.push(u32::try_from(next).map_err(|_| {
                        SolveError::Unsupported("fine seed shard exceeds u32 indexing".into())
                    })?);
                }
                let shard_entries = u64::from(*offsets.last().expect("shard final offset"));
                let arena_bytes = (shard_entries * 4).max(4);
                if arena_bytes > self.context.limits.max_storage_buffer_binding_size {
                    return Err(SolveError::Unsupported(format!(
                        "fine seed shard needs {:.3} GiB",
                        arena_bytes as f64 / 1024_f64.powi(3)
                    )));
                }
                Ok(FineSeedShard {
                    counts: shard_counts,
                    offsets_buffer: self.context.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("fine seed shard offsets"),
                            contents: bytemuck::cast_slice(&offsets),
                            usage: wgpu::BufferUsages::STORAGE,
                        },
                    ),
                    cursors_buffer: self.context.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("fine seed shard cursors"),
                        size: half_table_bytes,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    arena: self.context.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("fine seed shard arena"),
                        size: arena_bytes,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }),
                })
            })
            .collect()
    }

    fn fine_seed_shard_bind_group(
        &self,
        shard: &FineSeedShard,
        layout: &wgpu::BindGroupLayout,
        label: &'static str,
    ) -> wgpu::BindGroup {
        let counts = self
            .context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fine shard input counts"),
                contents: bytemuck::cast_slice(&shard.counts),
                usage: wgpu::BufferUsages::STORAGE,
            });
        fine_arena_bind_group(
            &self.context,
            layout,
            label,
            &shard.offsets_buffer,
            &counts,
            &shard.arena,
        )
    }

    fn scatter_fine_seed_shards(
        &self,
        shards: &[FineSeedShard],
        histogram: &FineHistogram,
        resources: &FineShardSeedResources<'_>,
    ) -> Result<(), SolveError> {
        for (half, shard) in shards.iter().enumerate() {
            let output = fine_arena_bind_group(
                &self.context,
                resources.arena_layout,
                "fine seed shard scatter arena",
                &shard.offsets_buffer,
                &shard.cursors_buffer,
                &shard.arena,
            );
            let mut encoder =
                self.context
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("fine seed shard scatter"),
                    });
            encoder.clear_buffer(&shard.cursors_buffer, 0, None);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fine seed shard scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(if half == 0 {
                resources.scatter_low_pipeline
            } else {
                resources.scatter_high_pipeline
            });
            pass.set_bind_group(0, resources.current_lean_bind_group, &[]);
            pass.set_bind_group(1, &histogram.bind_group, &[]);
            pass.set_bind_group(2, &output, &[]);
            pass.dispatch_workgroups(resources.groups_for_words, 1, 1);
            drop(pass);
            self.context.run(encoder, "fine shard scatter")?;
        }
        Ok(())
    }

    fn count_sharded_fine_trim(
        &self,
        shards: &[FineSeedShard],
        resources: &FineShardSeedResources<'_>,
    ) -> Result<FineHistogram, SolveError> {
        let table_bytes = u64::from(FINE_BUCKETS) * 4;
        let buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fine sharded trim counts"),
            size: table_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = single_storage_bind_group(
            &self.context,
            resources.count_layout,
            "fine sharded trim count binding",
            &buffer,
        );
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fine sharded trim count"),
                });
        encoder.clear_buffer(&buffer, 0, None);
        for shard in shards {
            let input = self.fine_seed_shard_bind_group(
                shard,
                resources.arena_layout,
                "fine shard trim input",
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fine shard trim count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(resources.trim_count_pipeline);
            pass.set_bind_group(0, resources.current_lean_bind_group, &[]);
            pass.set_bind_group(1, &bind_group, &[]);
            pass.set_bind_group(2, &input, &[]);
            pass.dispatch_workgroups(FINE_BUCKETS / 2, 1, 1);
        }
        self.context.run(encoder, "shard trim count")?;
        let counts = self
            .context
            .read_u32(&buffer, table_bytes, "fine sharded trim counts")?;
        Ok(FineHistogram {
            counts,
            _buffer: buffer,
            bind_group,
            elapsed: Duration::ZERO,
        })
    }

    fn scatter_sharded_fine_trim(
        &self,
        shards: &[FineSeedShard],
        histogram: &FineHistogram,
        output: &FineStorage,
        resources: &FineShardSeedResources<'_>,
    ) -> Result<(), SolveError> {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("fine sharded trim scatter"),
                });
        encoder.clear_buffer(&output.cursors, 0, None);
        for shard in shards {
            let input = self.fine_seed_shard_bind_group(
                shard,
                resources.arena_layout,
                "fine shard scatter input",
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fine shard trim scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(resources.trim_scatter_pipeline);
            pass.set_bind_group(0, resources.current_lean_bind_group, &[]);
            pass.set_bind_group(1, &histogram.bind_group, &[]);
            pass.set_bind_group(2, &input, &[]);
            pass.set_bind_group(3, &output.bind_group, &[]);
            pass.dispatch_workgroups(FINE_BUCKETS / 2, 1, 1);
        }
        self.context.run(encoder, "shard trim scatter")
    }

    pub(super) fn build_sharded_fine_seed_and_trim(
        &self,
        survivor_count: u64,
        resources: FineShardSeedResources<'_>,
    ) -> Result<FineCsrArena, SolveError> {
        let started = Instant::now();
        let seed_histogram = self.sharded_seed_histogram(survivor_count, &resources)?;
        let shards = self.allocate_fine_seed_shards(&seed_histogram.counts)?;
        self.scatter_fine_seed_shards(&shards, &seed_histogram, &resources)?;
        let trim_histogram = self.count_sharded_fine_trim(&shards, &resources)?;
        let output_survivors: u64 = trim_histogram
            .counts
            .iter()
            .map(|&count| u64::from(count))
            .sum();
        let output = self.allocate_fine_storage(
            output_survivors,
            &trim_histogram.counts,
            resources.arena_layout,
        )?;
        self.scatter_sharded_fine_trim(&shards, &trim_histogram, &output, &resources)?;
        Ok(FineCsrArena {
            counts: trim_histogram.counts,
            survivor_count: output_survivors as u32,
            offsets_buffer: output.offsets,
            arena: output.arena,
            histogram_elapsed: Duration::ZERO,
            scatter_elapsed: started.elapsed(),
        })
    }
}
