use std::sync::atomic::AtomicBool;

#[cfg(target_os = "macos")]
use grin_cuckatoo_miner::solver::gpu_metal::GpuMetalSolver;
use grin_cuckatoo_miner::{
    solver::{
        SolveOutcome, SolveRequest, Solver, TrimmingMode,
        gpu_wgpu::{GpuWgpuConfig, GpuWgpuSolver},
    },
    verify::verify_cycle,
};

/// Slow C32 acceptance gate for capable GPUs.
#[test]
#[ignore = "requires a capable GPU and a long C32 run"]
fn c32_lean_finds_gate_nonces() {
    assert_fixed_seed_gate(Box::new(GpuWgpuSolver::new().expect("wgpu adapter")));
}

#[test]
#[ignore = "requires a capable GPU and a long C32 run"]
fn c32_slean_finds_gate_nonces() {
    assert_fixed_seed_gate(Box::new(
        GpuWgpuSolver::new_with_config(slean_config()).expect("wgpu adapter"),
    ));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires native Metal and a long C32 run"]
fn c32_metal_finds_gate_nonces() {
    assert_fixed_seed_gate(Box::new(
        GpuMetalSolver::new(slean_config()).expect("Metal adapter"),
    ));
}

fn slean_config() -> GpuWgpuConfig {
    GpuWgpuConfig {
        trimming: TrimmingMode::Slean,
        slean_parts: 4,
        local_ram_kib: 32,
    }
}

fn assert_fixed_seed_gate(mut solver: Box<dyn Solver>) {
    let mut found = Vec::new();
    let cancel = AtomicBool::new(false);
    for nonce in 0..100 {
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce,
            live_work: false,
            edge_bits: 32,
            cycle_length: 42,
            rounds: 128,
        };
        let keys = request.sip_keys();
        if let SolveOutcome::Proof(proof) = solver.solve(request, &cancel).expect("C32 solve") {
            verify_cycle(keys, 32, 42, &proof).expect("backend proof must verify");
            found.push(nonce);
        }
    }
    assert_eq!(found, [45, 74]);
}
