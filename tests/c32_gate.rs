use std::sync::atomic::AtomicBool;

use grin_cuckatoo_miner::{
    solver::{SolveOutcome, SolveRequest, Solver, gpu_wgpu::GpuWgpuSolver},
    verify::verify_cycle,
};

/// Full acceptance gate. On the M5 Air this takes roughly 35 minutes, so it
/// is opt-in rather than part of every `cargo test` invocation.
#[test]
#[ignore = "requires a capable GPU and a long C32 run"]
fn fixed_seed_c32_finds_exactly_45_and_74() {
    let mut solver = GpuWgpuSolver::new().expect("wgpu adapter");
    let mut found = Vec::new();
    let cancel = AtomicBool::new(false);
    for nonce in 0..100 {
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce,
            job: None,
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
