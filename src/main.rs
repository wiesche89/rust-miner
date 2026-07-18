use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result, bail};
use clap::Parser;

use grin_cuckatoo_miner::{
    cli::{Backend, Cli, Command, SolverArgs},
    coordinator::{MineConfig, mine},
    solver::{
        SolveOutcome, SolveRequest, Solver, TrimmingMode,
        cpu_lean::CpuLeanSolver,
        gpu_metal::GpuMetalSolver,
        gpu_wgpu::{GpuWgpuConfig, GpuWgpuSolver},
    },
    stratum::StratumClient,
    verify::verify_cycle,
};

fn make_solver(args: &SolverArgs) -> Result<Box<dyn Solver>> {
    let gpu_config = || GpuWgpuConfig {
        trimming: args.trimming,
        slean_parts: args.slean_parts,
        local_ram_kib: args.local_ram_kib,
    };
    match args.backend {
        Backend::Auto => {
            #[cfg(target_os = "macos")]
            {
                if GpuMetalSolver::is_native()
                    && args.edge_bits >= 18
                    && args.trimming != TrimmingMode::Lean
                {
                    match GpuMetalSolver::new(gpu_config()) {
                        Ok(solver) => return Ok(Box::new(solver)),
                        Err(error) => {
                            eprintln!("native Metal unavailable ({error}); trying wgpu");
                        }
                    }
                }
            }
            match GpuWgpuSolver::new_with_config(gpu_config()) {
                Ok(solver) => Ok(Box::new(solver)),
                Err(error) => {
                    eprintln!("GPU unavailable ({error}); falling back to CPU");
                    Ok(Box::new(CpuLeanSolver::default()))
                }
            }
        }
        Backend::Metal => Ok(Box::new(GpuMetalSolver::new(gpu_config())?)),
        Backend::Cpu => Ok(Box::new(CpuLeanSolver::default())),
        Backend::Gpu => Ok(Box::new(GpuWgpuSolver::new_with_config(gpu_config())?)),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Mine(args) => {
            if [
                "GRIN_MINER_DIAGNOSTIC_NODE_MASK_BITS",
                "GRIN_MINER_DIAGNOSTIC_SIPHASH_ONLY",
                "GRIN_MINER_DIAGNOSTIC_PER_ROUND",
                "GRIN_MINER_DIAGNOSTIC_EARLY_PHASES",
                "GRIN_MINER_DIAGNOSTIC_BUCKET_ROUND0",
                "GRIN_MINER_DIAGNOSTIC_BUCKETS",
                "GRIN_MINER_DIAGNOSTIC_SURVIVOR_COUNTS",
                "GRIN_MINER_DIAGNOSTIC_FINE_CSR",
                "GRIN_MINER_DIAGNOSTIC_FINE_END_ROUND",
                "GRIN_MINER_DIAGNOSTIC_SLEAN_PHASES",
            ]
            .iter()
            .any(|name| std::env::var_os(name).is_some())
            {
                bail!("GPU diagnostic environment variables are forbidden in mine mode");
            }
            let solver = make_solver(&args.solver)?;
            let client = StratumClient::connect(
                &args.node_host,
                args.node_port,
                &args.login,
                &args.password,
            )
            .await?;
            mine(
                client,
                solver,
                MineConfig {
                    edge_bits: args.solver.edge_bits,
                    cycle_length: args.solver.cycle_length,
                    rounds: args.solver.rounds,
                    max_graphs: args.max_graphs,
                    nonce_start: args.nonce_start,
                },
            )
            .await
        }
        Command::Gate(args) => {
            let pre_pow = hex::decode(&args.pre_pow).context("invalid --pre-pow hex")?;
            if pre_pow.is_empty() {
                bail!("--pre-pow must not be empty");
            }
            let mut solver = make_solver(&args.solver)?;
            let mut found = Vec::new();
            let nonce_end = args
                .nonce_start
                .checked_add(args.count)
                .context("--nonce-start + --count overflows u64")?;
            let cancel = AtomicBool::new(false);
            for nonce in args.nonce_start..nonce_end {
                let request = SolveRequest {
                    pre_pow: pre_pow.clone(),
                    nonce,
                    job: None,
                    edge_bits: args.solver.edge_bits,
                    cycle_length: args.solver.cycle_length,
                    rounds: args.solver.rounds,
                };
                let keys = request.sip_keys();
                match solver.solve(request, &cancel)? {
                    SolveOutcome::Proof(proof) => {
                        verify_cycle(
                            keys,
                            args.solver.edge_bits,
                            args.solver.cycle_length,
                            &proof,
                        )
                        .context("gate backend returned invalid proof")?;
                        println!("nonce {nonce}: POW_OK {:?}", proof.nonces);
                        found.push(nonce);
                    }
                    SolveOutcome::NoCycle => {}
                    SolveOutcome::Cancelled => bail!("offline gate was unexpectedly cancelled"),
                    SolveOutcome::Inconclusive(reason) => {
                        println!("nonce {nonce}: INCONCLUSIVE {reason}");
                    }
                }
            }
            println!("gate complete: {} cycle(s), nonces={found:?}", found.len());
            if args.solver.edge_bits == 32
                && args.pre_pow.eq_ignore_ascii_case("00")
                && args.nonce_start == 0
                && args.count == 100
                && found != [45, 74]
            {
                bail!("C32 acceptance gate failed: expected nonces [45, 74]");
            }
            Ok(())
        }
    }
}
