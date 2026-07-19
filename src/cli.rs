use clap::{Parser, Subcommand, ValueEnum};

use crate::solver::TrimmingMode;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Backend {
    Auto,
    Metal,
    Cpu,
    #[value(alias = "gpu")]
    Wgpu,
}

#[derive(Debug, Parser)]
#[command(name = "grin-cuckatoo-miner", version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Mine jobs from a Grin Stratum V1 server.
    Mine(MineArgs),
    /// Run a deterministic pre_pow/nonce range without Stratum.
    Gate(GateArgs),
}

#[derive(Debug, clap::Args)]
pub struct SolverArgs {
    #[arg(
        long,
        value_enum,
        default_value_t = Backend::Auto,
        help = "Solver backend; auto selects native Metal on macOS when available, then wgpu, then CPU for C28 and smaller"
    )]
    pub backend: Backend,
    #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u8).range(1..=32))]
    pub edge_bits: u8,
    #[arg(long, default_value_t = 42)]
    pub cycle_length: usize,
    #[arg(
        long,
        default_value_t = 128,
        value_parser = clap::value_parser!(u32).range(1..),
        help = "Maximum trim rounds; an exact round-128 verdict may finish earlier"
    )]
    pub rounds: u32,
    #[arg(
        long,
        value_enum,
        default_value_t = TrimmingMode::Auto,
        help = "GPU trimming mode; auto prefers slean and falls back to lean"
    )]
    pub trimming: TrimmingMode,
    #[arg(
        long,
        default_value_t = 4,
        value_parser = parse_power_of_two,
        help = "Power-of-two edge partitions used by slean"
    )]
    pub slean_parts: u32,
    #[arg(
        long,
        default_value_t = 32,
        value_parser = parse_power_of_two,
        help = "Target workgroup-local bitmap size in KiB"
    )]
    pub local_ram_kib: u32,
}

fn parse_power_of_two(value: &str) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|error| format!("expected an integer: {error}"))?;
    if parsed < 2 || !parsed.is_power_of_two() {
        return Err("value must be a power of two and at least 2".into());
    }
    Ok(parsed)
}

#[derive(Debug, clap::Args)]
pub struct MineArgs {
    #[command(flatten)]
    pub solver: SolverArgs,
    #[arg(long, default_value = "127.0.0.1")]
    pub node_host: String,
    #[arg(long, default_value_t = 3416)]
    pub node_port: u16,
    #[arg(long, default_value = "rust-miner")]
    pub login: String,
    #[arg(long, default_value = "x")]
    pub password: String,
    #[arg(long, default_value_t = 0, help = "0 means unlimited")]
    pub max_graphs: u64,
    #[arg(long, default_value_t = 0)]
    pub nonce_start: u64,
}

#[derive(Debug, clap::Args)]
pub struct GateArgs {
    #[command(flatten)]
    pub solver: SolverArgs,
    #[arg(long, default_value = "00")]
    pub pre_pow: String,
    #[arg(long, default_value_t = 0)]
    pub nonce_start: u64,
    #[arg(long, default_value_t = 100)]
    pub count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_is_a_wgpu_alias() {
        for name in ["wgpu", "gpu"] {
            let cli = Cli::try_parse_from(["miner", "gate", "--backend", name]).unwrap();
            let Command::Gate(args) = cli.command else {
                unreachable!();
            };
            assert!(matches!(args.solver.backend, Backend::Wgpu));
        }
    }
}
