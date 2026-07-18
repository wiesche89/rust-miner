pub mod cpu_lean;
pub mod d2;
pub mod gpu_metal;
pub mod gpu_wgpu;
pub mod peel;

use std::sync::atomic::AtomicBool;

use thiserror::Error;

use crate::{keys::derive_keys, siphash::SipKeys, verify::Proof};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TrimmingMode {
    Auto,
    Lean,
    Slean,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobContext {
    pub height: u64,
    pub job_id: u64,
    pub difficulty: u64,
}

#[derive(Debug, Clone)]
pub struct SolveRequest {
    pub pre_pow: Vec<u8>,
    pub nonce: u64,
    pub job: Option<JobContext>,
    pub edge_bits: u8,
    pub cycle_length: usize,
    pub rounds: u32,
}

impl SolveRequest {
    pub fn sip_keys(&self) -> SipKeys {
        derive_keys(&self.pre_pow, self.nonce)
    }

    pub(crate) fn graph_params(&self) -> GraphParams {
        GraphParams {
            keys: self.sip_keys(),
            edge_bits: self.edge_bits,
            cycle_length: self.cycle_length,
            rounds: self.rounds,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GraphParams {
    pub keys: SipKeys,
    pub edge_bits: u8,
    pub cycle_length: usize,
    pub rounds: u32,
}

#[cfg(test)]
pub(crate) static NEVER_CANCEL: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Copy)]
pub struct BackendCapabilities {
    pub min_edge_bits: u8,
    pub max_edge_bits: u8,
    pub cycle_length: usize,
}

#[derive(Debug)]
pub enum SolveOutcome {
    Proof(Proof),
    NoCycle,
    Cancelled,
    Inconclusive(String),
}

pub fn validate_request(
    request: &SolveRequest,
    capabilities: BackendCapabilities,
) -> Result<(), SolveError> {
    if request.pre_pow.is_empty() {
        return Err(SolveError::InvalidConfig(
            "pre_pow must not be empty".into(),
        ));
    }
    if request.edge_bits == 0 || request.edge_bits > 32 {
        return Err(SolveError::InvalidConfig(
            "edge_bits must be in 1..=32".into(),
        ));
    }
    if request.edge_bits < capabilities.min_edge_bits
        || request.edge_bits > capabilities.max_edge_bits
    {
        return Err(SolveError::Unsupported(format!(
            "backend supports edge_bits {}..={}, requested {}",
            capabilities.min_edge_bits, capabilities.max_edge_bits, request.edge_bits
        )));
    }
    if request.cycle_length == 0
        || !request.cycle_length.is_multiple_of(2)
        || request.cycle_length > 64
    {
        return Err(SolveError::InvalidConfig(
            "cycle_length must be even and in 2..=64".into(),
        ));
    }
    if request.job.is_some() && request.cycle_length != capabilities.cycle_length {
        return Err(SolveError::Unsupported(format!(
            "live backend requires cycle_length {}",
            capabilities.cycle_length
        )));
    }
    if request.rounds == 0 {
        return Err(SolveError::InvalidConfig(
            "rounds must be at least 1".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum SolveError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("backend cannot solve this request: {0}")]
    Unsupported(String),
    #[error("GPU error: {0}")]
    Gpu(String),
    #[error("exact cycle search exceeded its resource limit: {0}")]
    SearchLimit(String),
}

pub trait Solver: Send {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> BackendCapabilities;
    fn solve(
        &mut self,
        request: SolveRequest,
        cancel: &AtomicBool,
    ) -> Result<SolveOutcome, SolveError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capabilities() -> BackendCapabilities {
        BackendCapabilities {
            min_edge_bits: 1,
            max_edge_bits: 32,
            cycle_length: 42,
        }
    }

    #[test]
    fn rejects_dangerous_work_configuration_centrally() {
        let mut request = SolveRequest {
            pre_pow: vec![0],
            nonce: 0,
            job: None,
            edge_bits: 32,
            cycle_length: 42,
            rounds: 0,
        };
        assert!(matches!(
            validate_request(&request, capabilities()),
            Err(SolveError::InvalidConfig(_))
        ));
        request.rounds = 64;
        request.cycle_length = 1_000;
        assert!(matches!(
            validate_request(&request, capabilities()),
            Err(SolveError::InvalidConfig(_))
        ));
    }

    #[test]
    fn work_request_derives_reference_keys_from_original_context() {
        let request = SolveRequest {
            pre_pow: vec![0],
            nonce: 45,
            job: Some(JobContext {
                height: 1,
                job_id: 2,
                difficulty: 3,
            }),
            edge_bits: 32,
            cycle_length: 42,
            rounds: 128,
        };
        assert_eq!(request.sip_keys(), derive_keys(&[0], 45));
    }
}
