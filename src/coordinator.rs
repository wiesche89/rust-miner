use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use anyhow::{Context, Result, bail};

use crate::{
    solver::{JobContext, SolveOutcome, SolveRequest, Solver},
    stratum::{Job, StratumClient, SubmitParams},
    verify::{proof_difficulty, verify_cycle},
};

pub struct MineConfig {
    pub edge_bits: u8,
    pub cycle_length: usize,
    pub rounds: u32,
    pub max_graphs: u64,
    pub nonce_start: u64,
}

fn job_is_stale(current: &Job, latest: &Job) -> bool {
    // Grin keeps every block-template version in current_block_versions and
    // accepts its job_id until the chain advances to another height.
    latest.height != current.height
}

pub async fn mine(
    mut client: StratumClient,
    mut solver: Box<dyn Solver>,
    config: MineConfig,
) -> Result<()> {
    if config.cycle_length != 42 {
        bail!("Grin Stratum submission requires --cycle-length 42");
    }
    let mut current_job = client.get_job().await?;
    let mut pre_pow = current_job.pre_pow_bytes()?;
    let mut nonce = config.nonce_start;
    let mut graphs = 0_u64;
    let mut job_updates = client.job_updates();
    let _ = job_updates.borrow_and_update();
    println!(
        "job height={} id={} difficulty={} backend={}",
        current_job.height,
        current_job.job_id,
        current_job.difficulty,
        solver.name()
    );

    while config.max_graphs == 0 || graphs < config.max_graphs {
        let request = SolveRequest {
            pre_pow: pre_pow.clone(),
            nonce,
            job: Some(JobContext {
                height: current_job.height,
                job_id: current_job.job_id,
                difficulty: current_job.difficulty,
            }),
            edge_bits: config.edge_bits,
            cycle_length: config.cycle_length,
            rounds: config.rounds,
        };
        let keys = request.sip_keys();
        let cancel = Arc::new(AtomicBool::new(false));
        let solver_cancel = Arc::clone(&cancel);
        let mut solve_task = tokio::task::spawn_blocking(move || {
            let result = solver.solve(request, &solver_cancel);
            (solver, result)
        });
        loop {
            tokio::select! {
                joined = &mut solve_task => {
                    let (returned_solver, result) = joined.context("solver task panicked")?;
                    solver = returned_solver;
                    let outcome = result.with_context(|| format!("solving nonce {nonce}"))?;
                    graphs += 1;

                    let next_job = client.get_job().await?;
                    if job_is_stale(&current_job, &next_job) {
                        current_job = next_job;
                        pre_pow = current_job.pre_pow_bytes()?;
                        nonce = config.nonce_start;
                        println!(
                            "new job height={} id={} difficulty={}; previous solve discarded",
                            current_job.height, current_job.job_id, current_job.difficulty
                        );
                    } else {
                        match outcome {
                            SolveOutcome::Proof(proof) => {
                                verify_cycle(keys, config.edge_bits, config.cycle_length, &proof)
                                    .context("solver returned an invalid proof; refusing submission")?;
                                let difficulty =
                                    proof_difficulty(&proof, config.edge_bits, current_job.height);
                                if difficulty >= current_job.difficulty {
                                    let result = client
                                        .submit(SubmitParams {
                                            edge_bits: config.edge_bits,
                                            height: current_job.height,
                                            job_id: current_job.job_id,
                                            nonce,
                                            pow: &proof.nonces,
                                        })
                                        .await?;
                                    println!(
                                        "accepted proof nonce={nonce} difficulty={difficulty} result={result}"
                                    );
                                } else {
                                    println!(
                                        "verified cycle nonce={nonce}, below share difficulty {difficulty} < {}",
                                        current_job.difficulty
                                    );
                                }
                                nonce = nonce.wrapping_add(1);
                            }
                            SolveOutcome::NoCycle => nonce = nonce.wrapping_add(1),
                            SolveOutcome::Cancelled => {
                                println!("solve nonce={nonce} cancelled for a newer height");
                            }
                            SolveOutcome::Inconclusive(reason) => {
                                eprintln!("solve nonce={nonce} inconclusive: {reason}; skipping nonce");
                                nonce = nonce.wrapping_add(1);
                            }
                        }
                    }
                    break;
                }
                changed = job_updates.changed() => {
                    if changed.is_err() {
                        cancel.store(true, Ordering::Relaxed);
                        continue;
                    }
                    if let Some(latest) = job_updates.borrow_and_update().clone()
                        && job_is_stale(&current_job, &latest)
                        && !cancel.swap(true, Ordering::Relaxed)
                    {
                        println!(
                            "height update {} -> {}; cancelling nonce={nonce}",
                            current_job.height, latest.height
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn job(height: u64, job_id: u64, pre_pow: &str) -> Job {
        Job {
            height,
            job_id,
            difficulty: 1,
            pre_pow: pre_pow.into(),
        }
    }

    #[test]
    fn same_height_template_versions_remain_submit_eligible() {
        let current = job(10, 0, "aa");
        assert!(!job_is_stale(&current, &job(10, 3, "bb")));
        assert!(job_is_stale(&current, &job(11, 0, "cc")));
    }
}
