use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};

use crate::{
    solver::{SolveError, SolveOutcome, SolveRequest, Solver},
    stratum::{Job, StratumClient, SubmitOutcome, SubmitParams},
    verify::{unscaled_proof_difficulty, verify_cycle},
};

pub struct MineConfig {
    pub edge_bits: u8,
    pub cycle_length: usize,
    pub rounds: u32,
    pub max_graphs: u64,
    pub nonce_start: u64,
}

fn job_is_stale(current: &Job, latest: &Job) -> bool {
    // Same-height template versions remain valid until the chain advances.
    latest.height != current.height
}

fn next_nonce(nonce: u64) -> Result<u64> {
    nonce
        .checked_add(1)
        .ok_or_else(|| anyhow!("nonce space exhausted at u64::MAX"))
}

async fn reconnect_and_get_job(client: &mut StratumClient) -> Job {
    let mut delay = Duration::from_secs(1);
    loop {
        match client.reconnect().await {
            Ok(()) => match client.get_job().await {
                Ok(job) => return job,
                Err(error) => eprintln!("Stratum job request failed after reconnect: {error:#}"),
            },
            Err(error) => eprintln!("Stratum reconnect failed: {error:#}"),
        }
        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(Duration::from_secs(30));
    }
}

pub async fn mine(
    mut client: StratumClient,
    mut solver: Box<dyn Solver>,
    config: MineConfig,
) -> Result<()> {
    if config.cycle_length != 42 {
        bail!("Grin Stratum submission requires --cycle-length 42");
    }
    let mut job_updates = client.job_updates();
    let mut connection_updates = client.connection_updates();
    let mut current_job = match client.get_job().await {
        Ok(job) => job,
        Err(error) => {
            eprintln!("initial Stratum job request failed: {error:#}");
            reconnect_and_get_job(&mut client).await
        }
    };
    if let Some(latest) = job_updates.borrow_and_update().clone()
        && job_is_stale(&current_job, &latest)
    {
        current_job = latest;
    }
    let mut pre_pow = current_job.pre_pow_bytes()?;
    let mut nonce = config.nonce_start;
    let mut graphs = 0_u64;
    let _ = connection_updates.borrow_and_update();
    let mut consecutive_solver_errors = 0_u8;
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
            live_work: true,
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
        let mut job_channel_open = true;
        let mut connection_channel_open = true;
        loop {
            tokio::select! {
                joined = &mut solve_task => {
                    let (returned_solver, result) = joined.context("solver task panicked")?;
                    solver = returned_solver;
                    let outcome = match result {
                        Ok(outcome) => {
                            consecutive_solver_errors = 0;
                            outcome
                        }
                        Err(error @ SolveError::Gpu(_)) => {
                            consecutive_solver_errors += 1;
                            eprintln!("solver failed for nonce {nonce}: {error}");
                            if consecutive_solver_errors >= 3 {
                                return Err(error).context("solver failed three times in a row");
                            }
                            let recovery = tokio::task::spawn_blocking(move || {
                                let result = solver.recover();
                                (solver, result)
                            });
                            let (returned_solver, recovery_result) = recovery
                                .await
                                .context("solver recovery task panicked")?;
                            solver = returned_solver;
                            if let Err(recovery_error) = recovery_result {
                                eprintln!("solver recovery failed: {recovery_error}");
                            }
                            nonce = next_nonce(nonce)?;
                            break;
                        }
                        Err(error) => {
                            return Err(error).with_context(|| format!("solving nonce {nonce}"));
                        }
                    };
                    graphs += 1;

                    if !client.is_connected() {
                        current_job = reconnect_and_get_job(&mut client).await;
                        pre_pow = current_job.pre_pow_bytes()?;
                        nonce = config.nonce_start;
                        println!(
                            "Stratum reconnected at height={} id={} difficulty={}; previous solve discarded",
                            current_job.height, current_job.job_id, current_job.difficulty
                        );
                    } else if let Some(latest) = job_updates.borrow_and_update().clone()
                        && job_is_stale(&current_job, &latest)
                    {
                        current_job = latest;
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
                                    unscaled_proof_difficulty(&proof, config.edge_bits);
                                if difficulty >= current_job.difficulty {
                                    match client
                                        .submit(SubmitParams {
                                            edge_bits: config.edge_bits,
                                            height: current_job.height,
                                            job_id: current_job.job_id,
                                            nonce,
                                            pow: &proof.nonces,
                                        })
                                        .await
                                    {
                                        Ok(SubmitOutcome::Accepted(result)) => println!(
                                            "accepted proof nonce={nonce} difficulty={difficulty} result={result}"
                                        ),
                                        Ok(SubmitOutcome::Rejected(error))
                                            if error.is_expected_share_rejection() =>
                                        {
                                            eprintln!(
                                                "share rejected nonce={nonce} code={} message={}",
                                                error.code, error.message
                                            )
                                        }
                                        Ok(SubmitOutcome::Rejected(error)) => {
                                            bail!(
                                                "verified share rejected with code {}: {}",
                                                error.code,
                                                error.message
                                            );
                                        }
                                        Err(error) => {
                                            eprintln!("share submission failed: {error:#}");
                                            // Delivery is unknown, so do not risk sending it twice.
                                            current_job = reconnect_and_get_job(&mut client).await;
                                            pre_pow = current_job.pre_pow_bytes()?;
                                            nonce = config.nonce_start;
                                            break;
                                        }
                                    }
                                } else {
                                    println!(
                                        "verified cycle nonce={nonce}, below share difficulty {difficulty} < {}",
                                        current_job.difficulty
                                    );
                                }
                                nonce = next_nonce(nonce)?;
                            }
                            SolveOutcome::NoCycle => nonce = next_nonce(nonce)?,
                            SolveOutcome::Cancelled => {
                                println!("solve nonce={nonce} cancelled for a newer height");
                            }
                            SolveOutcome::Inconclusive(reason) => {
                                eprintln!("solve nonce={nonce} inconclusive: {reason}; skipping nonce");
                                nonce = next_nonce(nonce)?;
                            }
                        }
                    }
                    break;
                }
                changed = job_updates.changed(), if job_channel_open => {
                    if changed.is_err() {
                        cancel.store(true, Ordering::Relaxed);
                        job_channel_open = false;
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
                changed = connection_updates.changed(), if connection_channel_open => {
                    if changed.is_err() {
                        connection_channel_open = false;
                        cancel.store(true, Ordering::Relaxed);
                    } else if !*connection_updates.borrow_and_update() {
                        cancel.store(true, Ordering::Relaxed);
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
    use crate::{solver::BackendCapabilities, verify::Proof};
    use serde_json::{Value, json};
    use std::time::Instant;
    use tokio::{
        io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines, ReadHalf, WriteHalf},
        net::{TcpListener, TcpStream},
    };

    struct WaitForCancellation;

    const C18_PROOF: [u64; 42] = [
        10314, 14021, 15134, 30319, 33503, 33645, 34282, 38657, 46697, 51560, 78930, 82234, 89639,
        104092, 104555, 105565, 106157, 127348, 132671, 138459, 153192, 153214, 157224, 160279,
        161096, 164039, 166165, 176219, 177336, 180942, 189572, 190168, 194849, 197668, 214343,
        224188, 230161, 236195, 244331, 251293, 254432, 261863,
    ];

    struct ProofOnce(bool);

    impl Solver for ProofOnce {
        fn name(&self) -> &'static str {
            "proof-once"
        }

        fn capabilities(&self) -> BackendCapabilities {
            test_capabilities()
        }

        fn solve(
            &mut self,
            request: SolveRequest,
            _cancel: &AtomicBool,
        ) -> Result<SolveOutcome, SolveError> {
            if self.0 {
                return Ok(SolveOutcome::NoCycle);
            }
            assert_eq!(request.nonce, 69);
            self.0 = true;
            Ok(SolveOutcome::Proof(Proof {
                nonces: C18_PROOF.to_vec(),
            }))
        }
    }

    struct SlowRecovery(bool);

    impl Solver for SlowRecovery {
        fn name(&self) -> &'static str {
            "slow-recovery"
        }

        fn capabilities(&self) -> BackendCapabilities {
            test_capabilities()
        }

        fn recover(&mut self) -> Result<(), SolveError> {
            std::thread::sleep(Duration::from_millis(200));
            Ok(())
        }

        fn solve(
            &mut self,
            _request: SolveRequest,
            _cancel: &AtomicBool,
        ) -> Result<SolveOutcome, SolveError> {
            if self.0 {
                Ok(SolveOutcome::NoCycle)
            } else {
                self.0 = true;
                Err(SolveError::Gpu("test failure".into()))
            }
        }
    }

    fn test_capabilities() -> BackendCapabilities {
        BackendCapabilities {
            min_edge_bits: 1,
            max_edge_bits: 32,
            cycle_length: 42,
        }
    }

    impl Solver for WaitForCancellation {
        fn name(&self) -> &'static str {
            "wait-for-cancellation"
        }

        fn capabilities(&self) -> BackendCapabilities {
            test_capabilities()
        }

        fn solve(
            &mut self,
            _request: SolveRequest,
            cancel: &AtomicBool,
        ) -> Result<SolveOutcome, SolveError> {
            while !cancel.load(Ordering::Relaxed) {
                std::thread::yield_now();
            }
            Ok(SolveOutcome::Cancelled)
        }
    }

    fn job(height: u64, job_id: u64, pre_pow: &str) -> Job {
        Job {
            height,
            job_id,
            difficulty: 1,
            pre_pow: pre_pow.into(),
        }
    }

    #[test]
    fn same_height_job_stays_valid() {
        let current = job(10, 0, "aa");
        assert!(!job_is_stale(&current, &job(10, 3, "bb")));
        assert!(job_is_stale(&current, &job(11, 0, "cc")));
    }

    #[test]
    fn nonce_exhaustion_is_explicit() {
        assert_eq!(next_nonce(41).unwrap(), 42);
        assert!(next_nonce(u64::MAX).is_err());
    }

    #[tokio::test]
    async fn job_push_cancels_without_polling() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = tokio::io::split(stream);
            let mut lines = BufReader::new(reader).lines();

            let login: Value =
                serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
            writer
                .write_all(
                    format!(
                        "{}\n",
                        json!({"id": login["id"], "result": "ok", "error": null})
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            let getjob: Value =
                serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
            writer
                .write_all(
                    format!(
                        "{}\n",
                        json!({
                            "id": getjob["id"],
                            "result": {"height": 10, "job_id": 0, "difficulty": 1, "pre_pow": "00"},
                            "error": null
                        })
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(20)).await;
            writer
                .write_all(
                    format!(
                        "{}\n",
                        json!({
                            "id": "Stratum",
                            "method": "job",
                            "params": {"height": 11, "job_id": 0, "difficulty": 1, "pre_pow": "01"}
                        })
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();

            if let Ok(Ok(Some(frame))) =
                tokio::time::timeout(Duration::from_millis(100), lines.next_line()).await
            {
                panic!("unexpected request after job push: {frame}");
            }
        });

        let client = StratumClient::connect("127.0.0.1", address.port(), "test", "x")
            .await
            .unwrap();
        mine(
            client,
            Box::new(WaitForCancellation),
            MineConfig {
                edge_bits: 18,
                cycle_length: 42,
                rounds: 1,
                max_graphs: 1,
                nonce_start: 0,
            },
        )
        .await
        .unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn rejection_keeps_mining() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = tokio::io::split(stream);
            let mut lines = BufReader::new(reader).lines();
            reply_to(&mut lines, &mut writer, json!("ok"), Value::Null).await;
            reply_to(
                &mut lines,
                &mut writer,
                json!({"height": 10, "job_id": 0, "difficulty": 1, "pre_pow": "00"}),
                Value::Null,
            )
            .await;
            let submit: Value =
                serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
            assert_eq!(submit["method"], "submit");
            write_response(
                &mut writer,
                &submit["id"],
                Value::Null,
                json!({"code": -32503, "message": "too late"}),
            )
            .await;
            assert!(lines.next_line().await.unwrap().is_none());
        });

        let client = StratumClient::connect("127.0.0.1", address.port(), "test", "x")
            .await
            .unwrap();
        mine(client, Box::new(ProofOnce(false)), mine_config(2))
            .await
            .unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn lost_submit_is_not_repeated() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = tokio::io::split(stream);
            let mut lines = BufReader::new(reader).lines();
            reply_to(&mut lines, &mut writer, json!("ok"), Value::Null).await;
            reply_to(
                &mut lines,
                &mut writer,
                json!({"height": 10, "job_id": 0, "difficulty": 1, "pre_pow": "00"}),
                Value::Null,
            )
            .await;
            let submit: Value =
                serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
            assert_eq!(submit["method"], "submit");
            drop(lines);
            drop(writer);

            let (stream, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = tokio::io::split(stream);
            let mut lines = BufReader::new(reader).lines();
            reply_to(&mut lines, &mut writer, json!("ok"), Value::Null).await;
            reply_to(
                &mut lines,
                &mut writer,
                json!({"height": 11, "job_id": 0, "difficulty": 1, "pre_pow": "01"}),
                Value::Null,
            )
            .await;
            assert!(lines.next_line().await.unwrap().is_none());
        });

        let client = StratumClient::connect("127.0.0.1", address.port(), "test", "x")
            .await
            .unwrap();
        mine(client, Box::new(ProofOnce(false)), mine_config(1))
            .await
            .unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn recovery_does_not_block_tokio() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = tokio::io::split(stream);
            let mut lines = BufReader::new(reader).lines();
            reply_to(&mut lines, &mut writer, json!("ok"), Value::Null).await;
            reply_to(
                &mut lines,
                &mut writer,
                json!({"height": 10, "job_id": 0, "difficulty": 1, "pre_pow": "00"}),
                Value::Null,
            )
            .await;
            assert!(lines.next_line().await.unwrap().is_none());
        });

        let client = StratumClient::connect("127.0.0.1", address.port(), "test", "x")
            .await
            .unwrap();
        let started = Instant::now();
        let tick = tokio::spawn(async {
            tokio::time::sleep(Duration::from_millis(20)).await;
            Instant::now()
        });
        let mining = tokio::spawn(mine(client, Box::new(SlowRecovery(false)), mine_config(1)));
        let ticked_at = tick.await.unwrap();
        assert!(ticked_at.duration_since(started) < Duration::from_millis(150));
        mining.await.unwrap().unwrap();
        server.await.unwrap();
    }

    fn mine_config(max_graphs: u64) -> MineConfig {
        MineConfig {
            edge_bits: 18,
            cycle_length: 42,
            rounds: 128,
            max_graphs,
            nonce_start: 69,
        }
    }

    async fn reply_to(
        lines: &mut Lines<BufReader<ReadHalf<TcpStream>>>,
        writer: &mut WriteHalf<TcpStream>,
        result: Value,
        error: Value,
    ) {
        let request: Value =
            serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        write_response(writer, &request["id"], result, error).await;
    }

    async fn write_response(
        writer: &mut WriteHalf<TcpStream>,
        id: &Value,
        result: Value,
        error: Value,
    ) {
        writer
            .write_all(
                format!("{}\n", json!({"id": id, "result": result, "error": error})).as_bytes(),
            )
            .await
            .unwrap();
    }
}
