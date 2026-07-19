use std::{
    collections::HashMap,
    io,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader, WriteHalf},
    net::TcpStream,
    sync::{Mutex, oneshot, watch},
    task::JoinHandle,
    time::timeout,
};

const MAX_STRATUM_FRAME_BYTES: usize = 1024 * 1024;
const RPC_LOW_DIFFICULTY: i64 = -32501;
const RPC_SUBMITTED_TOO_LATE: i64 = -32503;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Job {
    pub height: u64,
    pub job_id: u64,
    pub difficulty: u64,
    pub pre_pow: String,
}

impl Job {
    pub fn pre_pow_bytes(&self) -> Result<Vec<u8>> {
        hex::decode(&self.pre_pow).context("job contains invalid pre_pow hex")
    }
}

#[derive(Debug, Serialize)]
pub struct SubmitParams<'a> {
    pub edge_bits: u8,
    pub height: u64,
    pub job_id: u64,
    pub nonce: u64,
    pub pow: &'a [u64],
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(default)]
    pub data: Option<Value>,
}

impl RpcError {
    pub fn is_expected_share_rejection(&self) -> bool {
        matches!(self.code, RPC_LOW_DIFFICULTY | RPC_SUBMITTED_TOO_LATE)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum SubmitOutcome {
    Accepted(Value),
    Rejected(RpcError),
}

pub struct StratumClient {
    host: String,
    port: u16,
    login: String,
    password: String,
    writer: Arc<Mutex<WriteHalf<TcpStream>>>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    next_id: u64,
    job_tx: watch::Sender<Option<Job>>,
    job_rx: watch::Receiver<Option<Job>>,
    connected_tx: watch::Sender<bool>,
    connected_rx: watch::Receiver<bool>,
    reader_generation: Arc<AtomicU64>,
    reader_task: Option<JoinHandle<()>>,
}

impl StratumClient {
    pub async fn connect(host: &str, port: u16, login: &str, password: &str) -> Result<Self> {
        let (reader, writer) = connect_stream(host, port).await?;
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let (job_tx, job_rx) = watch::channel(None);
        let (connected_tx, connected_rx) = watch::channel(false);
        let reader_generation = Arc::new(AtomicU64::new(1));
        let reader_task = spawn_reader(
            reader,
            Arc::clone(&pending),
            job_tx.clone(),
            connected_tx.clone(),
            Arc::clone(&reader_generation),
            1,
        );
        let mut client = Self {
            host: host.into(),
            port,
            login: login.into(),
            password: password.into(),
            writer: Arc::new(Mutex::new(writer)),
            pending,
            next_id: 1,
            job_tx,
            job_rx,
            connected_tx,
            connected_rx,
            reader_generation,
            reader_task: Some(reader_task),
        };
        client.connected_tx.send_replace(true);
        if let Err(error) = client.login().await {
            client.connected_tx.send_replace(false);
            return Err(error);
        }
        Ok(client)
    }

    pub fn job_updates(&self) -> watch::Receiver<Option<Job>> {
        self.job_rx.clone()
    }

    pub fn connection_updates(&self) -> watch::Receiver<bool> {
        self.connected_rx.clone()
    }

    pub fn is_connected(&self) -> bool {
        *self.connected_rx.borrow()
    }

    pub async fn reconnect(&mut self) -> Result<()> {
        let generation = self.reader_generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.connected_tx.send_replace(false);
        self.stop_reader().await;
        let _ = self.writer.lock().await.shutdown().await;
        self.pending.lock().await.clear();
        let (reader, writer) = connect_stream(&self.host, self.port).await?;
        let pending = Arc::new(Mutex::new(HashMap::new()));
        self.writer = Arc::new(Mutex::new(writer));
        self.pending = Arc::clone(&pending);
        self.reader_task = Some(spawn_reader(
            reader,
            pending,
            self.job_tx.clone(),
            self.connected_tx.clone(),
            Arc::clone(&self.reader_generation),
            generation,
        ));
        self.connected_tx.send_replace(true);
        if let Err(error) = self.login().await {
            self.connected_tx.send_replace(false);
            return Err(error);
        }
        Ok(())
    }

    async fn stop_reader(&mut self) {
        if let Some(task) = self.reader_task.take() {
            task.abort();
            let _ = task.await;
        }
    }

    pub async fn get_job(&mut self) -> Result<Job> {
        let response = self.request("getjobtemplate", Value::Null).await?;
        if let Some(error) = response.get("error").filter(|value| !value.is_null()) {
            bail!("getjobtemplate rejected: {error}");
        }
        let job: Job =
            serde_json::from_value(response.get("result").cloned().unwrap_or(Value::Null))
                .context("getjobtemplate returned no usable job")?;
        job.pre_pow_bytes()?;
        if let Some(pushed) = self.job_rx.borrow().clone()
            && pushed.height > job.height
        {
            return Ok(pushed);
        }
        self.job_tx.send_replace(Some(job.clone()));
        Ok(job)
    }

    pub async fn submit(&mut self, params: SubmitParams<'_>) -> Result<SubmitOutcome> {
        let response = self
            .request("submit", serde_json::to_value(params)?)
            .await?;
        if let Some(error) = response.get("error").filter(|value| !value.is_null()) {
            let error = serde_json::from_value(error.clone()).unwrap_or_else(|_| RpcError {
                code: 0,
                message: error.to_string(),
                data: None,
            });
            return Ok(SubmitOutcome::Rejected(error));
        }
        Ok(SubmitOutcome::Accepted(
            response.get("result").cloned().unwrap_or(Value::Null),
        ))
    }

    async fn login(&mut self) -> Result<()> {
        let response = self
            .request(
                "login",
                json!({
                    "login": self.login,
                    "pass": self.password,
                    "agent": "grin-cuckatoo-rust/0.1"
                }),
            )
            .await?;
        ensure_ok(&response, "login")
    }

    async fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let frame = json!({"id": id, "jsonrpc": "2.0", "method": method, "params": params});
        let (response_tx, response_rx) = oneshot::channel();
        self.pending.lock().await.insert(id, response_tx);
        let write_result = async {
            let mut writer = self.writer.lock().await;
            writer
                .write_all(serde_json::to_string(&frame)?.as_bytes())
                .await?;
            writer.write_all(b"\n").await?;
            writer.flush().await
        }
        .await;
        if let Err(error) = write_result {
            self.pending.lock().await.remove(&id);
            return Err(error.into());
        }
        match timeout(Duration::from_secs(30), response_rx).await {
            Ok(response) => {
                response.map_err(|_| anyhow!("Stratum connection closed before response"))
            }
            Err(error) => {
                self.pending.lock().await.remove(&id);
                Err(error).context("Stratum response timed out")
            }
        }
    }
}

impl Drop for StratumClient {
    fn drop(&mut self) {
        if let Some(task) = self.reader_task.take() {
            task.abort();
        }
    }
}

async fn connect_stream(
    host: &str,
    port: u16,
) -> Result<(tokio::io::ReadHalf<TcpStream>, WriteHalf<TcpStream>)> {
    let stream = TcpStream::connect((host, port))
        .await
        .with_context(|| format!("connecting to {host}:{port}"))?;
    stream.set_nodelay(true)?;
    Ok(tokio::io::split(stream))
}

fn spawn_reader(
    reader: tokio::io::ReadHalf<TcpStream>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    job_tx: watch::Sender<Option<Job>>,
    connected_tx: watch::Sender<bool>,
    reader_generation: Arc<AtomicU64>,
    generation: u64,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut reader = BufReader::new(reader);
        loop {
            if reader_generation.load(Ordering::Acquire) != generation {
                break;
            }
            let frame = match read_frame(&mut reader).await {
                Ok(Some(frame)) => frame,
                Ok(None) => break,
                Err(error) => {
                    eprintln!("Stratum reader error: {error}");
                    break;
                }
            };
            if reader_generation.load(Ordering::Acquire) != generation {
                break;
            }
            let value: Value = match serde_json::from_slice(&frame) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("invalid Stratum JSON ignored: {error}");
                    continue;
                }
            };
            if value.get("method").and_then(Value::as_str) == Some("job") {
                match value
                    .get("params")
                    .cloned()
                    .ok_or_else(|| anyhow!("job push has no params"))
                    .and_then(|params| serde_json::from_value::<Job>(params).map_err(Into::into))
                {
                    Ok(job) => {
                        if let Err(error) = job.pre_pow_bytes() {
                            eprintln!("invalid Stratum job push ignored: {error}");
                        } else {
                            job_tx.send_replace(Some(job));
                        }
                    }
                    Err(error) => eprintln!("invalid Stratum job push ignored: {error}"),
                }
                continue;
            }
            let response_id = value.get("id").and_then(|id| {
                id.as_u64()
                    .or_else(|| id.as_str().and_then(|text| text.parse().ok()))
            });
            if let Some(id) = response_id
                && let Some(sender) = pending.lock().await.remove(&id)
            {
                let _ = sender.send(value);
            }
        }
        pending.lock().await.clear();
        if reader_generation.load(Ordering::Acquire) == generation {
            connected_tx.send_replace(false);
        }
    })
}

async fn read_frame<R: AsyncBufRead + Unpin>(reader: &mut R) -> io::Result<Option<Vec<u8>>> {
    let mut frame = Vec::new();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return if frame.is_empty() {
                Ok(None)
            } else {
                Ok(Some(frame))
            };
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let take = newline.map_or(available.len(), |position| position + 1);
        if frame.len() + take > MAX_STRATUM_FRAME_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Stratum frame exceeds 1 MiB",
            ));
        }
        frame.extend_from_slice(&available[..take]);
        reader.consume(take);
        if newline.is_some() {
            frame.pop();
            if frame.last() == Some(&b'\r') {
                frame.pop();
            }
            return Ok(Some(frame));
        }
    }
}

fn ensure_ok(response: &Value, operation: &str) -> Result<()> {
    if let Some(error) = response.get("error").filter(|value| !value.is_null()) {
        bail!("{operation} rejected: {error}");
    }
    match response.get("result") {
        Some(Value::String(value)) if value == "ok" => Ok(()),
        Some(value) if !value.is_null() => Ok(()),
        _ => bail!("{operation} returned no success result"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::{io::AsyncWriteExt, net::TcpListener};

    #[test]
    fn parses_getjob_and_push_shape() {
        let job: Job = serde_json::from_value(json!({
            "height": 7,
            "job_id": 3,
            "difficulty": 1,
            "pre_pow": "00"
        }))
        .unwrap();
        assert_eq!(job.pre_pow_bytes().unwrap(), vec![0]);
    }

    #[test]
    fn submit_uses_decimal_nonce() {
        let proof = vec![1, 2, 3, 4];
        let value = serde_json::to_value(SubmitParams {
            edge_bits: 10,
            height: 9,
            job_id: 2,
            nonce: 42,
            pow: &proof,
        })
        .unwrap();
        assert_eq!(value["nonce"], 42);
        assert_eq!(value["pow"], json!([1, 2, 3, 4]));
    }

    #[tokio::test]
    async fn oversized_frame_is_rejected() {
        let bytes = vec![b'x'; MAX_STRATUM_FRAME_BYTES + 1];
        let mut reader = BufReader::new(bytes.as_slice());
        let error = read_frame(&mut reader).await.unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn rejection_is_an_outcome() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = tokio::io::split(stream);
            let mut lines = BufReader::new(reader).lines();
            let login: Value =
                serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
            write_response(&mut writer, &login["id"], json!("ok"), Value::Null).await;
            let submit: Value =
                serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
            write_response(
                &mut writer,
                &submit["id"],
                Value::Null,
                json!({"code": -32503, "message": "Solution Submitted too late"}),
            )
            .await;
        });

        let mut client = StratumClient::connect("127.0.0.1", address.port(), "test", "x")
            .await
            .unwrap();
        let proof = [1, 2, 3, 4];
        let outcome = client
            .submit(SubmitParams {
                edge_bits: 10,
                height: 9,
                job_id: 2,
                nonce: 42,
                pow: &proof,
            })
            .await
            .unwrap();
        assert!(matches!(
            outcome,
            SubmitOutcome::Rejected(RpcError { code: -32503, .. })
        ));
    }

    #[tokio::test]
    async fn reconnect_closes_old_reader() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (closed_tx, closed_rx) = oneshot::channel();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = tokio::io::split(stream);
            let mut lines = BufReader::new(reader).lines();
            let login: Value =
                serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
            write_response(&mut writer, &login["id"], json!("ok"), Value::Null).await;
            assert!(lines.next_line().await.unwrap().is_none());
            closed_tx.send(()).unwrap();

            let (stream, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = tokio::io::split(stream);
            let mut lines = BufReader::new(reader).lines();
            let login: Value =
                serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
            write_response(&mut writer, &login["id"], json!("ok"), Value::Null).await;
            let getjob: Value =
                serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
            write_response(
                &mut writer,
                &getjob["id"],
                json!({"height": 10, "job_id": 0, "difficulty": 1, "pre_pow": "00"}),
                Value::Null,
            )
            .await;
        });

        let mut client = StratumClient::connect("127.0.0.1", address.port(), "test", "x")
            .await
            .unwrap();
        client.reconnect().await.unwrap();
        timeout(Duration::from_secs(2), closed_rx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(client.get_job().await.unwrap().height, 10);
    }

    #[tokio::test]
    async fn reader_publishes_job_push() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (push_tx, push_rx) = oneshot::channel();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (reader, mut writer) = tokio::io::split(stream);
            let mut lines = BufReader::new(reader).lines();

            let login: Value =
                serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
            writer
                .write_all(
                    format!(
                        "{}\n",
                        json!({"id": login["id"], "jsonrpc": "2.0", "result": "ok", "error": null})
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
                            "jsonrpc": "2.0",
                            "result": {"height": 10, "job_id": 0, "difficulty": 1, "pre_pow": "00"},
                            "error": null
                        })
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();

            push_rx.await.unwrap();
            writer
                .write_all(
                    format!(
                        "{}\n{}\n",
                        json!({
                            "id": "Stratum",
                            "jsonrpc": "2.0",
                            "method": "job",
                            "params": {"height": 99, "job_id": 0, "difficulty": 1, "pre_pow": "zz"}
                        }),
                        json!({"id": 9999, "jsonrpc": "2.0", "result": "ignored"})
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            writer
                .write_all(
                    format!(
                        "{}\n",
                        json!({
                            "id": "Stratum",
                            "jsonrpc": "2.0",
                            "method": "job",
                            "params": {"height": 11, "job_id": 0, "difficulty": 1, "pre_pow": "01"}
                        })
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
        });

        let mut client = StratumClient::connect("127.0.0.1", address.port(), "test", "x")
            .await
            .unwrap();
        assert_eq!(client.get_job().await.unwrap().height, 10);
        let mut updates = client.job_updates();
        let _ = updates.borrow_and_update();
        push_tx.send(()).unwrap();
        timeout(Duration::from_secs(2), async {
            loop {
                if updates
                    .borrow()
                    .as_ref()
                    .is_some_and(|job| job.height == 11)
                {
                    break;
                }
                updates.changed().await.unwrap();
            }
        })
        .await
        .expect("height-11 push was not published before timeout");
    }

    async fn write_response(
        writer: &mut WriteHalf<TcpStream>,
        id: &Value,
        result: Value,
        error: Value,
    ) {
        writer
            .write_all(
                format!(
                    "{}\n",
                    json!({"id": id, "jsonrpc": "2.0", "result": result, "error": error})
                )
                .as_bytes(),
            )
            .await
            .unwrap();
    }
}
