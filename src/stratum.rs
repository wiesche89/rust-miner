use std::{collections::HashMap, sync::Arc, time::Duration};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, WriteHalf},
    net::TcpStream,
    sync::{Mutex, oneshot, watch},
    time::timeout,
};

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

pub struct StratumClient {
    writer: Arc<Mutex<WriteHalf<TcpStream>>>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    next_id: u64,
    job_tx: watch::Sender<Option<Job>>,
    job_rx: watch::Receiver<Option<Job>>,
}

impl StratumClient {
    pub async fn connect(host: &str, port: u16, login: &str, password: &str) -> Result<Self> {
        let stream = TcpStream::connect((host, port))
            .await
            .with_context(|| format!("connecting to {host}:{port}"))?;
        stream.set_nodelay(true)?;
        let (reader, writer) = tokio::io::split(stream);
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let (job_tx, job_rx) = watch::channel(None);
        spawn_reader(reader, Arc::clone(&pending), job_tx.clone());
        let mut client = Self {
            writer: Arc::new(Mutex::new(writer)),
            pending,
            next_id: 1,
            job_tx,
            job_rx,
        };
        let response = client
            .request(
                "login",
                json!({
                    "login": login,
                    "pass": password,
                    "agent": "grin-cuckatoo-rust/0.1"
                }),
            )
            .await?;
        ensure_ok(&response, "login")?;
        Ok(client)
    }

    pub fn job_updates(&self) -> watch::Receiver<Option<Job>> {
        self.job_rx.clone()
    }

    pub async fn get_job(&mut self) -> Result<Job> {
        let response = self.request("getjobtemplate", Value::Null).await?;
        if let Some(error) = response.get("error").filter(|value| !value.is_null()) {
            bail!("getjobtemplate rejected: {error}");
        }
        let job: Job =
            serde_json::from_value(response.get("result").cloned().unwrap_or(Value::Null))
                .context("getjobtemplate returned no usable job")?;
        if let Some(pushed) = self.job_rx.borrow().clone()
            && pushed.height > job.height
        {
            return Ok(pushed);
        }
        self.job_tx.send_replace(Some(job.clone()));
        Ok(job)
    }

    pub async fn submit(&mut self, params: SubmitParams<'_>) -> Result<Value> {
        let response = self
            .request("submit", serde_json::to_value(params)?)
            .await?;
        if let Some(error) = response.get("error").filter(|value| !value.is_null()) {
            return Err(anyhow!("share rejected: {error}"));
        }
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
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

fn spawn_reader(
    reader: tokio::io::ReadHalf<TcpStream>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
    job_tx: watch::Sender<Option<Job>>,
) {
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        loop {
            let line = match lines.next_line().await {
                Ok(Some(line)) => line,
                Ok(None) => break,
                Err(error) => {
                    eprintln!("Stratum reader error: {error}");
                    break;
                }
            };
            let value: Value = match serde_json::from_str(&line) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("invalid Stratum JSON ignored: {error}: {line}");
                    continue;
                }
            };
            if value.get("method").and_then(Value::as_str) == Some("job") {
                match value
                    .get("params")
                    .cloned()
                    .ok_or_else(|| anyhow!("job push has no params"))
                    .and_then(|params| serde_json::from_value(params).map_err(Into::into))
                {
                    Ok(job) => {
                        job_tx.send_replace(Some(job));
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
    });
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
    fn submit_shape_uses_decimal_nonce_and_array() {
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
    async fn background_reader_publishes_job_pushes_without_a_request() {
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
}
