# grin-cuckatoo-miner

A standalone Cuckatoo miner for Grin with CPU, portable wgpu, and native Metal
backends. Every discovered cycle is verified in-process before it can be
submitted.

## Build

```bash
cargo build --release
cargo test --release
```

The long C32 hardware gate is ignored by default:

```bash
cargo test --release --test c32_gate -- --ignored --nocapture
```

## Test a known C32 solution

```bash
./target/release/grin-cuckatoo-miner gate \
  --backend auto \
  --trimming slean \
  --edge-bits 32 \
  --cycle-length 42 \
  --rounds 128 \
  --pre-pow 00 \
  --nonce-start 45 \
  --count 1
```

The complete fixed-seed gate over nonces `0..99` must find exactly nonces
`45` and `74`.

## Mine against a local testnet node

Enable the Stratum server in the Grin node and set `burn_reward = true` for
testnet mining. With a node listening on port `13416`:

```bash
./target/release/grin-cuckatoo-miner mine \
  --backend auto \
  --trimming slean \
  --node-host 127.0.0.1 \
  --node-port 13416 \
  --login m1-testnet
```

Use `--max-graphs 1` for a one-graph smoke test.

## Important options

- `--backend auto|metal|gpu|cpu`
- `--trimming auto|lean|slean`
- `--slean-parts 4` for a typical 16 GB unified-memory machine
- `--local-ram-kib 32`
- `--edge-bits 32 --cycle-length 42 --rounds 128`

Backend selection:

- On macOS C18–C32, `--backend auto` with `--trimming auto` or `slean`
  selects native Metal slean.
- `--backend gpu` with `--trimming auto` or `slean` selects portable wgpu
  slean.
- `--trimming lean` uses portable wgpu with `--backend auto` or `gpu`.
- Native Metal does not support lean; use `--backend gpu` instead.

On an M5 Air with 16 GB, C32/128 and `slean-parts=4` measured about
7.2–7.4 s (0.135–0.139 G/s) with native Metal, 10.4 s (0.096 G/s) with
wgpu slean, and 14.6 s (0.068 G/s) with wgpu lean. Other systems use wgpu,
with a CPU fallback when no GPU is available.

Arena overflow, cancellation, and inconclusive searches never produce or
submit an unchecked proof.

## License

MIT. See `LICENSE`.
