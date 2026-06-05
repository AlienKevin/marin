# Qwen3.6-35B-A3B-FP8 — v6e-4 throughput + #6136 coherence grid

Harness used for the v6e-4 (TP=4, single-host) throughput + degeneration grid posted to
[marin#6133](https://github.com/marin-community/marin/issues/6133#issuecomment-4620467315).

**These are standalone test scripts** — the serving itself uses the `openthoughts-agent:tpu`
(vLLM-TPU / tpu_inference **0.20.0**) docker image, not marin source. Model + XLA cache live in GCS.

## Files
- `run_grid_on_tpu.sh` — runs ON a v6e-4 TPU VM. Loops 4 server configs (32k/131k × bf16/fp8),
  holds each server up while sweeping client concurrency (throughput) + one coherence probe at peak
  batch. Resumable (GCS CSV + GCS-persistent XLA cache).
- `coherence_probe.py` — fires N concurrent **real** chat prompts at the live server and scans
  outputs for #6136-style degeneration (repeated-4gram fraction, longest consecutive-repeat run,
  empty-`</think>` count). Prints one JSON line.

## Environment
- TPU VM: `v6e-4`, runtime `v2-alpha-tpuv6e`, preemptible, SA `iris-worker@hai-gcp-models`.
- Model staged at `gs://marin-models-us/ot-agent/models/Qwen/Qwen3.6-35B-A3B-FP8` (37.5 GB, FP8).
- Results CSV + per-config probe JSONs: `gs://marin-models-us/ot-agent/qwen36-v6e4-grid/`.

## Exact serve command (per config; `<L>`=max_model_len, `<S>`=max_num_seqs; add `--kv-cache-dtype fp8` for fp8 KV)
```bash
sudo docker run -d --net=host --name qwen36 --privileged --shm-size=100gb -v /tmp:/tmp \
  -e JAX_PLATFORMS=tpu,cpu -e MODEL_IMPL_TYPE=vllm -e TOKENIZERS_PARALLELISM=false \
  -e JAX_ENABLE_COMPILATION_CACHE=1 \
  -e JAX_COMPILATION_CACHE_DIR=gs://marin-models-us/ot-agent/qwen36-v6e4-grid/xla-cache \
  -e JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 -e JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 \
  -e VLLM_XLA_CACHE_PATH=/tmp/jax-cache -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  --entrypoint /opt/openthoughts/.venv/bin/vllm \
  us-docker.pkg.dev/hai-gcp-models/ghcr-mirror/open-thoughts/openthoughts-agent:tpu \
  serve gs://marin-models-us/ot-agent/models/Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 127.0.0.1 --port 8000 --trust-remote-code \
    --served-model-name Qwen3.6-35B-A3B-FP8 --load-format runai_streamer \
    --tensor-parallel-size 4 --max-model-len <L> --max-num-seqs <S> \
    --limit-mm-per-prompt '{"image":0,"video":0}'
```
Configs: `32768 bf16 64` · `32768 fp8 64` · `131072 bf16 24` · `131072 fp8 48`.

## Exact throughput command (per concurrency `<cc>`)
```bash
sudo docker exec qwen36 /opt/openthoughts/.venv/bin/vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
  --model Qwen3.6-35B-A3B-FP8 --served-model-name Qwen3.6-35B-A3B-FP8 \
  --tokenizer /tmp/qwen36-tok --trust-remote-code \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --num-prompts $((cc*8)) --max-concurrency <cc> --request-rate inf --ignore-eos \
  --save-result --result-dir /tmp --result-filename r.json
```
Concurrency swept over `{1,4,8,16,32,64}` ∩ `≤ max_num_seqs`. Read `output_throughput`
(mean) + `max_output_tokens_per_s` (peak) from `r.json`.

## Exact coherence-probe command (at peak batch; `<S>`=max_num_seqs)
```bash
sudo docker exec qwen36 /opt/openthoughts/.venv/bin/python /tmp/coherence_probe.py \
  --n <S> --model Qwen3.6-35B-A3B-FP8 --max-tokens 512
```

## Run it
```bash
# on the v6e-4 VM, with the two scripts + coherence_probe.py present and gcloud auth:
bash run_grid_on_tpu.sh   # resumable; writes gs://.../qwen36-v6e4-grid/grid.csv
```

## Caveats
- **Non-agentic.** Throughput = random tokens; probe = 16 short single-turn generic prompts. This is
  a floor for #6136, not the agentic datagen distribution where it was originally observed.
- Must set `JAX_PLATFORMS=tpu,cpu` (image pins `tpu`; the vllm weight-loader calls `jax.devices("cpu")`).
- `--limit-mm-per-prompt '{"image":0,"video":0}'` because it's a VL checkpoint.
