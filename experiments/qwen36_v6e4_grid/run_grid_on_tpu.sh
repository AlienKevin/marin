#!/usr/bin/env bash
# Runs ON the v6e-4 TPU. DESCOPED grid for Qwen3.6-35B-A3B-FP8: 4 server launches
# (max_model_len {32k,131k} x kv {bf16,fp8}), each held up while sweeping CLIENT concurrency
# (throughput) + one coherence probe at max batch (#6136 degeneration check). Resumable via GCS CSV.
# GCS-persistent XLA cache so re-provisioned nodes skip recompiles.
set -uo pipefail
echo $$ > "$HOME/grid.pid"
IMAGE=us-docker.pkg.dev/hai-gcp-models/ghcr-mirror/open-thoughts/openthoughts-agent:tpu
MODEL=gs://marin-models-us/ot-agent/models/Qwen/Qwen3.6-35B-A3B-FP8
GCS=gs://marin-models-us/ot-agent/qwen36-v6e4-grid
CSV=/tmp/v6e4_qwen36_grid.csv
TOK=/tmp/qwen36-tok
VLLM=/opt/openthoughts/.venv/bin/vllm
PY=/opt/openthoughts/.venv/bin/python
HDR="ctx,kv,max_num_seqs,concurrency,in_len,out_len,req_per_s,mean_out_tok_s,peak_out_tok_s,mean_ttft_ms,mean_tpot_ms,q_n,q_degen,q_pct,q_maxconsec,q_meanrep4,launch_s,status,ts_pt"
INLEN=1024; OUTLEN=1024
# 4 server configs: "ctx kv max_num_seqs"  (max_num_seqs per OOM-safe design)
CONFIGS="32768 bf16 64
32768 fp8 64
131072 bf16 24
131072 fp8 48"
log(){ echo "[$(TZ=America/Los_Angeles date '+%H:%M:%S')] $*"; }

sudo gcloud auth configure-docker us-docker.pkg.dev --quiet 2>/dev/null
log "pull image"; sudo docker pull "$IMAGE" >/dev/null 2>&1 || { log "PULL FAILED"; exit 1; }
gcloud storage cp "$GCS/coherence_probe.py" /tmp/coherence_probe.py 2>/dev/null
if [ ! -f "$TOK/tokenizer_config.json" ]; then mkdir -p "$TOK"
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json added_tokens.json configuration.json chat_template.jinja; do
    gcloud storage cp "$MODEL/$f" "$TOK/" 2>/dev/null; done; log "tokenizer staged"; fi
gcloud storage cp "$GCS/grid.csv" "$CSV" 2>/dev/null || true
[ -s "$CSV" ] && head -1 "$CSV" | grep -q '^ctx,' || echo "$HDR" > "$CSV"

is_done(){ tail -n +2 "$CSV" | awk -F, '{print $1","$2","$4}' | grep -qx "$1"; }  # key ctx,kv,concurrency

launch_server(){ local L=$1 KV=$2 S=$3 kvflag=""
  [ "$KV" = "fp8" ] && kvflag="--kv-cache-dtype fp8"
  sudo docker rm -f qwen36 >/dev/null 2>&1
  sudo docker run -d --net=host --name qwen36 --privileged --shm-size=100gb -v /tmp:/tmp \
    -e JAX_PLATFORMS=tpu,cpu -e MODEL_IMPL_TYPE=vllm -e TOKENIZERS_PARALLELISM=false \
    -e JAX_ENABLE_COMPILATION_CACHE=1 -e JAX_COMPILATION_CACHE_DIR=gs://marin-models-us/ot-agent/qwen36-v6e4-grid/xla-cache \
    -e JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 -e JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 -e VLLM_XLA_CACHE_PATH=/tmp/jax-cache \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    --entrypoint "$VLLM" "$IMAGE" \
    serve "$MODEL" --host 127.0.0.1 --port 8000 --trust-remote-code \
    --served-model-name Qwen3.6-35B-A3B-FP8 --load-format runai_streamer --tensor-parallel-size 4 \
    --max-model-len "$L" --max-num-seqs "$S" --limit-mm-per-prompt '{"image":0,"video":0}' $kvflag >/dev/null 2>&1
}
wait_ready(){ local t=0
  while [ $t -lt 1500 ]; do
    local st=$(sudo docker inspect -f '{{.State.Status}}' qwen36 2>/dev/null)
    [ "$st" = "exited" ] && { sudo docker logs --tail 8 qwen36 2>&1 | grep -iE 'error|oom|exhaust|out of memory|RESOURCE_EXHAUSTED|assert' | tail -1; return 1; }
    curl -s --max-time 5 localhost:8000/v1/models 2>/dev/null | grep -q Qwen3.6 && return 0
    sleep 10; t=$((t+10)); done; echo "timeout_1500s"; return 1; }
run_bench(){ local C=$1 N=$(( $1*8 )); [ $N -lt 64 ] && N=64
  sudo docker exec qwen36 "$VLLM" bench serve --backend openai-chat \
    --base-url http://127.0.0.1:8000 --endpoint /v1/chat/completions \
    --model Qwen3.6-35B-A3B-FP8 --served-model-name Qwen3.6-35B-A3B-FP8 --tokenizer "$TOK" --trust-remote-code \
    --dataset-name random --random-input-len $INLEN --random-output-len $OUTLEN \
    --num-prompts $N --max-concurrency $C --request-rate inf --ignore-eos \
    --save-result --result-dir /tmp --result-filename r.json > /tmp/bench.out 2>&1; }
parse_metrics(){ python3 - <<'PY' 2>/dev/null
import json
try: d=json.load(open('/tmp/r.json'))
except Exception: print(",,,,"); raise SystemExit
print("%s,%s,%s,%s,%s"%(round(d.get('request_throughput',0),3),round(d.get('output_throughput',0),1),
 round(d.get('max_output_tokens_per_s',0),1) if d.get('max_output_tokens_per_s') is not None else '',
 round(d.get('mean_ttft_ms',0),1),round(d.get('mean_tpot_ms',0),2)))
PY
}
run_probe(){ timeout 240 sudo docker exec qwen36 "$PY" /tmp/coherence_probe.py --n "$1" --model Qwen3.6-35B-A3B-FP8 --max-tokens 512 > /tmp/probe.json 2>/tmp/probe.err; }
parse_probe(){ python3 - <<'PY' 2>/dev/null
import json
try: d=json.load(open('/tmp/probe.json'))
except Exception: print(",,,,"); raise SystemExit
pa=d.get('pct_any'); mr=d.get('mean_rep4')
print("%s,%s,%s,%s,%s"%(d.get('ok',''),d.get('degen',''),pa if pa is not None else '',d.get('max_consec',''),mr if mr is not None else ''))
PY
}

echo "$CONFIGS" | while read L KV S; do
  [ -z "$L" ] && continue
  # build concurrency list <= S, ensure S included
  CCS=""; for c in 1 4 8 16 32 64; do [ $c -le $S ] && CCS="$CCS $c"; done
  echo "$CCS" | grep -qw "$S" || CCS="$CCS $S"
  TOPC=$(echo $CCS | tr ' ' '\n' | sort -n | tail -1)
  # skip whole config if all concurrency rows present
  alldone=1; for C in $CCS; do is_done "$L,$KV,$C" || alldone=0; done
  [ $alldone -eq 1 ] && { log "config $L,$KV done; skip"; continue; }
  log "CONFIG $L,$KV seqs=$S : launching server"; t0=$(date +%s)
  launch_server "$L" "$KV" "$S"
  if reason=$(wait_ready); then
    launch_s=$(( $(date +%s) - t0 )); log "CONFIG $L,$KV : ready ${launch_s}s"
    for C in $(echo $CCS | tr ' ' '\n' | sort -n); do
      is_done "$L,$KV,$C" && { log "  cc=$C done; skip"; continue; }
      log "  bench cc=$C"; run_bench "$C"; m=$(parse_metrics); [ -z "$m" ] && m=",,,,"
      q=",,,,"
      if [ "$C" = "$TOPC" ] && [ "$m" != ",,,," ]; then
        log "  coherence probe n=$C (max batch)"; run_probe "$C"; q=$(parse_probe); [ -z "$q" ] && q=",,,,"
        gcloud storage cp /tmp/probe.json "$GCS/probe_${L}_${KV}_${C}.json" 2>/dev/null
      fi
      st="OK"; [ "$m" = ",,,," ] && st="BENCH_FAIL"
      row="$L,$KV,$S,$C,$INLEN,$OUTLEN,$m,$q,$launch_s,$st,$(TZ=America/Los_Angeles date '+%Y-%m-%dT%H:%M:%S')"
      echo "$row" >> "$CSV"; gcloud storage cp "$CSV" "$GCS/grid.csv" 2>/dev/null
      log "  recorded -> $row"
    done
  else
    log "CONFIG $L,$KV : server FAIL (${reason:-notready})"
    row="$L,$KV,$S,$TOPC,$INLEN,$OUTLEN,,,,,,,,,,,$(( $(date +%s)-t0 )),FAIL:${reason:-notready},$(TZ=America/Los_Angeles date '+%Y-%m-%dT%H:%M:%S')"
    : # field count: 6 + 5(metrics) + 5(quality) + launch_s,status,ts = 19 ; 11 commas after out_len
    echo "$row" >> "$CSV"; gcloud storage cp "$CSV" "$GCS/grid.csv" 2>/dev/null
  fi
  sudo docker rm -f qwen36 >/dev/null 2>&1
done
gcloud storage cp "$CSV" "$GCS/grid.csv" 2>/dev/null
echo done > /tmp/GRID_DONE; gcloud storage cp /tmp/GRID_DONE "$GCS/DONE" 2>/dev/null
log "ALL CONFIGS DONE"
