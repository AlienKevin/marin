#!/usr/bin/env bash
# Launch 10 sharded eval jobs for arm (b) = full-transcript SFT @ 10K.
# Reuses experiments/exp5611_eval_qwen3_1_7b_swe_zero_8k.py with overridden
# MODEL_PATH / MODEL_NAME env vars.
#
# Daytona budget: 10 shards × 4 concurrent = 40 in-flight instances. Stay well
# under the 100-instance cap. Do NOT run alongside another eval — wait for this
# to fully finish before launching arm (c).
#
# Cross-region egress: the eval shard itself is a CPU job, but it spawns a
# vLLM-serving TPU via Ray + MARIN_VLLM_MODE=native. Without --region, that TPU
# can land in any of the v6e-4 zones (europe-west4-a, us-east1-d, us-east5-b),
# and if it lands outside us-east5 the 1.7B safetensors (~3.4 GB) get read
# cross-region. SFT_REGION pins both the spawned TPU and (transitively) the
# eval shard to a single GCS region; set it to the bucket region holding
# MODEL_PATH. Default us-east5 matches the standard SFT output location.
#
# Usage:
#   bash scripts/launch_eval_arm_b.sh
#   SFT_REGION=us-central1 MODEL_PATH=gs://... bash scripts/launch_eval_arm_b.sh
#
# Assumes the arm (b) SFT job /kevin/exp5611-sft-qwen3-1-7b-10k-echo has
# completed and the checkpoint exists at the expected path.

set -euo pipefail

cd "$(dirname "$0")/.."

# Path is derived from the SFT step name + the executor's hash suffix; verify
# with `gcloud storage ls` before running.
MODEL_PATH=${MODEL_PATH:-"gs://marin-us-east5/checkpoints/exp5611_sft_qwen3_1_7b_swe_zero_10k_8192tokens_arch32k_echo_v5p8-8028d4/hf/step-1249"}
MODEL_NAME=${MODEL_NAME:-"qwen3-1.7b-swe-zero-10k-echo"}
HARBOR_RUN_ID=${HARBOR_RUN_ID:-"run-10k-echo"}
# Pin TPU placement to the model's home region to avoid cross-region weight reads.
SFT_REGION=${SFT_REGION:-"us-east5"}
# Iris job-name suffix; derived from MODEL_NAME by stripping the qwen3-1.7b-... prefix
# so the dashboard names track the model under test (e.g. "1m-echo" for the 1M echo run).
JOB_TAG=${JOB_TAG:-"${MODEL_NAME#qwen3-1.7b-swe-zero-}"}

echo "Eval arm (b): full-transcript SFT"
echo "MODEL_PATH=$MODEL_PATH"
echo "MODEL_NAME=$MODEL_NAME"
echo "HARBOR_RUN_ID=$HARBOR_RUN_ID"
echo "SFT_REGION=$SFT_REGION  (pins spawned vLLM TPU to this region)"
echo "JOB_TAG=$JOB_TAG       (used in iris job names)"
echo ""

# Split the 100 task list into 10 shards of 10.
ALL_TASKS_JSON=$(.venv/bin/python -c "
import json
import sys
sys.path.insert(0, '.')
from experiments.exp5611_eval_qwen3_1_7b_swe_zero_8k import ALL_TASK_NAMES
print(json.dumps(ALL_TASK_NAMES))
")

.venv/bin/python -c "
import json
tasks = json.loads(r'''$ALL_TASKS_JSON''')
assert len(tasks) == 100, f'expected 100, got {len(tasks)}'
shards = [tasks[i*10:(i+1)*10] for i in range(10)]
for i, s in enumerate(shards):
    with open(f'/tmp/eval_arm_b_shard_{i:02d}.json', 'w') as f:
        json.dump(s, f)
print('wrote 10 shard files to /tmp/eval_arm_b_shard_*.json')
"

for i in 00 01 02 03 04 05 06 07 08 09; do
    TASKS_JSON=$(cat /tmp/eval_arm_b_shard_${i}.json)
    uv run iris --config lib/iris/examples/marin.yaml job run \
        --cpu 0.5 --memory 4GB --disk 10GB \
        --region "${SFT_REGION}" \
        --job-name "exp5611-eval-qwen3-1-7b-${JOB_TAG}-shard${i}" \
        -e DAYTONA_API_KEY "${DAYTONA_API_KEY}" \
        -e WANDB_API_KEY "${WANDB_API_KEY}" \
        -e WANDB_ENTITY marin-community -e WANDB_PROJECT harbor \
        -e HF_TOKEN "${HF_TOKEN}" -e MARIN_PREFIX gs://marin-us-central2 \
        -e ENV_TYPE daytona -e MARIN_VLLM_MODE native \
        -e HARBOR_RUN_ID "${HARBOR_RUN_ID}" \
        -e HARBOR_SHARD_ID "shard-${i}" \
        -e HARBOR_TASK_NAMES_JSON "${TASKS_JSON}" \
        -e MODEL_PATH "${MODEL_PATH}" \
        -e MODEL_NAME "${MODEL_NAME}" \
        --no-wait \
        -- python experiments/exp5611_eval_qwen3_1_7b_swe_zero_8k.py
done

echo ""
echo "All 10 shards submitted."
echo "Watch progress:"
echo "  uv run iris --config lib/iris/examples/marin.yaml job list | grep eval-qwen3-1-7b-10k-echo"
