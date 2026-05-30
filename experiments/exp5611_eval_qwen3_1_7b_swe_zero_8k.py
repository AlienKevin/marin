# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval Qwen3-1.7B-Base SFT (10K @ 8192-token truncation) on SWE-bench Verified.

Companion to experiments/exp5611_sft_qwen3_1_7b_swe_zero_8k.py. Evaluates the
SFT checkpoint at 32K context (eval-time max_model_len) to test whether the
`!!!` degeneration we observed on Marin-8B is base-model-specific or general
to any 8K-truncated SWE-ZERO SFT.

Differences from exp5611_eval_marin_8b_swe_zero_10k_8k.py:
- model_path: Qwen3-1.7B-Base SFT checkpoint (us-east5)
- vLLM tokenizer: Qwen/Qwen3-1.7B-Base
- max_model_len: 32768 (user direction; tests longer-context inference behavior)
- max_input_tokens / max_output_tokens scaled accordingly
- output_dir: marin-8b-swe-zero-... -> qwen3-1.7b-swe-zero-...

Tracked in: https://github.com/marin-community/marin/issues/5611

Usage:
  for r in run-1 run-2 run-3; do
    uv run iris --config lib/iris/examples/marin.yaml job run \
      --cpu 0.5 --memory 4GB --disk 10GB \
      --job-name exp5611-eval-qwen3-1-7b-$r \
      -e DAYTONA_API_KEY ${DAYTONA_API_KEY} \
      -e WANDB_API_KEY ${WANDB_API_KEY} \
      -e WANDB_ENTITY marin-community \
      -e WANDB_PROJECT harbor \
      -e HF_TOKEN ${HF_TOKEN} \
      -e MARIN_PREFIX gs://marin-us-central2 \
      -e ENV_TYPE daytona \
      -e MARIN_VLLM_MODE native \
      -e HARBOR_RUN_ID $r \
      --no-wait \
      -- python experiments/exp5611_eval_qwen3_1_7b_swe_zero_8k.py
  done
"""

import json
import logging
import os

from experiments.evals.evals import evaluate_harbor
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Same 100 SWE-bench Verified instances as exp4898 / exp5611 Marin-8B eval, so
# the result is directly comparable to the parent 10K @ 32K and my Marin-8B 10K @ 8K runs.
ALL_TASK_NAMES = [
    "astropy__astropy-13236",
    "astropy__astropy-14369",
    "astropy__astropy-14508",
    "astropy__astropy-14598",
    "astropy__astropy-14995",
    "astropy__astropy-8872",
    "django__django-10097",
    "django__django-11138",
    "django__django-11141",
    "django__django-11206",
    "django__django-11276",
    "django__django-11333",
    "django__django-11433",
    "django__django-11477",
    "django__django-11490",
    "django__django-11728",
    "django__django-11820",
    "django__django-12050",
    "django__django-12276",
    "django__django-12308",
    "django__django-12406",
    "django__django-13109",
    "django__django-13128",
    "django__django-13315",
    "django__django-13346",
    "django__django-13363",
    "django__django-13401",
    "django__django-13410",
    "django__django-13449",
    "django__django-13516",
    "django__django-13670",
    "django__django-13925",
    "django__django-13933",
    "django__django-14017",
    "django__django-14053",
    "django__django-14238",
    "django__django-14315",
    "django__django-14855",
    "django__django-14999",
    "django__django-15037",
    "django__django-15128",
    "django__django-15252",
    "django__django-15277",
    "django__django-15278",
    "django__django-15368",
    "django__django-15467",
    "django__django-15499",
    "django__django-15987",
    "django__django-16082",
    "django__django-16485",
    "django__django-16527",
    "django__django-16595",
    "matplotlib__matplotlib-20826",
    "matplotlib__matplotlib-24870",
    "matplotlib__matplotlib-25332",
    "matplotlib__matplotlib-25960",
    "matplotlib__matplotlib-26466",
    "psf__requests-2317",
    "pydata__xarray-3151",
    "pydata__xarray-3305",
    "pydata__xarray-4629",
    "pydata__xarray-4687",
    "pydata__xarray-6938",
    "pylint-dev__pylint-4551",
    "pylint-dev__pylint-6386",
    "pylint-dev__pylint-6903",
    "pytest-dev__pytest-10051",
    "pytest-dev__pytest-10081",
    "pytest-dev__pytest-5840",
    "pytest-dev__pytest-7324",
    "pytest-dev__pytest-7521",
    "pytest-dev__pytest-8399",
    "scikit-learn__scikit-learn-12973",
    "scikit-learn__scikit-learn-13135",
    "scikit-learn__scikit-learn-13142",
    "scikit-learn__scikit-learn-14087",
    "scikit-learn__scikit-learn-15100",
    "scikit-learn__scikit-learn-25931",
    "scikit-learn__scikit-learn-26194",
    "sphinx-doc__sphinx-11445",
    "sphinx-doc__sphinx-7440",
    "sphinx-doc__sphinx-7757",
    "sphinx-doc__sphinx-8721",
    "sphinx-doc__sphinx-9229",
    "sphinx-doc__sphinx-9230",
    "sphinx-doc__sphinx-9698",
    "sympy__sympy-13372",
    "sympy__sympy-13480",
    "sympy__sympy-14531",
    "sympy__sympy-15017",
    "sympy__sympy-16450",
    "sympy__sympy-16597",
    "sympy__sympy-18199",
    "sympy__sympy-19495",
    "sympy__sympy-20801",
    "sympy__sympy-21379",
    "sympy__sympy-21847",
    "sympy__sympy-22456",
    "sympy__sympy-23413",
    "sympy__sympy-24443",
]

_task_names_json = os.environ.get("HARBOR_TASK_NAMES_JSON")
TASK_NAMES = json.loads(_task_names_json) if _task_names_json else ALL_TASK_NAMES

# Default MODEL_PATH is filled in after SFT lands; override via env var meanwhile.
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "gs://marin-us-east5/checkpoints/exp5611_sft_qwen3_1_7b_swe_zero_1m_8192tokens_arch32k_v5p32-a26bea/hf/step-15624",
)
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3-1.7b-swe-zero-1m-8192tokens")
HARBOR_RUN_ID = os.environ.get("HARBOR_RUN_ID", "run-1")
HARBOR_SHARD_ID = os.environ.get("HARBOR_SHARD_ID", "")
ENV_TYPE = os.environ.get("ENV_TYPE", "daytona")
N_CONCURRENT = int(os.environ.get("HARBOR_N_CONCURRENT", "4"))

_TPU_VARIANT = os.environ.get("TPU_VARIANT", "v6e-4")
# Pin to us-east5 (has v6e-4); without this the scheduler sometimes picks us-west4
# which has no groups in the marin cluster.
_EVAL_REGION = os.environ.get("EVAL_REGION", "us-east5")
RESOURCES = ResourceConfig.with_tpu(_TPU_VARIANT, regions=[_EVAL_REGION])

# Eval at 32K context (user direction). Qwen3-1.7B-Base supports max_position_embeddings=32768
# architecturally with rope_scaling=null, so this is in-distribution for the base model.
VLLM_ENGINE_KWARGS = {
    "max_model_len": 32768,
    "max_num_seqs": N_CONCURRENT,
    "tensor_parallel_size": 4,
}

# vLLM tokenizer override - keep the Qwen3 tokenizer + chat template (with {% generation %}
# markers that render transparently at inference).
os.environ.setdefault("VLLM_TOKENIZER", "Qwen/Qwen3-1.7B-Base")
os.environ.setdefault("MSWEA_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

AGENT_KWARGS = {
    "temperature": 1.0,
    # Match talkie-coder eval setup: step_limit/max_turns generous, max_tokens=4096.
    # eos patch (config.json eos_token_id=[151643,151645]) keeps avg output ~100-200
    # tokens/turn so we rarely hit 4096; the cap is just a safety net for runaway turns.
    "max_turns": 50,
    "model_info": {
        "max_input_tokens": 32768 - 4096,
        "max_output_tokens": 4096,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    },
}

_RUN_SUBDIR = HARBOR_RUN_ID if not HARBOR_SHARD_ID else f"{HARBOR_RUN_ID}/{HARBOR_SHARD_ID}"
OUTPUT_DIR = os.path.join(
    "evaluation",
    "harbor",
    "swebench-verified",
    MODEL_NAME,
    "mini-swe-agent-v1",
    _RUN_SUBDIR,
)

logger.info("=" * 60)
logger.info("exp5611 qwen3-1.7b eval: %s on SWE-bench Verified (100 tasks)", MODEL_NAME)
logger.info("Run id: %s", HARBOR_RUN_ID)
logger.info("Model: %s", MODEL_PATH)
logger.info("Agent: mini-swe-agent-v1, Env: %s, Concurrent: %d", ENV_TYPE, N_CONCURRENT)
logger.info(
    "Engine: max_model_len=%d, max_input=%d, max_output=%d, temp=%s",
    VLLM_ENGINE_KWARGS["max_model_len"],
    AGENT_KWARGS["model_info"]["max_input_tokens"],
    AGENT_KWARGS["model_info"]["max_output_tokens"],
    AGENT_KWARGS["temperature"],
)
logger.info("=" * 60)

if __name__ == "__main__":
    step = evaluate_harbor(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        dataset="swebench-verified",
        version="1.0",
        task_names=TASK_NAMES,
        resource_config=RESOURCES,
        engine_kwargs=VLLM_ENGINE_KWARGS,
        wandb_tags=[
            "harbor",
            "swebench-verified",
            "mini-swe-agent-v1",
            ENV_TYPE,
            MODEL_NAME,
            HARBOR_RUN_ID,
            "exp5611",
            "qwen3-1.7b-ablation",
            "swe-zero-8k-truncation",
        ],
        agent="mini-swe-agent-v1",
        n_concurrent=N_CONCURRENT,
        env=ENV_TYPE,
        agent_kwargs=AGENT_KWARGS,
    ).with_output_path(OUTPUT_DIR)

    executor_main(steps=[step])
