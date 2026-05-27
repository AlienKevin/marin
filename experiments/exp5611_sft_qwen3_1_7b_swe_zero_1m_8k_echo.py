# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-ZERO 1M @ 8192-token SFT on Qwen3-1.7B-Base — *full-transcript* variant.

Arm (b) of the TerminalWorld Phase 0 derisk at the 1M scale
(terminalworld.vercel.app, marin#5866). Scale-up of
``exp5611_sft_qwen3_1_7b_swe_zero_100k_8k_echo.py``: same loss-mask change
(``{% generation %}`` markers around user/tool messages as well as assistant
ones, so the SFT cross-entropy is also computed over terminal-output tokens),
applied to 1M rows instead of 100K.

Headline gap to arm (a) shrinks with scale:
- 10K: 7/100 (a) vs 5/100 (b), gap 2 pp
- 100K: 9/100 (a) vs 8/100 (b), gap 1 pp
- 1M: 11/100 (a) reference; (b) is this experiment.

If (b) 1M lands at or above (a) 1M's 11/100, ECHO-style SFT-time unmasking
fully recovers at scale (pure dilution). If it stays ~1-2 pp below, the
residual is consistent with a small env-token contamination signal that
doesn't wash out with more data.

Differences from the 100K full-transcript SFT script:
- Subset size 100K -> 1M (different jsonl GCS path)
- TPU v5p-16 -> v5p-32 (more compute for 10x larger dataset)
- Global batch 16 -> 64 (matches arm (a) 1M, which used batch=64 on v5p-32)
- DATASET_NAME suffix updated (separate tokenization cache from 100K + from arm (a) 1M)
- Everything else (chat template, tokenizer, optimizer settings, checkpoint cadence) identical

Usage:
  uv run iris --config lib/iris/examples/marin.yaml job run \\
      --tpu v5p-32 --region us-east5 \\
      -e WANDB_ENTITY marin-community -e WANDB_PROJECT marin \\
      -e WANDB_API_KEY ${WANDB_API_KEY} -e HF_TOKEN ${HF_TOKEN} \\
      -e MARIN_PREFIX gs://marin-us-east5 \\
      -e SFT_REGION us-east5 \\
      --job-name exp5611-sft-qwen3-1-7b-1m-echo \\
      --no-wait \\
      -- python experiments/exp5611_sft_qwen3_1_7b_swe_zero_1m_8k_echo.py
"""

import math
import os
import dataclasses

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.qwen3 import qwen3_1_7b
from experiments.qwen3_chat_template_full_transcript import (
    QWEN_3_CHAT_TEMPLATE_FULL_TRANSCRIPT,
)
from experiments.simple_sft_config import SimpleSFTConfig, compute_per_device_parallelism
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

# Separate tokenization cache: the {% generation %} placement differs from
# arm (a) 1M (so the assistant_masks differ); SUBSET_SIZE differs from the
# 100K echo run. Each (scale, mask) combo gets its own cache.
DATASET_NAME = "swe_zero_1m_qwen3_2f328e1d_echo"
PREBUILT_JSONL_GLOB = "gs://marin-us-east5/datasets/swe-zero-12m-jsonl-1m-2f328e1d/data/*.jsonl.gz"
SUBSET_SIZE = 1_000_000
MAX_SEQ_LEN = 8192

QWEN3_TOKENIZER = "Qwen/Qwen3-1.7B-Base"
QWEN3_INIT = "Qwen/Qwen3-1.7B-Base"

TPU_VARIANT = os.environ.get("TPU_VARIANT", "v5p-32")
SFT_REGION = os.environ.get("SFT_REGION", "us-east5")
RESOURCES = ResourceConfig.with_tpu(TPU_VARIANT, regions=[SFT_REGION])

TARGET_EPOCHS = 1
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "64"))
MICROBATCH_SIZE = TRAIN_BATCH_SIZE
NUM_TRAIN_STEPS = max(1, math.ceil(TARGET_EPOCHS * SUBSET_SIZE / TRAIN_BATCH_SIZE))

LEARNING_RATE = 2e-5

CHECKPOINT_INTERVAL = 25
HF_KEEP_INTERVAL = max(1, NUM_TRAIN_STEPS // 4)


def create_tokenization_step():
    return default_tokenize(
        name=DATASET_NAME,
        dataset=PREBUILT_JSONL_GLOB,
        tokenizer=QWEN3_TOKENIZER,
        format=ChatLmDatasetFormat(
            chat_template=QWEN_3_CHAT_TEMPLATE_FULL_TRANSCRIPT,
            pack=1,
        ),
    )


tokenized = create_tokenization_step()

mixture_config = lm_mixture_data_config(
    {DATASET_NAME: tokenized},
    {DATASET_NAME: float(SUBSET_SIZE)},
    shuffle=SUBSET_SIZE,
    missing_weights_are_validation=True,
    mixture_block_size=12288,
)

qwen3_1_7b_arch_32k = dataclasses.replace(qwen3_1_7b, max_seq_len=32768)
RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "")

sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=QWEN3_TOKENIZER,
    initialize_from_hf=QWEN3_INIT,
    pad_tokenizer_to_match_model=True,
    train_batch_size=TRAIN_BATCH_SIZE,
    per_device_parallelism=compute_per_device_parallelism(TRAIN_BATCH_SIZE, MICROBATCH_SIZE, RESOURCES),
    per_device_eval_parallelism=8,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    max_seq_len=MAX_SEQ_LEN,
    seed=42,
    steps_per_checkpoint=CHECKPOINT_INTERVAL,
    steps_per_hf_export=HF_KEEP_INTERVAL,
    lr_schedule="cosine",
    warmup=0.03,
    decay=0.9,
    min_lr_ratio=0.1,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    epsilon=1e-8,
    max_grad_norm=30.0,
)

exp5611_qwen3_sft_echo_1m = default_sft(
    name=f"exp5611_sft_qwen3_1_7b_swe_zero_1m_{MAX_SEQ_LEN}tokens_arch32k_echo_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_1_7b_arch_32k,
    sft_config=sft_config,
    tags=[
        "qwen3",
        "qwen3-1.7b-base",
        "swe-zero",
        "sft",
        "1m",
        "echo",
        "terminalworld",
        f"{MAX_SEQ_LEN}tokens",
        RESOURCE_SUFFIX,
    ],
)

exp5611_qwen3_checkpoint_echo_1m = exp5611_qwen3_sft_echo_1m.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== exp5611 arm (b) @ 1M: Qwen3-1.7B-Base SWE-ZERO full-transcript SFT ===")
    print(f"Dataset: {PREBUILT_JSONL_GLOB} ({SUBSET_SIZE:,} sampled @ 2f328e1d)")
    print("Chat template: QWEN_3_CHAT_TEMPLATE_FULL_TRANSCRIPT (user + assistant in loss)")
    print(f"Truncation: pack=1, right-truncate examples > {MAX_SEQ_LEN} tokens")
    print(f"Training steps: {NUM_TRAIN_STEPS:,} (1 epoch)")
    print(f"Batch size: {TRAIN_BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"Resources: {RESOURCES.device.variant} in {SFT_REGION}")
    print(f"Checkpoint: {exp5611_qwen3_checkpoint_echo_1m}")
    executor_main(steps=[exp5611_qwen3_sft_echo_1m])
