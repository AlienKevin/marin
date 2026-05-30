# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-ZERO 1M @ 8192-token SFT on Qwen3-1.7B-Base.

Superset of the 100K SFT: 100K trajectories + 900K more random rows from the
same commit (built by scripts/sample_swe_zero_supersets_qwen3.py).

Compute target: finish within 24h.
  Per-step time observed at v5p-8 batch=8: ~3.3 s.
  At v5p-32 batch=64 (2 per chip on 32 chips): same memory profile, similar step time.
  1,000,000 / 64 = 15,625 steps * 3.3 s = ~14.3h. Comfortably under 24h.

Tracked in: https://github.com/marin-community/marin/issues/5611
"""

import math
import os

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
import dataclasses

from experiments.qwen3 import qwen3_1_7b
from experiments.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.simple_sft_config import SimpleSFTConfig, compute_per_device_parallelism
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

DATASET_NAME = "swe_zero_1m_qwen3_2f328e1d"
PREBUILT_JSONL_GLOB = "gs://marin-us-east5/datasets/swe-zero-12m-jsonl-1m-2f328e1d/data/*.jsonl.gz"
SUBSET_SIZE = 1_000_000
MAX_SEQ_LEN = 8192

QWEN3_TOKENIZER = "Qwen/Qwen3-1.7B-Base"
QWEN3_INIT = "Qwen/Qwen3-1.7B-Base"

TPU_VARIANT = os.environ.get("TPU_VARIANT", "v5p-32")
SFT_REGION = os.environ.get("SFT_REGION", "us-east5")
RESOURCES = ResourceConfig.with_tpu(TPU_VARIANT, regions=[SFT_REGION])

TARGET_EPOCHS = 1
# Global batch=64 (2 per chip on v5p-32's 32 chips). Same per-chip memory profile as the
# 10K run's batch=8 on v5p-8 (2 per chip on 4 chips). Scales linearly via FSDP/data-parallel.
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
        format=ChatLmDatasetFormat(chat_template=QWEN_3_CHAT_TEMPLATE, pack=1),
    )


tokenized = create_tokenization_step()

mixture_config = lm_mixture_data_config(
    {DATASET_NAME: tokenized},
    {DATASET_NAME: float(SUBSET_SIZE)},
    shuffle=SUBSET_SIZE,
    missing_weights_are_validation=True,
    mixture_block_size=12288,
)

# Marin's qwen3_1_7b config has max_seq_len=4096 (training shape). Expand to 32K so
# Llama3 RoPE precomputes scaled positions out to 32K, keeping anything past 8K
# in-distribution at eval time. Data side stays at 8K (sft_config.max_seq_len=8192).
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

exp5611_qwen3_1m_sft = default_sft(
    name=f"exp5611_sft_qwen3_1_7b_swe_zero_1m_{MAX_SEQ_LEN}tokens_arch32k_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_1_7b_arch_32k,
    sft_config=sft_config,
    tags=[
        "qwen3",
        "qwen3-1.7b-base",
        "swe-zero",
        "sft",
        "1m",
        f"{MAX_SEQ_LEN}tokens",
        RESOURCE_SUFFIX,
    ],
)

exp5611_qwen3_1m_checkpoint = exp5611_qwen3_1m_sft.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print(f"=== exp5611: Qwen3-1.7B-Base SWE-ZERO 1M SFT @ {MAX_SEQ_LEN}-token context ===")
    print(f"Dataset: {PREBUILT_JSONL_GLOB} ({SUBSET_SIZE:,} sampled @ 2f328e1d)")
    print(f"Training steps: {NUM_TRAIN_STEPS:,} (1 epoch, batch={TRAIN_BATCH_SIZE})")
    print(f"Learning rate: {LEARNING_RATE}, checkpoint every {CHECKPOINT_INTERVAL} steps")
    print(f"Resources: {RESOURCES.device.variant} in {SFT_REGION}")
    print(f"Checkpoint: {exp5611_qwen3_1m_checkpoint}")
    executor_main(steps=[exp5611_qwen3_1m_sft])
