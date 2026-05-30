# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-ZERO 100K @ 8192-token SFT on Qwen3-1.7B-Base (scale-up of exp5611's 10K run).

Superset of the 10K SFT: same 10K trajectories used in
exp5611_sft_qwen3_1_7b_swe_zero_8k.py + 90K additional random rows from the same
commit (built by scripts/sample_swe_zero_supersets_qwen3.py).

Differences from the 10K SFT script:
- Subset size 10K -> 100K (different jsonl GCS path)
- TPU v5p-8 -> v5p-16 (more compute for 10x larger dataset; ~5.7h at batch=16)
- Global batch 8 -> 16 (2 per chip on v5p-16, same memory profile as 10K run)
- Everything else (chat template, tokenizer, optimizer settings, checkpoint cadence) identical

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

DATASET_NAME = "swe_zero_100k_qwen3_2f328e1d"
PREBUILT_JSONL_GLOB = "gs://marin-us-east5/datasets/swe-zero-12m-jsonl-100k-2f328e1d/data/*.jsonl.gz"
SUBSET_SIZE = 100_000
MAX_SEQ_LEN = 8192

QWEN3_TOKENIZER = "Qwen/Qwen3-1.7B-Base"
QWEN3_INIT = "Qwen/Qwen3-1.7B-Base"

TPU_VARIANT = os.environ.get("TPU_VARIANT", "v5p-16")
SFT_REGION = os.environ.get("SFT_REGION", "us-east5")
RESOURCES = ResourceConfig.with_tpu(TPU_VARIANT, regions=[SFT_REGION])

TARGET_EPOCHS = 1
# Global batch=16 (2 per chip on v5p-16's 8 chips). Doubles 10K run's batch=8 for the
# 10x-larger dataset; keeps per-chip memory profile constant.
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "16"))
MICROBATCH_SIZE = TRAIN_BATCH_SIZE
NUM_TRAIN_STEPS = max(1, math.ceil(TARGET_EPOCHS * SUBSET_SIZE / TRAIN_BATCH_SIZE))

# Match talkie-coder/sft (run_swe_sft_12h_v2_lr2e5.sh) optimizer settings.
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

exp5611_qwen3_100k_sft = default_sft(
    name=f"exp5611_sft_qwen3_1_7b_swe_zero_100k_{MAX_SEQ_LEN}tokens_arch32k_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_1_7b_arch_32k,
    sft_config=sft_config,
    tags=[
        "qwen3",
        "qwen3-1.7b-base",
        "swe-zero",
        "sft",
        "100k",
        f"{MAX_SEQ_LEN}tokens",
        RESOURCE_SUFFIX,
    ],
)

exp5611_qwen3_100k_checkpoint = exp5611_qwen3_100k_sft.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print(f"=== exp5611: Qwen3-1.7B-Base SWE-ZERO 100K SFT @ {MAX_SEQ_LEN}-token context ===")
    print(f"Dataset: {PREBUILT_JSONL_GLOB} ({SUBSET_SIZE:,} sampled @ 2f328e1d)")
    print(f"Training steps: {NUM_TRAIN_STEPS:,} (1 epoch, batch={TRAIN_BATCH_SIZE})")
    print(f"Learning rate: {LEARNING_RATE}, checkpoint every {CHECKPOINT_INTERVAL} steps")
    print(f"Resources: {RESOURCES.device.variant} in {SFT_REGION}")
    print(f"Checkpoint: {exp5611_qwen3_100k_checkpoint}")
    executor_main(steps=[exp5611_qwen3_100k_sft])
