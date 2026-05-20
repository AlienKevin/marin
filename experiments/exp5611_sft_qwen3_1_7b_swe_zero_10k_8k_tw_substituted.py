# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-ZERO 10K @ 8192-token SFT on Qwen3-1.7B-Base — *TerminalWorld substituted*.

Arm (c) of the TerminalWorld Phase 0 derisk: identical to
``exp5611_sft_qwen3_1_7b_swe_zero_8k.py`` (arm (a)) except the training data
comes from the TerminalWorld simulator's env predictions rather than real
Daytona executions. Agent commands are held fixed at the SWE-ZERO ground
truth; only the env-response messages are replaced. See
``scripts/substitute_env_with_simulator.py`` for the data prep step that
produces the substituted dataset.

The chat template, optimizer, schedule, batch size, and runtime are inherited
from arm (a) verbatim so the substitution is the only ablation.
"""

import math
import os
import dataclasses

from levanter.data.text import ChatLmDatasetFormat

from experiments.defaults import default_sft, default_tokenize
from experiments.qwen3 import qwen3_1_7b
from experiments.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.simple_sft_config import SimpleSFTConfig, compute_per_device_parallelism
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

DATASET_NAME = "swe_zero_tw_substituted_10k_qwen3"
PREBUILT_JSONL_GLOB = "gs://marin-us-east5/datasets/swe-zero-tw-substituted-10k-2f328e1d/data/*.jsonl.gz"
SUBSET_SIZE = 10_000
MAX_SEQ_LEN = 8192

QWEN3_TOKENIZER = "Qwen/Qwen3-1.7B-Base"
QWEN3_INIT = "Qwen/Qwen3-1.7B-Base"

TPU_VARIANT = os.environ.get("TPU_VARIANT", "v5p-8")
SFT_REGION = os.environ.get("SFT_REGION", "us-east5")
RESOURCES = ResourceConfig.with_tpu(TPU_VARIANT, regions=[SFT_REGION])

TARGET_EPOCHS = 1
TRAIN_BATCH_SIZE = 8
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

exp5611_qwen3_sft_tw_substituted = default_sft(
    name=f"exp5611_sft_qwen3_1_7b_swe_zero_10k_{MAX_SEQ_LEN}tokens_arch32k_tw_substituted_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_1_7b_arch_32k,
    sft_config=sft_config,
    tags=[
        "qwen3",
        "qwen3-1.7b-base",
        "swe-zero",
        "sft",
        "10k",
        "terminalworld",
        "tw-substituted",
        f"{MAX_SEQ_LEN}tokens",
        RESOURCE_SUFFIX,
    ],
)

exp5611_qwen3_checkpoint_tw = exp5611_qwen3_sft_tw_substituted.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print("=== exp5611 arm (c): Qwen3-1.7B-Base SWE-ZERO 10K TerminalWorld-substituted SFT ===")
    print(f"Dataset: {PREBUILT_JSONL_GLOB} ({SUBSET_SIZE:,} substituted trajectories)")
    print(f"Training steps: {NUM_TRAIN_STEPS:,} (1 epoch)")
    print(f"Batch size: {TRAIN_BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"Resources: {RESOURCES.device.variant} in {SFT_REGION}")
    print(f"Checkpoint: {exp5611_qwen3_checkpoint_tw}")
    executor_main(steps=[exp5611_qwen3_sft_tw_substituted])
