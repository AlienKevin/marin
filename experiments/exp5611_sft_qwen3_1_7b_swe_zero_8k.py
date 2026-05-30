# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SWE-ZERO 10K @ 8192-token SFT on Qwen3-1.7B-Base (ablation for #5611).

Base-model ablation: my Marin-8B 10K @ 8K SFT degenerated to 100% `!!!` walls on
SWE-bench Verified, but the Marin-8B *base* already degenerated at 78%, so the SFT
amplified a pre-existing base-model failure mode rather than creating a new one.

This experiment swaps the base model to Qwen3-1.7B-Base while keeping everything
else identical to the Marin-8B 10K @ 8K run:
- 10K random trajectories sampled from AlienKevin/SWE-ZERO-12M-trajectories at the
  *latest* commit (2f328e1dcea8...), via scripts/sample_swe_zero_10k_qwen3.py
- 8192-token max_seq_len, pack=1 (one chat per window, right-truncate at 8K)
- 625 training steps, batch=16, 1 epoch
- LR 5e-5 (matches parent Qwen3-0.6B SFT in exp4898_sft_qwen3_06b_swe_zero_140b.py)

Chat-template handling:
- Uses experiments.qwen3_chat_template.QWEN_3_CHAT_TEMPLATE with `{% generation %}`
  markers so Levanter can apply loss to assistant tokens only.
- The {% generation %} blocks are transparent at inference (HF tokenizer treats
  them as pass-through when return_assistant_tokens_mask=False, which is the
  default for vLLM chat completions).

Tokenizer:
- Qwen/Qwen3-1.7B-Base, vocab padded to 151,936 on the model side vs 151,669 in
  tokenizer. pad_tokenizer_to_match_model=True is required (else ValueError at
  train_lm step, as documented in exp4898_sft_qwen3_06b script).

Tracked in: https://github.com/marin-community/marin/issues/5611

Usage:
  uv run iris --config lib/iris/examples/marin.yaml job run \
      --tpu v5p-8 --region us-east5 \
      -e WANDB_ENTITY marin-community -e WANDB_PROJECT marin \
      -e WANDB_API_KEY ${WANDB_API_KEY} -e HF_TOKEN ${HF_TOKEN} \
      -e MARIN_PREFIX gs://marin-us-east5 \
      -e SFT_REGION us-east5 \
      --no-wait \
      -- python experiments/exp5611_sft_qwen3_1_7b_swe_zero_8k.py
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

# Prebuilt 10K-sample jsonl.gz from scripts/sample_swe_zero_10k_qwen3.py.
# Random sample (seed=42) from AlienKevin/SWE-ZERO-12M-trajectories@2f328e1d.
DATASET_NAME = "swe_zero_10k_qwen3_2f328e1d"
PREBUILT_JSONL_GLOB = "gs://marin-us-east5/datasets/swe-zero-12m-jsonl-10k-2f328e1d/data/*.jsonl.gz"
SUBSET_SIZE = 10_000
MAX_SEQ_LEN = 8192

QWEN3_TOKENIZER = "Qwen/Qwen3-1.7B-Base"
QWEN3_INIT = "Qwen/Qwen3-1.7B-Base"

TPU_VARIANT = os.environ.get("TPU_VARIANT", "v5p-8")
SFT_REGION = os.environ.get("SFT_REGION", "us-east5")
RESOURCES = ResourceConfig.with_tpu(TPU_VARIANT, regions=[SFT_REGION])
_NUM_CHIPS = int(RESOURCES.device.variant.split("-")[-1]) // 2

TARGET_EPOCHS = 1
# Global batch=8 matches talkie-coder (8 FSDP processes x per_device_bs=1). On v5p-8 (4 chips)
# this means 2 per chip; well within Qwen3-1.7B + 8K context memory budget.
TRAIN_BATCH_SIZE = 8
MICROBATCH_SIZE = TRAIN_BATCH_SIZE
NUM_TRAIN_STEPS = max(1, math.ceil(TARGET_EPOCHS * SUBSET_SIZE / TRAIN_BATCH_SIZE))

# LR=2e-5 from RicardoDominguez/talkie-coder/sft/run_swe_sft_12h_v2_lr2e5.sh (their best
# from a sweep on SWE SFT for talkie-1930-13b-base). The parent Qwen3-0.6B SFT used 5e-5
# but talkie's tuning on the same task family argues for the lower value.
LEARNING_RATE = 2e-5

# Levanter resume-checkpoint cadence. With preemptible v5p-8 hitting ~14-min preemption
# cycles (observed 2026-05-12 in this experiment), 625-step checkpoints (the default
# NUM_TRAIN_STEPS//4) means each preemption wipes all progress. 25 steps mirrors
# exp4898_sft_qwen3_06b's setting and exp4760 (Marin-32B SFT)'s fix for the same issue.
CHECKPOINT_INTERVAL = 25
HF_KEEP_INTERVAL = max(1, NUM_TRAIN_STEPS // 4)


def create_tokenization_step():
    # Direct GCS path -> TokenizeConfig (file paths), bypassing transform_conversation.
    # pack=1 -> max_segments_per_example=1, default slice_strategy="left" right-truncates.
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

# Marin's qwen3_1_7b config has max_seq_len=4096 (Marin training shape). For SFT on
# Qwen3-1.7B-Base we expand to 32768 so Llama3 RoPE precomputes scaled positions out
# to 32K. The data side (sft_config.max_seq_len=8192) still right-truncates training
# inputs to 8K. Earlier we set arch max_seq_len=8192 which made anything past pos 8191
# fall into the `!` attractor at eval time. Setting it to 32K means trained positions
# 0-8191 get task updates while positions 8192-32767 retain Llama3-extrapolated RoPE.
qwen3_1_7b_arch_32k = dataclasses.replace(qwen3_1_7b, max_seq_len=32768)
RESOURCE_SUFFIX = RESOURCES.device.variant.replace("-", "")

sft_config = SimpleSFTConfig(
    resources=RESOURCES,
    tokenizer=QWEN3_TOKENIZER,
    initialize_from_hf=QWEN3_INIT,
    # Qwen3 base embedding is padded to 151,936 vs 151,669 in tokenizer; needed
    # to avoid a ValueError at train_lm step (see exp4898_sft_qwen3_06b notes).
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
    # Match talkie-coder's optimizer settings:
    #   cosine_with_min_lr -> Levanter "cosine" + min_lr_ratio > 0 (approximation)
    #   warmup_ratio=0.03, weight_decay=0.1, max_grad_norm=30 (loose clip for chat-token spikes)
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

exp5611_qwen3_sft = default_sft(
    name=f"exp5611_sft_qwen3_1_7b_swe_zero_10k_{MAX_SEQ_LEN}tokens_arch32k_{RESOURCE_SUFFIX}",
    tokenized=mixture_config,
    model_config=qwen3_1_7b_arch_32k,
    sft_config=sft_config,
    tags=[
        "qwen3",
        "qwen3-1.7b-base",
        "swe-zero",
        "sft",
        "10k",
        f"{MAX_SEQ_LEN}tokens",
        RESOURCE_SUFFIX,
    ],
)

exp5611_qwen3_checkpoint = exp5611_qwen3_sft.cd(f"hf/step-{NUM_TRAIN_STEPS - 1}").nonblocking()

if __name__ == "__main__":
    print(f"=== exp5611: Qwen3-1.7B-Base SWE-ZERO 10K SFT @ {MAX_SEQ_LEN}-token context ===")
    print(f"Dataset: {PREBUILT_JSONL_GLOB} ({SUBSET_SIZE:,} sampled @ 2f328e1d)")
    print(f"Truncation: pack=1, right-truncate examples > {MAX_SEQ_LEN} tokens")
    print(f"Training steps: {NUM_TRAIN_STEPS:,} (1 epoch)")
    print(f"Batch size: {TRAIN_BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"Resources: {RESOURCES.device.variant} in {SFT_REGION}")
    print(f"Checkpoint: {exp5611_qwen3_checkpoint}")
    executor_main(steps=[exp5611_qwen3_sft])
