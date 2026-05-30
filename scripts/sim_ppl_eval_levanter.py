# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score env-token PPL using Levanter on TPU — same compute_forward pattern as
marin's save_logprobs.py (PR #4398), but with our pre-tokenized held-out as
direct LmExample batches instead of going through LMMixtureDatasetConfig.

For each held-out trajectory: render prefix+target through the model's chat
template, tokenize, build an LmExample with loss_weight=1 on target tokens
only, batch-forward, get per-token loss, sum NLL on target tokens.

Output: gs://${OUT_PATH}/sim_ppl_${MODEL_NAME}.jsonl (one row per traj + summary)
"""

import argparse
import json
import logging
import math
import os
import tempfile
from pathlib import Path

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_MODEL_LEN = 16384
SIM_SYSTEM_PROMPT = (
    "You are a Linux terminal simulator for a mini-swe-agent rollout. "
    "Given the task context and the conversation so far (a sequence of bash "
    "commands and their observations), you receive a new bash command from the "
    "agent and emit the terminal's response to that command, exactly as it would "
    "appear in the terminal under the prefix 'Observation: '. Be faithful to "
    "the repository state implied by the prior commands and outputs."
)


def download_ckpt(gs_path, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("gs")
    src = gs_path.removeprefix("gs://").rstrip("/")
    for entry in fs.ls(src):
        name = entry.rsplit("/", 1)[-1]
        log.info("downloading %s", name)
        with fs.open(f"gs://{entry}", "rb") as fin, open(local_dir / name, "wb") as fout:
            fout.write(fin.read())
    return local_dir


def build_messages_agent(traj):
    base = [
        {"role": "system", "content": traj["system"]},
        {"role": "user", "content": traj["task"]},
    ]
    for cmd, env in traj["prefix"]:
        base.append({"role": "assistant", "content": cmd})
        base.append({"role": "user", "content": env})
    base.append({"role": "assistant", "content": traj["target_cmd"]})
    full = base + [{"role": "user", "content": traj["target_env"]}]
    return base, full


def build_messages_sim(traj):
    msgs = [{"role": "system", "content": SIM_SYSTEM_PROMPT}]
    first_cmd, first_env = traj["prefix"][0]
    msgs.append({"role": "user", "content": f"Task context:\n{traj['task']}\n\nCommand:\n{first_cmd}"})
    msgs.append({"role": "assistant", "content": first_env})
    for cmd, env in traj["prefix"][1:]:
        msgs.append({"role": "user", "content": f"Command:\n{cmd}"})
        msgs.append({"role": "assistant", "content": env})
    msgs.append({"role": "user", "content": f"Command:\n{traj['target_cmd']}"})
    full = msgs + [{"role": "assistant", "content": traj["target_env"]}]
    return msgs, full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="GCS path to HF checkpoint")
    p.add_argument("--model-name", required=True)
    p.add_argument("--mode", choices=("agent", "sim"), required=True)
    p.add_argument("--heldout", default="gs://marin-us-east5/heldout/sim_ppl_heldout_100.jsonl")
    p.add_argument("--out-path", required=True)
    args = p.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="sim_ppl_lev_"))
    log.info("workdir: %s", workdir)
    local_ckpt = download_ckpt(args.model_path, workdir / "ckpt")

    # Tokenization
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(local_ckpt), trust_remote_code=True)
    log.info("tokenizer loaded; chat_template ok: %s", bool(tokenizer.chat_template))

    # Pre-tokenize held-out
    fs = fsspec.filesystem("gs")
    with fs.open(args.heldout, "r") as f:
        heldout = [json.loads(l) for l in f]
    log.info("loaded %d held-out trajectories", len(heldout))

    examples = []
    for traj in heldout:
        if args.mode == "sim":
            prefix_msgs, full_msgs = build_messages_sim(traj)
        else:
            prefix_msgs, full_msgs = build_messages_agent(traj)
        prefix_text = tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=False)
        full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        examples.append(dict(
            instance_id=traj["instance_id"], repo=traj["repo"],
            prefix_len=len(prefix_ids), full_len=len(full_ids),
            tokens=full_ids,
        ))
    log.info("tokenized %d examples (max len %d)", len(examples), max(e["full_len"] for e in examples))

    # Load model via Levanter HFCheckpointConverter
    import jax
    import jmp
    import equinox as eqx
    import numpy as np
    import haliax as hax
    from haliax import Axis
    from haliax.partitioning import round_axis_for_partitioning
    from levanter.compat.hf_checkpoints import HFCheckpointConverter
    from levanter.models.lm_model import LmExample
    from levanter.models.loss import next_token_loss
    from levanter.utils.tree_utils import inference_mode
    from levanter.trainer import TrainerConfig
    import dataclasses as dc
    import sys as _sys; _sys.path.insert(0, ".")
    from experiments.qwen3 import qwen3_1_7b

    trainer = TrainerConfig(mp=jmp.get_policy("c=bf16"))
    compute_axis_mapping = trainer.compute_axis_mapping
    parameter_axis_mapping = trainer.parameter_axis_mapping
    mp = trainer.mp

    # Use the canonical Qwen3-1.7B config from experiments/qwen3.py, with
    # arch max_seq_len=32768 to match SFT and accommodate the held-out prompts.
    model_config = dc.replace(qwen3_1_7b, max_seq_len=MAX_MODEL_LEN)

    with trainer.use_device_mesh(), hax.axis_mapping(parameter_axis_mapping):
        Pos = model_config.Pos
        Vocab = round_axis_for_partitioning(
            Axis("vocab", len(tokenizer)),
            compute_axis_mapping,
        )

        log.info("loading model from HF checkpoint via Levanter")
        converter: HFCheckpointConverter = model_config.hf_checkpoint_converter()
        converter = converter.replaced(reference_checkpoint=str(local_ckpt), tokenizer=tokenizer)
        model = converter.load_pretrained(model_config.model_type, ref=str(local_ckpt), dtype=mp.compute_dtype)
        model = inference_mode(model, True)
        model = mp.cast_to_compute(model)
        log.info("model loaded")

        key = jax.random.PRNGKey(0)

        @hax.named_jit
        def compute_loss(tokens_arr, mask_arr):
            example = LmExample.causal(tokens=tokens_arr, loss_weight=mask_arr)
            activations = model.activations(example.tokens, example.attn_mask, key=key)
            logits = hax.dot(activations, model.get_lm_head(), axis=model.Embed)
            loss = next_token_loss(model.Pos, model.Vocab, logits=logits, true_ids=example.tokens,
                                    loss_weight=example.loss_weight, reduction=None)
            return loss

        results = []
        sum_nll_total = 0.0
        n_tok_total = 0
        for i, ex in enumerate(examples):
            tokens = ex["tokens"]
            T = len(tokens)
            if T > MAX_MODEL_LEN:
                results.append(dict(instance_id=ex["instance_id"], repo=ex["repo"],
                                     n_target_tokens=0, sum_nll=0, mean_nll=None, ppl=None,
                                     error=f"too_long_{T}>{MAX_MODEL_LEN}"))
                continue
            # Build padded tokens + loss mask (1 on target, 0 on prefix)
            padded = np.zeros(MAX_MODEL_LEN, dtype=np.int32)
            padded[:T] = tokens
            mask = np.zeros(MAX_MODEL_LEN, dtype=np.float32)
            # Target = tokens[prefix_len : T]; mark those positions
            mask[ex["prefix_len"]:T] = 1.0
            tokens_arr = hax.named(padded, axis=Pos)
            mask_arr = hax.named(mask, axis=Pos)
            loss = compute_loss(tokens_arr, mask_arr)
            loss_np = np.array(loss.array)
            target_loss = loss_np[ex["prefix_len"]:T]
            target_mask = mask[ex["prefix_len"]:T]
            # next_token_loss already applies loss_weight; sum over target window
            s_nll = float(target_loss.sum())
            n_used = int(target_mask.sum())
            if n_used == 0:
                results.append(dict(instance_id=ex["instance_id"], repo=ex["repo"],
                                     n_target_tokens=0, sum_nll=0, mean_nll=None, ppl=None))
                continue
            mean_nll = s_nll / n_used
            ppl = math.exp(mean_nll)
            sum_nll_total += s_nll
            n_tok_total += n_used
            results.append(dict(instance_id=ex["instance_id"], repo=ex["repo"],
                                 n_target_tokens=n_used, sum_nll=s_nll, mean_nll=mean_nll, ppl=ppl,
                                 full_len=ex["full_len"], prefix_len=ex["prefix_len"]))
            if (i + 1) % 5 == 0:
                log.info("[%3d/%d] running mean NLL=%.3f  ppl=%.2f",
                         i + 1, len(examples), sum_nll_total / n_tok_total,
                         math.exp(sum_nll_total / n_tok_total))

    out_path = args.out_path
    out_strip = out_path.removeprefix("gs://").rstrip("/")
    with fs.open(f"gs://{out_strip}", "w") as fout:
        for r in results:
            fout.write(json.dumps(r) + "\n")
        summary = dict(
            _summary=True,
            model_name=args.model_name,
            model_path=args.model_path,
            mode=args.mode,
            n_trajectories=len(heldout),
            n_scored=sum(1 for r in results if r.get("n_target_tokens", 0) > 0),
            total_target_tokens=n_tok_total,
            total_sum_nll=sum_nll_total,
            mean_nll_per_token=sum_nll_total / n_tok_total if n_tok_total else None,
            ppl=math.exp(sum_nll_total / n_tok_total) if n_tok_total else None,
        )
        fout.write(json.dumps(summary) + "\n")
    log.info("=== SUMMARY ===")
    for k, v in summary.items(): log.info("  %s: %s", k, v)
    log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
