# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score env-token perplexity using HF transformers + torch directly.

vllm-tpu doesn't expose `prompt_logprobs`, so we fall back to running the
model's forward() ourselves on a CPU iris worker (1.7B fits comfortably in
128GB RAM; ~30-60min per checkpoint for 100 prompts × ~5K tokens).

Per-trajectory: render prefix + target chat, tokenize both separately to
find the target span, forward the full ids, compute log_softmax over logits
shifted by 1, sum NLL over the target span.
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

MAX_MODEL_LEN = 16384  # cap to avoid CPU OOM on extreme cases

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
    p.add_argument("--model-path", required=True,
                   help="GCS path (gs://...) to download, or a HF Hub model id (no gs:// prefix)")
    p.add_argument("--model-name", required=True)
    p.add_argument("--mode", choices=("agent", "sim"), required=True)
    p.add_argument("--heldout", default="gs://marin-us-east5/heldout/sim_ppl_heldout_100.jsonl")
    p.add_argument("--out-path", required=True)
    p.add_argument("--chat-template-from", default=None,
                   help="GCS path to a checkpoint dir whose chat_template.jinja should override the "
                        "tokenizer's template — required for base models (which ship no chat template) "
                        "so prompt rendering is byte-identical to the SFT runs.")
    args = p.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="sim_ppl_hf_"))
    log.info("workdir: %s", workdir)

    # model-path may be a GCS checkpoint (download it) or a HF Hub id (load directly).
    if args.model_path.startswith("gs://"):
        model_src = str(download_ckpt(args.model_path, workdir / "ckpt"))
    else:
        model_src = args.model_path  # HF Hub id; from_pretrained will fetch it
    log.info("model source: %s", model_src)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    log.info("loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_src, trust_remote_code=True)

    # Override the chat template so a base model renders prompts identically to the SFT runs.
    if args.chat_template_from:
        fs0 = fsspec.filesystem("gs")
        tmpl_path = args.chat_template_from.removeprefix("gs://").rstrip("/") + "/chat_template.jinja"
        with fs0.open(f"gs://{tmpl_path}", "r") as tf:
            tokenizer.chat_template = tf.read()
        log.info("overrode chat_template from %s (len=%d)", args.chat_template_from, len(tokenizer.chat_template))
    if not tokenizer.chat_template:
        raise ValueError("tokenizer has no chat_template and --chat-template-from not given; "
                         "prompts cannot be rendered comparably.")

    log.info("loading model (bf16, CPU)")
    model = AutoModelForCausalLM.from_pretrained(
        model_src,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to("cpu").eval()

    fs = fsspec.filesystem("gs")
    with fs.open(args.heldout, "r") as f:
        heldout = [json.loads(l) for l in f]
    log.info("loaded %d held-out trajectories", len(heldout))

    results = []
    sum_nll_total = 0.0
    n_tok_total = 0
    for idx, traj in enumerate(heldout):
        if args.mode == "sim":
            prefix_msgs, full_msgs = build_messages_sim(traj)
        else:
            prefix_msgs, full_msgs = build_messages_agent(traj)
        prefix_text = tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=False)
        full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

        prefix_ids = tokenizer(prefix_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        n_target = full_ids.shape[1] - prefix_ids.shape[1]
        full_len = full_ids.shape[1]

        if n_target <= 0:
            results.append(dict(instance_id=traj["instance_id"], repo=traj["repo"],
                                 n_target_tokens=0, sum_nll=0, mean_nll=None, ppl=None,
                                 full_len=full_len, prefix_len=prefix_ids.shape[1],
                                 error="zero_target_tokens"))
            continue
        if full_len > MAX_MODEL_LEN:
            results.append(dict(instance_id=traj["instance_id"], repo=traj["repo"],
                                 n_target_tokens=0, sum_nll=0, mean_nll=None, ppl=None,
                                 full_len=full_len, prefix_len=prefix_ids.shape[1],
                                 error=f"too_long_{full_len}>{MAX_MODEL_LEN}"))
            continue

        with torch.no_grad():
            out = model(full_ids)
            logits = out.logits  # (1, T, V)
            # Predicting position t comes from logits at t-1.
            # Target positions: [full_len-n_target .. full_len-1]
            # Their predicting logits: [full_len-n_target-1 .. full_len-2]
            target_positions = torch.arange(full_len - n_target, full_len)
            pred_logits = logits[0, target_positions - 1, :]  # (n_target, V)
            target_ids = full_ids[0, target_positions]
            log_probs = torch.nn.functional.log_softmax(pred_logits.float(), dim=-1)
            nlls = -log_probs[torch.arange(n_target), target_ids]
            s_nll = nlls.sum().item()
        n_used = n_target
        mean_nll = s_nll / n_used
        ppl = math.exp(mean_nll)
        sum_nll_total += s_nll
        n_tok_total += n_used
        results.append(dict(instance_id=traj["instance_id"], repo=traj["repo"],
                             n_target_tokens=n_used, sum_nll=s_nll, mean_nll=mean_nll, ppl=ppl,
                             full_len=full_len, prefix_len=prefix_ids.shape[1]))
        if (idx + 1) % 5 == 0:
            log.info("[%3d/%d] running mean NLL=%.3f  ppl=%.2f",
                     idx + 1, len(heldout), sum_nll_total / n_tok_total,
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
