# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score env-token perplexity for a single checkpoint on the SWE-ZERO held-out set.

Uses vLLM's in-process LLM class with `prompt_logprobs=1` to extract per-prompt-token
logprobs in batched fashion — much more reliable than the HTTP API for scoring.

For each held-out trajectory, render the full chat (prefix + target message) with
the model's chat template, tokenize, run prompt_logprobs scoring, then sum the
NLL over the target tokens (boundary found by tokenizing prefix vs full).

Output: gs://${OUT_DIR}/sim_ppl_${MODEL_NAME}.jsonl with one row per trajectory
plus a single summary line.
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

MAX_MODEL_LEN = 32768

SIM_SYSTEM_PROMPT = (
    "You are a Linux terminal simulator for a mini-swe-agent rollout. "
    "Given the task context and the conversation so far (a sequence of bash "
    "commands and their observations), you receive a new bash command from the "
    "agent and emit the terminal's response to that command, exactly as it would "
    "appear in the terminal under the prefix 'Observation: '. Be faithful to "
    "the repository state implied by the prior commands and outputs."
)


def download_ckpt(gs_path: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("gs")
    src = gs_path.removeprefix("gs://").rstrip("/")
    for entry in fs.ls(src):
        name = entry.rsplit("/", 1)[-1]
        log.info("downloading %s", name)
        with fs.open(f"gs://{entry}", "rb") as fin, open(local_dir / name, "wb") as fout:
            fout.write(fin.read())
    return local_dir


def build_messages_agent(traj: dict) -> tuple[list[dict], list[dict]]:
    """Return (prefix_messages, prefix_plus_target_messages) for agent framing."""
    base = [
        {"role": "system", "content": traj["system"]},
        {"role": "user", "content": traj["task"]},
    ]
    for cmd, env in traj["prefix"]:
        base.append({"role": "assistant", "content": cmd})
        base.append({"role": "user", "content": env})
    base.append({"role": "assistant", "content": traj["target_cmd"]})
    with_target = base + [{"role": "user", "content": traj["target_env"]}]
    return base, with_target


def build_messages_sim(traj: dict) -> tuple[list[dict], list[dict]]:
    """Return (prefix_messages, prefix_plus_target_messages) for sim framing.

    Sim training format (from scripts/rewrite_swe_zero_for_sim.py):
      [system: SIM_SYSTEM, user: "Task context:\\n{task}\\n\\nCommand:\\n{cmd_0}",
       assistant: env_0, user: "Command:\\n{cmd_1}", assistant: env_1, ...]
    """
    msgs = [{"role": "system", "content": SIM_SYSTEM_PROMPT}]
    first_cmd, first_env = traj["prefix"][0]
    msgs.append({"role": "user", "content": f"Task context:\n{traj['task']}\n\nCommand:\n{first_cmd}"})
    msgs.append({"role": "assistant", "content": first_env})
    for cmd, env in traj["prefix"][1:]:
        msgs.append({"role": "user", "content": f"Command:\n{cmd}"})
        msgs.append({"role": "assistant", "content": env})
    msgs.append({"role": "user", "content": f"Command:\n{traj['target_cmd']}"})
    with_target = msgs + [{"role": "assistant", "content": traj["target_env"]}]
    return msgs, with_target


_debug_dumped = False


def run_all(args):
    workdir = Path(tempfile.mkdtemp(prefix="sim_ppl_"))
    log.info("workdir: %s", workdir)
    local_ckpt = download_ckpt(args.model_path, workdir / "ckpt")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(local_ckpt), trust_remote_code=True)
    log.info("tokenizer loaded; chat_template exists: %s", bool(tokenizer.chat_template))

    fs = fsspec.filesystem("gs")
    with fs.open(args.heldout, "r") as f:
        heldout = [json.loads(l) for l in f]
    log.info("loaded %d held-out trajectories", len(heldout))

    # Build (prefix_text, full_text, n_target_tokens) per trajectory
    prepared = []
    for traj in heldout:
        if args.mode == "sim":
            prefix_msgs, full_msgs = build_messages_sim(traj)
        else:
            prefix_msgs, full_msgs = build_messages_agent(traj)
        prefix_text = tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=False)
        full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        # Use the same tokenization both vLLM and we use, sans special tokens
        # (apply_chat_template already inserted them).
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        n_target = max(0, len(full_ids) - len(prefix_ids))
        prepared.append(dict(
            instance_id=traj["instance_id"], repo=traj["repo"],
            full_text=full_text, n_target=n_target,
            full_len=len(full_ids), prefix_len=len(prefix_ids),
        ))

    # Filter to those that fit in max_model_len and have n_target > 0
    keep_idx = [i for i, p in enumerate(prepared)
                if p["n_target"] > 0 and p["full_len"] <= MAX_MODEL_LEN - 2]
    log.info("scoring %d / %d trajectories (others either zero-target or too long)",
             len(keep_idx), len(prepared))

    # Direct vLLM with prompt_logprobs
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=str(local_ckpt),
        tensor_parallel_size=4,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )
    sampling = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)

    prompts = [prepared[i]["full_text"] for i in keep_idx]
    log.info("calling llm.generate on %d prompts...", len(prompts))
    outputs = llm.generate(prompts=prompts, sampling_params=sampling, use_tqdm=True)

    # Build results
    results = [None] * len(prepared)
    sum_nll_total = 0.0
    n_tok_total = 0
    for out_idx, out in enumerate(outputs):
        i = keep_idx[out_idx]
        meta = prepared[i]
        plp = out.prompt_logprobs or []  # list of {token_id: Logprob} or None
        n_target = meta["n_target"]
        nlls = []
        start = max(0, len(plp) - n_target)
        for ent in plp[start:]:
            if ent is None: continue
            # ent is a dict mapping token_id -> Logprob obj with .logprob, .rank
            # The "chosen" token has rank == 1.
            picked = None
            for tok_id, lp_obj in ent.items():
                rank = getattr(lp_obj, "rank", None) if not isinstance(lp_obj, dict) else lp_obj.get("rank")
                logprob = getattr(lp_obj, "logprob", None) if not isinstance(lp_obj, dict) else lp_obj.get("logprob")
                if rank == 1 and logprob is not None:
                    picked = logprob; break
            if picked is None:
                # Fallback: any logprob
                for tok_id, lp_obj in ent.items():
                    logprob = getattr(lp_obj, "logprob", None) if not isinstance(lp_obj, dict) else lp_obj.get("logprob")
                    if logprob is not None:
                        picked = logprob; break
            if picked is None: continue
            nlls.append(-picked)
        if not nlls:
            results[i] = dict(instance_id=meta["instance_id"], repo=meta["repo"],
                               n_target_tokens=0, sum_nll=0, mean_nll=None, ppl=None,
                               error="no_logprobs_extracted")
            continue
        s_nll = sum(nlls); n_used = len(nlls)
        mean_nll = s_nll / n_used
        ppl = math.exp(mean_nll)
        sum_nll_total += s_nll; n_tok_total += n_used
        results[i] = dict(instance_id=meta["instance_id"], repo=meta["repo"],
                           n_target_tokens=n_used, sum_nll=s_nll, mean_nll=mean_nll, ppl=ppl,
                           full_len=meta["full_len"], prefix_len=meta["prefix_len"])

    # Fill gaps (skipped trajectories)
    for i, p in enumerate(prepared):
        if results[i] is None:
            results[i] = dict(instance_id=p["instance_id"], repo=p["repo"],
                               n_target_tokens=0, sum_nll=0, mean_nll=None, ppl=None,
                               error="skipped_too_long_or_empty_target",
                               full_len=p["full_len"], prefix_len=p["prefix_len"])

    # Write per-traj + summary
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--mode", choices=("agent", "sim"), required=True)
    p.add_argument("--heldout", default="gs://marin-us-east5/heldout/sim_ppl_heldout_100.jsonl")
    p.add_argument("--out-path", required=True)
    args = p.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
