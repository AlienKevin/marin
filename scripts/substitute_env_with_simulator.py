# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replace SWE-ZERO 10K env observations with TerminalWorld simulator predictions.

The Phase 0 derisk version of arm (c): rather than running a full closed-loop
agent ↔ simulator rollout (which would burn many hours on 2x vLLM serving), we
hold the agent's commands fixed at the real SWE-ZERO trajectory and only
replace the env response tokens with the simulator's prediction conditioned
on (task context + previous (cmd, env_predicted) pairs + current cmd).

This isolates the "sim env vs real env" effect with no confounding from
agent-distribution shift. A positive Phase 0 result here would justify the
full agent ↔ simulator rollout build-out in Phase 0.5.

Inputs:
- gs://marin-us-east5/datasets/swe-zero-12m-jsonl-10k-2f328e1d/data/*.jsonl.gz
  (the original SWE-ZERO 10K trajectories)
- TerminalWorld simulator checkpoint (from
  experiments/exp5611_sft_qwen3_1_7b_swe_zero_sim_10k_8k.py)

Output:
- gs://marin-us-east5/datasets/swe-zero-tw-substituted-10k-2f328e1d/data/*.jsonl.gz
  (same structure as input, but each `user` message after position 1 is
  replaced with the simulator's prediction).

The simulator was trained on rewritten messages
([{system: sim_prompt, user: "Task...\\nCommand: ...", assistant: env_response,
  user: "Command: ...", assistant: env_response, ...}], cf.
scripts/rewrite_swe_zero_for_sim.py), so we rebuild that input format for each
step of each trajectory before asking the simulator for the next env.

Execution strategy: one batched ``llm.chat`` call per turn, across all
trajectories that still have an active command at that turn. vLLM's native
continuous batching gives near-optimal throughput; for 10K trajectories x
avg ~12 turns x ~200 tokens we expect ~30 minutes on a v6e-4.
"""

import argparse
import gzip
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Same simulator system prompt as in scripts/rewrite_swe_zero_for_sim.py — must
# match training-time formatting exactly.
SIM_SYSTEM_PROMPT = (
    "You are a Linux terminal simulator for a mini-swe-agent rollout. "
    "Given the task context and the conversation so far (a sequence of bash "
    "commands and their observations), you receive a new bash command from the "
    "agent and emit the terminal's response to that command, exactly as it would "
    "appear in the terminal under the prefix 'Observation: '. Be faithful to "
    "the repository state implied by the prior commands and outputs."
)


def build_sim_input(task: str, history: list[tuple[str, str]], current_cmd: str) -> list[dict]:
    """Reconstruct the simulator's training-time chat for one step."""
    out: list[dict] = [{"role": "system", "content": SIM_SYSTEM_PROMPT}]
    if not history:
        out.append(
            {
                "role": "user",
                "content": f"Task context:\n{task}\n\nCommand:\n{current_cmd}",
            }
        )
        return out
    first_cmd, first_env = history[0]
    out.append(
        {
            "role": "user",
            "content": f"Task context:\n{task}\n\nCommand:\n{first_cmd}",
        }
    )
    out.append({"role": "assistant", "content": first_env})
    for cmd, env in history[1:]:
        out.append({"role": "user", "content": f"Command:\n{cmd}"})
        out.append({"role": "assistant", "content": env})
    out.append({"role": "user", "content": f"Command:\n{current_cmd}"})
    return out


def load_shards(src_glob: str, tmp: Path) -> list[Path]:
    listing = subprocess.run(
        ["gcloud", "storage", "ls", src_glob],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    local_paths: list[Path] = []
    for src_url in listing:
        shard_name = src_url.rsplit("/", 1)[-1]
        dst = tmp / f"in_{shard_name}"
        log.info("downloading %s", src_url)
        subprocess.run(
            ["gcloud", "storage", "cp", src_url, str(dst)],
            check=True,
            capture_output=True,
        )
        local_paths.append(dst)
    return local_paths


def parse_trajectory(row: dict) -> tuple[str, list[tuple[str, str]]] | None:
    """Return (task, [(cmd, real_env), ...]) for a valid SWE-ZERO row, else None."""
    messages = row["messages"]
    if len(messages) < 4 or messages[1]["role"] != "user":
        return None
    task = messages[1]["content"]
    pairs: list[tuple[str, str]] = []
    i = 2
    while i + 1 < len(messages):
        cmd_msg, env_msg = messages[i], messages[i + 1]
        if cmd_msg["role"] != "assistant" or env_msg["role"] != "user":
            break
        pairs.append((cmd_msg["content"], env_msg["content"]))
        i += 2
    return (task, pairs) if pairs else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-glob",
        default="gs://marin-us-east5/datasets/swe-zero-12m-jsonl-10k-2f328e1d/data/*.jsonl.gz",
    )
    parser.add_argument(
        "--dst-prefix",
        default="gs://marin-us-east5/datasets/swe-zero-tw-substituted-10k-2f328e1d/data",
    )
    parser.add_argument(
        "--sim-model-path",
        required=True,
        help="GCS or local path to the simulator HF checkpoint dir.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=15,
        help="Cap on (cmd, env) pairs per trajectory (matches SWE-ZERO mini-swe-agent default).",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # If sim_model_path is GCS, download it locally first (vLLM can't load from GCS directly).
        local_sim_path = args.sim_model_path
        if local_sim_path.startswith("gs://"):
            local_dir = tmp_path / "sim_checkpoint"
            log.info("downloading simulator checkpoint to %s", local_dir)
            subprocess.run(
                ["gcloud", "storage", "cp", "-r", local_sim_path, str(local_dir)],
                check=True,
            )
            local_sim_path = str(local_dir)

        log.info("loading simulator from %s", local_sim_path)
        llm = LLM(
            model=local_sim_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
        )
        sampling_params = SamplingParams(
            temperature=1.0,
            max_tokens=args.max_tokens,
        )

        # Download shards
        shard_paths = load_shards(args.src_glob, tmp_path)

        # Load all rows from all shards into a single working set
        rows: list[dict] = []
        for sp in shard_paths:
            with gzip.open(sp, "rt") as fin:
                for line in fin:
                    rows.append(json.loads(line))
        log.info("loaded %d trajectories", len(rows))

        # Parse trajectories
        parsed: list[tuple[str, list[tuple[str, str]]] | None] = [parse_trajectory(r) for r in rows]

        # Per-trajectory state: (task, real_cmds, predicted_envs_so_far)
        state: list[dict] = []
        for parse_out, row in zip(parsed, rows, strict=True):
            if parse_out is None:
                state.append({"valid": False})
                continue
            task, pairs = parse_out
            n = min(len(pairs), args.max_turns)
            state.append(
                {
                    "valid": True,
                    "row": row,
                    "task": task,
                    "cmds": [c for c, _ in pairs[:n]],
                    "predicted_envs": [],
                }
            )
        n_valid = sum(1 for s in state if s["valid"])
        log.info("valid trajectories: %d / %d", n_valid, len(state))

        # Turn-by-turn batched substitution
        for turn in range(args.max_turns):
            active_indices: list[int] = []
            active_inputs: list[list[dict]] = []
            for i, s in enumerate(state):
                if not s["valid"]:
                    continue
                if turn >= len(s["cmds"]):
                    continue
                history = list(zip(s["cmds"][:turn], s["predicted_envs"], strict=True))
                msgs = build_sim_input(s["task"], history, s["cmds"][turn])
                active_indices.append(i)
                active_inputs.append(msgs)
            if not active_inputs:
                log.info("turn %d: nothing left to do", turn)
                break
            t0 = time.time()
            log.info("turn %d: %d active trajectories, batched chat...", turn, len(active_inputs))
            outputs = llm.chat(active_inputs, sampling_params=sampling_params, use_tqdm=False)
            elapsed = time.time() - t0
            log.info("turn %d: done in %.1fs (%.2f traj/sec)", turn, elapsed, len(active_inputs) / max(elapsed, 1e-3))
            for idx, out in zip(active_indices, outputs, strict=True):
                predicted = out.outputs[0].text
                state[idx]["predicted_envs"].append(predicted)

        # Reassemble trajectories with substituted env messages and write out
        # per-shard so the eventual SFT step finds files in the expected layout.
        rows_per_shard: list[list[dict]] = []
        # Rebuild per-shard partition order matching shard_paths
        offset = 0
        for sp in shard_paths:
            with gzip.open(sp, "rt") as fin:
                n = sum(1 for _ in fin)
            rows_per_shard.append(state[offset : offset + n])
            offset += n
        assert offset == len(state), f"partition mismatch: {offset} vs {len(state)}"

        for sp, partition in zip(shard_paths, rows_per_shard, strict=True):
            shard_name = sp.name.removeprefix("in_")
            local_dst = tmp_path / f"out_{shard_name}"
            n_written = 0
            with gzip.open(local_dst, "wt") as fout:
                for s in partition:
                    if not s["valid"] or not s["predicted_envs"]:
                        continue
                    original_messages = s["row"]["messages"]
                    new_messages = list(original_messages[:2])
                    for cmd, predicted_env in zip(s["cmds"], s["predicted_envs"], strict=True):
                        new_messages.append({"role": "assistant", "content": cmd})
                        new_messages.append({"role": "user", "content": predicted_env})
                    out_row = {
                        "instance_id": s["row"].get("instance_id"),
                        "repo": s["row"].get("repo"),
                        "messages": new_messages,
                        "trajectory_format": "mini-swe-agent-1",
                        "exit_status": "TerminalWorld-substituted",
                        "source_exit_status": s["row"].get("exit_status"),
                    }
                    fout.write(json.dumps(out_row, separators=(",", ":")) + "\n")
                    n_written += 1
            dst_url = f"{args.dst_prefix}/{shard_name}"
            log.info("uploading %s (%d rows)", dst_url, n_written)
            subprocess.run(
                ["gcloud", "storage", "cp", str(local_dst), dst_url],
                check=True,
            )


if __name__ == "__main__":
    main()
