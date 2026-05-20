# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rewrite SWE-ZERO 10K trajectories into TerminalWorld simulator training format.

Source: gs://marin-us-east5/datasets/swe-zero-12m-jsonl-10k-2f328e1d/data/*.jsonl.gz
Target: gs://marin-us-east5/datasets/swe-zero-sim-10k-2f328e1d/data/*.jsonl.gz

Original trajectory layout (mini-swe-agent v1):
    [system(mini-swe sysprompt), user(task), assistant(cmd_1), user(env_1),
     assistant(cmd_2), user(env_2), ...]

Rewritten layout for simulator SFT (standard assistant-only mask, no template change):
    [system(simulator sysprompt),
     user(f"Task context:\\n{task}\\n\\nCommand:\\n{cmd_1}"),
     assistant(env_1),
     user(f"Command:\\n{cmd_2}"),
     assistant(env_2), ...]

The simulator therefore learns p(env | task + history + most_recent_cmd) using
exactly the same chat template, optimizer, and recipe as arm (a) — the only
thing that changes is the *content* of the training examples.
"""

import argparse
import gzip
import json
import logging
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SIM_SYSTEM_PROMPT = (
    "You are a Linux terminal simulator for a mini-swe-agent rollout. "
    "Given the task context and the conversation so far (a sequence of bash "
    "commands and their observations), you receive a new bash command from the "
    "agent and emit the terminal's response to that command, exactly as it would "
    "appear in the terminal under the prefix 'Observation: '. Be faithful to "
    "the repository state implied by the prior commands and outputs."
)


def rewrite_messages(messages: list[dict]) -> list[dict] | None:
    if len(messages) < 4:
        return None
    if messages[0]["role"] != "system":
        return None
    if messages[1]["role"] != "user":
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
    if not pairs:
        return None

    out: list[dict] = [{"role": "system", "content": SIM_SYSTEM_PROMPT}]
    first_cmd, first_env = pairs[0]
    out.append(
        {
            "role": "user",
            "content": f"Task context:\n{task}\n\nCommand:\n{first_cmd}",
        }
    )
    out.append({"role": "assistant", "content": first_env})
    for cmd, env in pairs[1:]:
        out.append({"role": "user", "content": f"Command:\n{cmd}"})
        out.append({"role": "assistant", "content": env})
    return out


def process_shard(src_path: Path, dst_path: Path) -> tuple[int, int]:
    n_in = n_out = 0
    with gzip.open(src_path, "rt") as fin, gzip.open(dst_path, "wt") as fout:
        for line in fin:
            n_in += 1
            row = json.loads(line)
            new_msgs = rewrite_messages(row["messages"])
            if new_msgs is None:
                continue
            out_row = {
                "instance_id": row.get("instance_id"),
                "repo": row.get("repo"),
                "messages": new_msgs,
                "trajectory_format": "terminalworld-sim-1",
                "source_exit_status": row.get("exit_status"),
            }
            fout.write(json.dumps(out_row, separators=(",", ":")) + "\n")
            n_out += 1
    return n_in, n_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-glob",
        default="gs://marin-us-east5/datasets/swe-zero-12m-jsonl-10k-2f328e1d/data/*.jsonl.gz",
    )
    parser.add_argument(
        "--dst-prefix",
        default="gs://marin-us-east5/datasets/swe-zero-sim-10k-2f328e1d/data",
    )
    args = parser.parse_args()

    listing = subprocess.run(
        ["gcloud", "storage", "ls", args.src_glob],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    log.info("Found %d source shards", len(listing))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        total_in = total_out = 0
        for src_url in listing:
            shard_name = src_url.rsplit("/", 1)[-1]
            local_src = tmp_path / f"in_{shard_name}"
            local_dst = tmp_path / f"out_{shard_name}"
            log.info("→ downloading %s", src_url)
            subprocess.run(
                ["gcloud", "storage", "cp", src_url, str(local_src)],
                check=True,
                capture_output=True,
            )
            n_in, n_out = process_shard(local_src, local_dst)
            total_in += n_in
            total_out += n_out
            dst_url = f"{args.dst_prefix}/{shard_name}"
            log.info("→ uploading %s (%d/%d rows)", dst_url, n_out, n_in)
            subprocess.run(
                ["gcloud", "storage", "cp", str(local_dst), dst_url],
                check=True,
                capture_output=True,
            )
            local_src.unlink()
            local_dst.unlink()

        log.info("DONE: %d/%d trajectories rewritten", total_out, total_in)


if __name__ == "__main__":
    main()
