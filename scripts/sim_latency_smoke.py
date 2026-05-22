# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0.5 cheap validation v2: characterize simulator vLLM latency alone.

Single v6e-4 + one vllm-tpu serving the simulator. Drives synthetic batched
chat-completion requests built from real arm (a) trajectories at varying
turn depths, with concurrency matching the closed-loop target (=4). Records
per-call latency for each (concurrency, turn-depth) cell.

Combined with arm (a) Daytona eval wall-time (already on HF), this gives us
a per-turn closed-loop cost estimate without needing to run two vLLMs at once.

Outputs:
- gs://${MARIN_PREFIX}/sim-latency-smoke/run-${RUN_ID}/timing.csv
- gs://${MARIN_PREFIX}/sim-latency-smoke/run-${RUN_ID}/summary.json
"""

import argparse
import asyncio
import csv
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import fsspec
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SIM_CKPT_DEFAULT = (
    "gs://marin-us-east5/checkpoints/"
    "exp5611_sft_qwen3_1_7b_swe_zero_sim_10k_8192tokens_arch32k_v5p8-83663d/hf/step-1249"
)
ARM_A_EVAL_HF = "AlienKevin/SWE-ZERO-10K-Qwen3-1.7B-Base-eval"

PORT = 8000
MAX_MODEL_LEN = 32768
MAX_OUTPUT_TOKENS = 4096

# (concurrency, turn_depth) grid to sweep
TURN_DEPTHS = [1, 5, 10, 20, 35, 50]
CONCURRENCIES = [1, 2, 4, 8]
N_TRIALS_PER_CELL = 3

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


def start_vllm(ckpt: Path, log_dir: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("MODEL_IMPL_TYPE", "vllm")
    env.setdefault("TPU_MIN_LOG_LEVEL", "3")
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "vllm", "serve", str(ckpt),
        "--tensor-parallel-size", "4",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--trust-remote-code",
        "--served-model-name", "sim",
    ]
    log.info("starting vllm: %s", " ".join(cmd))
    stdout = open(log_dir / "vllm.stdout", "w")
    stderr = open(log_dir / "vllm.stderr", "w")
    return subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)


async def wait_for_server(timeout: float = 1800) -> None:
    url = f"http://127.0.0.1:{PORT}/v1/models"
    deadline = time.time() + timeout
    last_warn = 0.0
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                r = await client.get(url, timeout=5)
                if r.status_code == 200:
                    log.info("vllm ready after %.1fs", time.time() - (deadline - timeout))
                    return
            except Exception:
                pass
            await asyncio.sleep(5)
            if time.time() - last_warn > 60:
                log.info("waiting for vllm... %.0fs elapsed", time.time() - (deadline - timeout))
                last_warn = time.time()
    raise TimeoutError(f"vllm not ready in {timeout}s")


def load_real_histories() -> list[tuple[str, list[tuple[str, str]]]]:
    """Pull (task_prompt, [(cmd, env)*]) from arm (a) trajectories."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        repo_id=ARM_A_EVAL_HF,
        filename="trajectories.jsonl",
        repo_type="dataset",
        local_dir=tempfile.mkdtemp(),
    )
    out = []
    for line in open(p):
        r = json.loads(line)
        if r.get("not_attempted") or not r.get("trajectory"):
            continue
        traj = r["trajectory"]
        task_prompt = None
        pairs = []
        i = 0
        # first user msg = task prompt
        while i < len(traj):
            if traj[i].get("role") == "user":
                task_prompt = traj[i]["content"]
                i += 1
                break
            i += 1
        if task_prompt is None:
            continue
        # walk pairs of (assistant=bash, user=observation)
        while i + 1 < len(traj):
            a, u = traj[i], traj[i + 1]
            if a.get("role") == "assistant" and u.get("role") == "user":
                pairs.append((a["content"], u["content"]))
            i += 2
        if pairs:
            out.append((task_prompt, pairs))
    log.info("loaded %d real histories", len(out))
    return out


def build_prompt(task_prompt: str, history: list[tuple[str, str]], current_cmd: str) -> list[dict]:
    msgs = [{"role": "system", "content": SIM_SYSTEM_PROMPT}]
    if not history:
        msgs.append({"role": "user", "content": f"Task context:\n{task_prompt}\n\nCommand:\n{current_cmd}"})
        return msgs
    first_cmd, first_env = history[0]
    msgs.append({"role": "user", "content": f"Task context:\n{task_prompt}\n\nCommand:\n{first_cmd}"})
    msgs.append({"role": "assistant", "content": first_env})
    for c, e in history[1:]:
        msgs.append({"role": "user", "content": f"Command:\n{c}"})
        msgs.append({"role": "assistant", "content": e})
    msgs.append({"role": "user", "content": f"Command:\n{current_cmd}"})
    return msgs


async def chat_call(client: httpx.AsyncClient, messages: list[dict]) -> tuple[float, int]:
    payload = {
        "model": "sim",
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    t0 = time.time()
    r = await client.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    dt = time.time() - t0
    out = r.json()["choices"][0]["message"]["content"]
    return dt, len(out)


async def measure_cell(
    client: httpx.AsyncClient,
    histories: list[tuple[str, list[tuple[str, str]]]],
    concurrency: int,
    turn_depth: int,
    trials: list[dict],
) -> None:
    """At turn_depth d: each request has d preceding (cmd, env) pairs + 1 current cmd."""
    # pick first `concurrency` histories that have >= turn_depth+1 pairs
    pool = [(t, p) for t, p in histories if len(p) > turn_depth]
    if len(pool) < concurrency:
        log.warning("only %d histories have >%d turns; skipping cell", len(pool), turn_depth)
        return
    sample = pool[:concurrency]
    prompts = []
    for task_prompt, pairs in sample:
        history = pairs[:turn_depth]
        current_cmd = pairs[turn_depth][0]
        prompts.append(build_prompt(task_prompt, history, current_cmd))

    for trial in range(N_TRIALS_PER_CELL):
        t0 = time.time()
        results = await asyncio.gather(*(chat_call(client, p) for p in prompts))
        wall = time.time() - t0
        latencies = [r[0] for r in results]
        out_lens = [r[1] for r in results]
        log.info(
            "concurrency=%d turn_depth=%d trial=%d wall=%.2fs per_req=%.2f-%.2fs (mean %.2f) out_lens=%s",
            concurrency, turn_depth, trial, wall, min(latencies), max(latencies), sum(latencies)/len(latencies),
            out_lens,
        )
        for lat, out_len in zip(latencies, out_lens):
            trials.append(dict(
                concurrency=concurrency,
                turn_depth=turn_depth,
                trial=trial,
                wall_s=wall,
                per_request_s=lat,
                out_chars=out_len,
            ))


async def run_all(histories: list[tuple[str, list[tuple[str, str]]]], out_dir: str) -> None:
    trials: list[dict] = []
    async with httpx.AsyncClient() as client:
        for c in CONCURRENCIES:
            for d in TURN_DEPTHS:
                await measure_cell(client, histories, c, d, trials)

    # Write outputs
    fs = fsspec.filesystem("gs")
    out_strip = out_dir.removeprefix("gs://").rstrip("/")
    fields = ["concurrency", "turn_depth", "trial", "wall_s", "per_request_s", "out_chars"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for row in trials:
        w.writerow(row)
    with fs.open(f"gs://{out_strip}/timing.csv", "w") as f:
        f.write(buf.getvalue())

    # Aggregate summary: mean wall per (concurrency, turn_depth) cell
    summary = {"cells": []}
    from collections import defaultdict
    by_cell = defaultdict(list)
    for r in trials:
        by_cell[(r["concurrency"], r["turn_depth"])].append(r["wall_s"])
    for (c, d), walls in sorted(by_cell.items()):
        summary["cells"].append(dict(
            concurrency=c,
            turn_depth=d,
            mean_wall_s=sum(walls) / len(walls),
            min_wall_s=min(walls),
            max_wall_s=max(walls),
            n_trials=len(walls),
        ))
    with fs.open(f"gs://{out_strip}/summary.json", "w") as f:
        f.write(json.dumps(summary, indent=2))
    log.info("=== CELL SUMMARY ===")
    for c in summary["cells"]:
        log.info("  c=%d d=%d  mean_wall=%.2fs (n=%d)", c["concurrency"], c["turn_depth"], c["mean_wall_s"], c["n_trials"])
    log.info("wrote to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-ckpt", default=SIM_CKPT_DEFAULT)
    parser.add_argument(
        "--out-dir",
        default=f"gs://marin-us-east5/sim-latency-smoke/run-{int(time.time())}",
    )
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="sim_latency_"))
    log_dir = workdir / "logs"
    log.info("workdir: %s", workdir)

    sim_local = download_ckpt(args.sim_ckpt, workdir / "sim")
    vllm_proc = start_vllm(sim_local, log_dir)

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(wait_for_server())
        histories = load_real_histories()
        loop.run_until_complete(run_all(histories, args.out_dir))
    finally:
        log.info("killing vllm")
        try:
            vllm_proc.terminate()
            vllm_proc.wait(timeout=30)
        except Exception:
            vllm_proc.kill()


if __name__ == "__main__":
    main()
