# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0.5 cheap validation: closed-loop agent <-> simulator on one v6e-8.

Two vllm-tpu processes co-located on a v6e-8 (TP=4 each), serving:
- the arm (a) 10K agent checkpoint on port 8000 (chips 0..3)
- the 10K simulator checkpoint on port 8001 (chips 4..7)

Drives 10 SWE-bench Verified trajectories at 4 concurrent, with the simulator
replacing Daytona for the env step. Records per-turn latency for both halves
and writes trajectories + timing CSV + summary to GCS.

Outputs:
- gs://${MARIN_PREFIX}/closed-loop-smoke/${RUN_ID}/trajectories.jsonl
- gs://${MARIN_PREFIX}/closed-loop-smoke/${RUN_ID}/timing.csv
- gs://${MARIN_PREFIX}/closed-loop-smoke/${RUN_ID}/summary.json

Note: pass@1 against SWE-bench is NOT measured here — the simulator outputs
aren't grounded in a real repo, so verifier rewards aren't meaningful. The
point is throughput + behavior characterization for Phase 0.5 planning.
"""

import argparse
import asyncio
import csv
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import fsspec
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

AGENT_CKPT_DEFAULT = (
    "gs://marin-us-east5/checkpoints/"
    "exp5611_sft_qwen3_1_7b_swe_zero_10k_8192tokens_arch32k_v5p8-e5118c/hf/step-1249"
)
SIM_CKPT_DEFAULT = (
    "gs://marin-us-east5/checkpoints/"
    "exp5611_sft_qwen3_1_7b_swe_zero_sim_10k_8192tokens_arch32k_v5p8-83663d/hf/step-1249"
)
ARM_A_EVAL_HF = "AlienKevin/SWE-ZERO-10K-Qwen3-1.7B-Base-eval"

AGENT_PORT, SIM_PORT = 8000, 8001
N_CONCURRENT = 4
N_TASKS = 10
MAX_TURNS = 50
MAX_MODEL_LEN = 32768
MAX_OUTPUT_TOKENS = 4096

# Trim observation to head 5K + tail 5K chars before showing to agent (matches arm (a) eval).
OBS_TRUNC_CHARS = 10000

SIM_SYSTEM_PROMPT = (
    "You are a Linux terminal simulator for a mini-swe-agent rollout. "
    "Given the task context and the conversation so far (a sequence of bash "
    "commands and their observations), you receive a new bash command from the "
    "agent and emit the terminal's response to that command, exactly as it would "
    "appear in the terminal under the prefix 'Observation: '. Be faithful to "
    "the repository state implied by the prior commands and outputs."
)

TERMINATE_TOKENS = (
    "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT",
    "MINI_SWE_AGENT_FINAL_OUTPUT",
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


def start_vllm(ckpt: Path, port: int, chips: list[int], log_dir: Path) -> subprocess.Popen:
    env = os.environ.copy()
    # vllm-tpu picks TPU via JAX; restrict which chips this process sees.
    env["TPU_VISIBLE_CHIPS"] = ",".join(str(c) for c in chips)
    # tpu_inference defaults MODEL_IMPL_TYPE=auto -> flax_nnx fails without auto mesh
    env.setdefault("MODEL_IMPL_TYPE", "vllm")
    env.setdefault("TPU_MIN_LOG_LEVEL", "3")
    env.setdefault("TPU_STDERR_LOG_LEVEL", "3")
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "vllm", "serve", str(ckpt),
        "--tensor-parallel-size", str(len(chips)),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--trust-remote-code",
        "--served-model-name", f"vllm_{port}",
    ]
    log.info("starting vllm on port %d (chips %s): %s", port, chips, " ".join(cmd))
    stdout = open(log_dir / f"vllm_{port}.stdout", "w")
    stderr = open(log_dir / f"vllm_{port}.stderr", "w")
    return subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env)


async def wait_for_server(port: int, timeout: float = 900) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                r = await client.get(url, timeout=5)
                if r.status_code == 200:
                    log.info("vllm @ %d ready", port)
                    return
            except Exception:
                pass
            await asyncio.sleep(5)
    raise TimeoutError(f"vllm @ port {port} not ready in {timeout}s")


def load_arm_a_eval_seed() -> tuple[str, list[dict]]:
    """Pull system prompt + first N_TASKS task prompts from arm (a)'s HF dataset."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        repo_id=ARM_A_EVAL_HF,
        filename="trajectories.jsonl",
        repo_type="dataset",
        local_dir=tempfile.mkdtemp(),
    )
    rows = [json.loads(l) for l in open(p)]
    # First row's first message is the system prompt; the second is task prompt.
    sys_prompt = None
    tasks = []
    for r in rows:
        if r.get("not_attempted") or not r.get("trajectory"):
            continue
        traj = r["trajectory"]
        if traj[0].get("role") == "system" and sys_prompt is None:
            sys_prompt = traj[0]["content"]
        # task prompt is the first user msg
        for m in traj:
            if m.get("role") == "user":
                tasks.append({"task_id": r["task_id"], "task_prompt": m["content"]})
                break
        if len(tasks) >= N_TASKS:
            break
    assert sys_prompt is not None, "no system prompt found in arm (a) eval"
    return sys_prompt, tasks


def extract_bash(text: str) -> str | None:
    matches = re.findall(r"```bash\n(.+?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def is_terminal(bash: str) -> bool:
    return any(t in bash for t in TERMINATE_TOKENS)


def trim_obs(s: str) -> str:
    if len(s) <= OBS_TRUNC_CHARS:
        return s
    half = OBS_TRUNC_CHARS // 2
    return s[:half] + "\n... [truncated] ...\n" + s[-half:]


def build_sim_messages(task_prompt: str, history: list[tuple[str, str]], cmd: str) -> list[dict]:
    msgs = [{"role": "system", "content": SIM_SYSTEM_PROMPT}]
    if not history:
        msgs.append({"role": "user", "content": f"Task context:\n{task_prompt}\n\nCommand:\n{cmd}"})
        return msgs
    first_cmd, first_env = history[0]
    msgs.append({"role": "user", "content": f"Task context:\n{task_prompt}\n\nCommand:\n{first_cmd}"})
    msgs.append({"role": "assistant", "content": first_env})
    for c, e in history[1:]:
        msgs.append({"role": "user", "content": f"Command:\n{c}"})
        msgs.append({"role": "assistant", "content": e})
    msgs.append({"role": "user", "content": f"Command:\n{cmd}"})
    return msgs


async def chat_call(client: httpx.AsyncClient, port: int, model: str, messages: list[dict]) -> tuple[str, float]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": MAX_OUTPUT_TOKENS,
    }
    t0 = time.time()
    r = await client.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    return text, time.time() - t0


async def run_trajectory(
    client: httpx.AsyncClient,
    task: dict,
    sys_prompt: str,
    timings: list[dict],
) -> dict:
    task_id = task["task_id"]
    task_prompt = task["task_prompt"]
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": task_prompt},
    ]
    history: list[tuple[str, str]] = []
    log.info("[%s] starting", task_id)
    t_traj0 = time.time()
    terminated_reason = "max_turns"
    for turn in range(MAX_TURNS):
        # agent call
        agent_resp, agent_dt = await chat_call(client, AGENT_PORT, f"vllm_{AGENT_PORT}", messages)
        messages.append({"role": "assistant", "content": agent_resp})
        bash = extract_bash(agent_resp)
        if bash is None:
            timings.append(dict(task=task_id, turn=turn, agent_s=agent_dt, sim_s=0.0,
                                bash_len=0, sim_len=0, terminated="no_bash"))
            terminated_reason = "no_bash"
            break
        if is_terminal(bash):
            timings.append(dict(task=task_id, turn=turn, agent_s=agent_dt, sim_s=0.0,
                                bash_len=len(bash), sim_len=0, terminated="clean_submit"))
            terminated_reason = "clean_submit"
            break

        # sim call
        sim_msgs = build_sim_messages(task_prompt, history, bash)
        sim_obs, sim_dt = await chat_call(client, SIM_PORT, f"vllm_{SIM_PORT}", sim_msgs)
        obs_trimmed = trim_obs(sim_obs)
        rendered_obs = f"Observation: {obs_trimmed}"
        messages.append({"role": "user", "content": rendered_obs})
        history.append((bash, sim_obs))
        timings.append(dict(task=task_id, turn=turn, agent_s=agent_dt, sim_s=sim_dt,
                            bash_len=len(bash), sim_len=len(sim_obs), terminated=""))
    total = time.time() - t_traj0
    log.info("[%s] done: turn=%d  reason=%s  total=%.1fs", task_id, turn, terminated_reason, total)
    return dict(
        task_id=task_id,
        n_turns=turn + 1,
        terminated_reason=terminated_reason,
        total_s=total,
        messages=messages,
    )


async def run_all(sys_prompt: str, tasks: list[dict], out_dir: str) -> None:
    sem = asyncio.Semaphore(N_CONCURRENT)
    timings: list[dict] = []
    trajectories: list[dict] = []

    async with httpx.AsyncClient() as client:
        async def one(t):
            async with sem:
                return await run_trajectory(client, t, sys_prompt, timings)

        results = await asyncio.gather(*(one(t) for t in tasks))
        trajectories.extend(results)

    # Write outputs to GCS
    fs = fsspec.filesystem("gs")
    out_strip = out_dir.removeprefix("gs://").rstrip("/")
    # trajectories
    with fs.open(f"gs://{out_strip}/trajectories.jsonl", "w") as f:
        for r in trajectories:
            f.write(json.dumps(r) + "\n")
    # timings
    buf = io.StringIO()
    fields = ["task", "turn", "agent_s", "sim_s", "bash_len", "sim_len", "terminated"]
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    for row in timings:
        w.writerow(row)
    with fs.open(f"gs://{out_strip}/timing.csv", "w") as f:
        f.write(buf.getvalue())
    # summary
    n_finished = len(trajectories)
    n_clean = sum(1 for r in trajectories if r["terminated_reason"] == "clean_submit")
    mean_turns = sum(r["n_turns"] for r in trajectories) / max(n_finished, 1)
    mean_total = sum(r["total_s"] for r in trajectories) / max(n_finished, 1)
    agent_lat = [t["agent_s"] for t in timings]
    sim_lat = [t["sim_s"] for t in timings if t["sim_s"] > 0]
    summary = dict(
        n_tasks=n_finished,
        n_concurrent=N_CONCURRENT,
        max_turns=MAX_TURNS,
        n_clean_submit=n_clean,
        mean_turns_per_task=mean_turns,
        mean_total_s_per_task=mean_total,
        mean_agent_call_s=sum(agent_lat) / max(len(agent_lat), 1),
        mean_sim_call_s=sum(sim_lat) / max(len(sim_lat), 1),
        total_agent_calls=len(agent_lat),
        total_sim_calls=len(sim_lat),
    )
    with fs.open(f"gs://{out_strip}/summary.json", "w") as f:
        f.write(json.dumps(summary, indent=2))
    log.info("=== SUMMARY ===")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)
    log.info("wrote to %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-ckpt", default=AGENT_CKPT_DEFAULT)
    parser.add_argument("--sim-ckpt", default=SIM_CKPT_DEFAULT)
    parser.add_argument(
        "--out-dir",
        default=f"gs://marin-us-east5/closed-loop-smoke/run-{int(time.time())}",
    )
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="closed_loop_"))
    log_dir = workdir / "logs"
    log.info("workdir: %s", workdir)

    # 1. download both ckpts in parallel-ish (sequential here for simplicity)
    agent_local = download_ckpt(args.agent_ckpt, workdir / "agent")
    sim_local = download_ckpt(args.sim_ckpt, workdir / "sim")

    # 2. start both vllms
    agent_proc = start_vllm(agent_local, AGENT_PORT, chips=[0, 1, 2, 3], log_dir=log_dir)
    sim_proc = start_vllm(sim_local, SIM_PORT, chips=[4, 5, 6, 7], log_dir=log_dir)

    try:
        # 3. wait for both
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            asyncio.gather(wait_for_server(AGENT_PORT), wait_for_server(SIM_PORT))
        )

        # 4. load tasks
        sys_prompt, tasks = load_arm_a_eval_seed()
        log.info("loaded %d tasks", len(tasks))

        # 5. run
        loop.run_until_complete(run_all(sys_prompt, tasks, args.out_dir))
    finally:
        log.info("killing vllm processes")
        for p in (agent_proc, sim_proc):
            try:
                p.terminate()
                p.wait(timeout=30)
            except Exception:
                p.kill()


if __name__ == "__main__":
    main()
