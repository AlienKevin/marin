# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample full simulator responses at different context depths.

Companion to sim_latency_smoke.py which only recorded lengths. Picks 3 real
arm (a) histories per depth (1, 10, 35) and asks the sim to predict the next
observation, saving the full text. Also runs the same prompt twice — once with
max_tokens=4096 (default), once with max_tokens=512 (capped) — so we can see
whether the depth-10 'runaway' is just length truncation or content garbage.

Output: gs://marin-us-east5/sim-samples/run-${RUN_ID}/samples.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import fsspec
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SIM_CKPT = (
    "gs://marin-us-east5/checkpoints/"
    "exp5611_sft_qwen3_1_7b_swe_zero_sim_10k_8192tokens_arch32k_v5p8-83663d/hf/step-1249"
)
ARM_A_EVAL_HF = "AlienKevin/SWE-ZERO-10K-Qwen3-1.7B-Base-eval"

PORT = 8000
DEPTHS = [1, 10, 35]
N_PER_DEPTH = 3
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
    return subprocess.Popen(
        cmd,
        stdout=open(log_dir / "vllm.stdout", "w"),
        stderr=open(log_dir / "vllm.stderr", "w"),
        env=env,
    )


async def wait_for_server(timeout: float = 1800) -> None:
    url = f"http://127.0.0.1:{PORT}/v1/models"
    deadline = time.time() + timeout
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                r = await client.get(url, timeout=5)
                if r.status_code == 200:
                    log.info("vllm ready")
                    return
            except Exception:
                pass
            await asyncio.sleep(5)
    raise TimeoutError("vllm not ready")


def build_prompt(task_prompt: str, history: list[tuple[str, str]], cmd: str) -> list[dict]:
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


def load_histories():
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        repo_id=ARM_A_EVAL_HF, filename="trajectories.jsonl",
        repo_type="dataset", local_dir=tempfile.mkdtemp(),
    )
    out = []
    for line in open(p):
        r = json.loads(line)
        if r.get("not_attempted") or not r.get("trajectory"): continue
        traj = r["trajectory"]
        task = None
        pairs = []
        i = 0
        while i < len(traj):
            if traj[i].get("role") == "user":
                task = traj[i]["content"]
                i += 1
                break
            i += 1
        while i + 1 < len(traj):
            a, u = traj[i], traj[i+1]
            if a.get("role") == "assistant" and u.get("role") == "user":
                pairs.append((a["content"], u["content"]))
            i += 2
        if task and pairs:
            out.append((r["task_id"], task, pairs))
    return out


async def chat(client, messages, max_tokens):
    payload = {"model": "sim", "messages": messages, "temperature": 1.0, "max_tokens": max_tokens}
    t0 = time.time()
    r = await client.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    dt = time.time() - t0
    return r.json()["choices"][0]["message"]["content"], dt


async def run(out_dir: str) -> None:
    histories = load_histories()
    samples = []
    async with httpx.AsyncClient() as client:
        for depth in DEPTHS:
            pool = [h for h in histories if len(h[2]) > depth]
            sample = pool[:N_PER_DEPTH]
            for task_id, task_prompt, pairs in sample:
                history = pairs[:depth]
                current_cmd = pairs[depth][0]
                real_env = pairs[depth][1]
                msgs = build_prompt(task_prompt, history, current_cmd)
                for max_tok in (4096, 512):
                    text, dt = await chat(client, msgs, max_tok)
                    samples.append(dict(
                        task_id=task_id, depth=depth, max_tokens=max_tok,
                        current_cmd=current_cmd, real_env=real_env, sim_resp=text,
                        latency_s=dt, sim_resp_chars=len(text),
                    ))
                    log.info("task=%s depth=%d max_tok=%d  cmd=%r  sim_chars=%d (vs real_chars=%d)  lat=%.1fs",
                             task_id, depth, max_tok, current_cmd[:60], len(text), len(real_env), dt)
    fs = fsspec.filesystem("gs")
    out_strip = out_dir.removeprefix("gs://").rstrip("/")
    with fs.open(f"gs://{out_strip}/samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    log.info("wrote %d samples to %s", len(samples), out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=f"gs://marin-us-east5/sim-samples/run-{int(time.time())}")
    args = parser.parse_args()
    workdir = Path(tempfile.mkdtemp(prefix="sim_sample_"))
    sim_local = download_ckpt(SIM_CKPT, workdir / "sim")
    proc = start_vllm(sim_local, workdir / "logs")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(wait_for_server())
        loop.run_until_complete(run(args.out_dir))
    finally:
        try: proc.terminate(); proc.wait(timeout=30)
        except Exception: proc.kill()


if __name__ == "__main__":
    main()
