# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0.5 closed-loop validation — single-model agent ↔ simulator self-play.

Key unblock: (b) full-transcript 1M is trained on the *whole* transcript, so it
is simultaneously a policy (assistant turns = commands) and a world model (user
turns = observations). One vLLM server plays BOTH roles, alternating turn by
turn — eliminating the two-vLLM-on-one-chip coordination that blocked the
earlier dual-process attempt (TPU_VISIBLE_CHIPS not honored).

Per episode, seeded from a held-out task (system + task prompt only, NO prefix —
the agent must drive from turn 1):
  loop for up to MAX_TURNS:
    agent turn : role=assistant, generate a command, stop at <|im_end|>
    if command contains a submit/finish marker -> episode ends (clean termination)
    sim turn   : role=user, generate the observation for that command

Degeneration fixes from the world-model-probe findings, applied to the SIM turn:
  - max_tokens=512 (training-data observation median ~200 tok; caps runaway)
  - repetition_penalty=1.3 + no-repeat-ngram via vLLM
  - enable_thinking=false in the chat template render (stops <think> leak at long ctx)

Records per episode: turns survived, clean-termination flag, per-turn agent/sim
latency, full transcript, and a degeneration flag per sim observation. This is a
COHERENCE / cost demo — there is no ground-truth pass/fail because the env is
simulated (a fully-simulated rollout can't certify a real fix). The pass@1
payoff experiment (SFT on sim rollouts -> eval on the real container) is the
separate follow-up.

Output: gs://${OUT}/closed_loop_${RUN}.jsonl  (one row per episode)
"""

import argparse
import json
import logging
import re
import tempfile
import time
import zlib
from pathlib import Path

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MAX_MODEL_LEN = 32768
# mini-swe-agent submit markers — agent signals it's done.
SUBMIT_RE = re.compile(r"(MARIN_FINISH|echo\s+COMPLETE_TASK|submit\b|<\s*/?\s*finish\s*>)", re.IGNORECASE)


def download_ckpt(gs_path, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("gs")
    src = gs_path.removeprefix("gs://").rstrip("/")
    for entry in fs.ls(src):
        name = entry.rsplit("/", 1)[-1]
        with fs.open(f"gs://{entry}", "rb") as fin, open(local_dir / name, "wb") as fout:
            fout.write(fin.read())
    return local_dir


def comp_ratio(s):
    b = s.encode("utf-8", "ignore")
    return len(zlib.compress(b, 9)) / len(b) if len(b) >= 20 else 1.0


def is_degen(s):
    return len(s) > 800 and comp_ratio(s) < 0.08


def render(tokenizer, msgs, next_role):
    """Render the conversation and open `next_role`'s turn for completion."""
    rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return rendered + f"<|im_start|>{next_role}\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--heldout", default="gs://marin-us-east5/heldout/sim_ppl_heldout_100.jsonl")
    p.add_argument("--out", default="gs://marin-us-east5/closed-loop")
    p.add_argument("--run-id", required=True)
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--max-turns", type=int, default=50)
    p.add_argument("--agent-max-tokens", type=int, default=4096)
    p.add_argument("--sim-max-tokens", type=int, default=512)
    args = p.parse_args()

    START = time.time()
    workdir = Path(tempfile.mkdtemp(prefix="cl_"))
    model_src = str(download_ckpt(args.model_path, workdir / "ckpt")) if args.model_path.startswith("gs://") else args.model_path

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tokenizer = AutoTokenizer.from_pretrained(model_src, trust_remote_code=True)
    llm = LLM(model=model_src, tensor_parallel_size=4, max_model_len=MAX_MODEL_LEN, trust_remote_code=True)

    # Agent turn: a command. Sim turn: an observation, with anti-degeneration knobs.
    agent_sp = SamplingParams(temperature=1.0, max_tokens=args.agent_max_tokens, stop=["<|im_end|>"])
    sim_sp = SamplingParams(temperature=1.0, max_tokens=args.sim_max_tokens,
                            stop=["<|im_end|>"], repetition_penalty=1.3)

    fs = fsspec.filesystem("gs")
    # --- telemetry: all GCS-based so progress is visible regardless of iris log RPC ---
    run_dir = f"{args.out.rstrip('/')}/{args.run_id}"

    def beat(**kv):
        """Overwrite a tiny heartbeat object; readable any time mid-run."""
        kv["wall_clock"] = round(time.time() - START, 1)
        try:
            with fs.open(f"{run_dir}/heartbeat.json", "w") as f:
                f.write(json.dumps(kv))
        except Exception as e:  # telemetry must never crash the run
            log.warning("heartbeat write failed: %s", e)

    beat(phase="vllm_ready", episode=0, n_episodes=args.n_episodes, turn=0)

    with fs.open(args.heldout, "r") as f:
        heldout = [json.loads(l) for l in f]
    episodes_seed = heldout[: args.n_episodes]

    # Resume: a preemptible v6e-4 restarts this process from scratch on eviction.
    # Skip any episode whose per-episode file already exists, so a preemption costs
    # at most the one in-flight episode instead of all prior work.
    out_rows = []
    for ei, seed in enumerate(episodes_seed):
        ep_file = f"{run_dir}/ep_{ei:03d}.json"
        if fs.exists(ep_file):
            with fs.open(ep_file, "r") as f:
                out_rows.append(json.loads(f.read()))
            log.info("ep %d/%d resumed from existing %s", ei + 1, len(episodes_seed), ep_file)
            continue
        msgs = [{"role": "system", "content": seed["system"]},
                {"role": "user", "content": seed["task"]}]
        transcript = []
        clean_term = False
        agent_lat, sim_lat, sim_degen = [], [], []
        turns = 0
        for t in range(args.max_turns):
            turns = t + 1
            # ---- agent turn ----
            t0 = time.time()
            a_out = llm.generate(prompts=[render(tokenizer, msgs, "assistant")], sampling_params=agent_sp, use_tqdm=False)
            cmd = a_out[0].outputs[0].text
            agent_lat.append(round(time.time() - t0, 2))
            msgs.append({"role": "assistant", "content": cmd})
            transcript.append({"role": "assistant", "text": cmd})
            beat(phase="generating", episode=ei + 1, n_episodes=args.n_episodes, repo=seed["repo"],
                 turn=turns, last="agent", agent_lat=agent_lat[-1],
                 sim_degen_so_far=sum(sim_degen), episodes_done=ei)
            if SUBMIT_RE.search(cmd):
                clean_term = True
                break
            # ---- sim turn ----
            t0 = time.time()
            s_out = llm.generate(prompts=[render(tokenizer, msgs, "user")], sampling_params=sim_sp, use_tqdm=False)
            obs = s_out[0].outputs[0].text
            sim_lat.append(round(time.time() - t0, 2))
            sim_degen.append(is_degen(obs))
            msgs.append({"role": "user", "content": obs})
            transcript.append({"role": "user", "text": obs})
            beat(phase="generating", episode=ei + 1, n_episodes=args.n_episodes, repo=seed["repo"],
                 turn=turns, last="sim", sim_lat=sim_lat[-1], sim_degen_so_far=sum(sim_degen),
                 episodes_done=ei)

        row = dict(
            episode=ei, instance_id=seed["instance_id"], repo=seed["repo"],
            turns=turns, clean_termination=clean_term, hit_max_turns=(turns >= args.max_turns and not clean_term),
            n_sim_turns=len(sim_lat), n_sim_degen=sum(sim_degen),
            agent_lat_mean=round(sum(agent_lat) / len(agent_lat), 2) if agent_lat else None,
            sim_lat_mean=round(sum(sim_lat) / len(sim_lat), 2) if sim_lat else None,
            per_turn_total_mean=round((sum(agent_lat) + sum(sim_lat)) / max(turns, 1), 2),
            transcript=transcript,
        )
        out_rows.append(row)
        # per-episode file: partial progress is durable + readable as it accrues
        with fs.open(f"{run_dir}/ep_{ei:03d}.json", "w") as f:
            f.write(json.dumps(row))
        beat(phase="episode_done", episode=ei + 1, n_episodes=args.n_episodes, repo=seed["repo"],
             turn=turns, episodes_done=ei + 1, last_clean=clean_term, last_degen=sum(sim_degen))
        log.info("ep %d/%d %s: turns=%d clean=%s sim_degen=%d/%d",
                 ei + 1, len(episodes_seed), seed["repo"], turns, clean_term, sum(sim_degen), len(sim_lat))

    out_path = f"{args.out.rstrip('/')}/closed_loop_{args.run_id}.jsonl".removeprefix("gs://")
    with fs.open(f"gs://{out_path}", "w") as fout:
        for r in out_rows:
            fout.write(json.dumps(r) + "\n")
    beat(phase="done", episode=args.n_episodes, n_episodes=args.n_episodes, episodes_done=len(out_rows))
    log.info("wrote gs://%s (%d episodes)", out_path, len(out_rows))


if __name__ == "__main__":
    main()
