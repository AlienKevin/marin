# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample a held-out set of SWE-ZERO trajectories from repos NOT in the 1M SFT subset.

Used to build the env-token PPL probe (see scripts/sim_ppl_eval_hf.py and
terminalworld.vercel.app/world-model-probe). Produces 100 trajectories, each
from a distinct unseen repo, with >=15 (command, observation) pairs. For each,
we keep the system prompt, task prompt, the first 14 (cmd, env) pairs as the
scoring prefix, and the 15th (cmd, env) as the query command + target observation.

Two stages:
  1. Enumerate every (repo, instance_id) in the 1M training subset
     (swe-zero-12m-jsonl-1m-2f328e1d) so we can exclude them.
  2. Scan the full 12M corpus (swe-zero-12m-jsonl) in a shuffled, fixed-seed
     order, collect trajectories from unseen repos + unseen instance_ids with
     >=15 assistant turns, then round-robin across repos for max repo diversity.

Note: in the 12M corpus the repo/instance_id live under row["metadata"], not at
top level (unlike the pre-tokenized 1M subset where they are top-level).

Output: gs://marin-us-east5/heldout/sim_ppl_heldout_100.jsonl
"""

import argparse
import gzip
import io
import json
import logging
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRAIN_SUBSET_GLOB = "gs://marin-us-east5/datasets/swe-zero-12m-jsonl-1m-2f328e1d/data/"
FULL_CORPUS_GLOB = "gs://marin-us-east5/datasets/swe-zero-12m-jsonl/data/"
OUT_PATH = "gs://marin-us-east5/heldout/sim_ppl_heldout_100.jsonl"
N_HELDOUT = 100
MIN_TURNS = 15
PROBE_SHARDS = 100  # how many corpus shards to scan (shuffled); enough for 100 unseen-repo hits
SEED = 42


def _cat(url: str) -> bytes:
    return subprocess.run(f"gcloud storage cat '{url}'", shell=True, capture_output=True).stdout


def enumerate_train(glob: str) -> tuple[set[str], set[str]]:
    shards = subprocess.run(
        f"gcloud storage ls {glob}", shell=True, capture_output=True, text=True
    ).stdout.strip().split("\n")
    repos: set[str] = set()
    iids: set[str] = set()

    def scan(url):
        out = []
        with gzip.open(io.BytesIO(_cat(url)), "rt") as f:
            for line in f:
                r = json.loads(line)
                # pre-tokenized subset: repo/instance_id are top-level
                repo = r.get("repo") or r.get("metadata", {}).get("repo", "")
                iid = r.get("instance_id") or r.get("metadata", {}).get("instance_id", "")
                if repo or iid:
                    out.append((repo, iid))
        return out

    with ThreadPoolExecutor(max_workers=8) as ex:
        for rows in ex.map(scan, shards):
            for repo, iid in rows:
                if repo:
                    repos.add(repo)
                if iid:
                    iids.add(iid)
    log.info("train subset: %d repos, %d instance_ids", len(repos), len(iids))
    return repos, iids


def sample_heldout(train_repos: set[str], train_iids: set[str]) -> list[dict]:
    shards = subprocess.run(
        f"gcloud storage ls {FULL_CORPUS_GLOB}", shell=True, capture_output=True, text=True
    ).stdout.strip().split("\n")
    rng = random.Random(SEED)
    rng.shuffle(shards)

    def scan(url):
        out = []
        try:
            with gzip.open(io.BytesIO(_cat(url)), "rt") as f:
                for line in f:
                    r = json.loads(line)
                    md = r.get("metadata", {})
                    repo, iid = md.get("repo", ""), md.get("instance_id", "")
                    msgs = r.get("messages", [])
                    n_assist = sum(1 for m in msgs if m.get("role") == "assistant")
                    if repo and iid and n_assist >= MIN_TURNS and repo not in train_repos and iid not in train_iids:
                        out.append((repo, iid, msgs))
        except Exception:
            pass
        return out

    candidates, seen = [], set()
    with ThreadPoolExecutor(max_workers=24) as ex:
        futs = [ex.submit(scan, u) for u in shards[:PROBE_SHARDS]]
        for fut in as_completed(futs):
            for c in fut.result():
                if c[1] not in seen:
                    seen.add(c[1])
                    candidates.append(c)
    log.info("candidates: %d from %d unseen repos", len(candidates), len({c[0] for c in candidates}))

    # Round-robin across repos for maximum repo diversity.
    rng2 = random.Random(SEED + 1)
    by_repo: dict[str, list] = {}
    for c in candidates:
        by_repo.setdefault(c[0], []).append(c)
    order = sorted(by_repo, key=lambda r: rng2.random())
    queues = {r: rng2.sample(by_repo[r], len(by_repo[r])) for r in order}
    selected = []
    while len(selected) < N_HELDOUT and any(queues.values()):
        for r in order:
            if queues[r] and len(selected) < N_HELDOUT:
                selected.append(queues[r].pop())

    rows = []
    for repo, iid, msgs in selected:
        pairs = []
        i = 2  # msgs[0]=system, msgs[1]=task
        while i + 1 < len(msgs):
            a, u = msgs[i], msgs[i + 1]
            if a.get("role") == "assistant" and u.get("role") == "user":
                pairs.append((a["content"], u["content"]))
            i += 2
        if len(pairs) < MIN_TURNS:
            continue
        rows.append(dict(
            instance_id=iid, repo=repo,
            system=msgs[0]["content"], task=msgs[1]["content"],
            prefix=pairs[:14], target_cmd=pairs[14][0], target_env=pairs[14][1],
        ))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-path", default=OUT_PATH)
    args = p.parse_args()

    train_repos, train_iids = enumerate_train(TRAIN_SUBSET_GLOB)
    rows = sample_heldout(train_repos, train_iids)
    log.info("selected %d trajectories from %d unique repos", len(rows), len({r["repo"] for r in rows}))

    out_strip = args.out_path.removeprefix("gs://").rstrip("/")
    with fsspec.open(f"gs://{out_strip}", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log.info("wrote %s", args.out_path)


if __name__ == "__main__":
    main()
