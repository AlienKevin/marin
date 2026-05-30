"""Tiny diagnostic: run vLLM with prompt_logprobs=1 on one tiny prompt and dump
the raw structure of the response so we can see what shape vllm-tpu actually
returns. Output to GCS so we can read it back."""
import argparse, json, os, tempfile
from pathlib import Path
import fsspec


def download_ckpt(gs_path, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("gs")
    src = gs_path.removeprefix("gs://").rstrip("/")
    for entry in fs.ls(src):
        name = entry.rsplit("/", 1)[-1]
        with fs.open(f"gs://{entry}", "rb") as fin, open(local_dir / name, "wb") as fout:
            fout.write(fin.read())
    return local_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--out-path", required=True)
    args = p.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="probe_"))
    local_ckpt = download_ckpt(args.model_path, workdir / "ckpt")

    from vllm import LLM, SamplingParams
    llm = LLM(model=str(local_ckpt), tensor_parallel_size=4, max_model_len=2048, trust_remote_code=True)

    # Tiny prompt: just a few tokens
    prompt = "The quick brown fox jumps over the lazy dog"
    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)
    outs = llm.generate(prompts=[prompt], sampling_params=sp, use_tqdm=False)
    out = outs[0]

    dump = {
        "prompt": prompt,
        "prompt_token_ids": list(out.prompt_token_ids) if hasattr(out, "prompt_token_ids") else None,
        "n_prompt_tokens": len(out.prompt_token_ids) if hasattr(out, "prompt_token_ids") else None,
        "prompt_logprobs_type": str(type(out.prompt_logprobs)),
        "prompt_logprobs_len": len(out.prompt_logprobs) if out.prompt_logprobs else 0,
    }
    # Inspect first 3 non-None entries
    samples = []
    for i, ent in enumerate(out.prompt_logprobs or []):
        if ent is None:
            samples.append({"i": i, "value": None})
            continue
        sample = {"i": i, "type": str(type(ent))}
        if hasattr(ent, "items"):
            for tok_id, lp_obj in list(ent.items())[:2]:
                sample.setdefault("entries", []).append({
                    "tok_id": str(tok_id),
                    "lp_obj_type": str(type(lp_obj)),
                    "lp_obj_str": str(lp_obj)[:200],
                    "dir": [a for a in dir(lp_obj) if not a.startswith("_")][:20],
                    "getattr_logprob": str(getattr(lp_obj, "logprob", "MISSING")),
                    "getattr_rank": str(getattr(lp_obj, "rank", "MISSING")),
                    "is_dict": isinstance(lp_obj, dict),
                    "is_float": isinstance(lp_obj, float),
                })
        samples.append(sample)
        if len(samples) >= 5:
            break

    dump["sample_entries"] = samples

    # Write to GCS
    fs = fsspec.filesystem("gs")
    out_strip = args.out_path.removeprefix("gs://").rstrip("/")
    with fs.open(f"gs://{out_strip}", "w") as f:
        f.write(json.dumps(dump, indent=2, default=str))
    print("wrote", args.out_path)


if __name__ == "__main__":
    main()
