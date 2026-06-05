#!/usr/bin/env python3
"""Fire N concurrent REAL chat prompts at a live vLLM server and scan outputs for
#6136-style degeneration (repetition loops, think-overuse, incoherence).
Outputs one JSON line: {n, ok, degen, pct_any, max_consec, mean_rep4, worst}.
Dependency-free (urllib + concurrent.futures)."""
import json, sys, urllib.request, argparse, re
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "What is 17 * 23? Show your reasoning briefly.",
    "Write a Python function is_palindrome(s) and explain it in one line.",
    "Explain why the sky is blue in 3 sentences.",
    "List the first 6 prime numbers.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "If x + 3 = 10, what is x? One short sentence.",
    "Give 3 benefits of unit testing, as a short list.",
    "Translate 'good morning' into French and Spanish.",
    "What is the capital of Japan? Answer in one sentence.",
    "Write a haiku about autumn.",
    "Why might `for i in range(len(a)): a.pop(i)` misbehave? Briefly.",
    "Convert 100 degrees Fahrenheit to Celsius. Show the formula.",
    "Name three sorting algorithms with their average time complexity.",
    "What does HTTP status code 404 mean? One sentence.",
    "Define recursion in one sentence.",
    "What is 2 to the power of 10?",
]

def call(args, idx):
    body = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPTS[idx % len(PROMPTS)]}],
        "max_tokens": args.max_tokens, "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            d = json.load(r)
        ch = d["choices"][0]; msg = ch.get("message", {})
        txt = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        return {"i": idx, "finish": ch.get("finish_reason"), "text": txt, "err": None}
    except Exception as e:
        return {"i": idx, "finish": None, "text": "", "err": str(e)[:120]}

def scan(text):
    toks = text.split()
    n = len(toks)
    if n < 8:
        return {"n_tok": n, "rep4": 0.0, "max_consec": 0, "uniq": 1.0, "empty_think": 0, "degen": False}
    grams = [tuple(toks[i:i+4]) for i in range(n-3)]
    rep4 = 1 - (len(set(grams)) / len(grams)) if grams else 0.0
    # longest run of an immediately-repeating unit (len 1..3)
    max_consec = 0
    for L in (1, 2, 3):
        i = 0
        while i + L <= n:
            unit = toks[i:i+L]; run = 1; j = i + L
            while j + L <= n and toks[j:j+L] == unit:
                run += 1; j += L
            if run > max_consec:
                max_consec = run
            i += L if run == 1 else run * L
    uniq = len(set(toks)) / n
    empty_think = len(re.findall(r"<think>\s*</think>", text)) + len(re.findall(r"</think>\s*</think>", text))
    degen = (rep4 > 0.5) or (max_consec > 20) or (n >= 60 and uniq < 0.20) or (empty_think >= 5)
    return {"n_tok": n, "rep4": round(rep4, 3), "max_consec": max_consec,
            "uniq": round(uniq, 3), "empty_think": empty_think, "degen": degen}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=240)
    args = ap.parse_args()
    idxs = list(range(args.n))
    with ThreadPoolExecutor(max_workers=args.n) as ex:
        results = list(ex.map(lambda k: call(args, k), idxs))
    rows, degen, errs, worst = [], 0, 0, None
    for r in results:
        if r["err"]:
            errs += 1; continue
        s = scan(r["text"]); rows.append(s)
        if s["degen"]:
            degen += 1
            if worst is None or s["rep4"] > worst.get("rep4", 0):
                worst = {**s, "snippet": r["text"][:160].replace("\n", " ")}
    ok = len(rows)
    out = {
        "n": args.n, "ok": ok, "errors": errs, "degen": degen,
        "pct_any": round(degen / ok, 3) if ok else None,
        "max_consec": max([r["max_consec"] for r in rows], default=0),
        "mean_rep4": round(sum(r["rep4"] for r in rows) / ok, 3) if ok else None,
        "worst": worst,
    }
    print(json.dumps(out))

if __name__ == "__main__":
    main()
