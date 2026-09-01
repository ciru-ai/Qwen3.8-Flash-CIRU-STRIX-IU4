#!/usr/bin/env python3
"""Cold prefill/generation ladder vs the running CIRU server (WSL2/ROCDXG).
Protocol mirrors docs/BENCHMARKS.md: uncached exact-count prompts, 128 decode
tokens, MTP depth 3, one slot. Reports the server's own timings dict."""
import json, time, urllib.request

BASE = "http://127.0.0.1:8080/v1/chat/completions"
TARGETS = [512, 2048, 8192, 16384, 32768]
WORD = "persistent "  # ~1 token per word for Qwen tokenizer

def chat(prompt_words: int, n_gen: int = 128, timeout: int = 900):
    prompt = (WORD * prompt_words).strip()
    body = {
        "model": "Qwen3.8-Flash-CIRU-STRIX-IU4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "cache_prompt": False,       # keep measurement cold
        "max_tokens": n_gen,
    }
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    wall = time.time() - t0
    t = d.get("timings", {})
    u = d.get("usage", {})
    return {
        "wall_s": round(wall, 3),
        "prompt_n": u.get("prompt_tokens", t.get("prompt_n")),
        "prompt_tps": round(t.get("prompt_per_second", 0), 3),
        "prompt_ms": round(t.get("prompt_ms", 0), 1),
        "gen_tps": round(t.get("predicted_per_second", 0), 3),
        "gen_n": t.get("predicted_n", 0),
        "ttfp_s": round(t.get("prompt_ms", 0) / 1000, 3),
        "draft_accept": f"{t.get('draft_n_accepted',0)}/{t.get('draft_n',0)}"
                         f" ({100*t.get('draft_n_accepted',0)/max(t.get('draft_n',1),1):.1f}%)",
    }

if __name__ == "__main__":
    print(f"{'prompt':>7} {'prefill t/s':>10} {'gen t/s':>8} {'TTFP s':>7} "
          f"{'wall s':>7}  MTP accept")
    for n in TARGETS:
        try:
            r = chat(n)
            print(f"{r['prompt_n']:>7} {r['prompt_tps']:>10.1f} {r['gen_tps']:>8.1f} "
                  f"{r['ttfp_s']:>7.2f} {r['wall_s']:>7.1f}  {r['draft_accept']}")
        except Exception as e:
            print(f"{n:>7} ERROR: {e}")
        time.sleep(2)
