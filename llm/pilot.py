import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from llm.agents import final_answer, researcher, solver, verifier
from llm.client import LLM
from llm.tasks import all_tasks, grade

PICK = ["code00", "code04", "code08", "code11", "code16", "ana00", "ana02", "ana08", "ana12",
        "res00", "res04", "res08"]

if __name__ == "__main__":
    backend, model = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("ollama", "qwen2.5:3b")
    os.makedirs("runs", exist_ok=True)
    llm = LLM(backend, model, max_requests=300 if backend == "groq" else None)
    tasks = {t["id"]: t for t in all_tasks()}
    rows = []
    for tid in PICK:
        t = tasks[tid]
        evidence, seen = "", []
        if t["type"] == "research":
            for _ in range(2):
                ev, new, u = researcher(llm, t, seen)
                evidence, seen = evidence + "\n" + ev, seen + new
        for k in range(3):
            s = solver(llm, t, evidence=evidence if t["type"] == "research" else "")
            ok = grade(t, final_answer(t, s["text"]))
            v_pass, _, vu = verifier(llm, t, s["text"], evidence)
            rows.append(dict(id=tid, type=t["type"], ok=ok, conf=s["confidence"], v_pass=v_pass,
                             tin=s["tin"], tout=s["tout"], sec=s["seconds"]))
            print(f"{tid} sample {k}: correct={ok} conf={s['confidence']} verifier_pass={v_pass} "
                  f"({s['tout']} tok, {s['seconds']:.1f}s)", flush=True)
    json.dump(rows, open("runs/pilot.json", "w"))
    print("\n=== summary ===")
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    for ty, rs in by.items():
        v_right = np.mean([r["v_pass"] == r["ok"] for r in rs])
        print(f"{ty:10s} solver correct {np.mean([r['ok'] for r in rs]):.2f} | verifier agrees with truth {v_right:.2f} "
              f"| mean out tokens {np.mean([r['tout'] for r in rs]):.0f} | mean sec {np.mean([r['sec'] for r in rs]):.1f}")
    c = [r for r in rows if r["conf"] is not None]
    good, bad = [r["conf"] for r in c if r["ok"]], [r["conf"] for r in c if not r["ok"]]
    print(f"confidence reported in {len(c)}/{len(rows)} samples; mean when correct "
          f"{np.mean(good) if good else float('nan'):.2f}, when wrong {np.mean(bad) if bad else float('nan'):.2f}")
