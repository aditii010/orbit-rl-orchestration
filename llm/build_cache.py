import json
import os
import sys

from llm.agents import final_answer, researcher, solver, strip_confidence, verifier
from llm.client import LLM
from llm.tasks import all_tasks, grade

K = 4
K_GROQ = {"coding": {"0": 3, "1": 3}, "analytical": {"0": 3, "1": 2},
          "research": {"0": 2, "1": 3, "2": 3, "3": 2}}


def build_task(llm, t, ks=None):
    rec = {"id": t["id"], "type": t["type"], "notes": [], "levels": {}}
    levels = {}
    if t["type"] == "research":
        evidence, seen = "", []
        levels["0"] = ""
        for i in range(3):
            ev, new, u = researcher(llm, t, seen)
            rec["notes"].append({"text": ev, "docs": new, **u})
            evidence, seen = evidence + "\n" + ev, seen + new
            levels[str(i + 1)] = evidence
    else:
        ev, _, u = researcher(llm, t)
        rec["notes"].append({"text": ev, "docs": [], **u})
        levels = {"0": "", "1": ev}
    for lv, evidence in levels.items():
        rows = []
        for _ in range(ks[t["type"]][lv] if ks else K):
            s = solver(llm, t, evidence=evidence)
            passed, _, vu = verifier(llm, t, s["text"], evidence)
            rows.append({"text": s["text"], "conf": s["confidence"], "tin": s["tin"], "tout": s["tout"],
                         "sec": s["seconds"], "ok_raw": grade(t, strip_confidence(s["text"])),
                         "ok_final": grade(t, final_answer(t, s["text"])), "v_pass": passed,
                         "v_tin": vu["tin"], "v_tout": vu["tout"], "v_sec": vu["seconds"]})
        rec["levels"][lv] = rows
    return rec


if __name__ == "__main__":
    backend, model, out = (sys.argv[1:4] if len(sys.argv) > 3 else ("ollama", "qwen2.5:3b", "runs/cache_qwen3b.jsonl"))
    os.makedirs("runs", exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    llm = (LLM(backend, model, max_requests=650, min_tokens=1500, reasoning_effort="low")
           if backend == "groq" else LLM(backend, model))
    for t in all_tasks():
        if t["id"] in done:
            continue
        rec = build_task(llm, t, K_GROQ if backend == "groq" else None)
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(t["id"], "done", flush=True)
