"""Real-LLM comparison on the replayed trace cache. Pre-declared in docs/decisions.md.
Dev tasks (26) are used to tune the rule and to refit the DQN; test tasks (24) are only used to report.
Unit of analysis = task: per-task mean return over 100 seeded episodes, paired across methods."""
import itertools
import sys

import numpy as np

from baselines.policies import fixed_pipeline, make_rule_router, single_agent
from eval.runner import evaluate
from llm.trace_env import TraceEnv, load_cache, split_tasks
from rl.dqn import DoubleDQN
from rl.train import train

HIDE = (3, 14)
CACHE = sys.argv[1] if len(sys.argv) > 1 else "runs/cache_qwen3b.jsonl"
EPS_PER_TASK = 100


def per_task(policy, recs, per=EPS_PER_TASK):
    r = evaluate(policy, range(len(recs) * per), env_fn=lambda: TraceEnv(recs))
    return r["returns_raw"].reshape(per, len(recs)).mean(0), r      # seed % n = task, so column j = task j


def diff_ci(a, b):
    d = a - b
    return d.mean(), 2.07 * d.std(ddof=1) / np.sqrt(len(d))


if __name__ == "__main__":
    recs = load_cache(CACHE)
    dev, test = split_tasks(recs)
    types = np.array([r["type"] for r in test])
    print(f"{len(dev)} dev tasks, {len(test)} test tasks")
    grid = [dict(research_on=r, verify=v, max_retries=m, min_money_for_retry=mm, summarize=s)
            for r, v, m, mm, s in itertools.product([(), (2,), (1, 2), (0, 1, 2)], [True, False], [0, 1, 2, 3],
                                                    [.05, .1, .15, .25], [True, False])]
    dev_scores = [per_task(make_rule_router(**g), dev, 20)[0].mean() for g in grid]
    best = grid[int(np.argmax(dev_scores))]
    print(f"TUNED RULE (chosen on dev, dev return {max(dev_scores):+.3f}): {best}", flush=True)

    zs = [DoubleDQN(seed=s, hide=HIDE).load(f"checkpoints/blind_seed{s}.pt") for s in range(5)]
    refit = [train(episodes=20_000, seed=s, eval_every=10**9, hide=HIDE, env_fn=lambda: TraceEnv(dev))[0]
             for s in range(5)]
    print("refit done", flush=True)

    methods = {"single agent": [single_agent], "fixed pipeline": [fixed_pipeline],
               "tuned rule (dev)": [make_rule_router(**best)],
               "DQN zero-shot (sim-trained)": [a.policy() for a in zs],
               "DQN refit on real dev traces": [a.policy() for a in refit]}
    res = {}
    for name, pols in methods.items():
        runs = [per_task(p, test) for p in pols]
        res[name] = dict(task=np.mean([r[0] for r in runs], 0), seeds=[r[0].mean() for r in runs],
                         stats=[r[1] for r in runs])
    print("\n=== TEST tasks (24): mean return, success, cost; DQN rows average 5 seeds ===")
    for name, r in res.items():
        succ = np.mean([s["success"] for s in r["stats"]])
        cost = np.mean([s["cost"] for s in r["stats"]])
        by = {t: r["task"][types == t].mean() for t in ("coding", "analytical", "research")}
        print(f"{name:32s} return {r['task'].mean():+.3f} success {succ:.2f} cost {cost:.3f} | "
              f"coding {by['coding']:+.2f} analytical {by['analytical']:+.2f} research {by['research']:+.2f}")
        if len(r["seeds"]) > 1:
            print(" " * 32 + "per-seed: " + " ".join(f"{x:+.3f}" for x in r["seeds"]))
    print("\n=== paired differences (per-task, 95% CI over tasks) ===")
    for a, b in [("fixed pipeline", "single agent"), ("tuned rule (dev)", "single agent"),
                 ("tuned rule (dev)", "fixed pipeline"), ("DQN zero-shot (sim-trained)", "single agent"),
                 ("DQN zero-shot (sim-trained)", "tuned rule (dev)"),
                 ("DQN refit on real dev traces", "tuned rule (dev)"),
                 ("DQN refit on real dev traces", "single agent")]:
        m, ci = diff_ci(res[a]["task"], res[b]["task"])
        print(f"{a:32s} - {b:18s} {m:+.4f} ± {ci:.4f}")
    print("\n=== average calls per episode by task type (test) ===")
    for name in ("tuned rule (dev)", "DQN zero-shot (sim-trained)", "DQN refit on real dev traces"):
        for t in ("coding", "analytical", "research"):
            sub = [r for r in test if r["type"] == t]
            outs = [evaluate(p, range(len(sub) * 100), env_fn=lambda: TraceEnv(sub))
                    for p in methods[name]]
            c = {k: np.mean([o[f"calls_{k}"] for o in outs]) for k in ("researcher", "solver", "verifier", "summarizer")}
            print(f"{name:32s} {t:10s} " + " ".join(f"{k[:4]} {v:.2f}" for k, v in c.items())
                  + f" | success {np.mean([o['success'] for o in outs]):.2f}")
    print("\n=== ALL 50 tasks, no tuning needed (single, fixed, zero-shot DQN) ===")
    for name in ("single agent", "fixed pipeline", "DQN zero-shot (sim-trained)"):
        rs = [per_task(p, recs)[0].mean() for p in methods[name]]
        print(f"{name:32s} return {np.mean(rs):+.3f}")
