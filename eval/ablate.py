import numpy as np

from eval.runner import evaluate
from rl.dqn import DoubleDQN
from rl.train import train

TEST = range(100_000, 110_000)
SHIFT = {"type_probs": (0.1, 0.2, 0.7)}
VARIANTS = {
    "A no cost penalty (train lambda=0)": dict(config={"lambda_cost": 0.0}),
    "B no history (verify result, call counts, summarized)": dict(hide=(5, 6, 7, 8, 9, 10, 11, 15)),
    "C1 no difficulty estimate": dict(hide=(3,)),
    "C2 no confidence score": dict(hide=(14,)),
}

if __name__ == "__main__":
    full = [DoubleDQN(seed=s).load(f"checkpoints/s2_seed{s}.pt") for s in range(5)]
    ref = {}
    for w, cfg in (("train world", None), ("research-heavy", SHIFT)):
        ref[w] = [evaluate(a.policy(), TEST, cfg) for a in full]
        print(f"FULL DQN [{w}]: mean {np.mean([r['return'] for r in ref[w]]):+.3f} "
              f"(min {min(r['return'] for r in ref[w]):+.3f})", flush=True)
    for name, kw in VARIANTS.items():
        agents = [train(seed=s, eval_every=10**9, **kw)[0] for s in range(5)]
        for w, cfg in (("train world", None), ("research-heavy", SHIFT)):
            res = [evaluate(a.policy(), TEST, cfg) for a in agents]
            m = np.array([r["return"] for r in res])
            d = np.mean([r["returns_raw"] for r in res], 0) - np.mean([r["returns_raw"] for r in ref[w]], 0)
            print(f"{name:55s} [{w:14s}] return {m.mean():+.3f} (min {m.min():+.3f}) "
                  f"success {np.mean([r['success'] for r in res]):.2f} cost {np.mean([r['cost'] for r in res]):.3f} "
                  f"calls {np.mean([r['steps'] for r in res]):.2f} | vs full: {d.mean():+.4f} ± "
                  f"{1.96*d.std(ddof=1)/np.sqrt(len(d)):.4f}", flush=True)
