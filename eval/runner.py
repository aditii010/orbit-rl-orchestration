"""Benchmark runner: evaluate any policy on the same seeded tasks."""
import numpy as np

from env.orbit_env import OrbitEnv


def evaluate(policy, seeds, config=None, env_fn=None):
    env = env_fn() if env_fn else OrbitEnv(config)
    rets, succ, cost, steps = [], [], [], []
    calls = {"researcher": [], "solver": [], "verifier": [], "summarizer": []}
    for s in seeds:
        obs = env.reset(seed=int(s))
        total, done, info = 0.0, False, {}
        while not done:
            obs, r, done, info = env.step(policy(obs, env.action_mask()))
            total += r
        rets.append(total); succ.append(float(info["success"]))
        cost.append(info["total_cost"]); steps.append(info["steps"])
        for k in calls:
            calls[k].append(info["calls"][k])
    n = len(seeds)
    ret = np.array(rets)
    out = {
        "return": ret.mean(), "return_ci": 1.96 * ret.std(ddof=1) / np.sqrt(n),
        "success": np.mean(succ), "cost": np.mean(cost),
        "cost_per_success": np.sum(cost) / max(1.0, np.sum(succ)),
        "steps": np.mean(steps), "returns_raw": ret,
    }
    out.update({f"calls_{k}": np.mean(v) for k, v in calls.items()})
    return out


def table(results):
    print(f"{'method':32s} {'return':>14s} {'success':>8s} {'cost':>6s} {'cost/succ':>9s} {'steps':>6s}")
    for name, r in results.items():
        print(f"{name:32s} {r['return']:+7.3f}±{r['return_ci']:.3f} {r['success']:8.2f} "
              f"{r['cost']:6.3f} {r['cost_per_success']:9.3f} {r['steps']:6.2f}")
