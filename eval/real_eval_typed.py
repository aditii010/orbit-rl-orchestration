"""Typed rule family vs refit DQN on a trace cache. Pre-declared in docs/decisions.md."""
import itertools
import sys

import numpy as np

from baselines.policies import make_typed_rule
from eval.real_eval import HIDE, diff_ci, per_task
from llm.trace_env import TraceEnv, load_cache, split_tasks
from rl.train import train

CACHE = sys.argv[1]


if __name__ == "__main__":
    recs = load_cache(CACHE)
    dev, test = split_tasks(recs)
    grid = [dict(res_calls=(a, b, c), verify=(v1, v2, v3), max_retries=m, summarize=s)
            for a, b, c, v1, v2, v3, m, s in itertools.product([0, 1], [0, 1], [0, 1, 2, 3], [True, False],
                                                               [True, False], [True, False], [0, 1, 2], [True, False])]
    scores = [per_task(make_typed_rule(**g), dev, 20)[0].mean() for g in grid]
    best = grid[int(np.argmax(scores))]
    print(f"TYPED RULE (dev {max(scores):+.3f}): {best}", flush=True)
    rule_task, rule_stats = per_task(make_typed_rule(**best), test)
    agents = [train(episodes=20_000, seed=s, eval_every=10**9, hide=HIDE, env_fn=lambda: TraceEnv(dev))[0]
              for s in range(5)]
    runs = [per_task(a.policy(), test) for a in agents]
    dqn_task = np.mean([r[0] for r in runs], 0)
    print(f"typed rule  test return {rule_task.mean():+.3f} success {rule_stats['success']:.2f} cost {rule_stats['cost']:.3f}")
    print(f"refit DQN   test return {dqn_task.mean():+.3f} success {np.mean([r[1]['success'] for r in runs]):.2f} "
          f"cost {np.mean([r[1]['cost'] for r in runs]):.3f}  per-seed " + " ".join(f"{r[0].mean():+.3f}" for r in runs))
    m, ci = diff_ci(dqn_task, rule_task)
    print(f"refit DQN - typed rule: {m:+.4f} ± {ci:.4f}")
