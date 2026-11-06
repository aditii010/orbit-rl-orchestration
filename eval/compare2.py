import itertools

import numpy as np

from baselines.policies import make_rule_router, single_agent
from eval.compare import RULE
from eval.runner import evaluate, table
from rl.train import train

DEV = range(0, 5000)
TEST = range(100_000, 110_000)

if __name__ == "__main__":
    grid = itertools.product([(), (2,)], [True, False], [0, 1, 2, 3], [None, 0.6, 0.7], [0.1],
                             [True, False], [None, 0.5, 0.6, 0.7, 0.8], [False, True])
    scored = []
    for ro, v, mr, sk, mm, sm, gd, lr in grid:
        cfg = dict(research_on=ro, verify=v, max_retries=mr, skip_verify_conf=sk,
                   min_money_for_retry=mm, summarize=sm, giveup_diff=gd, lowconf_retry=lr)
        scored.append((evaluate(make_rule_router(**cfg), DEV)["return"], cfg))
    scored.sort(key=lambda x: -x[0])
    print(f"searched {len(scored)} rules on DEV. top 3:")
    for r, c in scored[:3]:
        print(f"  dev {r:+.3f}  {c}")
    best_cfg = scored[0][1]

    base = {"single agent": evaluate(single_agent, TEST),
            "old rule (no difficulty)": evaluate(make_rule_router(**RULE), TEST),
            "NEW rule (difficulty-aware)": evaluate(make_rule_router(**best_cfg), TEST)}
    new_raw = base["NEW rule (difficulty-aware)"]["returns_raw"]
    dqn = {}
    for seed in range(5):
        agent, _ = train(episodes=20000, seed=seed, eval_every=10**9)
        dqn[f"DQN seed {seed}"] = evaluate(agent.policy(), TEST)
    print()
    table({**base, **dqn})
    raws = np.array([r["returns_raw"] for r in dqn.values()])
    means = raws.mean(1)
    print(f"\nDQN mean {means.mean():+.3f} (spread {means.min():+.3f}..{means.max():+.3f}) "
          f"vs NEW rule {new_raw.mean():+.3f}")
    for i, d in enumerate(raws):
        diff = d - new_raw
        print(f"seed {i}: DQN - NEW rule = {diff.mean():+.4f} ± {1.96*diff.std(ddof=1)/np.sqrt(len(diff)):.4f}")
    print(f"relative gain over NEW rule: {100*(means.mean()/new_raw.mean()-1):+.1f}%  (GO needs >= +10%)")
