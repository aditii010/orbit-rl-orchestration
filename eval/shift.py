"""Held-out test: policies trained/tuned on the DEFAULT world are tested in shifted worlds.
Also a 'retuned rule' per world (an optimistic reference: a human who knew the shift in advance)."""
import itertools
import os

import numpy as np

from baselines.policies import make_rule_router, single_agent
from eval.compare2 import DEV as _  # noqa: F401  (keeps seed ranges documented in one place)
from eval.runner import evaluate
from eval.tune_dqn import NEW_RULE
from rl.train import train

TEST = range(100_000, 105_000)
RETUNE_DEV = range(0, 2000)
CONDITIONS = {
    "train world":      {},
    "tight budget":     {"budget_range": (0.3, 0.6)},
    "loose budget":     {"budget_range": (1.2, 2.0)},
    "research-heavy":   {"type_probs": (0.1, 0.2, 0.7)},
    "harder tasks":     {"difficulty_beta": (3.0, 2.0)},
    "easier tasks":     {"difficulty_beta": (2.0, 3.0)},
}


def retune(config):
    grid = itertools.product([(), (2,)], [True, False], [0, 1, 2, 3], [None, 0.6, 0.7], [0.1],
                             [True, False], [None, 0.5, 0.6, 0.7, 0.8], [False, True])
    best, best_cfg = -9, None
    for ro, v, mr, sk, mm, sm, gd, lr in grid:
        cfg = dict(research_on=ro, verify=v, max_retries=mr, skip_verify_conf=sk,
                   min_money_for_retry=mm, summarize=sm, giveup_diff=gd, lowconf_retry=lr)
        r = evaluate(make_rule_router(**cfg), RETUNE_DEV, config)["return"]
        if r > best:
            best, best_cfg = r, cfg
    return best_cfg


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    agents = []
    for seed in range(5):                                   # trained ONCE, on the default world
        agent, _h = train(seed=seed, eval_every=10**9, save=f"checkpoints/s2_seed{seed}.pt")
        agents.append(agent)
        print(f"trained seed {seed}", flush=True)
    rule = make_rule_router(**NEW_RULE)                     # tuned ONCE, on the default world
    print(f"\n{'condition':16s} {'single':>8s} {'rule(fixed)':>12s} {'rule(retuned)':>14s} "
          f"{'DQN mean':>9s} {'DQN min':>8s} {'DQN-rule':>16s}", flush=True)
    for name, cfg in CONDITIONS.items():
        single = evaluate(single_agent, TEST, cfg)["return"]
        r = evaluate(rule, TEST, cfg)
        retuned = evaluate(make_rule_router(**retune(cfg)), TEST, cfg)["return"]
        d = [evaluate(a.policy(), TEST, cfg) for a in agents]
        means = np.array([x["return"] for x in d])
        diff = np.mean([x["returns_raw"] for x in d], axis=0) - r["returns_raw"]   # seed-avg DQN vs rule
        print(f"{name:16s} {single:+8.3f} {r['return']:+12.3f} {retuned:+14.3f} {means.mean():+9.3f} "
              f"{means.min():+8.3f} {diff.mean():+8.4f}±{1.96*diff.std(ddof=1)/np.sqrt(len(diff)):.4f}",
              flush=True)
