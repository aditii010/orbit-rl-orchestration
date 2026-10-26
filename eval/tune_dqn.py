"""Plan A: three pre-declared DQN settings x 5 seeds. Select by DEV mean only, then report TEST."""
import numpy as np

from baselines.policies import make_rule_router
from eval.runner import evaluate
from rl.train import train

DEV = range(0, 5000)
TEST = range(100_000, 110_000)
# best difficulty-aware rule from eval/compare2.py
NEW_RULE = dict(research_on=(), verify=True, max_retries=3, skip_verify_conf=0.6,
                min_money_for_retry=0.1, summarize=True, giveup_diff=0.7, lowconf_retry=False)

SETTINGS = {
    "S1 longer":            dict(episodes=60_000, lr=1e-3, target_every=500, eps_frac=0.6),
    "S2 longer+slower":     dict(episodes=60_000, lr=3e-4, target_every=1000, eps_frac=0.6),
    "S3 +more exploration": dict(episodes=60_000, lr=3e-4, target_every=1000, eps_frac=0.8),
}

if __name__ == "__main__":
    rule = make_rule_router(**NEW_RULE)
    rule_dev, rule_test = evaluate(rule, DEV), evaluate(rule, TEST)
    print(f"NEW rule: dev {rule_dev['return']:+.3f}  test {rule_test['return']:+.3f}", flush=True)
    results = {}
    for name, kw in SETTINGS.items():
        dev, test = [], []
        for seed in range(5):
            agent, _ = train(seed=seed, eval_every=10**9, **kw)
            dev.append(evaluate(agent.policy(), DEV)["return"])
            test.append(evaluate(agent.policy(), TEST))
            print(f"  {name} seed {seed}: dev {dev[-1]:+.3f} test {test[-1]['return']:+.3f}", flush=True)
        results[name] = (dev, test)
        print(f"{name}: dev mean {np.mean(dev):+.3f} (min {min(dev):+.3f})", flush=True)
    chosen = max(results, key=lambda n: np.mean(results[n][0]))      # selection uses DEV only
    print(f"\nCHOSEN BY DEV: {chosen}")
    dev, test = results[chosen]
    raws = np.array([t["returns_raw"] for t in test])
    for i, d in enumerate(raws):
        diff = d - rule_test["returns_raw"]
        print(f"seed {i}: test {d.mean():+.3f}  DQN - rule = {diff.mean():+.4f} ± {1.96*diff.std(ddof=1)/np.sqrt(len(diff)):.4f}")
    print(f"TEST mean {raws.mean():+.3f} vs rule {rule_test['return']:+.3f}  "
          f"(gain {100*(raws.mean()/rule_test['return']-1):+.1f}%)")
