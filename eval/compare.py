"""Train DQN on several seeds, test each (final checkpoint) against baselines on the same TEST tasks."""
import numpy as np

from baselines.policies import fixed_pipeline, make_rule_router, single_agent
from eval.runner import evaluate, table
from rl.train import train

TEST = range(100_000, 110_000)
# best rule found on DEV by eval/run_baselines.py
RULE = dict(research_on=(), verify=True, max_retries=3, skip_verify_conf=0.6, min_money_for_retry=0.1)

if __name__ == "__main__":
    base = {"single agent": evaluate(single_agent, TEST),
            "fixed pipeline": evaluate(fixed_pipeline, TEST),
            "rule router (tuned)": evaluate(make_rule_router(**RULE), TEST)}
    rule_raw = base["rule router (tuned)"]["returns_raw"]
    single_raw = base["single agent"]["returns_raw"]
    dqn = {}
    for seed in range(5):
        agent, _ = train(episodes=20000, seed=seed, eval_every=10**9)
        dqn[f"DQN seed {seed}"] = evaluate(agent.policy(), TEST)
    table({**base, **dqn})
    raws = np.array([r["returns_raw"] for r in dqn.values()])
    means = raws.mean(1)
    print(f"\nDQN mean over seeds: {means.mean():+.3f}  (seed spread {means.min():+.3f} to {means.max():+.3f})")
    print(f"Rule router:         {rule_raw.mean():+.3f}")
    for i, d in enumerate(raws):
        diff = d - rule_raw                      # paired: same task, same seed
        print(f"seed {i}: DQN - rule = {diff.mean():+.3f} ± {1.96*diff.std(ddof=1)/np.sqrt(len(diff)):.3f}")
    print(f"relative gain of DQN mean over rule: {100*(means.mean()/rule_raw.mean()-1):+.1f}%  (GO needs >= +10%)")
    print("\nPaired vs single agent (same tasks):")
    for i, d in enumerate(raws):
        diff = d - single_raw
        print(f"seed {i}: DQN - single = {diff.mean():+.3f} ± {1.96*diff.std(ddof=1)/np.sqrt(len(diff)):.3f}")
    best_simple = max(rule_raw.mean(), single_raw.mean())
    print(f"\nbest simple baseline = {best_simple:+.3f}; DQN mean gain = {100*(means.mean()/best_simple-1):+.1f}% (GO needs >= +10%)")
