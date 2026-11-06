from collections import Counter, defaultdict

import numpy as np

from baselines.policies import MAX_STEPS, RETRY, STOP, make_rule_router
from env.orbit_env import ACTIONS, OrbitEnv
from eval.analyze import SHORT
from eval.compare import RULE
from eval.runner import evaluate
from rl.train import train

TEST = range(100_000, 110_000)


def low_conf_unchecked(o):
    return o[4] > 0.5 and o[5] > 0.5 and o[14] < 0.5


def part_a(seed):
    agent, _ = train(episodes=20000, seed=seed, eval_every=10**9)
    pol, env, rows = agent.policy(), OrbitEnv(), defaultdict(Counter)
    for s in range(100_000, 102_000):
        o, done = env.reset(seed=s), False
        while not done:
            a = pol(o, env.action_mask())
            if low_conf_unchecked(o):
                diff = "easy" if o[3] < 0.35 else "mid" if o[3] < 0.65 else "hard"
                money = "money<50%" if o[12] < 0.5 else "money>=50%"
                rows[(diff, money)][SHORT[ACTIONS[a]]] += 1
            o, _, done, _ = env.step(a)
    print(f"\n--- Part A: DQN seed {seed}, low-confidence unchecked candidates ---")
    for k in sorted(rows):
        n = sum(rows[k].values())
        print(f"difficulty {k[0]:5s} {k[1]:11s} (n={n:4d})  "
              + ", ".join(f"{a} {100*v/n:.0f}%" for a, v in rows[k].most_common()))


def with_low_conf_action(base, action_name):
    def pol(o, mask):
        if low_conf_unchecked(o) and action_name != "verify":
            n_solve = round(o[9] * MAX_STEPS)
            can_retry = (n_solve - 1) < RULE["max_retries"] and o[12] > RULE["min_money_for_retry"]
            if action_name == "retry" and can_retry and mask[RETRY]:
                return RETRY
            return STOP
        return base(o, mask)
    return pol


def part_b():
    base = make_rule_router(**RULE)
    res = {n: evaluate(with_low_conf_action(base, n), TEST) for n in ("verify", "retry", "stop")}
    print("\n--- Part B: hand rule, only the low-confidence action changes (10,000 paired tasks) ---")
    ref = res["verify"]["returns_raw"]
    for n, r in res.items():
        d = r["returns_raw"] - ref
        print(f"low-conf -> {n:6s} return={r['return']:+.3f} success={r['success']:.2f} "
              f"cost={r['cost']:.3f}   vs verify: {d.mean():+.4f} ± {1.96*d.std(ddof=1)/np.sqrt(len(d)):.4f}")


def part_c():
    env, wrong, total = OrbitEnv(), 0, 0
    pol = make_rule_router(**RULE)
    for s in range(100_000, 110_000):
        o, done = env.reset(seed=s), False
        while not done:
            if low_conf_unchecked(o):
                total += 1; wrong += (not env.candidate_correct)
            o, _, done, _ = env.step(pol(o, env.action_mask()))
    print(f"\n--- Part C: P(candidate truly wrong | unchecked, confidence<0.5) = {wrong/total:.2f}  (n={total}) ---")


if __name__ == "__main__":
    part_c()
    part_b()
    for seed in (0, 1):
        part_a(seed)
