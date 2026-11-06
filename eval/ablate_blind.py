import itertools

import numpy as np

from baselines.policies import make_rule_router
from eval.runner import evaluate
from rl.train import train

DEV, TEST = range(0, 5000), range(100_000, 110_000)


def blind(policy):
    def p(obs, mask):
        o = obs.copy()
        o[3] = 0.0
        o[14] = 0.0
        return policy(o, mask)
    return p


if __name__ == "__main__":
    grid = [dict(research_on=r, verify=v, max_retries=m, min_money_for_retry=mm, summarize=s)
            for r, v, m, mm, s in itertools.product([(), (2,), (1, 2), (0, 1, 2)], [True, False], [0, 1, 2, 3],
                                                    [.05, .1, .15, .25], [True, False])]
    best = max(grid, key=lambda g: evaluate(blind(make_rule_router(**g)), DEV)["return"])
    rule = evaluate(blind(make_rule_router(**best)), TEST)
    print(f"BLIND RULE (chosen on DEV) {best}\n  test {rule['return']:+.4f}", flush=True)
    rets = []
    for s in range(5):
        agent, _ = train(seed=s, eval_every=10**9, hide=(3, 14), save=f"checkpoints/blind_seed{s}.pt")
        r = evaluate(agent.policy(), TEST)
        d = np.array(r["returns_raw"]) - np.array(rule["returns_raw"])
        rets.append(r["return"])
        print(f"seed {s}: test {r['return']:+.4f}  DQN-blind - blind rule = {d.mean():+.4f} ± "
              f"{1.96*d.std(ddof=1)/np.sqrt(len(d)):.4f}", flush=True)
    print(f"MEAN blind DQN {np.mean(rets):+.4f} vs blind rule {rule['return']:+.4f} "
          f"(gain {100*(np.mean(rets)/rule['return']-1):+.1f}%)  [full-info DQN +0.189, full-info rule +0.173]")
