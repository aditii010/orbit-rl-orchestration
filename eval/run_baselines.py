"""Tune the rule router on DEV seeds, then report all baselines on separate TEST seeds."""
import itertools

from baselines.policies import fixed_pipeline, make_rule_router, single_agent
from eval.runner import evaluate, table

DEV = range(0, 1000)
TEST = range(100_000, 102_000)

if __name__ == "__main__":
    best, best_cfg = -9, None
    grid = itertools.product([(), (2,), (0, 1, 2)], [True, False], [0, 1, 2, 3],
                             [None, 0.6, 0.7], [0.1, 0.25])
    for research_on, verify, retries, skip, minmoney in grid:
        cfg = dict(research_on=research_on, verify=verify, max_retries=retries,
                   skip_verify_conf=skip, min_money_for_retry=minmoney)
        r = evaluate(make_rule_router(**cfg), DEV)["return"]
        if r > best:
            best, best_cfg = r, cfg
    print("best rule on DEV:", best_cfg, f"dev return={best:.3f}\n")
    results = {
        "single agent": evaluate(single_agent, TEST),
        "fixed pipeline": evaluate(fixed_pipeline, TEST),
        "rule router (tuned)": evaluate(make_rule_router(**best_cfg), TEST),
    }
    table(results)
