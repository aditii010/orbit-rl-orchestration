"""Learning curves for the report: train the final DQN setting (S2, hide nothing) for 5 seeds, evaluating on the 1,000 DEV tasks every 2,000 episodes.
Output: report/learning_curve.json = {seed: [[episode, dev_return, dev_success, dev_cost], ...]}. Descriptive figure only, no hypothesis attached."""
import json
import sys

from rl.train import train

seed = int(sys.argv[1])
_, hist = train(seed=seed, eval_every=2000)
json.dump(hist, open(f"report/learning_curve_seed{seed}.json", "w"))
print("done", seed)
