import json
import sys

from rl.train import train

seed = int(sys.argv[1])
_, hist = train(seed=seed, eval_every=2000)
json.dump(hist, open(f"report/learning_curve_seed{seed}.json", "w"))
print("done", seed)
