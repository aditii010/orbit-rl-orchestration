from collections import Counter, defaultdict

from baselines.policies import make_rule_router
from env.orbit_env import ACTIONS, OrbitEnv
from eval.compare import RULE
from rl.train import train

SHORT = {"INVOKE_RESEARCHER": "research", "INVOKE_SOLVER": "solve", "INVOKE_VERIFIER": "verify",
         "INVOKE_SUMMARIZER": "summarize", "RETRY_SOLVER": "retry", "STOP": "stop"}
TYPES = ["coding", "analytical", "research"]


def context(o):
    if not o[4]:
        return f"1 no candidate / {TYPES[int(o[:3].argmax())]}"
    if o[6] > 0.5:
        return "4 verifier said PASS"
    if o[7] > 0.5:
        return "5 verifier said FAIL, money " + ("low (<25%)" if o[12] < 0.25 else "ok")
    c = o[14]
    return "2 unchecked, confidence " + ("low (<0.5)" if c < 0.5 else "mid" if c < 0.65 else "high (>=0.65)")


def profile(policy, seeds=range(100_000, 102_000)):
    env, table = OrbitEnv(), defaultdict(Counter)
    for s in seeds:
        o, done = env.reset(seed=s), False
        while not done:
            a = policy(o, env.action_mask())
            table[context(o)][SHORT[ACTIONS[a]]] += 1
            o, _, done, _ = env.step(a)
    return table


def show(name, table):
    print(f"\n=== {name} ===")
    for ctx in sorted(table):
        n = sum(table[ctx].values())
        parts = ", ".join(f"{k} {100*v/n:.0f}%" for k, v in table[ctx].most_common())
        print(f"{ctx:42s} (n={n:5d})  {parts}")


if __name__ == "__main__":
    show("Rule router", profile(make_rule_router(**RULE)))
    for seed in (0, 4):
        agent, _ = train(episodes=20000, seed=seed, eval_every=10**9)
        show(f"DQN seed {seed}", profile(agent.policy()))
