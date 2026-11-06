import argparse
import os

import numpy as np
import torch

from env.orbit_env import OrbitEnv
from eval.runner import evaluate
from rl.dqn import DoubleDQN, ReplayBuffer

DEV = range(0, 1000)
TRAIN_SEED_BASE = 1_000_000


def train(episodes=60000, seed=0, eps_start=1.0, eps_end=0.05, eps_frac=0.6, buffer=50_000,
          batch=64, warmup=1000, target_every=1000, eval_every=2000, save=None, config=None,
          lr=3e-4, hide=(), env_fn=None):
    agent = DoubleDQN(seed=seed, lr=lr, hide=hide)
    buf = ReplayBuffer(buffer)
    env = env_fn() if env_fn else OrbitEnv(config)
    rng = np.random.default_rng(seed)
    step, history = 0, []
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (eps_frac * episodes))
        obs = env.reset(seed=TRAIN_SEED_BASE + seed * 10_000_000 + ep)
        done = False
        while not done:
            mask = env.action_mask()
            a = agent.act(obs, mask, eps)
            obs2, r, done, _ = env.step(a)
            m2 = np.ones(len(mask), bool) if done else env.action_mask()
            buf.add(obs * agent.keep, a, r, obs2 * agent.keep, float(done), m2)
            obs = obs2
            step += 1
            if buf.size >= warmup:
                agent.update(buf.sample(batch, rng))
            if step % target_every == 0:
                agent.sync_target()
        if (ep + 1) % eval_every == 0:
            r = evaluate(agent.policy(), DEV, config)
            history.append((ep + 1, r["return"], r["success"], r["cost"]))
            print(f"ep {ep+1:6d} eps={eps:.2f}  dev return={r['return']:+.3f} "
                  f"success={r['success']:.2f} cost={r['cost']:.3f}", flush=True)
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        torch.save(agent.q.state_dict(), save)
    return agent, history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=60000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="checkpoints/dqn_seed0.pt")
    a = p.parse_args()
    train(a.episodes, a.seed, save=a.save)
