import copy

import numpy as np
import torch
import torch.nn as nn

from env.orbit_env import ACTIONS, OBS_DIM

N_ACTIONS = len(ACTIONS)
NEG = -1e9


class QNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim=OBS_DIM, n_actions=N_ACTIONS):
        self.cap, self.i, self.size = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, obs_dim), np.float32)
        self.d = np.zeros(capacity, np.float32)
        self.m2 = np.ones((capacity, n_actions), bool)

    def add(self, s, a, r, s2, d, m2):
        i = self.i
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i], self.m2[i] = s, a, r, s2, d, m2
        self.i = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n, rng):
        idx = rng.integers(0, self.size, n)
        t = torch.as_tensor
        return (t(self.s[idx]), t(self.a[idx]), t(self.r[idx]), t(self.s2[idx]),
                t(self.d[idx]), t(self.m2[idx]))


class DoubleDQN:
    def __init__(self, gamma=0.99, lr=1e-3, hidden=128, seed=0, hide=()):
        torch.manual_seed(seed)
        self.keep = np.ones(OBS_DIM, np.float32)
        self.keep[list(hide)] = 0.0
        self.rng = np.random.default_rng(seed)
        self.q = QNet(hidden=hidden)
        self.target = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma

    def act(self, obs, mask, eps=0.0):
        obs = obs * self.keep
        legal = np.flatnonzero(mask)
        if self.rng.random() < eps:
            return int(self.rng.choice(legal))
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs).unsqueeze(0))[0].numpy()
        q = np.where(mask, q, NEG)
        return int(q.argmax())

    def update(self, batch):
        s, a, r, s2, d, m2 = batch
        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next_live = self.q(s2).masked_fill(~m2, NEG)
            best = q_next_live.argmax(1, keepdim=True)
            q_next = self.target(s2).gather(1, best).squeeze(1)
            y = r + self.gamma * (1.0 - d) * q_next
        loss = nn.functional.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return float(loss.detach())

    def load(self, path):
        self.q.load_state_dict(torch.load(path))
        self.sync_target()
        return self

    def sync_target(self):
        self.target.load_state_dict(self.q.state_dict())

    def policy(self):
        return lambda obs, mask: self.act(obs, mask, eps=0.0)
