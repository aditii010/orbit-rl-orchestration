# ORBIT: learning when to collaborate

A reinforcement learning controller (Double DQN) that decides which AI agent to call next,
or when to stop, to get good answers at low cost. Lab project for UCS760.

Status: Milestone 0 (repo skeleton). See `docs/decisions.md` for every choice we made.

## Layout
- `env/` simulator (tasks, agents, budget)
- `agents/` researcher, solver, verifier, summarizer
- `baselines/` single-agent, fixed pipeline, rule-based, LLM router
- `rl/` bandit, Double DQN, replay buffer
- `eval/` benchmark runner, metrics, plots
- `experiments/` one-off feasibility tests
- `tests/` reward and environment tests

Never commit API keys. Keep them in `.env` (git-ignored).
