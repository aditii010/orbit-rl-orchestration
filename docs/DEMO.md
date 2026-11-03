# ORBIT live demo: runbook

## 0. Five minutes before
1. Open Terminal, go to the project folder, run: `./run_demo.sh`
   It starts Ollama if needed, checks the model is there, warms it up (the first call is slow) and opens the dashboard at http://localhost:8501.
2. Click each of the five **Quick demo** buttons once. This confirms everything works on this laptop and warms the model.
3. Close other heavy apps. Plug in the charger. Turn off notifications.
4. Safety net: if anything misbehaves, set box 03 to **Replay (recorded, offline)**. It needs no model and no internet.
   (Fonts load from Google. Offline, the page falls back to plain serif and sans-serif and still works.)

## 1. The demo (about 6 minutes)
Use the **Quick demo** buttons. One click sets the task, the controller, the budget and runs it.

| # | Button | What you say | What the audience sees |
|---|--------|--------------|------------------------|
| 0 | (hero) | "Several AI helpers can work on a task. Calling all of them is slow and expensive; calling too few gives wrong answers. A reinforcement-learning controller decides which helper to call next, and when to stop." | The title, the three-step flow. |
| 1 | **Coding: solve, check, stop** | "A coding task. The controller asks the Solver, then runs a cheap test check, sees it passes and stops." | Solver, Verifier (PASS), STOP. Result Correct, cost about 0.11. Open "Why this action?" on a card to show the scores. |
| 2 | **Research: simulator-only DQN fails** | "This controller was trained only in a simulator. It has never met a real model. On a research question it never asks the Researcher." | Solver, STOP. The answer is "unknown". Result Wrong, cost 0.063. |
| 3 | **Research: retrained DQN succeeds** | "Same question, same task, but the controller was retrained on recorded real runs. Now it calls the Researcher first." | Researcher, Solver and more. Answer: 1991. Result Correct. Be honest: "it is not always tidy, it makes a few extra calls." Cost about 0.25. |
| 4 | **Failed check, then retry** (Replay) | "Here the first answer fails the test check. The controller sees that and retries, instead of stopping." | Solver, Verifier (FAIL), Solver (retry), Verifier (PASS), STOP. Result Correct, cost about 0.29. The single agent gets this task wrong. |
| 5 | **Tight budget** (Replay) | "Now it only has 0.15 to spend. Same question, but it drops the wasteful calls." | About 3 steps, Correct, cost about 0.12 instead of about 0.25 at budget 0.8. |
| 6 | Press **COMPARE ALL** (any task) | "All five controllers on the same task: who is correct, and at what cost." | Table plus the cost bars. |
| 7 | Scroll to **05 What we measured** | "In the simulator the learned controller beats the best hand-written rule by about 9 percent, and by about 16 percent when it cannot see difficulty or confidence. On real models, calling the right helper matters, and the limits are on the last tab." | The charts and the Limitations tab. |

## 2. What to expect (so nothing surprises you)
- Live qwen runs take 2 to 25 seconds. Coding is fast, maths problems are slower.
- Live runs can differ a little from the recorded ones because the model samples its answers.
- The Quick demo buttons 4 and 5 use Replay on purpose, because they depend on one specific recorded draw.
- "Money" is a cost unit built from token counts. One helper call costs about 0.02 to 0.15.

## 3. Questions you may be asked, and honest answers
- **Where is the reinforcement learning?** The controller is a Double DQN. State: task type, whether it has an answer, the last check result, how many calls so far, money and steps left. Actions: researcher, solver, verifier, summarizer, retry, stop. Reward: answer quality minus cost. It learns by trial and error.
- **Why not just write rules?** In the simulator the learned controller beat the best rule we could write (+9%). On real models a tuned rule is about as good, so the value of learning is that it finds the rule automatically and handles the cost trade-off.
- **Did RL win on the real models?** Mixed. Retrained on real runs, it ties the best hand rule on qwen and is slightly behind on gpt-oss. Both are in the Limitations tab. The tests were written before running them.
- **What is the "simulator only" controller for?** It shows the sim-to-real gap: skills learned in a simulator need retraining on real runs. That is why the retrained one exists.
- **What is a Q-value / "score"?** The controller's estimate of how good each next step is, counting the cost still to come. It picks the highest one.
- **Can it handle any question?** No. The research helper only knows 16 invented documents, and coding tasks come from a fixed list with tests.
- **Is the number small sample size a problem?** Yes. 24 held-out real tasks, so we report error bars and call it a pilot.

## 4. If something goes wrong
| Problem | Fix |
|---------|-----|
| Red box: "Could not reach the model" | Open the Ollama app, or choose Replay in box 03. |
| "The Groq free limit was hit" | Choose Ollama or Replay. |
| Page does not open | Run `./run_demo.sh` again. Log: `docs/logs/streamlit.log`. |
| Port busy | `pkill -f "streamlit run"` then `./run_demo.sh`. |
| Slow first run | Normal. The launcher warms the model; click a Quick demo button once before the audience arrives. |
| Stop everything | `pkill -f "streamlit run"` |

## 5. Developer notes
- Streamlit caches imported files, so after editing `app/*.py` restart with `./run_demo.sh`.
- Tests: `PYTHONPATH=. python3 -m pytest -q tests` (77 tests).
- Presets are defined in `app/dashboard.py` (PRESETS); the results numbers are in `app/results_data.py`, each checked against a log.
