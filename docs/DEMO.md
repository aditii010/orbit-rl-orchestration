# ORBIT live demo: runbook

## 0. Five minutes before
1. Open Terminal, go to the project folder, run: `./run_demo.sh`
   It starts Ollama if needed, checks the model is there, warms it up (the first call is slow) and opens the dashboard at http://localhost:8501.
2. Click each of the five **Quick demo** buttons once. This confirms everything works on this laptop and warms the model.
3. Close other heavy apps. Plug in the charger. Turn off notifications.
4. Safety net: if anything misbehaves, set box 03 to **Replay (recorded, offline)**. It needs no model and no internet.
   (Fonts load from Google. Offline, the page falls back to plain serif and sans-serif and still works.)

## 1. The demo (about 6 minutes)
The page is a **race**: a baseline in the left corner, the learned DQN in the right corner, the same task, both shown live and side by side.
Use the **Quick demo** buttons. One click sets the task, the two corners, the budget and starts the race. The page scrolls down to the race by itself.

| # | Button | What you say | What the audience sees |
|---|--------|--------------|------------------------|
| 0 | (hero) | "Several AI helpers can work on a task. Calling all of them is slow and expensive; calling too few gives wrong answers. A reinforcement-learning controller decides which helper to call next, and when to stop. Here it races a baseline." | Title and the three-step flow. |
| 1 | **Race: a coding task** (live) | "Left: the fixed pipeline, which always uses every helper. Right: our DQN." | Left takes 5 steps (researcher, solver, verifier, summarizer, stop), cost about 0.30. Right takes 3 (solver, verifier, stop), cost about 0.11. Both correct. The right corner gets the WINNER tag. |
| 2 | **Race: why we retrain** (live) | "Left: a DQN trained only in a simulator. Right: the same DQN retrained on real recorded runs. Same question." | Left never calls the Researcher and answers "unknown" (wrong). Right looks things up first and answers 1991 (correct). Be honest: "the right one makes a few extra calls; it is not always tidy." |
| 3 | **Race: failed check, retry** (replay) | "Left: a single agent. Right: our DQN. Watch what happens when the first answer is wrong." | Right: solver, verifier FAIL, solver (retry), verifier PASS, stop. Correct. Left stops after one try and is wrong. |
| 4 | **Race: tight budget** (replay) | "Only 0.15 to spend. Which one handles that better?" | Right finishes in about 3 steps for about 0.12, left uses about 0.21. |
| 5 | **Scoreboard: 24 unseen tasks** | "One task proves little, so here are 24 tasks the DQN never trained on, each raced 10 times." | A live tally counts up: the DQN wins 16, loses 8, average advantage +0.196. Bars for score and cost. |
| 6 | Change the **left corner** to *Tuned rule* and press **SCOREBOARD** | (only if asked) "Against a hand-tuned rule it does not win." | The page honestly reports the rule ahead, or too close to call. Do not hide this; it is in the Limitations tab. |
| 7 | Scroll to **05 What we measured** | "In the simulator the DQN beats the best hand-written rule by about 9 percent, and by about 16 percent when it cannot see difficulty or confidence. On real models, calling the right helper matters; the limits are on the last tab." | Charts and the Limitations tab. |

Note the winner is decided by the real score (answer quality minus cost penalty, the same reward the DQN was trained on). The page never declares the DQN the winner unless the numbers say so.

## 2. What to expect (so nothing surprises you)
- Live qwen runs take 2 to 25 seconds. Coding is fast, maths problems are slower.
- Live runs can differ a little from the recorded ones because the model samples its answers.
- Quick demo buttons 3 to 5 use Replay on purpose, because they depend on one specific recorded draw. In Replay each step is paused for 0.7 seconds so it is watchable and records well.
- "Money" is a cost unit built from token counts. One helper call costs about 0.02 to 0.15.

## 3. Questions you may be asked, and honest answers
- **Where is the reinforcement learning?** The controller is a Double DQN. State: task type, whether it has an answer, the last check result, how many calls so far, money and steps left. Actions: researcher, solver, verifier, summarizer, retry, stop. Reward: answer quality minus cost. It learns by trial and error.
- **Why not just write rules?** In the simulator the learned controller beat the best rule we could write (+9%). On real models a tuned rule is about as good, so the value of learning is that it finds the rule automatically and handles the cost trade-off.
- **Did RL win on the real models?** Mixed. Retrained on real runs, it ties the best hand rule on qwen and is slightly behind on gpt-oss. Both are in the Limitations tab. The tests were written before running them.
- **What is the "simulator only" controller for?** It shows the sim-to-real gap: skills learned in a simulator need retraining on real runs. That is why the retrained one exists.
- **What is a Q-value / "score"?** The controller's estimate of how good each next step is, counting the cost still to come. It picks the highest one.
- **Can it handle any question?** No. The research helper only knows 16 invented documents, and coding tasks come from a fixed list with tests.
- **Is the small sample size a problem?** Yes. 24 held-out real tasks, so we report error bars and call it a pilot.

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
- Tests: `PYTHONPATH=. python3 -m pytest -q tests` (about 90 tests).
- Presets are defined in `app/dashboard.py` (PRESETS); the results numbers are in `app/results_data.py`, each checked against a log.
