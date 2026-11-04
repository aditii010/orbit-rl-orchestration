"""Chapter 6, references and appendices."""
from rtf import *  # noqa: F401,F403
import rtf as R

REFS = [
    "Q. Wu et al., \"AutoGen: Enabling next-gen LLM applications via multi-agent conversation,\" arXiv:2308.08155, 2023.",
    "S. Hong et al., \"MetaGPT: Meta programming for a multi-agent collaborative framework,\" arXiv:2308.00352, 2023.",
    "S. Yao et al., \"ReAct: Synergizing reasoning and acting in language models,\" arXiv:2210.03629, 2022.",
    "R. S. Sutton and A. G. Barto, <i>Reinforcement Learning: An Introduction</i>, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.",
    "P. Lewis et al., \"Retrieval-augmented generation for knowledge-intensive NLP tasks,\" in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2020.",
    "L. Chen, M. Zaharia and J. Zou, \"FrugalGPT: How to use large language models while reducing cost and improving performance,\" arXiv:2305.05176, 2023.",
    "I. Ong et al., \"RouteLLM: Learning to route LLMs with preference data,\" arXiv:2406.18665, 2024.",
    "C. J. C. H. Watkins and P. Dayan, \"Q-learning,\" <i>Machine Learning</i>, vol. 8, no. 3-4, pp. 279-292, 1992.",
    "V. Mnih et al., \"Human-level control through deep reinforcement learning,\" <i>Nature</i>, vol. 518, no. 7540, pp. 529-533, 2015.",
    "L.-J. Lin, \"Self-improving reactive agents based on reinforcement learning, planning and teaching,\" <i>Machine Learning</i>, vol. 8, no. 3-4, pp. 293-321, 1992.",
    "H. van Hasselt, A. Guez and D. Silver, \"Deep reinforcement learning with Double Q-learning,\" in <i>Proc. AAAI Conf. Artificial Intelligence</i>, vol. 30, no. 1, pp. 2094-2100, 2016.",
    "D. P. Kingma and J. Ba, \"Adam: A method for stochastic optimization,\" in <i>Proc. International Conference on Learning Representations (ICLR)</i>, 2015.",
    "J. Tobin et al., \"Domain randomization for transferring deep neural networks from simulation to the real world,\" in <i>Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)</i>, 2017.",
    "P. Henderson et al., \"Deep reinforcement learning that matters,\" in <i>Proc. AAAI Conf. Artificial Intelligence</i>, vol. 32, no. 1, 2018.",
    "R. Agarwal et al., \"Deep reinforcement learning at the edge of the statistical precipice,\" in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2021.",
]


def build():
    heading(1, "6. CONCLUSION AND FUTURE SCOPE", newpage=True)
    heading(2, "6.1 Conclusion")
    p("This project formulated the orchestration of a small team of language-model agents as a Markov decision process and trained a Double DQN controller to choose the "
      "next agent or to stop. In a simulator, the controller beat every baseline: +0.189 against +0.173 for the best hand-written rule (+9.3%), +0.153 for a single agent and "
      "+0.003 for a fixed pipeline. It kept its edge in five of six shifted worlds, and with the difficulty and confidence signals removed it beat the best blind rule by "
      "15.7%. The cost penalty is the essential ingredient of the design: without it the return falls by 0.31.")
    p("On recorded outputs of two real models the picture is more modest and more informative. The choice of agents matters, because research questions cannot be answered "
      "without the researcher and a fixed pipeline wastes money. A controller trained only in the simulator does not transfer, since it never calls the researcher. "
      "Retraining on recorded traces removes that failure, and the retrained controller ties a tuned hand rule on the smaller model, but a rule tuned with task-specific "
      "options is slightly better on the larger one. The value of learning is that it discovers a policy of comparable quality automatically and handles the cost trade-off "
      "without hand-written rules.")
    p("Against the objectives of Section 1.3: the MDP and its loophole tests were built and passed (objective 1); the controller and fair baselines were implemented "
      "(objective 2); the simulator hypotheses were tested with decision rules written in advance, including the cases where the predictions were wrong (objective 3); "
      "the real-model comparison and the simulation-to-reality gap were measured (objective 4); and the live race with a scoreboard was delivered (objective 5).")
    heading(2, "6.2 Future Scope")
    numbered([
        "<b>More tasks and models.</b> Enlarge the real task set and add models to narrow the intervals and test whether the typed-rule advantage on the larger model is real.",
        "<b>Learning from live calls.</b> Train online on real model calls, and capture the effect of feedback on retries, which the replay environment omits.",
        "<b>A more realistic simulator.</b> Randomise skills, costs and verifier reliability, in the spirit of domain randomisation [13], so that a simulator-trained controller transfers without retraining.",
        "<b>A learned difficulty estimate.</b> Train a cheap predictor of task difficulty so that the controller can use the signal that helped most in the simulator.",
        "<b>An LLM router baseline.</b> Add a baseline in which a language model chooses the next agent, with its own routing calls counted in the cost.",
        "<b>Richer reporting.</b> Use performance profiles and interval estimates across many seeds and tasks, as recommended in [14] and [15].",
    ])

    heading(1, "REFERENCES", newpage=True)
    for i, t in enumerate(REFS, 1):
        reference("[%d] %s" % (i, t))

    heading(1, "APPENDIX A: CODE ORGANISATION", newpage=True)
    p("The repository has six code packages: <b>env</b> (simulator and Gym-style environment with action masks), <b>baselines</b> (single agent, fixed pipeline, rule routers), "
      "<b>rl</b> (Double DQN, replay buffer, training loop), <b>eval</b> (evaluation runner and the tuning, shift, ablation, blind and real-trace studies), <b>llm</b> "
      "(model client, the four agents, the 50 tasks with graders, the trace-cache builder and replay environment) and <b>app</b> (race engine, charts and the Streamlit "
      "dashboard). The folder <b>docs</b> holds the decision log, the specification, the demonstration runbook and the logs of every reported run; <b>tests</b> holds 87 "
      "automated tests covering reward loopholes, graders, replay, the race engine and the dashboard.")
    heading(1, "APPENDIX B: REPRODUCING THE RESULTS")
    p("From the repository root, with Python 3 and PyTorch installed: run the tests with <b>PYTHONPATH=. python3 -m pytest -q tests</b>. Run the main comparison and tuning with "
      "<b>python3 -m eval.tune_dqn</b>, the shift study with <b>python3 -m eval.shift</b>, the ablations with <b>python3 -m eval.ablate</b> and <b>python3 -m eval.ablate_blind</b>, and "
      "the real-trace studies with <b>python3 -m eval.real_eval runs/cache_qwen3b.jsonl</b> and the same command for <b>runs/cache_gptoss20b.jsonl</b>, followed by "
      "<b>python3 -m eval.real_eval_typed</b> with the cache path. The dashboard starts with <b>./run_demo.sh</b>. Logs of every run reported here are in <b>docs/logs</b>.")
    heading(1, "APPENDIX C: PRE-REGISTERED PREDICTIONS AND OUTCOMES", newpage=True)
    R.table_caption("C", "Predictions written before the runs, and what happened")
    R.table([2300, 3000, 3726], ["Experiment", "Prediction or rule", "Outcome"], [
        ["Feasibility, simple worlds", "Go if the optimum beats the best rule by 10%", "No-go: best rule within 0 to 1.3% of optimal; a richer world was designed first"],
        ["First DDQN vs rules", "10% over the best rule", "+12.7% over a weak rule; flaws found; fair rule gave 0%"],
        ["Skip the check at low confidence", "Skipping helps", "Not supported (+0.0075, not significant)"],
        ["Plan A: stabilise training", "Beat the rule beyond noise, no seed below it", "Met on all five seeds; +9.3%, just under the 10% bar"],
        ["Shift study", "Edge survives shifts", "Mostly supported: at or above the retuned rule in 5 of 6 worlds"],
        ["Ablations", "Difficulty hurts most; cost penalty large", "Cost penalty as predicted; memory of the episode hurt more than difficulty in the training world"],
        ["Blind test", "DDQN loses about half its edge but keeps +3% over the blind rule", "Passed (+15.7%); the loss was smaller than predicted (0.012)"],
        ["Real qwen: fixed pipeline", "Beats the single agent", "Wrong: -0.126 (interval includes 0)"],
        ["Real qwen: zero-shot, rule, refit", "Zero-shot near single agent; rule uses researcher; refit ties rule", "All right; zero-shot research success 0.00"],
        ["Real gpt-oss", "Single agent strong on coding and analysis; researcher decisive; refit ties rule", "Right; refit minus rule +0.100 (interval includes 0)"],
        ["Typed rule vs refit", "Tie on both models", "Tie on qwen (+0.005); typed rule better on gpt-oss (-0.048)"],
    ], aligns=["ql", "ql", "ql"], size=19)
