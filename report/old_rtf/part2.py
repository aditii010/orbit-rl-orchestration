"""Chapters 3 and 4 (lean version)."""
from rtf import *  # noqa: F401,F403
import rtf as R


def build():
    heading(1, "3. PROBLEM FORMULATION AND SYSTEM DESIGN", newpage=True)
    heading(2, "3.1 System Overview")
    p("A <b>controller</b> observes the situation and chooses the next action; four <b>agents</b> (researcher, solver, verifier, summariser) carry it out; an "
      "<b>environment</b> keeps the hidden state, charges the cost of every call and, at the end, grades the answer and pays the reward. The same controller runs on a "
      "simulator, on a replay of recorded real-model outputs, and on live model calls (Figure 3.1).")
    R.flow(["Task and budget arrive", "Controller observes 16 numbers, picks a legal action", "Agent runs, cost is charged",
            "State updates (answer, check result, money left)", "STOP: answer graded, reward paid"])
    R.figure_caption(3, "Orchestration loop. The middle three steps repeat until the controller chooses STOP or a limit is reached.")
    heading(2, "3.2 Tasks and Agents")
    p("<b>Coding</b> tasks ask for a small Python function. <b>Analytical</b> tasks are word problems with an exact numeric answer. <b>Research</b> tasks are questions "
      "about invented facts in a small document set, which no model can answer from memory. The <b>researcher</b> retrieves evidence, the <b>solver</b> drafts an answer, "
      "the <b>verifier</b> returns pass or fail for the current answer (for coding it runs the visible unit tests) and the <b>summariser</b> formats the final answer.")
    heading(2, "3.3 The Markov Decision Process")
    p("The controller sees 16 numbers (Table 3.1). The true difficulty and the true correctness of the current answer are hidden; the difficulty estimate and the "
      "confidence score are noisy signals about them. Table 3.2 lists the six actions. Illegal actions are masked in both action selection and the learning "
      "update, so the network is never asked to value an action that cannot be taken.")
    R.table_caption(3, "Observation vector seen by the controller")
    R.table([1000, 2700, 5326], ["Index", "Name", "Meaning"], [
        ["0 to 2", "Task type", "One-hot: coding, analytical, research"],
        ["3", "Difficulty estimate", "True difficulty plus Gaussian noise (sd 0.15), clipped to 0 to 1"],
        ["4", "Has answer", "1 once a solver call has produced a candidate answer"],
        ["5 to 7", "Last check", "One-hot: none yet, passed, failed (for the current answer)"],
        ["8 to 11", "Calls so far", "Researcher, solver, verifier, summariser, each divided by 8"],
        ["12, 13", "Money, steps left", "Fractions of the budget and of the 8-step limit still unused"],
        ["14", "Confidence", "Noisy score for the current answer (mean 0.70 if correct, 0.50 if wrong, sd 0.15)"],
        ["15", "Summarised", "1 if the current answer has been through the summariser"],
    ], aligns=["qc", "ql", "ql"])
    R.table_caption(3, "Action set and legality")
    R.table([2500, 3900, 2626], ["Action", "Effect", "Legal when"], [
        ["INVOKE_RESEARCHER", "Retrieves new evidence for later solver calls", "Always"],
        ["INVOKE_SOLVER", "Produces the first candidate answer", "No candidate yet"],
        ["INVOKE_VERIFIER", "Checks the current candidate", "A candidate exists"],
        ["INVOKE_SUMMARIZER", "Formats the current candidate", "Candidate not yet summarised"],
        ["RETRY_SOLVER", "Produces a new candidate", "A candidate exists"],
        ["STOP", "Hands in the current candidate", "Always"],
    ])
    p("<b>Reward.</b> The reward of an episode is R = q x 1[correct] - lambda x C / C_ref - 0.3 x 1[no answer], where C is the sum of the call costs, lambda = 0.5 and "
      "C_ref = 0.4. The quality q is 1, except in the simulator where a correct but never summarised answer earns 0.9. The cost is paid at the step where it occurs and "
      "the quality at the end, so nothing rewards producing more text. An episode ends at STOP, when the cost reaches the budget (uniform in 0.5 to 1.2), or after "
      "8 steps. Four automated tests check that the reward cannot be gamed: stopping at once scores below solve-then-stop; a useless repeat scores below stopping; a "
      "verified answer beats the same answer with wasted calls; and exhausting the budget with no answer scores worst.")
    heading(2, "3.4 The Simulator")
    p("One solver call is correct with probability p = skill x (1 - difficulty) x 0.85^n x g(e) x b, where n counts earlier solver calls, g(e) = 0.6 + 0.4e for "
      "research and 0.9 + 0.1e for other tasks, e is the evidence level (+0.35 per researcher call, at most 1), and b = 1.10 for a retry after a failed check. The "
      "verifier gives the right verdict with the reliability in Table 3.3. Difficulty is Beta(2, 2), the three task types are equally likely, and each cost is "
      "multiplied by a random factor in 0.8 to 1.2 (a retry costs 0.8 of a first call).")
    R.table_caption(3, "Simulator parameters")
    R.table([3400, 1800, 1800, 2026], ["Parameter", "Coding", "Analytical", "Research"], [
        ["Solver skill", "0.95", "0.85", "0.70"], ["Verifier reliability", "0.95", "0.85", "0.70"],
        ["Base cost: researcher", "0.08", "0.08", "0.10"], ["Base cost: solver", "0.10", "0.12", "0.15"],
        ["Base cost: verifier", "0.03", "0.05", "0.08"], ["Base cost: summariser", "0.03", "0.03", "0.03"],
    ], aligns=["ql", "qc", "qc", "qc"])
    heading(2, "3.5 The Real-Model Environment")
    R.table_caption(3, "The 50 real tasks (all automatically graded)")
    R.table([1500, 900, 6626], ["Type", "Count", "Content and grading"], [
        ["Coding", "20", "Python functions; 2 visible and 5 to 6 hidden unit tests; the verifier runs only the visible tests"],
        ["Analytical", "15", "Word problems with an exact numeric answer; the verifier is an LLM check"],
        ["Research", "15", "Questions over 16 invented documents, keyword retrieval; 6 need one document, 9 need two or three"],
    ], aligns=["ql", "qc", "ql"])
    p("<b>Models.</b> The main model is qwen2.5:3b, run locally through Ollama, which is weak enough to fail often. A stronger check uses openai/gpt-oss-20b through Groq. "
      "Research retrieval is a plain keyword search with no model call: an early version let the model summarise the passages, and on irrelevant ones it wrote "
      "\"none of these help\", which the solver copied, so research accuracy fell to 0 of 9 until the researcher was made pure retrieval.")
    p("<b>Trace cache and replay.</b> Agent outputs were recorded once: for each task and evidence level, several solver samples and the verifier verdict for each. "
      "Every policy replays the same recordings, so differences come from the policy and not from model luck. A retry replays the next unused sample; the effect of "
      "telling the solver why it failed is not recorded (a stated limitation).")
    p("<b>Cost mapping.</b> Real calls are priced as (tokens in + tokens out) / 2500 + 0.02, so that an average solver call costs about 0.12 as in the simulator; the 0.02 "
      "covers retrieval, tests and orchestration. This choice fixes what lambda means for real calls.")
    p("<b>What the controller sees.</b> In a pilot, qwen reported confidence 0.98 when right and 1.00 when wrong (36 samples), and no non-oracle difficulty estimate "
      "existed. In every real-model experiment the difficulty and confidence entries are therefore set to zero. Section 5.5 first measures what this blindness costs.")
    heading(2, "3.6 The Double DQN Controller")
    p("The controller is a network with two hidden layers of 128 units (16 inputs, 6 outputs). Episodes are played with an epsilon-greedy policy over legal actions, "
      "and each transition (with the next-state mask) goes into a replay buffer. A mini-batch gives the target y = r + gamma x Q_target(s', argmax over legal a' of "
      "Q_online(s', a')), with no future term at the end of an episode. The Huber loss is minimised with Adam (gradient norm clipped at 10), and the target network is "
      "refreshed every 1,000 steps. Table 3.5 gives the settings. The blind variant zeroes the difficulty and confidence inputs; the refit variant is trained from "
      "scratch on the replay environment built from the development tasks only.")
    R.table_caption(3, "Training settings of the final controller (S2)")
    R.table([4200, 4826], ["Setting", "Value"], [
        ["Episodes (simulator / real-trace refit)", "60,000 / 20,000"],
        ["Discount; learning rate (Adam)", "0.99; 0.0003"],
        ["Replay buffer; batch; warm-up", "50,000; 64; 1,000 steps"],
        ["Exploration", "Epsilon 1.0 to 0.05, linear over the first 60% of episodes"],
        ["Seeds; checkpoint", "5 seeds; final checkpoint (no picking)"],
    ])
    heading(2, "3.7 Baselines and Demonstration System")
    p("The baselines are a <b>single agent</b> (one solver call, then stop), a <b>fixed pipeline</b> (researcher, solver, verifier, summariser, stop, always), a "
      "<b>rule router</b> (best of 960 hand-built rules over research, verification, retries, a confidence threshold and a give-up difficulty, tuned on 5,000 "
      "development tasks), a <b>blind rule</b> (best of 256 rules that cannot use difficulty or confidence) and, for real traces, a <b>typed rule</b> (best of 768 rules "
      "with per-task-type options). A router in which an LLM picks the next agent was not built. A web application runs any two controllers on the same task, live or "
      "from recorded runs, reports a winner using the training reward (computed, never fixed in advance), draws the money spent step by step for both, and offers a "
      "scoreboard over the 24 held-out tasks (Section 5.9).")

    heading(1, "4. EXPERIMENTAL METHODOLOGY", newpage=True)
    heading(2, "4.1 Splits, Seeds and Statistics")
    p("In the simulator, rules are tuned on 5,000 development tasks (seeds 0 to 4,999) and everything is reported on 10,000 test tasks (seeds 100,000 to 109,999); training "
      "seeds start at 1,000,000 and never overlap. The shift study uses 5,000 tasks per world. For real models the 50 tasks are split by type, even positions to "
      "development (26 tasks: rule tuning and refit) and odd positions to test (24 tasks, 100 seeded episodes each).")
    p("Methods are compared on identical seeded tasks, so differences are paired. In the simulator the 95% interval is 1.96 standard errors of the paired difference. "
      "For real models the unit is the task: returns are averaged over a task's 100 episodes and the interval is 2.07 times the standard deviation of the 24 paired "
      "differences divided by the square root of 24, because episodes of one task are not independent. A method is said to beat another only if the interval "
      "excludes zero, and for DDQN results at least four of five seeds must also lie above the comparison.")
    heading(2, "4.2 Pre-Registration and Fairness Controls")
    p("Before each experiment the question, settings, predictions and decision rule were written into the decision log and committed to version control; afterwards the log "
      "records the result and scores each prediction, including the wrong ones (Appendix C). The initial rule was that learning is pursued only if the optimum beats "
      "the best hand rule by at least 10%. Four controls support fairness:")
    bullet("<b>Tuning data.</b> Rules are tuned on development tasks only. Tuning 300 rules on 1,000 tasks once gave a lucky winner (+0.177 development, +0.137 test), so the set was enlarged to 5,000.")
    bullet("<b>Strong baselines.</b> The rule family was extended until it could use every signal the controller uses; a type-specific family was added for real traces.")
    bullet("<b>No picking.</b> Final checkpoints of all five seeds are reported; the demonstration model is seed 0, fixed before any test result was seen.")
    bullet("<b>Like for like.</b> The blind controller is compared with a blind rule.")
    heading(2, "4.3 Implementation")
    p("The code is Python with PyTorch: about 3,300 lines in six packages with 87 automated tests. Models run on a personal laptop (qwen2.5:3b through Ollama at about "
      "14 tokens per second; gpt-oss-20b through Groq, capped at 650 requests), and training one simulator controller takes a few minutes. The repository holds the decision "
      "log, the logs of every reported run and the recorded traces.")
