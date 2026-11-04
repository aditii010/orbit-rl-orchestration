"""Chapter 5 (lean version). Every number is copied from a log in docs/logs (Appendix C lists the experiments)."""
import json
from rtf import *  # noqa: F401,F403
import rtf as R

LC = json.load(open("report/lc_summary.json"))


def build():
    heading(1, "5. RESULTS AND DISCUSSION", newpage=True)
    p("Returns are the reward of Section 3.3, averaged over tasks (higher is better). Simulator numbers use 10,000 unseen test tasks unless stated; real-model numbers "
      "use the 24 held-out tasks.")

    heading(2, "5.1 Validity Checks and Corrections")
    p("Feasibility tests in simple worlds found the best hand rule within 0 to 1.3% of the optimum, a no-go, so a richer world was designed before it was tested. The first "
      "DDQN comparison looked promising (+0.173 on average, 12.7% above the best simple baseline), but three flaws in our own comparison were found before any claim was "
      "made: the rule search could not skip the summariser; 300 rules tuned on 1,000 tasks produced a lucky winner (+0.177 development, +0.137 test); and the rules never read "
      "the difficulty estimate that the DDQN reads. With difficulty-aware rules (960 rules, 5,000 tuning tasks) the best rule reached +0.173 and the DDQN advantage fell to 0%. "
      "The original 12.7% came from a weak baseline.")
    p("One mechanism was tested directly. When confidence is below 0.5 the answer is wrong 92% of the time (6,214 cases), which suggested that skipping the check there "
      "would help. Replacing the check by an immediate stop changed the return by only +0.0075 (interval 0.0063) and by an immediate retry by -0.0004. The hypothesis was "
      "not supported: the check is cheap and the return is flat in that region.")

    heading(2, "5.2 Training Stability")
    p("With the original setting (20,000 episodes) the five seeds scored +0.185, +0.183, +0.181, +0.181 and +0.136, so one seed was clearly below the rule. Three longer "
      "settings declared in advance were compared on 5,000 development tasks: S1 (60,000 episodes, learning rate 0.001, target update 500) scored +0.181, S2 (learning "
      "rate 0.0003, target update 1,000) scored +0.203 and S3 (S2 with exploration over 80% of training) scored +0.198. S2 was chosen on development return alone, and "
      "it removed the unstable seed: all five test scores (+0.183 to +0.195) are above the best rule. Figure 5.1 shows its learning curve (mean of five seeds on "
      "1,000 development tasks): the return rises quickly during the first 20,000 episodes and then flattens near +0.205.")
    pts = LC["mean"][1::2]
    R.vchart(pts, 0.25, rows=8, labels=["%dk" % (4 * (i + 1)) for i in range(len(pts))], colour="navy", col_w=550, label_w=700)
    R.figure_caption(5, "Learning curve: mean development return of 5 seeds against training episodes (thousands).")

    heading(2, "5.3 Main Simulator Result")
    R.table_caption(5, "Simulator results on 10,000 test tasks")
    R.table([3300, 1650, 1450, 1250, 1376], ["Method", "Return", "Success", "Cost", "Steps"], [
        ["Fixed pipeline", "+0.003", "0.37", "0.293", "5.0"],
        ["Single agent", "+0.153", "0.34", "0.123", "2.0"],
        ["Best rule (uses difficulty and confidence)", "+0.173", "0.46", "0.233", "4.0"],
        ["<b>Double DQN (5 seeds)</b>", "<b>+0.189</b>", "<b>0.47</b>", "<b>0.215</b>", "<b>3.7</b>"],
    ], aligns=["ql", "qc", "qc", "qc", "qc"])
    R.bars([("Fixed pipeline", 0.0034, "greyblue", "+0.003"), ("Single agent", 0.1528, "greyblue", "+0.153"),
            ("Best rule", 0.1732, "greyblue", "+0.173"), ("Double DQN", 0.189, "navy", "+0.189")], 0.2)
    R.figure_caption(5, "Mean return on 10,000 test tasks (longer bar is better).")
    p("The DDQN earns +0.189, 9.3% above the best rule and 24% above the single agent, with a higher success rate than the single agent (0.47 against 0.34) at a cost "
      "of 0.215 against 0.123, which the reward judges worth paying. The fixed pipeline pays 0.293 for almost no gain. The 9.3% is just under the 10% bar set at the "
      "start, and both numbers are reported. Paired with the rule on identical tasks, every seed is ahead by +0.009 to +0.022, each beyond its 95% interval (about 0.007). "
      "The advantage is small in absolute terms (about 0.016 per task). The policy always solves first, stops or tidies after a passed check, retries after a failed one "
      "and gives up early on hard tasks, a structure it found without being told.")

    heading(2, "5.4 Robustness Under Distribution Shift")
    p("Controllers and rule were trained or tuned once and tested in six other worlds (Table 5.2). A rule retuned by a human who knew the shift is a reference. In the "
      "research-heavy world, research tasks are 70% of the mix.")
    R.table_caption(5, "Return in shifted worlds (5,000 tasks each; DDQN is the mean of 5 seeds, lowest seed in brackets)")
    R.table([2300, 1100, 1450, 1450, 1500, 1226], ["World", "Single", "Rule (fixed)", "Rule (retuned)", "DDQN", "DDQN minus rule"], [
        ["Training world", "+0.153", "+0.166", "+0.167", "+0.187 (+0.181)", "+0.020"],
        ["Tight budget", "+0.153", "+0.172", "+0.172", "+0.167 (+0.161)", "-0.005"],
        ["Loose budget", "+0.153", "+0.166", "+0.167", "+0.181 (+0.170)", "+0.016"],
        ["Research-heavy", "+0.067", "+0.029", "+0.072", "+0.077 (+0.072)", "+0.048"],
        ["Harder tasks", "+0.088", "+0.085", "+0.093", "+0.103 (+0.090)", "+0.018"],
        ["Easier tasks", "+0.213", "+0.246", "+0.241", "+0.263 (+0.257)", "+0.016"],
    ], aligns=["ql", "qc", "qc", "qc", "qc", "qc"], size=19)
    rows = []
    for w, rule, dqn in [("Training world", .166, .187), ("Tight budget", .172, .167), ("Loose budget", .166, .181),
                         ("Research-heavy", .029, .077), ("Harder tasks", .085, .103), ("Easier tasks", .246, .263)]:
        rows += [(w + ": fixed rule", rule, "greyblue", "%+.3f" % rule), (w + ": DDQN", dqn, "navy", "%+.3f" % dqn)]
    R.bars(rows, 0.28, label_w=3300, bar_w=4500, val_w=1226, size=18)
    R.figure_caption(5, "Return in six worlds for the fixed rule and the DDQN, both trained or tuned once.")
    p("The fixed rule breaks in the research-heavy world (+0.029, below the single agent at +0.067). The DDQN keeps +0.077, level with the retuned rule at +0.072, without "
      "retuning. It is at or above the retuned rule in five of six worlds and loses only under tight budgets (+0.167 against +0.172, within noise); that range partly overlaps "
      "training, and the retuned rule used only 2,000 tuning tasks. The value of the DDQN is robustness more than peak return.")

    heading(2, "5.5 Which Signals Matter: Ablations and the Blind Controller")
    p("Each variant was retrained for five seeds with the final setting and tested on the same 10,000 tasks as the full controller, using the true reward (Table 5.3).")
    R.table_caption(5, "Change in return relative to the full DDQN (paired, 5 seeds)")
    R.table([3600, 1700, 1700, 2026], ["Variant", "Training world", "Research-heavy", "Remark"], [
        ["A: no cost penalty in training", "-0.312", "-0.406", "7.2 calls, cost 0.54"],
        ["B: no memory of the episode", "-0.021", "-0.004", "stops verifying"],
        ["C1: no difficulty estimate", "-0.010", "-0.008", ""],
        ["C2: no confidence score", "-0.008", "-0.007", ""],
        ["C3: neither signal (blind)", "-0.012", "not run", "return +0.177"],
    ], aligns=["ql", "qc", "qc", "ql"])
    R.bars([("A: no cost penalty", 0.3123, "copper", "-0.312"), ("B: no memory", 0.0205, "copper", "-0.021"), ("C1: no difficulty", 0.0104, "copper", "-0.010"),
            ("C2: no confidence", 0.0075, "copper", "-0.008"), ("C3: neither signal", 0.0122, "copper", "-0.012")], 0.34)
    R.figure_caption(5, "Loss in return when an ingredient is removed (training world; bar length is the loss).")
    p("The cost penalty is by far the most important ingredient: without it the controller learns that calls are free, makes 7.2 calls per task at cost 0.54, and its true "
      "return falls to -0.123. Of the signals, the memory of the episode matters most in the training world, because without the last check result the controller stops "
      "verifying and behaves like a single agent. The difficulty estimate and the confidence score each cost 0.01 or less because they overlap: both say that the answer is "
      "probably wrong. Our predictions were only partly right: the cost-penalty prediction held, the difficulty estimate hurts most among the signals only in the "
      "research-heavy world, and the memory of the episode hurt more than expected in the training world.")
    p("The blind test removes difficulty and confidence together, because real models give neither (Section 3.5). The best blind rule is the single agent in disguise "
      "(+0.153). The blind DDQN reaches +0.177, 15.7% above it, with all five seeds ahead by +0.017 to +0.033, each beyond its interval; the pre-registered bar (at least 3% "
      "and at least four of five seeds) was passed. The prediction that the DDQN would lose half of its edge was wrong: it lost 0.012, while the best rule lost 0.020, so "
      "blindness hurts the rules more than the learner. The blind rule family is small, which may inflate the gap.")

    heading(2, "5.6 Real Models: What the Recorded Traces Show")
    p("Table 5.4 shows how much room orchestration has. It gives the fraction of final answers that are correct and how often the verifier agrees with the truth, for "
      "qwen2.5:3b (4 samples per setting) and gpt-oss-20b (2 to 3).")
    R.table_caption(5, "Recorded solver success and verifier agreement by task type and evidence level")
    R.table([3000, 1500, 1500, 1513, 1513], ["Setting", "qwen success", "gpt-oss success", "qwen verifier", "gpt-oss verifier"], [
        ["Coding, no plan", "0.68", "1.00", "0.91", "1.00"], ["Coding, with plan", "0.79", "0.97", "0.86", "1.00"],
        ["Analytical, no plan", "0.95", "1.00", "0.92", "1.00"], ["Analytical, with plan", "0.93", "0.87", "0.88", "0.90"],
        ["Research, 0 researcher calls", "0.00", "0.00", "1.00", "0.93"], ["Research, 1 call", "0.40", "0.53", "0.80", "0.67"],
        ["Research, 2 calls", "0.43", "0.93", "0.70", "0.93"], ["Research, 3 calls", "0.43", "0.93", "0.70", "0.93"],
    ], aligns=["ql", "qc", "qc", "qc", "qc"])
    p("Research questions cannot be answered without the researcher: success is 0% with no call and, for the stronger model, 93% after two. The smaller model cannot chain "
      "facts across documents, so it stops near 43% however much evidence it receives. Both models already solve most coding and analytical tasks, so orchestration has "
      "little to gain there. Confidence is useless for qwen (Section 3.5), so the controller must act on the task type and on what has happened so far.")

    heading(2, "5.7 Real Models: Controller Comparison")
    p("The rules and the refit DDQN were fitted on the 26 development tasks and every policy was replayed on the 24 held-out tasks (Table 5.5). The first tuned rule is: call "
      "the researcher once on research tasks, solve once, stop. The typed rule found, for qwen, one researcher call on research tasks, visible-test verification for coding and "
      "two retries; for gpt-oss, two researcher calls on research tasks and nothing else.")
    R.table_caption(5, "Return, success rate and cost on 24 held-out real tasks")
    R.table([2900, 1050, 1000, 1000, 1050, 1000, 1026], ["Policy", "qwen return", "success", "cost", "gpt-oss return", "success", "cost"], [
        ["Single agent", "+0.462", "0.59", "0.106", "+0.563", "0.71", "0.116"],
        ["Fixed pipeline", "+0.336", "0.75", "0.334", "+0.282", "0.75", "0.374"],
        ["Tuned rule (researcher once)", "+0.569", "0.72", "0.120", "+0.670", "0.83", "0.131"],
        ["Typed rule", "+0.576", "0.76", "0.146", "+0.819", "1.00", "0.145"],
        ["DDQN, zero-shot (simulator only)", "+0.438", "0.64", "0.162", "+0.530", "0.71", "0.143"],
        ["DDQN, refit on traces", "+0.580", "0.77", "0.154", "+0.770", "0.96", "0.153"],
    ], aligns=["ql", "qc", "qc", "qc", "qc", "qc", "qc"], size=19)
    rows = []
    for name, q, g in [("Single agent", .462, .563), ("Fixed pipeline", .336, .282), ("Typed rule", .576, .819), ("DDQN zero-shot", .438, .530), ("DDQN refit", .580, .770)]:
        col = "navy" if "refit" in name else "greyblue"
        rows += [(name + ": qwen", q, col, "%+.3f" % q), (name + ": gpt-oss", g, "slate" if col == "greyblue" else "navy", "%+.3f" % g)]
    R.bars(rows, 0.9, label_w=3300, bar_w=4500, val_w=1226, size=18)
    R.figure_caption(5, "Return on held-out real tasks for each policy and model (DDQN refit is the mean of 5 seeds).")
    R.table_caption(5, "Paired task-level differences (95% interval over 24 tasks)")
    R.table([4200, 2413, 2413], ["Comparison", "qwen", "gpt-oss"], [
        ["Tuned rule minus fixed pipeline", "+0.233 (0.134)", "+0.388 (0.140)"],
        ["Fixed pipeline minus single agent", "-0.126 (0.202)", "-0.281 (0.217)"],
        ["Zero-shot DDQN minus single agent", "-0.024 (0.055)", "-0.033 (0.009)"],
        ["Refit DDQN minus single agent", "+0.118 (0.137)", "+0.207 (0.142)"],
        ["Refit DDQN minus tuned rule", "+0.011 (0.044)", "+0.100 (0.111)"],
        ["Refit DDQN minus typed rule", "+0.005 (0.018)", "-0.048 (0.041)"],
    ], aligns=["ql", "qc", "qc"])
    p("A well-chosen rule beats the fixed pipeline on both models (+0.233 and +0.388), because the pipeline pays about three times the cost of a single agent for steps most "
      "tasks do not need. On gpt-oss the refit DDQN beats the single agent (+0.207), and the zero-shot controller is slightly but clearly worse than the single agent "
      "(-0.033). The refit DDQN equals the first tuned rule, which could call the researcher only once. When the rule family may be task-specific, the refit DDQN ties "
      "the typed rule on qwen (+0.005) and is slightly but clearly behind it on gpt-oss (-0.048): it calls the researcher about 3.3 times on gpt-oss research tasks where two "
      "suffice, and one of five seeds scores noticeably lower (+0.656 against +0.79 to +0.81). With 26 training tasks this is unsurprising. On these traces a hand-tuned "
      "rule is as good as the learned policy or slightly better, and the learned policy finds a comparable rule without being told which options exist.")

    heading(2, "5.8 The Simulation-to-Reality Gap and Retraining")
    p("The simulator-trained blind DDQN, run unchanged on real traces, never calls the researcher on research tasks, so its success there is 0.00 on both models "
      "(Table 5.7). In the simulator the researcher only raised the success of later solver calls and was rarely worth its cost; with real models it is indispensable. "
      "Retraining on the traces repairs the gap: the refit controller calls the researcher 1.40 (qwen) and 3.33 (gpt-oss) times per research task, and research success becomes 0.46 and 0.87.")
    R.table_caption(5, "Calls per research episode (researcher, solver, verifier, summariser) and success on held-out tasks")
    R.table([2800, 3600, 2626], ["Model and policy", "Calls (res / sol / ver / sum)", "Success"], [
        ["qwen, zero-shot", "0.00 / 1.00 / 0.00 / 0.13", "0.00"], ["qwen, refit", "1.40 / 1.07 / 0.00 / 0.00", "0.46"],
        ["gpt-oss, zero-shot", "0.00 / 1.00 / 0.00 / 0.05", "0.00"], ["gpt-oss, refit", "3.33 / 1.01 / 0.00 / 0.04", "0.87"],
    ], aligns=["ql", "qc", "qc"])
    p("Coding shows the opposite lesson. The zero-shot controller verifies and summarises coding answers (1.4 verifier calls on qwen), which helps there (success 0.86 "
      "against 0.74 for the first tuned rule) but wastes money on gpt-oss, where the first answer is already right. The refit controller verifies only where it pays.")

    heading(2, "5.9 The Race and the Scoreboard")
    p("The demonstration runs the saved refit controller (training seed 0, fixed in advance) against any other policy, and each race is decided by the training reward. In the "
      "four prepared races the DDQN wins every time; for example, on a coding task whose first answer fails the visible tests, it verifies, sees the failure, retries and ends "
      "correct at cost 0.29 (score +0.64), while the single agent hands in the wrong answer at cost 0.13 (score -0.16). One task is a small sample, so the scoreboard races "
      "two policies on all 24 held-out tasks, each averaged over 10 recorded draws (Table 5.8).")
    R.table_caption(5, "Scoreboard of the demonstration controller (right) against each opponent (left), 24 tasks")
    R.table([3300, 1300, 1000, 1300, 2126], ["Opponent", "Right wins", "Ties", "Left wins", "Mean advantage"], [
        ["Fixed pipeline", "16", "0", "8", "+0.196"], ["Single agent", "6", "7", "11", "+0.068"],
        ["Typed rule", "1", "15", "8", "-0.056"], ["DDQN, simulator only", "12", "7", "5", "+0.064"],
    ], aligns=["ql", "qc", "qc", "qc", "qc"])
    p("The DDQN wins clearly against the fixed pipeline and the simulator-only controller, is ahead on average against the single agent but wins fewer tasks than it "
      "loses (6 against 11), and is behind the typed rule. The demonstration model scores +0.536 on these tasks, below the five-seed average of +0.580 in Table 5.5, "
      "because it is a single training run; the page reports the scoreboard as computed.")

    heading(2, "5.10 Discussion, Limitations and Threats to Validity")
    p("<b>What the evidence supports.</b> In a simulator whose parameters were fixed before training, a Double DQN finds a cost-aware policy that beats the best hand-written "
      "rule by 9.3%, stays ahead under most shifts and beats a blind rule by 15.7%; the cost penalty is the essential ingredient. On real models the choice of agents matters "
      "(research needs the researcher; a fixed pipeline wastes money) and the policy retrained on real traces finds a rule as good as a human-tuned one in about one case out of two.")
    p("<b>What it does not support.</b> The learned controller does not clearly beat a well-tuned task-specific rule on real traces, and it does not transfer from the simulator "
      "without retraining. Limitations and threats to validity:")
    bullet("The real-model study has 24 test tasks, two models and a few recorded samples per setting, so intervals are wide and results show direction, not proof.")
    bullet("A retry replays the next recorded sample, so the effect of feedback on the solver is not captured; verifier verdicts are fixed per sample.")
    bullet("The token-to-cost conversion (divisor 2,500, overhead 0.02) is a design choice; another choice would change the best policy.")
    bullet("The rule families are finite. The typed family was declared after the real results suggested it, and was therefore tuned on development tasks only.")
    bullet("The simulator is stylised and its parameters were chosen by the author, not fitted to data; the blind rule family is small.")
    bullet("The demonstration model is one training run, and the scoreboard tasks are also the test tasks of the real-model study.")
    bullet("Tasks are few and partly invented; retrieval is keyword-based over 16 documents, so research results say nothing about open-domain retrieval.")
