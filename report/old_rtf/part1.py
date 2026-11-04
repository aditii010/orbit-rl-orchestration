"""Front matter and chapters 1 and 2."""
from rtf import *  # noqa: F401,F403
import rtf as R

TITLE = "ORBIT: Learning When to Call Which Agent"
SUB = "Cost-Aware Orchestration of Multi-Agent Language Model Systems with Double Deep Q-Learning"


def build():
    # ------------------------------------------------------------ title page (no heading, so it stays out of the contents)
    line("THAPAR INSTITUTE OF ENGINEERING AND TECHNOLOGY", 32, bold=True, sb=240, sa=40)
    line("(Deemed to be University), Patiala, Punjab 147004", 22, sa=40)
    line("Department of Electronics and Communication Engineering", 26, bold=True, sa=480)
    line("A PROJECT REPORT ON", 24, sa=240)
    line("ORBIT", 64, bold=True, color="navy", sa=60)
    line("Learning When to Call Which Agent", 36, bold=True, sa=120)
    line("Cost-Aware Orchestration of Multi-Agent Language Model Systems with Double Deep Q-Learning", 26, italic=True, sa=480)
    line("Submitted in partial fulfilment of the requirements of the laboratory evaluation for", 22, sa=20)
    line("Course UCS760 (Reinforcement Learning and Conversational AI)", 24, bold=True, sa=480)
    R.add("\\trowd\\trgaph120\\trleft0\\clcbpat3\\cellx4513\\clcbpat3\\cellx9026\n")
    R.add("\\pard\\plain\\intbl\\qc\\b\\fs24 Submitted to\\cell\\pard\\plain\\intbl\\qc\\b\\fs24 Submitted by\\cell\\row\n")
    R.add("\\trowd\\trgaph120\\trleft0\\cellx4513\\cellx9026\n")
    R.add("\\pard\\plain\\intbl\\qc\\sa60\\fs24 <hl>[Name of faculty member]</hl>\\par\\pard\\plain\\intbl\\qc\\sa60\\fs24 <hl>[Designation]</hl>\\par"
          "\\pard\\plain\\intbl\\qc\\sa60\\fs24 Department of ECE, TIET Patiala\\cell".replace("<hl>", "{\\highlight7 ").replace("</hl>", "}"))
    R.add("\\pard\\plain\\intbl\\qc\\sa60\\b\\fs24 Aditi Sikarwar\\par\\pard\\plain\\intbl\\qc\\sa60\\fs24 Roll No.: {\\highlight7 [Roll number]}\\par"
          "\\pard\\plain\\intbl\\qc\\sa60\\fs24 Electronics and Communication Engineering, Class of 2027\\cell\\row\n")
    line("November 2026", 24, bold=True, sb=480, sa=20)
    line("Patiala, Punjab", 22)

    # ------------------------------------------------------------ certificate
    front_title("CERTIFICATE")
    spacer(12)
    p("This is to certify that the project report entitled <b>\"ORBIT: Learning When to Call Which Agent\"</b>, submitted by <b>Aditi Sikarwar</b> "
      "(Roll No. <hl>[Roll number]</hl>), a student of the Department of Electronics and Communication Engineering, Thapar Institute of "
      "Engineering and Technology, Patiala, in partial fulfilment of the requirements of the laboratory evaluation for course UCS760, "
      "is a bonafide record of the work carried out by the student under my guidance and supervision during October and November 2026.")
    p("The work reported in this document has not been submitted, in part or in full, for the award of any other degree, diploma or certificate.")
    spacer(36)
    line("<hl>[Name of supervisor]</hl>", 24, align="qr", bold=True, sa=20)
    line("<hl>[Designation]</hl>", 24, align="qr", sa=20)
    line("Department of Electronics and Communication Engineering", 22, align="qr", sa=20)
    line("Thapar Institute of Engineering and Technology, Patiala", 22, align="qr", sa=240)
    line("Date: <hl>[Date]</hl>", 24, align="ql", sa=20)
    line("Place: Patiala", 24, align="ql")

    # ------------------------------------------------------------ declaration
    front_title("DECLARATION")
    spacer(12)
    p("I hereby declare that the project report entitled <b>\"ORBIT: Learning When to Call Which Agent\"</b> is a record of my own project work, "
      "carried out for the laboratory evaluation of course UCS760 under the guidance of <hl>[Name of supervisor]</hl>. "
      "All sources of information, published work and software libraries that I used are acknowledged in the text and in the list of references.")
    p("<b>Disclosure of AI assistance.</b> The software, experiments and drafts of this report were developed with the help of an AI coding assistant "
      "(Claude Code, Anthropic). The assistant wrote and ran code and drafted text under my direction. I defined the project, chose the research questions and "
      "evaluation plan, reviewed the results and the claims made from them, and take responsibility for the contents of this report. Every experiment is "
      "recorded in the decision log of the project repository, including the mistakes that were found and corrected.")
    p("The results reported here were produced by the code in the accompanying repository and can be reproduced with the commands listed in Appendix B.")
    spacer(36)
    line("Aditi Sikarwar", 24, align="qr", bold=True, sa=20)
    line("Roll No. <hl>[Roll number]</hl>", 24, align="qr", sa=240)
    line("Date: <hl>[Date]</hl>", 24, align="ql", sa=20)
    line("Place: Patiala", 24, align="ql")

    # ------------------------------------------------------------ acknowledgement
    front_title("ACKNOWLEDGEMENT")
    spacer(12)
    p("I thank my supervisor, <hl>[Name of supervisor]</hl>, for the guidance, feedback and encouragement that shaped this project, and for the freedom to "
      "treat a negative result as a result. I am grateful to the Department of Electronics and Communication Engineering and to Thapar Institute of "
      "Engineering and Technology for the laboratory facilities and for a course that connects reinforcement learning with conversational systems.")
    p("I acknowledge the open-source tools on which this work stands: PyTorch for the learning code, Ollama for running a language model locally, "
      "and the Groq inference service for access to a second, larger model. I also thank the authors of the papers listed in the references, whose "
      "work defined the problem and the standards of evidence that this report tries to meet. Finally, I thank my family and friends for their support.")
    spacer(24)
    line("Aditi Sikarwar", 24, align="qr", bold=True)

    # ------------------------------------------------------------ abstract
    front_title("ABSTRACT")
    spacer(6)
    p("Systems built from several language-model agents, for example a researcher, a solver, a verifier and a summariser, can answer harder questions "
      "than a single model, but every extra call costs tokens and time. Fixed pipelines call every agent on every task and spend much of that "
      "budget on tasks that did not need it. This project asks whether a learned controller can decide, step by step, which agent to call next and when "
      "to stop, so that answer quality is maximised after a cost penalty.")
    p("The orchestration problem is formulated as a Markov decision process with a 16-number observation, six masked actions and a reward equal to answer "
      "quality minus a scaled cost. A Double Deep Q-Network (DDQN) controller is trained in a simulator and compared with a single agent, a fixed pipeline and "
      "hand-tuned rule routers on 10,000 unseen tasks, using paired comparisons and decision rules written down before the experiments. The DDQN reaches a mean "
      "return of +0.189, against +0.173 for the best rule (+9.3%), +0.153 for a single agent and +0.003 for the fixed pipeline. It keeps its advantage in five of six "
      "shifted task distributions and beats the best rule by 15.7% when the difficulty and confidence signals are hidden. Training without the cost penalty "
      "collapses (return change of -0.31), which makes the penalty the essential ingredient.")
    p("A second study replays recorded outputs of two real models (qwen2.5:3b and gpt-oss-20b) on 50 tasks. A controller trained only in the simulator fails on "
      "research questions because it never calls the researcher. Retraining on recorded traces repairs this and ties a tuned rule on the smaller model, but does not "
      "beat a rule tuned with task-specific options on the larger one. A live demonstration lets two controllers race on the same task, and a scoreboard "
      "over 24 unseen tasks reports wins, ties and losses. The results support learned orchestration as an automatic way to find cost-aware policies, and they "
      "show where hand-tuned rules remain competitive.")
    p("<b>Keywords:</b> reinforcement learning; Double DQN; multi-agent language model systems; cost-aware orchestration; sim-to-real transfer; "
      "pre-registered evaluation.", indent=False)

    # ------------------------------------------------------------ contents (filled after the main text is registered)
    R.add("%%CONTENTS%%")

    # ============================================================ chapter 1
    heading(1, "1. INTRODUCTION", newpage=True)
    heading(2, "1.1 Background and Motivation")
    p("Applications built on large language models (LLMs) increasingly split one task across specialised agents. A researcher retrieves evidence, a solver "
      "drafts an answer, a verifier checks it and a summariser formats it [1], [2], [3]. Such pipelines can raise accuracy, but each call consumes tokens, "
      "time and money, and the cost grows with every agent that is added.")
    p("Two simple designs dominate practice. A single agent is cheap but cannot recover from its own mistakes. A fixed pipeline calls every agent on every task. "
      "Our measurements show that the second is not automatically better. In the simulator of this project the fixed pipeline earns a mean return of +0.003, "
      "which is no better than doing nothing, against +0.153 for a single agent, because most tasks do not need all four calls. On recorded outputs of two real "
      "models it also scored below the single agent.")
    p("The right amount of effort depends on the task, on what has already happened (for example whether a check passed) and on the money left. Choosing the next "
      "action from the current situation, with a reward that arrives only at the end, is a sequential decision problem. It is the setting that reinforcement "
      "learning (RL) was designed for [4].")
    heading(2, "1.2 Problem Statement")
    p("Given a task and a cost budget, choose a sequence of agent calls that ends with STOP, so as to maximise the expected answer quality minus a penalty "
      "proportional to the total cost. The controller sees only an observation of the situation (task type, whether an answer exists, the result of the last check, "
      "how many calls of each kind were made, and the money and steps left) and not the hidden correctness of the current answer. Chapter 3 gives the formal "
      "definition.")
    heading(2, "1.3 Objectives")
    numbered([
        "Formulate multi-agent orchestration as a Markov decision process with an explicit reward that is tested against loopholes.",
        "Implement a Double DQN controller and fair baselines: a single agent, a fixed pipeline and tuned rule routers.",
        "Test, with decision rules written before the runs, whether the learned controller beats the best hand-written rule in a simulator, under distribution shift and when informative signals are hidden.",
        "Repeat the comparison on recorded outputs of real language models and measure the gap between simulation and reality.",
        "Provide a live demonstration in which two controllers handle the same task side by side.",
    ])
    heading(2, "1.4 Contributions")
    bullet("A loophole-tested MDP formulation of cost-aware orchestration with action masking and a reward that charges for every call.")
    bullet("A simulator study with a documented correction history: three flaws in our own first comparison were found and fixed before any claim was made.")
    bullet("A feature study (ablations and a blind variant) showing which observations matter and that the cost penalty is essential.")
    bullet("A real-model study on two LLMs using a trace-replay environment, a calibrated cost mapping, held-out tasks and task-level paired statistics.")
    bullet("An open, tested code base (87 automated tests) and a live race dashboard in which controllers compete on the same task.")
    heading(2, "1.5 Scope and Limitations")
    p("The tasks are single-turn and automatically graded (code with hidden unit tests, numeric problems and fact questions over a small invented document set). "
      "The simulator is a stylised model of agent behaviour, and the real-model study uses 50 tasks and two models, so its confidence intervals are wide. "
      "A router that asks an LLM to choose the next agent was not built; it is listed as future work. These limits are repeated, with numbers, in Section 5.10. Chapter 2 reviews related work, Chapter 3 defines the problem and system, Chapter 4 the protocol, Chapter 5 the results, and Chapter 6 concludes.")
    # ============================================================ chapter 2
    heading(1, "2. LITERATURE REVIEW AND BACKGROUND", newpage=True)
    heading(2, "2.1 Multi-Agent Language Model Systems")
    p("AutoGen [1] lets developers compose applications from conversing agents that can use models, tools and human input. MetaGPT [2] encodes standard operating "
      "procedures into prompts and assigns roles to agents in an assembly-line style for software tasks. ReAct [3] interleaves reasoning steps with actions such as "
      "search, so that a single model alternates between thinking and acting. Retrieval-augmented generation [5] grounds an answer in retrieved documents, "
      "which motivates the researcher agent used here. These systems define what the agents do. The order in which they are called is usually written by hand.")
    heading(2, "2.2 Cost-Aware Use of Language Models")
    p("FrugalGPT [6] reduces the cost of using LLM APIs by learning a cascade: a cheap model answers first, and a learned scorer decides whether to accept the "
      "answer or escalate to a more expensive model. RouteLLM [7] trains routers that choose between a strong and a weak model for each query, using preference data. "
      "Both choose among models. This project chooses among different roles (retrieve, solve, verify, tidy, retry, stop) one step at a time, with a state that includes "
      "the result of the previous check and the budget left. That makes the problem sequential, and it is why we model it as an MDP and not as a single routing decision.")
    heading(2, "2.3 Reinforcement Learning Background")
    p("A Markov decision process (MDP) is defined by states, actions, transition probabilities and a reward; the agent seeks a policy that maximises the expected "
      "discounted return [4]. Q-learning [8] learns the value of each action in each state. The Deep Q-Network (DQN) [9] approximates this function with a neural "
      "network and stabilises learning with two devices: an experience replay buffer [10], which breaks the correlation between consecutive samples, and a slowly updated "
      "target network. Standard DQN tends to overestimate action values because the same network both selects and evaluates the best next action. Double DQN [11] removes "
      "most of this bias by letting the online network select the action and the target network evaluate it. We train with the Adam optimiser [12] and a Huber loss, and we "
      "apply action masking, which restricts every maximisation to the actions that are legal in the current state.")
    heading(2, "2.4 Simulation-to-Reality Transfer and Evaluation Practice")
    p("Policies trained in simulation often degrade when applied to the real system. Domain randomisation [13] widens the simulator so that reality looks like one more "
      "variation. We examine the gap directly by running a simulator-trained controller on recorded real traces and then retraining on those traces. "
      "Deep RL results are also sensitive to seeds, hyper-parameters and implementation details [14], and point estimates hide uncertainty [15]. We therefore use "
      "several seeds, report final checkpoints without picking the best, give confidence intervals, and compare methods on identical seeded tasks.")
    heading(2, "2.5 Research Gap")
    p("Prior work either defines multi-agent systems with hand-written control flow or routes between models once per query. We found no study that learns a sequential "
      "controller over heterogeneous roles under an explicit cost penalty, evaluates it against fair tuned baselines with decision rules fixed in advance, and reports "
      "how the policy transfers to real models. This report addresses that gap at the scale of a laboratory project.")
