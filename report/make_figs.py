"""Charts for the report (matplotlib, report palette). Numbers are copied from docs/logs (see Appendix C of the report)."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

os.makedirs("report/img", exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#47565A", "axes.labelcolor": "#2B373B", "xtick.color": "#2B373B", "ytick.color": "#2B373B"})
SLATE, PERI, GREY, COPPER, OLIVE = "#47565A", "#5E6FA8", "#ADB6CE", "#A5601F", "#7C8120"


def save(fig, name):
    fig.savefig("report/img/%s.png" % name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---- Fig 3.1 orchestration loop
fig, ax = plt.subplots(figsize=(6.4, 1.75))
ax.set_xlim(0, 100); ax.set_ylim(0, 25); ax.axis("off")
boxes = ["Task and\nbudget\narrive", "Controller\nobserves 16 numbers,\npicks a legal\naction", "Agent runs,\ncost is\ncharged", "State updates:\nanswer, check,\nmoney left", "STOP: answer\ngraded, reward\npaid"]
xs = [1, 21, 41, 61, 81]
for i, (x, t) in enumerate(zip(xs, boxes)):
    ax.add_patch(FancyBboxPatch((x, 5), 17.5, 15, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#E8ECF6" if i not in (1, 4) else "#DDE3F2", ec=SLATE, lw=1))
    ax.text(x + 8.75, 12.5, t, ha="center", va="center", fontsize=7.3, color="#2B373B")
    if i < 4:
        ax.add_patch(FancyArrowPatch((x + 17.9, 12.5), (xs[i + 1] - 0.4, 12.5), arrowstyle="-|>", mutation_scale=10, color=SLATE, lw=1.2))
ax.annotate("", xy=(30, 4.6), xytext=(70, 4.6), arrowprops=dict(arrowstyle="-|>", color=OLIVE, lw=1.2, connectionstyle="arc3,rad=-0.0"))
ax.text(50, 1.6, "repeat until the controller chooses STOP, or the budget or step limit is reached", ha="center", fontsize=8, style="italic", color=OLIVE)
save(fig, "fig_loop")

# ---- Fig 5.1 learning curve
lc = json.load(open("report/lc_summary.json"))
fig, ax = plt.subplots(figsize=(6.2, 2.2))
x = [e / 1000 for e in lc["eps"]]
ax.fill_between(x, lc["min"], lc["max"], color=GREY, alpha=0.55, label="lowest to highest seed")
ax.plot(x, lc["mean"], color=SLATE, lw=2, label="mean of 5 seeds")
ax.axhline(0.1732, color=COPPER, ls="--", lw=1.1); ax.text(61, 0.176, "best rule", color=COPPER, fontsize=8, ha="right", va="bottom")
ax.set_xlabel("training episodes (thousands)"); ax.set_ylabel("development return"); ax.set_xlim(2, 60); ax.set_ylim(-0.06, 0.24)
ax.legend(frameon=False, loc="lower right", fontsize=8); ax.grid(axis="y", color="#E3E6EB", lw=0.6)
save(fig, "fig_curve")

# ---- Fig 5.2 simulator main result + ablation (two panels)
fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.25), gridspec_kw={"width_ratios": [1, 1.05]})
names = ["Fixed\npipeline", "Single\nagent", "Best\nrule", "Double\nDQN"]; vals = [0.0034, 0.1528, 0.1732, 0.189]
bars = a.bar(names, vals, color=[GREY, GREY, GREY, SLATE], width=0.62)
for r, v in zip(bars, vals): a.text(r.get_x() + r.get_width() / 2, v + 0.004, "%+.3f" % v, ha="center", fontsize=8.5)
a.set_ylabel("mean return"); a.set_ylim(0, 0.23); a.set_title("(a) Simulator, 10,000 test tasks", fontsize=9.5, loc="left"); a.tick_params(axis="x", labelsize=8)
lab = ["A: no cost penalty", "B: no memory", "C1: no difficulty", "C2: no confidence", "C3: neither signal"]; dv = [-0.312, -0.021, -0.010, -0.008, -0.012]
y = list(range(len(lab)))[::-1]
br = b.barh(y, dv, color=[COPPER, GREY, GREY, GREY, GREY], height=0.6)
for yy, v in zip(y, dv): b.text(v - 0.008, yy, "%+.3f" % v, ha="right", va="center", fontsize=8.5)
b.axvline(0, color=SLATE, lw=0.8); b.set_yticks(y); b.set_yticklabels(lab, fontsize=8); b.yaxis.tick_right(); b.set_xlim(-0.40, 0.0)
b.spines["left"].set_visible(False); b.spines["right"].set_visible(True); b.set_xlabel("change in return"); b.set_title("(b) Ablations (training world)", fontsize=9.5, loc="left")
fig.tight_layout(); save(fig, "fig_sim")

# ---- Fig 5.3 real-model comparison
pol = ["Single\nagent", "Fixed\npipeline", "Tuned\nrule", "Typed\nrule", "DDQN\nzero-shot", "DDQN\nrefit"]
q = [0.462, 0.336, 0.569, 0.576, 0.438, 0.580]; g = [0.563, 0.282, 0.670, 0.819, 0.530, 0.770]
fig, ax = plt.subplots(figsize=(6.4, 2.3))
w = 0.38; xs = range(len(pol))
b1 = ax.bar([i - w / 2 for i in xs], q, w, color=GREY, label="qwen2.5:3b"); b2 = ax.bar([i + w / 2 for i in xs], g, w, color=SLATE, label="gpt-oss-20b")
for r, v in list(zip(b1, q)) + list(zip(b2, g)): ax.text(r.get_x() + r.get_width() / 2, v + 0.012, "%.2f" % v, ha="center", fontsize=7.8)
ax.set_xticks(list(xs)); ax.set_xticklabels(pol, fontsize=8); ax.set_ylabel("return on 24 held-out tasks"); ax.set_ylim(0, 0.95)
ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8.5)
save(fig, "fig_real")

# ---- screenshots: shrink to a sensible size and compress (JPEG)
for n in ("full", "task", "running", "result", "detail"):
    im = Image.open("report/img/shot_%s.png" % n).convert("RGB")
    if n == "detail":
        im = im.crop((0, 0, im.width, 1330))
    r = 1700 / im.width
    im = im.resize((1700, int(im.height * r)), Image.LANCZOS)
    im.save("report/img/shot_%s.jpg" % n, quality=84, optimize=True)
    print(n, im.size, os.path.getsize("report/img/shot_%s.jpg" % n) // 1024, "KB")
print("figures done")
