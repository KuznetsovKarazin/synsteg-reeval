import json, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
out = sys.argv[2] if len(sys.argv) > 2 else "fig_tradeoff.pdf"
g = defaultdict(list)
for r in d["runs"]:
    g[(r["series"], r["setting"])].append(r)
names = {"ds2023": "DS-2023", "gpt4o": "GPT-4o", "luna": "Luna", "qwen3": "Qwen3"}
style = {"ds2023": ("o", "-", "0.0"), "gpt4o": ("s", "--", "0.35"), "luna": ("^", "-.", "0.55"), "qwen3": ("D", ":", "0.2")}
mean = lambda v, f: sum(x[f] for x in v) / len(v)
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(5.0, 3.4))
for s, nm in names.items():
    pts = sorted([(mean(v, "bpw"), mean(v, "auc"), min(x["auc"] for x in v), max(x["auc"] for x in v), v[0]["rate"])
                  for (ss, st), v in g.items() if ss == s and v[0]["rate"] is not None])
    if not pts:
        continue
    mk, ls, col = style[s]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.errorbar(xs, ys, yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]], marker=mk, ls=ls,
                color=col, ms=4, lw=1, capsize=2, label=nm)
    if s == "luna":
        for p in pts:
            ax.annotate(f"{p[4]:.0%}", (p[0], p[1]), textcoords="offset points", xytext=(-20, 3), fontsize=7, color="0.3")
ax.axhline(0.5, color="0.7", lw=0.8)
ax.set_xlabel("Capacity, bits per cover word")
ax.set_ylabel("Steganalysis AUC (whole text, TF-IDF + LR)")
ax.set_ylim(0.45, 1.02)
ax.set_xlim(left=0)
ax.legend(frameon=False, loc="lower right", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"), dpi=150)
