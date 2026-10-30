import json, sys, statistics as st, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1]))
cfg = d["config"]
names = {"old_8662": "DeepStego", "gpt4o_1231": "GPT-4o", "gpt6_luna_b5": "Luna", "qwen3_30b_b3": "Qwen3"}
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(4.6, 3.2))
xs = [i / 100 for i in range(0, 51)]
ax.plot(xs, [1 - x / 2 for x in xs], color="0.3", lw=1, label="Bound $1-r/2$")
markers = {"BERT-LSTM-Att": ("o", "black"), "TS-CNN": ("s", "0.55")}
for det, (mk, col) in markers.items():
    for s, nm in names.items():
        a = [r["test"]["acc"] for r in d["runs"] if r["series"] == s and r["detector"] == det]
        r_ = cfg["identical_pairs"][s] / cfg["n_pairs"][s]
        ax.errorbar(r_, st.mean(a), yerr=st.pstdev(a), fmt=mk, color=col, ms=5, capsize=2,
                    label=det if s == "old_8662" else None)
        if det == "BERT-LSTM-Att":
            ax.annotate(nm, (r_, st.mean(a)), textcoords="offset points", xytext=(6, -10 if nm != "GPT-4o" else -12), fontsize=8)
ax.set_xlabel("Share of unchanged stego sentences, $r$")
ax.set_ylabel("Test accuracy (protocol A)")
ax.set_xlim(-0.02, 0.5); ax.set_ylim(0.5, 1.02)
ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("fig_ceiling.pdf")

