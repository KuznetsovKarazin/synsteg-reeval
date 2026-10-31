import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1])); out = sys.argv[2]
arms = [f"stego_b{b}" for b in range(1, 6)]
ch = [d[a]["changed"] for a in arms]
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
st = {"D1": ("o", "-", "0.0"), "D2": ("s", "--", "0.3"), "D3": ("^", "-.", "0.5"), "D4": ("D", ":", "0.2"), "D5": ("v", "-", "0.65")}
lab = {"D1": "D1 RoBERTa (GPT-2)", "D2": "D2 HC3", "D3": "D3 DeBERTa (RAID)", "D4": "D4 Binoculars", "D5": "D5 Fast-DetectGPT"}
for k, (mk, ls, c) in st.items():
    axs[0].plot(ch, [d[a][k]["auc_vs_human"] for a in arms], marker=mk, ls=ls, color=c, ms=4, lw=1, label=lab[k])
    axs[0].scatter([0], [d["cover"][k]["auc_vs_human"]], marker=mk, color=c, s=14)
    axs[1].plot(ch, [max(v, 1 - v) for v in (d[a][k]["auc_cover_vs_stego"] for a in arms)], marker=mk, ls=ls, color=c, ms=4, lw=1)
axs[1].plot(ch, [d[a]["steg_auc"] for a in arms], marker="*", color="k", ms=7, lw=1.5, label="TF-IDF + LR (trained)")
for ax in axs:
    ax.axhline(0.5, color="0.75", lw=0.8); ax.set_xlabel("Share of changed words"); ax.set_xlim(-0.03, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("AUC"); axs[0].set_ylim(0, 1.03)
axs[0].set_title("(a) Machine text vs human text", fontsize=9)
axs[1].set_title("(b) Stego vs cover, two-sided", fontsize=9)
h0, l0 = axs[0].get_legend_handles_labels(); h1, l1 = axs[1].get_legend_handles_labels()
axs[1].legend(h0 + h1, l0 + l1, frameon=False, fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
