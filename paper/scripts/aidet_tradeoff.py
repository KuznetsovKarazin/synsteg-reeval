"""Trade-off across b=1..5: replacement share, AI-detector AUC (vs human, vs cover), whole-text steganalysis AUC."""
import json, difflib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import roc_auc_score, accuracy_score
AID = "paper/data/aidet"
recs = [json.loads(l) for l in open(f"{AID}/corpus.jsonl")]
T = {(r["arm"], r["uuid"]): r for r in recs if r["arm"] != "cand_b5"}
uu = sorted(u for a, u in T if a == "cover")
DETS = ["D1_roberta_openai", "D2_chatgpt_roberta", "D3_desklib", "D4_binoculars", "D5_fast_detectgpt"]
S = {d: {} for d in DETS}
for d in DETS:
    for l in open(f"{AID}/scores/{d}.jsonl"):
        x = json.loads(l); S[d][(x["arm"], x["uuid"])] = x["score"]
def warden():
    return make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                                    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
                         LogisticRegression(C=10, max_iter=3000))
def changed(a, b):
    a, b = a.split(), b.split(); m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return 1 - sum(x.size for x in m.get_matching_blocks()) / len(a)
def stegan(arm, seed):
    rng = np.random.default_rng(seed); g = np.empty(len(uu), int); g[rng.permutation(len(uu))] = np.arange(len(uu)) % 5
    sc, y = [], []
    for f in range(5):
        tr = [u for u, gg in zip(uu, g) if gg != f]; te = [u for u, gg in zip(uu, g) if gg == f]
        m = warden().fit([T["cover", u]["text"] for u in tr] + [T[arm, u]["text"] for u in tr], [0]*len(tr) + [1]*len(tr))
        sc += list(m.predict_proba([T["cover", u]["text"] for u in te] + [T[arm, u]["text"] for u in te])[:, 1]); y += [0]*len(te) + [1]*len(te)
    return roc_auc_score(y, sc), accuracy_score(y, np.array(sc) > .5)
out = {}
for arm in ["cover"] + [f"stego_b{b}" for b in range(1, 6)]:
    r = {}
    r["bpw"] = float(np.mean([T[arm, u]["bits"] / T[arm, u]["n_words"] for u in uu]))
    r["changed"] = float(np.mean([changed(T["cover", u]["text"], T[arm, u]["text"]) for u in uu]))
    for d in DETS:
        h = [S[d]["human", u] for u in uu]; c = [S[d]["cover", u] for u in uu]; x = [S[d][arm, u] for u in uu]
        lab = [0]*len(uu) + [1]*len(uu)
        r[d[:2]] = dict(auc_vs_human=roc_auc_score(lab, h + x), auc_cover_vs_stego=roc_auc_score(lab, c + x),
                        tpr5=float(np.mean(np.array(x) > np.quantile(h, .95))))
    if arm != "cover":
        v = [stegan(arm, s) for s in (0, 1, 2)]
        r["steg_auc"], r["steg_acc"] = float(np.mean([a for a, _ in v])), float(np.mean([c for _, c in v]))
    out[arm] = r
    print(arm, json.dumps(r), flush=True)
json.dump(out, open("paper/results/aidet/tradeoff.json", "w"), indent=1)
