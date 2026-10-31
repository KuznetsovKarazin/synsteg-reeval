"""Guide x evaluator matrix for key selection among K=16 permutations (b=5).
Evaluators: D1..D5 = AUC vs human (as in the dissertation); LR = whole-text TF-IDF+LR cover vs stego, grouped 5-fold, retrained on the selected texts.
Guides: random (k=0), D1..D5, ensemble (mean rank), LR (white-box, fold model trained without the text)."""
import json, numpy as np
from collections import defaultdict
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
AID = "paper/data/aidet"; K = 16
recs = [json.loads(l) for l in open(f"{AID}/corpus.jsonl")]
cover = {r["uuid"]: r["text"] for r in recs if r["arm"] == "cover"}
cand = defaultdict(dict)
for r in recs:
    if r["arm"] == "cand_b5": cand[r["uuid"]][r["cand"]] = r["text"]
uu = sorted(cover)
DETS = ["D1_roberta_openai", "D2_chatgpt_roberta", "D3_desklib", "D4_binoculars", "D5_fast_detectgpt"]
S = {d: {} for d in DETS}
for d in DETS:
    for l in open(f"{AID}/scores/{d}.jsonl"):
        x = json.loads(l); S[d][(x["arm"], x["uuid"], x["cand"])] = x["score"]
def warden():
    return make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                                    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
                         LogisticRegression(C=10, max_iter=3000))
rng = np.random.default_rng(0); g = np.empty(len(uu), int); g[rng.permutation(len(uu))] = np.arange(len(uu)) % 5
folds = [([u for u, gg in zip(uu, g) if gg != f], [u for u, gg in zip(uu, g) if gg == f]) for f in range(5)]
def lr_auc(st):
    sc, y = [], []
    for tr, te in folds:
        m = warden().fit([cover[u] for u in tr] + [st[u] for u in tr], [0]*len(tr) + [1]*len(tr))
        sc += list(m.predict_proba([cover[u] for u in te] + [st[u] for u in te])[:, 1]); y += [0]*len(te) + [1]*len(te)
    return roc_auc_score(y, sc)
sel = {"random": {u: 0 for u in uu}}
for d in DETS:
    sel[d[:2]] = {u: int(np.argmin([S[d][("cand_b5", u, k)] for k in range(K)])) for u in uu}
allc = [(u, k) for u in uu for k in range(K)]
ranks = {d: dict(zip(allc, rankdata([S[d][("cand_b5", u, k)] for u, k in allc]))) for d in DETS}
sel["ensemble"] = {u: int(np.argmin([np.mean([ranks[d][(u, k)] for d in DETS]) for k in range(K)])) for u in uu}
# LR guide: fold models trained on random candidate k=0
s = {}
for tr, te in folds:
    m = warden().fit([cover[u] for u in tr] + [cand[u][0] for u in tr], [0]*len(tr) + [1]*len(tr))
    for u in te: s[u] = int(np.argmin(m.predict_proba([cand[u][k] for k in range(K)])[:, 1]))
sel["LR"] = s
out = {}
for name, ch in sel.items():
    row = {}
    for d in DETS:
        h = [S[d][("human", u, -1)] for u in uu]; x = [S[d][("cand_b5", u, ch[u])] for u in uu]
        row[d[:2]] = roc_auc_score([0]*len(uu) + [1]*len(uu), h + x)
    row["LR"] = lr_auc({u: cand[u][ch[u]] for u in uu})
    out[name] = row; print(name, {k: round(v, 3) for k, v in row.items()}, flush=True)
json.dump(out, open("paper/results/aidet/matrix.json", "w"), indent=1)
