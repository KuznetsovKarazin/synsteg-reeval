"""Key selection (16 bit->synonym permutations, b=5) against the whole-text TF-IDF+LR steganalyser.
Uses corpus.jsonl of Chernov's experiment (re-built bit-exactly) and his detector scores.
"""
import json, sys, numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold
from scipy.stats import rankdata
AID = "paper/data/aidet"
K = 16
recs = [json.loads(l) for l in open(f"{AID}/corpus.jsonl")]
cover = {r["uuid"]: r["text"] for r in recs if r["arm"] == "cover"}
human = {r["uuid"]: r["text"] for r in recs if r["arm"] == "human"}
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

def cv_eval(stego_of, seed):
    """grouped 5-fold: train cover vs stego_of[u], test on held-out pairs. returns auc, acc, model per fold"""
    rng = np.random.default_rng(seed); order = rng.permutation(len(uu)); g = np.empty(len(uu), int); g[order] = np.arange(len(uu)) % 5
    sc, y, pr = [], [], []
    models = {}
    for f in range(5):
        tr = [u for u, gg in zip(uu, g) if gg != f]; te = [u for u, gg in zip(uu, g) if gg == f]
        m = warden().fit([cover[u] for u in tr] + [stego_of[u] for u in tr], [0] * len(tr) + [1] * len(tr))
        p = m.predict_proba([cover[u] for u in te] + [stego_of[u] for u in te])[:, 1]
        sc += list(p); y += [0] * len(te) + [1] * len(te); models[f] = (m, te)
    sc = np.array(sc)
    return roc_auc_score(y, sc), accuracy_score(y, sc > .5), models

out = {}
for seed in (0, 1, 2):
    res = {}
    rand = {u: cand[u][0] for u in uu}
    auc0, acc0, models = cv_eval(rand, seed)
    res["random key (k=0)"] = (auc0, acc0)
    # white-box selection: model of the fold that holds u (trained without u) picks min-score candidate
    sel = {}
    for f, (m, te) in models.items():
        for u in te:
            p = m.predict_proba([cand[u][k] for k in range(K)])[:, 1]; sel[u] = int(np.argmin(p))
    # evaluate the selected texts with the SAME fold models (non-adaptive warden)
    sc, y = [], []
    for f, (m, te) in models.items():
        sc += list(m.predict_proba([cover[u] for u in te] + [cand[u][sel[u]] for u in te])[:, 1]); y += [0]*len(te) + [1]*len(te)
    res["LR-selected, same warden"] = (roc_auc_score(y, sc), accuracy_score(y, np.array(sc) > .5))
    # warden retrained on selected texts
    a, c, _ = cv_eval({u: cand[u][sel[u]] for u in uu}, seed)
    res["LR-selected, retrained warden"] = (a, c)
    # selection by AI-detector ensemble (mean rank, Chernov's rule) evaluated by LR
    allc = [(u, k) for u in uu for k in range(K)]
    ranks = {d: dict(zip(allc, rankdata([S[d][("cand_b5", u, k)] for u, k in allc]))) for d in DETS}
    ens = {u: int(np.argmin([np.mean([ranks[d][(u, k)] for d in DETS]) for k in range(K)])) for u in uu}
    a, c, _ = cv_eval({u: cand[u][ens[u]] for u in uu}, seed)
    res["AI-ensemble-selected, retrained warden"] = (a, c)
    # what LR selection does to AI detectors (AUC vs human, as in dissertation)
    hum = {d: [S[d][("human", u, -1)] for u in uu] for d in DETS}
    for d in DETS:
        for name, ch in (("random", {u: 0 for u in uu}), ("LR-selected", sel)):
            x = [S[d][("cand_b5", u, ch[u])] for u in uu]
            res[f"{d[:2]} AUC vs human, {name}"] = (roc_auc_score([0]*len(uu)+[1]*len(uu), hum[d]+x), None)
    out[seed] = res
    print(seed, {k: tuple(round(v, 3) if v is not None else None for v in t) for k, t in res.items()}, flush=True)
json.dump({s: {k: list(v) for k, v in r.items()} for s, r in out.items()}, open("paper/results/aidet/keysel_lr.json", "w"), indent=1)
