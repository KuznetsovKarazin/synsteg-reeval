"""Ablation without phrasal substitutes and cross-series transfer of the whole-text detector."""
import json, random, sys, re, math
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
import sweep
from sweep import load, SERIES, TRAIL
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.sparse import hstack

def single_only(doc):
    d = dict(doc); k = []
    for tab in doc["key"]:
        if tab:
            base, codes = next(iter(tab.items()))
            k.append({base: {c: v for c, v in codes.items() if v and " " not in v.strip()}})
        else:
            k.append(tab)
    d["key"] = k
    return d

def feats(tr, te):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)
    A = hstack([w.fit_transform(tr), c.fit_transform(tr)]).tocsr()
    B = hstack([w.transform(te), c.transform(te)]).tocsr()
    return A, B

res = {"ablation": {}, "transfer": {}}
data = {s: load(s) for s in SERIES}
for s, docs in data.items():
    covers = [d["cover"] for d in docs]; groups = [d["id"] for d in docs]
    sdocs = [single_only(d) for d in docs]
    res["ablation"][s] = {}
    for rate in (0.05, 0.10, 0.20):
        out = {}
        for name, dd in (("all", docs), ("single", sdocs)):
            rng = random.Random(7)
            emb = [sweep.embed(d, rate, rng) for d in dd]
            st = [e[0] for e in emb]; bits = sum(e[1] for e in emb); nw = sum(len(d["toks"]) for d in dd)
            ev = sweep.evaluate(covers, st, groups, 0)
            out[name] = {"auc": ev["auc"], "acc": ev["acc"], "len_acc": ev["len_acc"], "bpw": bits / nw}
        res["ablation"][s][rate] = out
        print(s, rate, json.dumps(out), flush=True)
# transfer: train on deployed series A (all pairs), test on deployed series B
for a in SERIES:
    for b in SERIES:
        if a == b: continue
        A = data[a]; B = data[b]
        Xtr = [d["cover"] for d in A] + [d["stego"] for d in A]; ytr = [0]*len(A) + [1]*len(A)
        Xte = [d["cover"] for d in B] + [d["stego"] for d in B]; yte = [0]*len(B) + [1]*len(B)
        Ftr, Fte = feats(Xtr, Xte)
        m = LogisticRegression(C=10, max_iter=3000).fit(Ftr, ytr)
        p = m.predict_proba(Fte)[:, 1]
        res["transfer"][f"{a}->{b}"] = {"acc": accuracy_score(yte, p > .5), "auc": roc_auc_score(yte, p)}
        print(a, b, res["transfer"][f"{a}->{b}"], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
