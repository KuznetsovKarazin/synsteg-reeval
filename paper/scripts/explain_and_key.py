"""Explainability, confidence intervals and key information cost for the deployed series.

1. Pair-bootstrap 95% CI of whole-text TF-IDF+LR accuracy/AUC (out-of-fold, seed 0).
2. Interpretable detectors: word-unigram LR (top coefficients) and a 9-feature Zipf-histogram LR.
3. Key information cost: compact encoding of the tables vs the zlib-JSON key.
"""
import json, math, random, sys, zlib
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.sparse import hstack
from wordfreq import zipf_frequency
sys.path.insert(0, __import__("os").path.dirname(__file__))
from sweep import load, embed, SERIES

BINS = [0, 1, 2, 3, 4, 5, 6, 7, 99]


def zipf_hist(t):
    ws = [w.strip(".,;:!?\"'()[]").lower() for w in t.split()]
    ws = [w for w in ws if w]
    z = [zipf_frequency(w, "en") for w in ws]
    h = np.histogram(z, bins=BINS)[0] / max(1, len(z))
    return list(h) + [float(np.mean(z))]


def oof(X, y, g, seed, kind):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for tr, te in cv.split(np.zeros(len(y)), y, g):
        if kind == "full":
            w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)
            A = hstack([w.fit_transform([X[i] for i in tr]), c.fit_transform([X[i] for i in tr])]).tocsr()
            B = hstack([w.transform([X[i] for i in te]), c.transform([X[i] for i in te])]).tocsr()
            m = LogisticRegression(C=10, max_iter=3000)
        elif kind == "word1":
            w = TfidfVectorizer(ngram_range=(1, 1), min_df=2, sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
            A, B = w.fit_transform([X[i] for i in tr]), w.transform([X[i] for i in te])
            m = LogisticRegression(C=10, max_iter=3000)
        else:  # zipf histogram
            H = np.array(X)
            mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-9
            A, B = (H[tr] - mu) / sd, (H[te] - mu) / sd
            m = LogisticRegression(max_iter=2000)
        p[te] = m.fit(A, y[tr]).predict_proba(B)[:, 1]
    return p


def boot(y, p, g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    accs, aucs = [], []
    for _ in range(n):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        accs.append(accuracy_score(y[s], p[s] > 0.5)); aucs.append(roc_auc_score(y[s], p[s]))
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)),
            float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def top_words(covers, stegos, k=12):
    w = TfidfVectorizer(ngram_range=(1, 1), min_df=3, sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    A = w.fit_transform(covers + stegos)
    y = np.array([0] * len(covers) + [1] * len(stegos))
    m = LogisticRegression(C=10, max_iter=3000).fit(A, y)
    v = np.array(w.get_feature_names_out()); c = m.coef_[0]
    o = np.argsort(c)
    st, cv = list(v[o[::-1][:k]]), list(v[o[:k]])
    zs = float(np.mean([zipf_frequency(x, "en") for x in v[o[::-1][:50]]]))
    zc = float(np.mean([zipf_frequency(x, "en") for x in v[o[:50]]]))
    # correlation of coefficient with Zipf frequency over all features
    zall = np.array([zipf_frequency(x, "en") for x in v])
    r = float(np.corrcoef(c, zall)[0, 1])
    return st, cv, zs, zc, r


def key_cost(docs, series):
    b = {"luna": 5, "qwen3": 3}[series]
    vocab = Counter()
    for d in docs:
        for tab in d["key"]:
            if tab:
                for base, codes in tab.items():
                    vocab.update(v.lower() for v in codes.values() if v)
    V = len(vocab); tot = sum(vocab.values())
    H = -sum(c / tot * math.log2(c / tot) for c in vocab.values())
    rows = []
    for d in docs:
        mbits = len(d["message"])
        used = math.ceil(mbits / b)
        n = len(d["toks"])
        ideal_V = n + used * (2 ** b - 1) * math.ceil(math.log2(V))
        ideal_H = n + used * (2 ** b - 1) * H
        zl = 8 * len(zlib.compress(json.dumps(d["key"], separators=(",", ":"), ensure_ascii=False).encode(), 9))
        rows.append((mbits, zl / mbits, ideal_V / mbits, ideal_H / mbits))
    out = {}
    for L in sorted(set(r[0] for r in rows)):
        rr = [r for r in rows if r[0] == L]
        out[L] = [float(np.median([r[i] for r in rr])) for i in (1, 2, 3)]
    return {"V": V, "H_bits": H, "by_len": out}


def main():
    res = {}
    for s in SERIES:
        docs = load(s)
        if s in ("luna", "qwen3"):
            import glob
            from sweep import REP
            msg = {json.load(open(f))["uuid"]: json.load(open(f))["message"] for f in glob.glob(f"{REP}/{SERIES[s]}/*.json")}
            for d in docs:
                d["message"] = msg[d["id"]]
        covers = [d["cover"] for d in docs]; stegos = [d["stego"] for d in docs]
        g = np.array([d["id"] for d in docs] * 2)
        y = np.array([0] * len(docs) + [1] * len(docs))
        X = covers + stegos
        r = {"n_pairs": len(docs)}
        p = oof(X, y, g, 0, "full")
        r["full"] = {"acc": accuracy_score(y, p > .5), "auc": roc_auc_score(y, p), "ci": boot(y, p, g)}
        p = oof(X, y, g, 0, "word1")
        r["word1"] = {"acc": accuracy_score(y, p > .5), "auc": roc_auc_score(y, p)}
        H = [zipf_hist(t) for t in X]
        p = oof(H, y, g, 0, "zipf")
        r["zipf9"] = {"acc": accuracy_score(y, p > .5), "auc": roc_auc_score(y, p), "ci": boot(y, p, g)}
        st, cv, zs, zc, rc = top_words(covers, stegos)
        r["top"] = {"stego": st, "cover": cv, "zipf_top50_stego": zs, "zipf_top50_cover": zc, "corr_coef_zipf": rc}
        # zipf-histogram detector on the sweep at low rates
        r["zipf9_sweep"] = {}
        for rate in (0.05, 0.10, 0.20):
            rng = random.Random(7)
            st2 = [embed(d, rate, rng)[0] for d in docs]
            H2 = [zipf_hist(t) for t in covers + st2]
            p = oof(H2, y, g, 0, "zipf")
            r["zipf9_sweep"][rate] = {"acc": accuracy_score(y, p > .5), "auc": roc_auc_score(y, p)}
        if s in ("luna", "qwen3"):
            r["key"] = key_cost(docs, s)
        res[s] = r
        print(s, json.dumps(r, default=str)[:1500], flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
