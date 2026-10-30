"""Replacement-rate sweep for SynSteg-style synonym substitution.

For every saved report (cover text + per-word synonym table) we re-embed random bits
at a capped replacement rate r and evaluate a whole-text steganalyser with a split by
cover-stego pair. No LLM calls: the covers and tables are the ones already generated.
"""
import json, glob, math, random, re, sys, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.sparse import hstack
from wordfreq import zipf_frequency

REP = os.environ.get("SYNSTEG_REPORTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "reports"))
SERIES = {"luna": "gpt6_luna_missing", "qwen3": "qwen_missing_b3",
          "gpt4o": "gpt_omni_reports", "ds2023": "json_reports"}
RATES = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40]
SEEDS = [0, 1, 2]
TRAIL = re.compile(r"^(.*?)([^\w\-’']*)$")


def load(series):
    docs = []
    for fn in sorted(glob.glob(f"{REP}/{SERIES[series]}/*.json")):
        d = json.load(open(fn))
        toks = d["container"].split()
        if len(toks) != len(d["secret_key"]):
            continue
        docs.append({"id": d["uuid"], "cover": d["container"], "stego": d["encoded_message"],
                     "toks": toks, "key": d["secret_key"]})
    return docs


def eligible(tok, table):
    core = TRAIL.match(tok).group(1)
    if not table:
        return None
    base, codes = next(iter(table.items()))
    if base.lower() != core.lower():
        return None
    vals = [v for v in codes.values() if v and v.lower() != base.lower()]
    if len(vals) < 2 or len(set(v.lower() for v in vals)) != len(vals):
        return None
    return vals


def embed(doc, rate, rng):
    toks = doc["toks"]
    cand = [(i, v) for i, (t, tab) in enumerate(zip(toks, doc["key"])) if (v := eligible(t, tab))]
    k = min(len(cand), max(0, round(rate * len(toks))))
    chosen = rng.sample(cand, k) if k else []
    out, bits, zdelta = list(toks), 0, []
    for i, vals in chosen:
        core, trail = TRAIL.match(toks[i]).groups()
        new = rng.choice(vals)
        if core[:1].isupper():
            new = new[:1].upper() + new[1:]
        out[i] = new + trail
        bits += int(math.floor(math.log2(len(vals))))  # conservative: only distinct substitutes count
        zdelta.append(zipf_frequency(new.split()[0], "en") - zipf_frequency(core, "en"))
    return " ".join(out), bits, k, len(cand), zdelta


def feats_len(texts):
    return np.array([[len(t.split()), len(t), np.mean([len(w) for w in t.split()] or [0])] for t in texts])


def evaluate(covers, stegos, groups, seed):
    X = covers + stegos
    y = np.array([0] * len(covers) + [1] * len(stegos))
    g = np.array(groups + groups)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pt, pl, yt = np.zeros(len(X)), np.zeros(len(X)), y
    for tr, te in cv.split(X, y, g):
        w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200000)
        Xtr = hstack([w.fit_transform([X[i] for i in tr]), c.fit_transform([X[i] for i in tr])]).tocsr()
        Xte = hstack([w.transform([X[i] for i in te]), c.transform([X[i] for i in te])]).tocsr()
        m = LogisticRegression(C=10, max_iter=3000).fit(Xtr, y[tr])
        pt[te] = m.predict_proba(Xte)[:, 1]
        L = feats_len(X)
        mu, sd = L[tr].mean(0), L[tr].std(0) + 1e-9
        ml = LogisticRegression(max_iter=1000).fit((L[tr] - mu) / sd, y[tr])
        pl[te] = ml.predict_proba((L[te] - mu) / sd)[:, 1]
    return {"acc": accuracy_score(yt, pt > 0.5), "auc": roc_auc_score(yt, pt),
            "len_acc": accuracy_score(yt, pl > 0.5), "len_auc": roc_auc_score(yt, pl)}


def main():
    res = []
    for s in SERIES:
        docs = load(s)
        groups = [d["id"] for d in docs]
        covers = [d["cover"] for d in docs]
        nwords = sum(len(d["toks"]) for d in docs)
        settings = [("deployed", None)] + [(f"r={r:.2f}", r) for r in RATES]
        for name, r in settings:
            accs = []
            for seed in SEEDS:
                rng = random.Random(1000 * seed + 7)
                if r is None:
                    stegos = [d["stego"] for d in docs]
                    bits = ks = cands = 0
                    zd = []
                else:
                    emb = [embed(d, r, rng) for d in docs]
                    stegos = [e[0] for e in emb]
                    bits, ks, cands = sum(e[1] for e in emb), sum(e[2] for e in emb), sum(e[3] for e in emb)
                    zd = [z for e in emb for z in e[4]]
                ev = evaluate(covers, stegos, groups, seed)
                ident = sum(a == b for a, b in zip(covers, stegos))
                accs.append(ev)
                row = {"series": s, "setting": name, "rate": r, "seed": seed, "n_pairs": len(docs),
                       "identical": ident, "edit_rate": ks / nwords if r is not None else None,
                       "bpw": bits / nwords if r is not None else None,
                       "eligible_per_word": cands / nwords if r is not None else None,
                       "zipf_delta_mean": float(np.mean(zd)) if zd else None, **ev}
                res.append(row)
                print(json.dumps(row), flush=True)
    json.dump({"config": {"series": SERIES, "rates": RATES, "seeds": SEEDS,
                          "detector": "TF-IDF word1-2 + char_wb2-5, LR C=10, StratifiedGroupKFold(5) by pair",
                          "length_baseline": "LR on n_words, n_chars, mean word length"},
               "runs": res}, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
