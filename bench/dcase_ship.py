"""Shipping scorer on DCASE 2024 eval embodiment machines, few-shot, matched condition (source domain)."""
import os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from bench.fewshot import clips, DATA, CACHE, MACHINES, score_pooled_raw, score_frame_knn
from tickover_core import engine as E

def feats(name, m):
    z = np.load(os.path.join(CACHE, f"{name}__{m}.npz"))
    return {k.replace("|", os.sep): z[k] for k in z.files}

def grid(a, nb=4):
    a = np.asarray(a, np.float32); return a.reshape(a.shape[0], nb, -1)

def zscores(P, enr, idx):
    mu = P[enr].mean(0)
    sc = np.array([E._band_scores(P[enr], P[i], mu).mean() for i in idx])
    loo = np.array([E._band_scores(P[[x for x in enr if x != j]], P[j], P[[x for x in enr if x != j]].mean(0)).mean() for j in enr])
    med = np.median(loo); mad = 1.4826 * np.median(np.abs(loo - med)) + 0.15 * abs(med) + 1e-6
    return (sc - med) / mad

rows = []
for m in MACHINES:
    train, test, y, d = clips(m)
    keys = [os.path.relpath(p, DATA) for p in train + test]
    src_tr = [i for i, p in enumerate(train) if "_source_" in p]
    te = [len(train) + i for i in range(len(test)) if d[i] == 0]
    yt = y[d == 0]
    Ps = {}
    for L in ("L5", "L8", "L12"):
        F = feats(f"ced-base_{L}", m)
        G = [grid(F[k]) for k in keys]; T = min(g.shape[0] for g in G)
        Ps[L] = np.stack([E._pools(g[:T]) for g in G])
    mn = feats("mn10", m); MN = [mn[k] for k in keys]
    for k in (3, 5, 10):
        acc = {}
        for s in range(5):
            rng = np.random.default_rng(s); enr = list(rng.choice(src_tr, k, replace=False))
            zs = {L: zscores(Ps[L], enr, te) for L in Ps}
            cand = {"ced_L5": zs["L5"], "ced_L8": zs["L8"], "ced_L12": zs["L12"],
                    "ced_L5+L12": (zs["L5"] + zs["L12"]) / 2, "ced_fuse3": (zs["L5"] + zs["L8"] + zs["L12"]) / 3,
                    "mn10_pooled": score_pooled_raw([MN[i] for i in enr], [MN[i] for i in te]),
                    "mn10_frameknn": score_frame_knn([MN[i] for i in enr], [MN[i] for i in te])}
            for c, sc in cand.items():
                acc.setdefault(c, []).append((roc_auc_score(yt, sc), roc_auc_score(yt, sc, max_fpr=0.1)))
        for c, v in acc.items():
            v = np.array(v); rows.append((c, m, k, v[:, 0].mean(), v[:, 1].mean()))
import collections
agg = collections.defaultdict(list)
for c, m, k, a, p in rows: agg[(c, k)].append((m, a, p))
print(f"{'system':16s} k  meanAUC meanpAUC  " + " ".join(f"{m[:9]:>9s}" for m in MACHINES))
for (c, k), v in sorted(agg.items(), key=lambda kv: (kv[0][1], -np.mean([x[1] for x in kv[1]]))):
    per = {m: a for m, a, p in v}
    print(f"{c:16s} {k:2d}  {np.mean([x[1] for x in v]):.3f}   {np.mean([x[2] for x in v]):.3f}   " + " ".join(f"{per[m]:9.3f}" for m in MACHINES))
