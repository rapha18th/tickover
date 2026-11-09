import os, sys, numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from bench.fewshot import clips, DATA, CACHE, MACHINES, _l2
from bench.ens_test import loo_z, mn_frame, mn_pooled, ced_fn, feats
from nzeve_core import engine as E

def tok_frame(enr, tst, q=0.9):
    """Per-band token 1-NN, centred; q-quantile over (time, band) tokens."""
    bank = np.concatenate(enr)  # (N*T, 4, D)
    mu = bank.mean(0)
    out = []
    for t in tst:
        s = []
        for b in range(t.shape[1]):
            B = _l2(bank[:, b] - mu[b]); Q = _l2(t[:, b] - mu[b])
            s.append(1 - (Q @ B.T).max(1))
        out.append(np.quantile(np.concatenate(s), q))
    return np.array(out)

out = {}
for m in MACHINES:
    train, test, y, d = clips(m)
    keys = [os.path.relpath(p, DATA) for p in train + test]
    src = [i for i, p in enumerate(train) if "_source_" in p]
    te = [len(train) + i for i in range(len(test)) if d[i] == 0]; yt = y[d == 0]
    Gs = {}
    for L in ("L5", "L12"):
        F = feats(f"ced-base_{L}", m); G = [np.asarray(F[k], np.float32).reshape(-1, 4, 768) for k in keys]
        T = min(g.shape[0] for g in G); Gs[L] = [g[:T] for g in G]
    P5 = [E._pools(g) for g in Gs["L5"]]
    MN = [feats("mn10", m)[k] for k in keys]
    for k in (5, 10):
        for s in range(5):
            rng = np.random.default_rng(s); enr = list(rng.choice(src, k, replace=False))
            zc = loo_z(ced_fn, [P5[i] for i in enr], [P5[i] for i in te])
            zt5 = loo_z(tok_frame, [Gs["L5"][i] for i in enr], [Gs["L5"][i] for i in te])
            zt12 = loo_z(tok_frame, [Gs["L12"][i] for i in enr], [Gs["L12"][i] for i in te])
            zf = loo_z(mn_frame, [MN[i] for i in enr], [MN[i] for i in te])
            zm = loo_z(mn_pooled, [MN[i] for i in enr], [MN[i] for i in te])
            for nm, sc in {"ced": zc, "tok5": zt5, "tok12": zt12, "ced+tok5": zc + zt5, "ced+tok12": zc + zt12,
                           "ced+tok5+tok12": zc + zt5 + zt12, "mn_all3": zc + zm + zf, "all5": zc + zt5 + zt12 + zm + zf}.items():
                out.setdefault((nm, k), {}).setdefault(m, []).append(roc_auc_score(yt, sc))
for (nm, k), v in sorted(out.items(), key=lambda kv: (kv[0][1], -np.mean([np.mean(x) for x in kv[1].values()]))):
    print(f"{nm:16s} {k:2d} {np.mean([np.mean(x) for x in v.values()]):.3f}  " + " ".join(f"{np.mean(v[m]):.3f}" for m in MACHINES))
