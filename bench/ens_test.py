import os, sys, numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from bench.fewshot import clips, DATA, CACHE, MACHINES, _l2
from earshot_core import engine as E

def loo_z(score_fn, enr_items, test_items):
    sc = score_fn(enr_items, test_items)
    loo = np.array([score_fn(enr_items[:j] + enr_items[j+1:], [enr_items[j]])[0] for j in range(len(enr_items))])
    med = np.median(loo); mad = 1.4826*np.median(np.abs(loo-med)) + 0.15*abs(med) + 1e-6
    return (sc - med) / mad

def mn_pooled(enr, tst):
    E_ = _l2(np.stack([f.mean(0) for f in enr])); T_ = _l2(np.stack([f.mean(0) for f in tst]))
    return 1 - (T_ @ E_.T).max(1)

def mn_frame(enr, tst, q=0.9):
    B = _l2(np.concatenate(enr)); return np.array([np.quantile(1 - (_l2(f) @ B.T).max(1), q) for f in tst])

def ced_fn(enr, tst):
    mu = np.stack(enr).mean(0)
    return np.array([E._band_scores(np.stack(enr), t, mu).mean() for t in tst])

def feats(name, m):
    z = np.load(os.path.join(CACHE, f"{name}__{m}.npz")); return {k.replace("|", os.sep): z[k] for k in z.files}

def _main():
  pass
if __name__ == "__main__":
    out = {}
    for m in MACHINES:
        train, test, y, d = clips(m)
        keys = [os.path.relpath(p, DATA) for p in train + test]
        src = [i for i, p in enumerate(train) if "_source_" in p]
        te = [len(train) + i for i in range(len(test)) if d[i] == 0]; yt = y[d == 0]
        F = feats("ced-base_L5", m); G = [np.asarray(F[k], np.float32).reshape(-1, 4, 768) for k in keys]
        T = min(g.shape[0] for g in G); P = [E._pools(g[:T]) for g in G]
        MN = [feats("mn10", m)[k] for k in keys]
        for k in (5, 10):
            for s in range(5):
                rng = np.random.default_rng(s); enr = list(rng.choice(src, k, replace=False))
                zc = loo_z(ced_fn, [P[i] for i in enr], [P[i] for i in te])
                zm = loo_z(mn_pooled, [MN[i] for i in enr], [MN[i] for i in te])
                zf = loo_z(mn_frame, [MN[i] for i in enr], [MN[i] for i in te])
                for nm, sc in {"ced": zc, "mn_pool": zm, "mn_frame": zf, "ced+mn_pool": zc + zm, "ced+mn_frame": zc + zf, "all3": zc + zm + zf}.items():
                    out.setdefault((nm, k), {}).setdefault(m, []).append(roc_auc_score(yt, sc))
    for (nm, k), v in sorted(out.items(), key=lambda kv: (kv[0][1], -np.mean([np.mean(x) for x in kv[1].values()]))):
        print(f"{nm:14s} {k:2d} {np.mean([np.mean(x) for x in v.values()]):.3f}  " + " ".join(f"{np.mean(v[m]):.3f}" for m in MACHINES))
