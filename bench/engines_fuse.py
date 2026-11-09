import glob, os, sys, re
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from nzeve_core import engine as E
D = r"C:\Users\USER\Projects\earshot-data\engines\hf_car\MyDataSet"
files = sorted(glob.glob(os.path.join(D, "*.wav")))
lab = [re.sub(r"_\d+\.wav$", "", os.path.basename(f)) for f in files]
Ps = {}
for L in ("L5", "L8", "L12"):
    G = list(np.load(f"bench/cache/engines_{L}.npz", allow_pickle=True)["G"])
    T = min(g.shape[0] for g in G)
    Ps[L] = np.stack([E._pools(np.asarray(g[:T], dtype=np.float32)) for g in G])
h = [i for i, l in enumerate(lab) if l == "healthy"]
def layer_scores(P, enr):
    mu = P[enr].mean(0)
    sc = np.array([E._band_scores(P[enr], P[i], mu).mean() for i in range(len(P))])
    loo = []
    for j in enr:
        keep = [x for x in enr if x != j]
        loo.append(E._band_scores(P[keep], P[j], P[keep].mean(0)).mean())
    loo = np.array(loo); med = np.median(loo); mad = 1.4826*np.median(np.abs(loo-med)) + 0.15*abs(med) + 1e-6
    return (sc - med) / mad
for combo in (("L8",), ("L5","L12"), ("L5","L8","L12")):
    for k in (3, 5, 10):
        res = {c: [] for c in ("misfire", "tappet_issue", "belt_issue")}; fa = []
        for s in range(10):
            rng = np.random.default_rng(s); enr = list(rng.choice(h, k, replace=False))
            z = np.mean([layer_scores(Ps[L], enr) for L in combo], 0)
            rest = [i for i in h if i not in enr]
            fa.append(np.mean(z[rest] >= 3))
            for c in res:
                f = [i for i, l in enumerate(lab) if l == c]
                res[c].append(roc_auc_score([0]*len(rest)+[1]*len(f), np.r_[z[rest], z[f]]))
        print("+".join(combo), k, {c: round(float(np.mean(v)), 3) for c, v in res.items()}, "false_drift@3σ", round(float(np.mean(fa)),3))
