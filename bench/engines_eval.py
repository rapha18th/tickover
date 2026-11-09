"""Few-shot drift on phone recordings of car engines (healthy vs misfire/tappet/belt)."""
import glob, os, sys, re
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from tickover_core import audio, encoder, engine as E

D = r"C:\Users\USER\Projects\earshot-data\engines\hf_car\MyDataSet"
layer = sys.argv[1] if len(sys.argv) > 1 else "L8"
enc = encoder.get()
files = sorted(glob.glob(os.path.join(D, "*.wav")))
lab = [re.sub(r"_\d+\.wav$", "", os.path.basename(f)) for f in files]
cache = os.path.join(os.path.dirname(__file__), "cache", f"engines_{layer}.npz")
if os.path.exists(cache):
    G = list(np.load(cache, allow_pickle=True)["G"])
else:
    G = [enc.grid(audio.load(f), layer).astype(np.float32) for f in files]
    np.savez(cache, G=np.array(G, dtype=object))
T = min(g.shape[0] for g in G)
G = [g[:T] for g in G]
P = np.stack([E._pools(g) for g in G])
h = [i for i, l in enumerate(lab) if l == "healthy"]
for k in (3, 5, 10):
    res = {c: [] for c in ("misfire", "tappet_issue", "belt_issue")}
    for s in range(10):
        rng = np.random.default_rng(s)
        enr = list(rng.choice(h, k, replace=False))
        mu = P[enr].mean(0)
        sc = np.array([E._band_scores(P[enr], P[i], mu).mean() for i in range(len(G))])
        rest_h = [i for i in h if i not in enr]
        for c in res:
            f = [i for i, l in enumerate(lab) if l == c]
            res[c].append(roc_auc_score([0] * len(rest_h) + [1] * len(f), np.r_[sc[rest_h], sc[f]]))
    print(layer, k, {c: round(float(np.mean(v)), 3) for c, v in res.items()})
