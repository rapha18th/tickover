"""Benchmark the shipping engine (earshot.engine) few-shot.

DCASE 2024 Task 2 eval, embodiment machines (robot arm, drone, brushless motor, 3D printer,
air compressor): enroll k source-domain normals, score the source-domain test set.
Car engines (phone recordings, HF RashmiKarunarathne/MyDataSet): enroll k healthy, score
held-out healthy vs misfire / tappet / belt.
Reports AUC, pAUC(0.1) and the false-alarm rate at the 3-sigma DRIFT line.
"""
import glob, json, os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from earshot_core import engine as E, audio
from bench.fewshot import clips, DATA, CACHE, MACHINES

def fused(F, enr, idx):
    loo = np.stack([E._bank_scores(F, [j for j in enr if j != i], F[i]) for i in enr])
    med, mad = E._robust(loo)
    S = np.stack([E._bank_scores(F, enr, F[i]) for i in idx])
    Z = (S - med) / mad
    return Z[:, :3].mean(1), Z[:, 0], Z[:, 1], Z[:, 2]

def dcase_feats(m):
    train, test, y, d = clips(m)
    keys = [os.path.relpath(p, DATA) for p in train + test]
    c = np.load(os.path.join(CACHE, f"ced-base_L5__{m}.npz")); mn = np.load(os.path.join(CACHE, f"mn10__{m}.npz"))
    F = []
    for k in keys:
        kk = k.replace(os.sep, "|"); g = np.asarray(c[kk], np.float32).reshape(-1, 4, 768)
        F.append({"grid": g, "pools": E._pools(g), "frames": np.asarray(mn[kk], np.float32)})
    src = [i for i, p in enumerate(train) if "_source_" in p]
    te = [len(train) + i for i in range(len(test)) if d[i] == 0]
    return F, src, te, y[d == 0]

def engine_feats():
    D = r"C:\Users\USER\Projects\earshot-data\engines\hf_car\MyDataSet"
    cache = os.path.join(CACHE, "engines_ship.npz")
    files = sorted(glob.glob(os.path.join(D, "*.wav")))
    lab = [re.sub(r"_\d+\.wav$", "", os.path.basename(f)) for f in files]
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); F = list(z["F"])
    else:
        F = [E._features(audio.load(f)[:audio.SR * 10]) for f in files]
        np.savez(cache, F=np.array(F, dtype=object))
    return F, lab

def main():
    res = {}
    for m in MACHINES:
        F, src, te, yt = dcase_feats(m)
        for k in (3, 5, 10):
            for s in range(5):
                enr = list(np.random.default_rng(s).choice(src, k, replace=False))
                zs = fused(F, enr, te)
                for nm, z in zip(("earshot", "ced_only", "mn10_clip", "mn10_frames"), zs):
                    r = res.setdefault(("dcase", m, nm, k), {"auc": [], "pauc": [], "fa": []})
                    r["auc"].append(roc_auc_score(yt, z)); r["pauc"].append(roc_auc_score(yt, z, max_fpr=0.1))
                    r["fa"].append(float(np.mean(z[yt == 0] >= E.DRIFT)))
        print("done", m, flush=True)
    F, lab = engine_feats()
    h = [i for i, l in enumerate(lab) if l == "healthy"]
    for k in (3, 5, 10):
        for s in range(10):
            enr = list(np.random.default_rng(s).choice(h, k, replace=False))
            rest = [i for i in h if i not in enr]
            idx = list(range(len(F)))
            zs = fused(F, enr, idx)
            for nm, z in zip(("earshot", "ced_only", "mn10_clip", "mn10_frames"), zs):
                for c in ("misfire", "tappet_issue", "belt_issue"):
                    f = [i for i, l in enumerate(lab) if l == c]
                    yy = [0] * len(rest) + [1] * len(f); sc = np.r_[z[rest], z[f]]
                    r = res.setdefault(("engines", c, nm, k), {"auc": [], "pauc": [], "fa": []})
                    r["auc"].append(roc_auc_score(yy, sc)); r["pauc"].append(roc_auc_score(yy, sc, max_fpr=0.1))
                    r["fa"].append(float(np.mean(z[rest] >= E.DRIFT)))
    out = [{"set": a, "machine": b, "system": c, "k": k, **{x: float(np.mean(v)) for x, v in r.items()}} for (a, b, c, k), r in res.items()]
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "results_ship.json"), "w"), indent=1)
    import collections
    agg = collections.defaultdict(list)
    for o in out: agg[(o["set"], o["system"], o["k"])].append(o)
    for key in sorted(agg):
        v = agg[key]
        print(f"{key[0]:8s} {key[1]:12s} k={key[2]:2d} AUC {np.mean([o['auc'] for o in v]):.3f} pAUC {np.mean([o['pauc'] for o in v]):.3f} false@3σ {np.mean([o['fa'] for o in v]):.3f}  " +
              " ".join(f"{o['machine'][:8]}={o['auc']:.3f}" for o in v))


if __name__ == "__main__":
    main()
