"""Few-shot machine-drift benchmark on DCASE 2024 Task 2 embodiment machines.

Earshot enrolls a machine from a handful of healthy clips and scores new clips.
This harness reproduces that regime: k healthy clips (k in 3, 5, 10) drawn from
the official training normals, scored on the official labelled test set.

Usage:
  python bench/fewshot.py embed  <embedder> [machines...]   # cache features
  python bench/fewshot.py score  [embedders...]             # print the table
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

DATA = os.environ.get("EARSHOT_DATA", r"C:\Users\USER\Projects\earshot-data\dcase24")
CACHE = os.path.join(os.path.dirname(__file__), "cache")
MACHINES = ["RoboticArm", "HoveringDrone", "BrushlessMotor", "3DPrinter", "AirCompressor"]
SR = 16000


def load_wav(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def clips(machine: str):
    train = sorted(glob.glob(os.path.join(DATA, "train", machine, "train", "*.wav")))
    gt = {}
    with open(os.path.join(DATA, "gt", "ground_truth_data", f"ground_truth_{machine}_section_00_test.csv")) as f:
        for row in csv.reader(f):
            gt[row[0]] = int(row[1])
    dom = {}
    with open(os.path.join(DATA, "gt", "ground_truth_domain", f"ground_truth_{machine}_section_00_test.csv")) as f:
        for row in csv.reader(f):
            dom[row[0]] = int(row[1])  # 0 source, 1 target
    test = sorted(gt)
    return (train, [os.path.join(DATA, machine, "test", t) for t in test],
            np.array([gt[t] for t in test]), np.array([dom[t] for t in test]))


# ---------------------------------------------------------------- embedders
def get_embedder(name: str):
    """Return fn(wave16k) -> (T, D) frame features."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
    from bench import embedders  # noqa
    return embedders.load(name)


def cmd_embed(name: str, machines):
    fn = get_embedder(name)
    os.makedirs(CACHE, exist_ok=True)
    for m in machines:
        out = os.path.join(CACHE, f"{name}__{m}.npz")
        if os.path.exists(out) or glob.glob(os.path.join(CACHE, f"{name}_*__{m}.npz")):
            print("cached", out); continue
        train, test, y, d = clips(m)
        t0 = time.time()
        feats = {}
        for p in train + test:
            o = fn(load_wav(p))
            o = o if isinstance(o, dict) else {"": o}
            for tag, v in o.items():
                feats.setdefault(tag, {})[os.path.relpath(p, DATA)] = v
        dt = (time.time() - t0) / (len(train) + len(test))
        for tag, fd in feats.items():
            nm = f"{name}_{tag}" if tag else name
            np.savez(os.path.join(CACHE, f"{nm}__{m}.npz"), **{k.replace(os.sep, "|"): v for k, v in fd.items()})
        print(f"{name} {m}: {len(train)+len(test)} clips, {dt*1000:.0f} ms/clip")


# ---------------------------------------------------------------- scoring
def _l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def score_pooled_knn(enr, tst):
    """Mean-pooled embedding, centred on the enrolled mean, 1-NN cosine distance."""
    E = np.stack([f.mean(0) for f in enr]); T = np.stack([f.mean(0) for f in tst])
    mu = E.mean(0)
    E, T = _l2(E - mu), _l2(T - mu)
    return 1 - (T @ E.T).max(1)


def score_pooled_raw(enr, tst):
    E = _l2(np.stack([f.mean(0) for f in enr])); T = _l2(np.stack([f.mean(0) for f in tst]))
    return 1 - (T @ E.T).max(1)


def score_meanstd_knn(enr, tst):
    """Mean+std pooled (captures temporal variability), centred, 1-NN cosine."""
    pool = lambda f: np.concatenate([f.mean(0), f.std(0)])
    E = np.stack([pool(f) for f in enr]); T = np.stack([pool(f) for f in tst])
    mu, sd = E.mean(0), E.std(0) + E.std(0).mean() + 1e-6
    E, T = _l2((E - mu) / sd), _l2((T - mu) / sd)
    return 1 - (T @ E.T).max(1)


def score_frame_knn(enr, tst, q=0.9):
    """Each test frame's distance to its nearest enrolled frame; score = q-quantile.
    Catches short events (a click, a stall) that mean pooling washes out."""
    bank = np.concatenate(enr)
    mu = bank.mean(0)
    B = _l2(bank - mu)
    out = []
    for f in tst:
        F = _l2(f - mu)
        d = 1 - (F @ B.T).max(1)
        out.append(np.quantile(d, q))
    return np.array(out)


def score_frame_knn_mean(enr, tst):
    return score_frame_knn(enr, tst, q=0.5)


def score_ens(enr, tst):
    """Rank-free ensemble: z-normalise each scorer by leave-one-out enrolled scores."""
    parts = []
    for fn in (score_pooled_knn, score_frame_knn):
        s_t = fn(enr, tst)
        loo = np.array([fn(enr[:i] + enr[i + 1:], [enr[i]])[0] for i in range(len(enr))])
        parts.append((s_t - loo.mean()) / (loo.std() + 1e-6 + 0.1 * abs(loo.mean())))
    return np.mean(parts, 0)


def _crops(f, win, hop):
    """Split a (T, D) clip into overlapping windows to densify a tiny memory bank."""
    T = len(f)
    if T <= win:
        return [f]
    return [f[s:s + win] for s in range(0, T - win + 1, hop)]


def _rdp(f, gamma):
    """Relative deviation pooling (Wilkinghoff et al. 2026): weight frames by distance from the clip mean."""
    m = f.mean(0)
    d = np.linalg.norm(f - m, axis=1)
    w = (1 + d / (d.max() + 1e-9)) ** gamma
    return (w[:, None] * f).sum(0) / w.sum()


def _bands(v, nb):
    """(D,) -> (nb, D/nb) sub-band split of an embedding (BEAM)."""
    D = v.shape[-1] - v.shape[-1] % nb
    return v[..., :D].reshape(*v.shape[:-1], nb, D // nb)


def make_beam(nb=8, gamma=10.0, crop=None, center=True):
    """Sub-band 1-NN memory bank, mean+RDP pooled (DMM), crops for bank density."""
    def fn(enr, tst):
        bank = []
        for f in enr:
            f = f.astype(np.float32)
            parts = _crops(f, crop, max(1, crop // 3)) if crop else [f]
            bank += parts
        def pools(f):
            return [f.mean(0), _rdp(f, gamma)]
        B = [np.stack([pools(f)[i] for f in bank]) for i in range(2)]
        mus = [b.mean(0) if center else 0 for b in B]
        out = []
        for f in tst:
            f = f.astype(np.float32)
            P = pools(f)
            s = 0
            for i in range(2):
                b = _l2(_bands(B[i] - mus[i], nb))      # (N, nb, d)
                q = _l2(_bands(P[i] - mus[i], nb))      # (nb, d)
                sim = np.einsum("nbd,bd->nb", b, q).max(0)  # best match per band
                s += (1 - sim).mean()
            out.append(s / 2)
        return np.array(out)
    return fn


def make_cal(base):
    """Wrap a scorer with leave-one-out self-calibration: report z against the machine's own healthy spread."""
    def fn(enr, tst):
        s = base(enr, tst)
        if len(enr) < 3:
            return s
        loo = np.array([base(enr[:i] + enr[i + 1:], [enr[i]])[0] for i in range(len(enr))])
        return (s - np.median(loo)) / (1.4826 * np.median(np.abs(loo - np.median(loo))) + 1e-6)
    return fn


SCORERS = {"pooled_raw": score_pooled_raw, "pooled_knn": score_pooled_knn, "meanstd_knn": score_meanstd_knn,
           "frame_knn_q90": score_frame_knn, "frame_knn_q50": score_frame_knn_mean, "ens": score_ens,
           "beam1": make_beam(nb=1), "beam8": make_beam(nb=8), "beam8_g1": make_beam(nb=8, gamma=1.0),
           "beam8_crop": make_beam(nb=8, crop=20), "beam16": make_beam(nb=16), "beam4": make_beam(nb=4), "beam4_g1": make_beam(nb=4, gamma=1.0), "beam4_crop": make_beam(nb=4, crop=20), "beam4_g16": make_beam(nb=4, gamma=16.0)}
if os.environ.get("SCORERS"):
    SCORERS = {k: v for k, v in SCORERS.items() if k in os.environ["SCORERS"].split(",")}


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s)


def pauc(y, s, p=0.1):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s, max_fpr=p)


def cmd_score(names, ks=(3, 5, 10), seeds=5):
    rows = []
    for name in names:
        for m in MACHINES:
            path = os.path.join(CACHE, f"{name}__{m}.npz")
            if not os.path.exists(path):
                continue
            z = np.load(path)
            feats = {k.replace("|", os.sep): z[k] for k in z.files}
            train, test, y, d = clips(m)
            tr = [feats[os.path.relpath(p, DATA)] for p in train]
            src = [f for p, f in zip(train, tr) if "_source_" in p]
            te = [feats[os.path.relpath(p, DATA)] for p in test]
            # matched condition = source-domain test clips (enrol and test in the same setup)
            msk = d == 0
            te_src = [f for f, k in zip(te, msk) if k]
            for sname, fn in SCORERS.items():
                for k in ks:
                    a, pa = [], []
                    for s in range(seeds):
                        rng = np.random.default_rng(s)
                        enr = [src[i] for i in rng.choice(len(src), k, replace=False)]
                        sc = fn(enr, te_src)
                        a.append(auc(y[msk], sc)); pa.append(pauc(y[msk], sc))
                    rows.append(dict(emb=name, machine=m, scorer=sname, k=k, auc=float(np.mean(a)), pauc=float(np.mean(pa))))
    # summary: harmonic-free mean over machines
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["emb"], r["scorer"], r["k"])].append(r["auc"])
    print(f"{'embedder':14s} {'scorer':14s} {'k':>3s}  meanAUC  " + "  ".join(f"{m[:8]:>8s}" for m in MACHINES))
    for (e, s, k), v in sorted(agg.items(), key=lambda kv: -np.mean(kv[1])):
        per = {r["machine"]: r["auc"] for r in rows if r["emb"] == e and r["scorer"] == s and r["k"] == k}
        print(f"{e:14s} {s:14s} {k:3d}  {np.mean(v):.3f}    " + "  ".join(f"{per.get(m, float('nan')):8.3f}" for m in MACHINES))
    json.dump(rows, open(os.path.join(CACHE, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "embed":
        cmd_embed(sys.argv[2], sys.argv[3:] or MACHINES)
    else:
        cmd_score(sys.argv[2:])
