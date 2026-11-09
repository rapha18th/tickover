"""Condition-matched few-shot: same machine, same operating condition (DCASE hidden attributes).

For each DCASE 2024 eval machine, test clips are grouped by their ground-truth attribute string
(machine id, speed, mic, load...). Within a group, Earshot enrolls k normal clips and scores the
other normals and the anomalies of that group. This mirrors real use: one machine, recorded the
same way each time.
"""
import collections, csv, os, re, sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
from bench.fewshot import DATA, CACHE, MACHINES
from bench.eval_ship import fused
from tickover_core import engine as E

out = collections.defaultdict(list)
for m in MACHINES:
    attrs = {}
    for row in csv.reader(open(os.path.join(DATA, "gt", "ground_truth_attributes", f"ground_truth_{m}_section_00_test.csv"))):
        name, orig = row[0], row[-1]
        mm = re.match(r"section_00_(source|target)(?:_test)?_(normal|anomaly)_\d+_?(.*?)(?:\.wav)?$", orig)
        if not mm:
            continue
        at = re.sub(r"_?Bckg\w+", "", mm.group(3)).replace("noAttribute", "") or "-"
        attrs[name] = (mm.group(1), mm.group(2), at)
    c = np.load(os.path.join(CACHE, f"ced-base_L5__{m}.npz")); mn = np.load(os.path.join(CACHE, f"mn10__{m}.npz"))
    names = sorted(attrs)
    F = []
    for n in names:
        kk = f"{m}|test|{n}"
        g = np.asarray(c[kk], np.float32).reshape(-1, 4, 768)
        F.append({"grid": g, "pools": E._pools(g), "frames": np.asarray(mn[kk], np.float32)})
    groups = collections.defaultdict(list)
    for i, n in enumerate(names):
        dom, lab, at = attrs[n]
        groups[(dom, at)].append((i, lab == "anomaly"))
    for (dom, at), items in groups.items():
        norm = [i for i, a in items if not a]; anom = [i for i, a in items if a]
        for k in (3, 5):
            if len(norm) < k + 2 or not anom:
                continue
            for s in range(5):
                enr = list(np.random.default_rng(s).choice(norm, k, replace=False))
                rest = [i for i in norm if i not in enr]
                z = fused(F, enr, rest + anom)[0]
                y = [0] * len(rest) + [1] * len(anom)
                out[(m, k)].append((roc_auc_score(y, z), np.mean(z[:len(rest)] >= E.DRIFT), np.mean(z[len(rest):] >= E.DRIFT), len(rest), len(anom), dom, at))
    print("done", m, flush=True)
for (m, k), v in sorted(out.items(), key=lambda x: (x[0][1], x[0][0])):
    a = np.array([x[:3] for x in v], float)
    print(f"{m:15s} k={k} groups={len(set(x[5]+x[6] for x in v))} AUC {a[:,0].mean():.3f}  false@3σ {a[:,1].mean():.3f}  caught@3σ {a[:,2].mean():.3f}")
for k in (3, 5):
    a = np.array([x[:3] for (m, kk), v in out.items() if kk == k for x in v], float)
    print(f"ALL k={k} AUC {a[:,0].mean():.3f} false@3σ {a[:,1].mean():.3f} caught@3σ {a[:,2].mean():.3f}")
