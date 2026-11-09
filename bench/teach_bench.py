"""Validate tickover_teach on public labelled recordings.

For each case and each random draw: teach k recordings per state, read the detector's own
held-out test (n of N), then classify every recording not used in teaching (external n of N).
Recordings are real; which ones are drawn is random with a fixed seed. Nothing is tuned here.

Run: python bench/teach_bench.py [draws]
"""
import glob
import soundfile as sf
import hashlib
import json
import os
import random
import sys
import tempfile

os.environ["TICKOVER_HOME"] = tempfile.mkdtemp(prefix="tk_teach_bench_")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tickover_core import engine, teach  # noqa: E402

_cache, _raw = {}, engine._features


def _cached(x):
    k = hashlib.sha1(x.tobytes()).hexdigest()
    if k not in _cache:
        _cache[k] = _raw(x)
    return _cache[k]


engine._features = _cached
D = os.environ.get("TICKOVER_DATA", r"C:\Users\USER\Projects\earshot-data")
CASES = {
    "pump id_00 (DCASE 2020): normal vs abnormal": {
        "normal": sorted(glob.glob(os.path.join(D, "dcase20", "pump", "train", "*.wav")))[:40],
        "abnormal": sorted(glob.glob(os.path.join(D, "dcase20", "pump", "test", "anomaly_*.wav")))},
    "fan id_00 (DCASE 2020): normal vs abnormal (drift fails here, AUC 0.48)": {
        "normal": sorted(glob.glob(os.path.join(D, "dcase20", "fan", "train", "*.wav")))[:40],
        "abnormal": sorted(glob.glob(os.path.join(D, "dcase20", "fan", "test", "anomaly_*.wav")))},
    "car engines (HF phone recordings): healthy vs misfire vs belt vs tappet": {
        lab: sorted(glob.glob(os.path.join(D, "engines", "hf_car", "MyDataSet", f"{lab}_*.wav")))[:30]
        for lab in ("healthy", "misfire", "belt_issue", "tappet_issue")},
}




def _dcase24(machine, groups):
    """Normal test clips of one DCASE 2024 machine, grouped by a recorded operating condition."""
    import csv
    out = {lab: [] for lab in groups}
    gt = os.path.join(D, "dcase24", "gt", "ground_truth_attributes", f"ground_truth_{machine}_section_00_test.csv")
    for row in csv.reader(open(gt)):
        if "normal" not in row[-1]:
            continue
        for lab, keys in groups.items():
            if any(k in row[-1] for k in keys):
                out[lab].append(os.path.join(D, "dcase24", machine, "test", row[0]))
    return out


# Operating states: the question teach is for. Same machine, healthy, in two recorded conditions.
STATES = {
    "3D printer (DCASE 2024): print speed 1 vs 2, same printer": _dcase24("3DPrinter", {"speed 1": ["mdl_A_spd_1"], "speed 2": ["mdl_A_spd_2"]}),
    "brushless motor (DCASE 2024): slow vs fast": _dcase24("BrushlessMotor", {"slow": ["spd1000", "spd1100", "spd1200"], "fast": ["spd1700", "spd1800", "spd1900"]}),
    "robot arm (DCASE 2024): no payload vs 35": _dcase24("RoboticArm", {"empty": ["weight00"], "loaded": ["weight35"]}),
    "air compressor (DCASE 2024): flow 0 vs 2": _dcase24("AirCompressor", {"flow 0": ["fl_0_"], "flow 2": ["fl_2_"]}),
    "hovering drone (DCASE 2024): speed 1 vs 3": _dcase24("HoveringDrone", {"speed 1": ["spd_1_"], "speed 3": ["spd_3_"]}),
    "which machine is running: pump vs fan (DCASE 2020)": {"pump": CASES["pump id_00 (DCASE 2020): normal vs abnormal"]["normal"],
                                                           "fan": CASES["fan id_00 (DCASE 2020): normal vs abnormal (drift fails here, AUC 0.48)"]["normal"]},
}
FAULTS = dict(CASES)
CASES = {**{"state: " + k: v for k, v in STATES.items()}, **{"fault: " + k: v for k, v in FAULTS.items()}}


def _long(fs, s=2.0):
    return [f for f in fs if sf.info(f).duration >= s]


def run(draws=5, k=3, ext_per_label=8):
    rng = random.Random(0)
    report = {}
    for case, files in CASES.items():
        rows = []
        for d in range(draws):
            name = f"bench-{abs(hash(case)) % 10**6}-{k}-{d}"
            test_files, refused = {}, False
            for lab, fs in files.items():
                pool = _long(fs)
                pick = rng.sample(pool, min(len(pool), k + ext_per_label))
                test_files[lab] = pick[k:]
                try:
                    teach.add(name, lab, pick[:k])
                except ValueError:  # the quality gate rejected every recording of this state
                    refused = True
            if refused:
                rows.append({"held_out": "0/0", "active": False, "external": "0/0", "external_unknown": 0,
                             "external_uncertain": 0, "chance_of_luck": None, "median_margin_sigma": None,
                             "refused": "quality gate rejected the recordings"})
                print(case[:60], "k", k, "draw", d, rows[-1], flush=True)
                try:
                    teach.delete(name)
                except KeyError:
                    pass
                continue
            t = teach.test(name)
            right = total = unknown = unsure = 0
            for lab, fs in test_files.items():
                for f in fs:
                    try:
                        c = teach.classify(name, f)
                    except ValueError:  # clip under 2 s
                        continue
                    total += 1
                    right += c["label"] == lab
                    unknown += c["label"] == "unknown"
                    unsure += c["label"] == "uncertain"
            rows.append({"held_out": f"{t['right']}/{t['total']}", "active": t["active"],
                         "external": f"{right}/{total}", "external_unknown": unknown, "external_uncertain": unsure,
                         "chance_of_luck": t.get("chance_of_luck"), "median_margin_sigma": t.get("median_margin_sigma")})
            print(case[:60], "k", k, "draw", d, rows[-1], flush=True)
            teach.delete(name)
        ext_r = sum(int(r["external"].split("/")[0]) for r in rows)
        ext_t = sum(int(r["external"].split("/")[1]) for r in rows)
        def tot(rs):
            return f"{sum(int(r['external'].split('/')[0]) for r in rs)}/{sum(int(r['external'].split('/')[1]) for r in rs)}"
        on, off = [r for r in rows if r["active"]], [r for r in rows if not r["active"]]
        report[case] = {"k_per_state": k, "draws": rows, "active_draws": len(on), "of_draws": len(rows),
                        "external_total": f"{ext_r}/{ext_t}", "external_when_on": tot(on), "external_when_off": tot(off),
                        "wrong_label_when_on": sum(int(r["external"].split("/")[1]) - int(r["external"].split("/")[0])
                                                   - r["external_unknown"] - r["external_uncertain"] for r in on)}
    return report


if __name__ == "__main__":
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = {f"k={k}": run(draws, k) for k in (3, 5)}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "teach_results.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({k: {c: {x: v[x] for x in ("active_draws", "of_draws", "external_when_on", "external_when_off",
                                                 "wrong_label_when_on")} for c, v in r.items()} for k, r in out.items()}, indent=1))
