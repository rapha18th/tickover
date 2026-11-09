"""Drift trend per machine: is it getting worse, and when does it cross the line?

Uses the Theil-Sen slope (median of pairwise slopes) over the recent check-ins, which
shrugs off one odd recording. With at least three check-ins and a positive slope, it
forecasts the days until drift reaches the alert line. Every number comes with the count
of check-ins it rests on.
"""
from __future__ import annotations

import time

import numpy as np

DAY = 86400.0


def analyse(history: list[dict], drift_line: float = 3.0, watch_line: float = 2.0, window: int = 8,
            every_days: float | None = None, now: float | None = None) -> dict:
    now = now or time.time()
    h = [x for x in history if x.get("z") is not None][-window:]
    out = {"n": len(h), "state": "no data", "slope_per_week": None, "days_to_drift": None, "overdue_days": None}
    if every_days and history:
        late = (now - history[-1]["ts"]) / DAY - every_days
        out["overdue_days"] = round(late, 1) if late > 1 else None
    if not h:
        return out
    z = np.array([x["z"] for x in h], float)
    t = np.array([x["ts"] for x in h], float) / DAY
    last = float(z[-1])
    out["last_z"] = round(last, 2)
    if len(h) >= 3:
        sl = [(z[j] - z[i]) / (t[j] - t[i]) for i in range(len(h)) for j in range(i + 1, len(h)) if t[j] - t[i] > 1e-3]
        slope = float(np.median(sl)) if sl else 0.0
        out["slope_per_week"] = round(slope * 7, 2)
        if slope > 0 and last < drift_line:
            out["days_to_drift"] = round((drift_line - last) / slope, 1)
    if last >= drift_line:
        out["state"] = "drift"
    elif last >= watch_line:
        out["state"] = "watch"
    elif out["slope_per_week"] is not None and out["slope_per_week"] >= 0.4 and out["days_to_drift"] is not None:
        out["state"] = "worsening"
    else:
        out["state"] = "steady"
    recent = [x["verdict"] for x in h[-3:]]
    out["escalate"] = last >= drift_line or recent.count("watch") + recent.count("drift") >= 2
    return out
