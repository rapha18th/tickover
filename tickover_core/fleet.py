"""The fleet card: every machine on one image, for a weekly chat report."""
from __future__ import annotations

import time

from PIL import Image, ImageDraw

from .card import BG, INK, MUTED, RULE, VERDICT, _font

STATE = {"drift": VERDICT["drift"], "watch": VERDICT["watch"], "worsening": ((217, 119, 6), "WORSENING"),
         "steady": VERDICT["healthy"], "no data": ((161, 161, 170), "NO DATA")}


def _ago(ts: float, now: float) -> str:
    d = (now - ts) / 86400
    if d < 1:
        return "today"
    if d < 2:
        return "yesterday"
    return f"{int(d)} days ago"


def render(path: str, rows: list[dict], now: float | None = None, title: str = "Fleet report") -> str:
    """rows: [{name, kind, location, history:[{ts,z}], trend:{...}}]"""
    now = now or time.time()
    W = 1200
    RH = 128
    H = 150 + RH * max(1, len(rows)) + 60
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((48, 36), "TICKOVER", font=_font(18, True), fill=MUTED)
    d.text((48, 62), title, font=_font(40, True), fill=INK)
    d.text((W - 48 - d.textlength(time.strftime("%d %b %Y", time.localtime(now)), font=_font(20)), 80),
           time.strftime("%d %b %Y", time.localtime(now)), font=_font(20), fill=MUTED)
    y = 140
    for r in rows:
        tr = r["trend"]
        d.line((48, y, W - 48, y), fill=RULE)
        col, word = STATE.get(tr["state"], STATE["no data"])
        d.text((48, y + 18), r["name"], font=_font(28, True), fill=INK)
        sub = " · ".join(x for x in (r.get("kind"), r.get("location")) if x)
        last = r["history"][-1]["ts"] if r["history"] else None
        sub2 = f"last heard {_ago(last, now)}" if last else "not heard yet"
        if tr.get("overdue_days"):
            sub2 += f" · check-in {int(tr['overdue_days'])} days overdue"
        d.text((48, y + 56), sub, font=_font(18), fill=MUTED)
        d.text((48, y + 82), sub2, font=_font(18), fill=VERDICT["watch"][0] if tr.get("overdue_days") else MUTED)
        # status pill
        pw = d.textlength(word, font=_font(18, True)) + 28
        d.rounded_rectangle((430, y + 20, 430 + pw, y + 54), radius=17, fill=col)
        d.text((444, y + 26), word, font=_font(18, True), fill=(255, 255, 255))
        # trend sentence
        if tr.get("last_z") is not None:
            msg = f"{tr['last_z']:+.1f} σ"
            if tr.get("slope_per_week") is not None:
                msg += f" · {tr['slope_per_week']:+.1f} σ/week"
            if tr.get("days_to_drift") is not None and tr["state"] in ("worsening", "watch"):
                msg += f" · drift line in ~{int(round(tr['days_to_drift']))} days"
            d.text((430, y + 66), msg, font=_font(18), fill=INK)
            d.text((430, y + 90), f"from {tr['n']} check-ins", font=_font(15), fill=MUTED)
        # sparkline
        sx, sw, sh = 820, 330, 84
        sy = y + 22
        d.rectangle((sx, sy, sx + sw, sy + sh), fill=(240, 253, 244))
        zs = [x["z"] for x in r["history"]][-12:]
        lo, hi = -2.0, max(5.0, max(zs) + 0.5) if zs else 5.0
        ys = lambda v: sy + sh - (v - lo) / (hi - lo) * sh
        for line, c in ((2.0, VERDICT["watch"][0]), (3.0, VERDICT["drift"][0])):
            d.line((sx, ys(line), sx + sw, ys(line)), fill=c, width=1)
        if zs:
            pts = [(sx + (i + 0.5) * sw / len(zs), ys(v)) for i, v in enumerate(zs)]
            if len(pts) > 1:
                d.line(pts, fill=INK, width=2)
            for (px, py), v in zip(pts, zs):
                c = VERDICT["drift"][0] if v >= 3 else VERDICT["watch"][0] if v >= 2 else VERDICT["healthy"][0]
                d.ellipse((px - 5, py - 5, px + 5, py + 5), fill=c)
        y += RH
    d.line((48, y, W - 48, y), fill=RULE)
    d.text((48, y + 18), "Drift in robust sigmas against each machine's own healthy recordings. Orange line: watch. Red line: drift.",
           font=_font(15), fill=MUTED)
    img.save(path)
    return path
