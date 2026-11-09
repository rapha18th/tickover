"""The Earshot card: one PNG that carries the whole verdict into any chat.

Layout (light theme): header with machine and verdict, the clip's spectrogram with
the drift map laid over it, a band panel naming the frequency ranges that moved,
and a sparkline of recent checks against the healthy band.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 675
BG, INK, MUTED, RULE = (250, 250, 248), (24, 24, 27), (113, 113, 122), (228, 228, 231)
VERDICT = {"healthy": ((22, 163, 74), "HEALTHY"), "watch": ((217, 119, 6), "WATCH"),
           "drift": ((220, 38, 38), "DRIFT"), "insufficient": ((113, 113, 122), "NEEDS BASELINE")}


def _font(size: int, bold: bool = False):
    for name in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _gray_spec(mel: np.ndarray) -> np.ndarray:
    """(T, F) log-mel -> (F, T) uint8, low frequencies at the bottom."""
    m = mel.T[::-1]
    lo, hi = np.percentile(m, 5), np.percentile(m, 99.5)
    g = np.clip((m - lo) / (hi - lo + 1e-9), 0, 1)
    return (235 - 190 * g).astype(np.uint8)


def render(path: str, *, machine: str, verdict: str, z: float, mel: np.ndarray, amap: np.ndarray,
           band_labels: list[str], band_z: list[float], history: list[float], n_enrolled: int,
           enrolled_on: str, encoder: str) -> str:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    col, word = VERDICT.get(verdict, VERDICT["watch"])

    # header
    d.text((48, 36), "EARSHOT", font=_font(18, True), fill=MUTED)
    d.text((48, 62), machine, font=_font(40, True), fill=INK)
    pw = d.textlength(word, font=_font(22, True)) + 36
    d.rounded_rectangle((W - 48 - pw, 50, W - 48, 94), radius=22, fill=col)
    d.text((W - 48 - pw + 18, 59), word, font=_font(22, True), fill=(255, 255, 255))
    sub = (f"{z:+.1f} σ from its healthy self" if verdict != "insufficient" else "send more healthy clips to finish the baseline")
    d.text((48, 112), sub, font=_font(22), fill=INK)

    # spectrogram + drift overlay
    x0, y0, sw, sh = 48, 160, 820, 360
    spec = Image.fromarray(_gray_spec(mel)).resize((sw, sh), Image.BILINEAR).convert("RGB")
    # amap: 2.0 = the healthy 95th percentile of token distance. Colour only what lies beyond it.
    a = np.clip((np.asarray(amap, np.float32) - 2.0) / 3.0, 0, 1)
    a_img = Image.fromarray((a[::-1] * 255).astype(np.uint8)).resize((sw, sh), Image.BICUBIC)
    alpha = (np.asarray(a_img, dtype=np.float32) / 255.0) ** 0.8 * 0.8
    rgb = np.asarray(spec, dtype=np.float32)
    rgb = rgb * (1 - alpha[..., None]) + np.array(col, np.float32) * alpha[..., None]
    img.paste(Image.fromarray(rgb.astype(np.uint8)), (x0, y0))
    d.rectangle((x0, y0, x0 + sw, y0 + sh), outline=RULE, width=1)
    d.text((x0, y0 + sh + 8), "time →", font=_font(16), fill=MUTED)
    d.text((x0 - 40, y0 - 2), "8 kHz", font=_font(13), fill=MUTED)
    d.text((x0 - 16, y0 + sh - 14), "0", font=_font(13), fill=MUTED)
    d.text((x0 + sw - 160, y0 + sh + 8), "colour = drift from baseline", font=_font(16), fill=MUTED)

    # band panel
    bx = x0 + sw + 40
    d.text((bx, y0 - 2), "Where it changed", font=_font(20, True), fill=INK)
    nb = len(band_labels)
    bh = (sh - 40) / nb
    for i, (lab, bz) in enumerate(zip(band_labels[::-1], band_z[::-1])):
        yy = y0 + 40 + i * bh
        frac = float(np.clip(bz / 6.0, 0, 1))
        c = VERDICT["drift"][0] if bz >= 3 else VERDICT["watch"][0] if bz >= 2 else (161, 161, 170)
        d.text((bx, yy + 4), lab, font=_font(16), fill=MUTED)
        d.rounded_rectangle((bx, yy + 28, bx + 200, yy + 44), radius=8, fill=RULE)
        d.rounded_rectangle((bx, yy + 28, bx + max(16, int(200 * frac)), yy + 44), radius=8, fill=c)
        d.text((bx + 212, yy + 24), f"{bz:+.1f}σ", font=_font(16, True), fill=INK)

    # sparkline
    sy = y0 + sh + 50
    d.text((48, sy), "Recent checks", font=_font(16, True), fill=INK)
    hx, hw, hh = 200, 660, 50
    d.rectangle((hx, sy - 4, hx + hw, sy - 4 + hh), fill=(240, 253, 244))
    hist = list(history)[-30:]
    if hist:
        lo, hi = min(-1.0, min(hist)), max(4.0, max(hist))
        ys = lambda v: sy - 4 + hh - (v - lo) / (hi - lo) * hh
        d.line((hx, ys(2.0), hx + hw, ys(2.0)), fill=VERDICT["watch"][0], width=1)
        pts = [(hx + (i + 0.5) * hw / max(1, len(hist)), ys(v)) for i, v in enumerate(hist)]
        if len(pts) > 1:
            d.line(pts, fill=INK, width=2)
        for (px, py), v in zip(pts, hist):
            c = VERDICT["drift"][0] if v >= 3 else VERDICT["watch"][0] if v >= 2 else VERDICT["healthy"][0]
            d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=c)

    # footer
    d.line((48, H - 52, W - 48, H - 52), fill=RULE)
    d.text((48, H - 40), f"Baseline: {n_enrolled} healthy clips, {enrolled_on}   ·   Encoder: {encoder}   ·   Local, on-device",
           font=_font(15), fill=MUTED)
    img.save(path)
    return path
