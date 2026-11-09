"""Frozen audio encoder: CED (Apache-2.0, mispeech) run through onnxruntime.

CED is a ViT distilled on AudioSet. Its published ONNX graph takes a raw 16 kHz
waveform and returns class logits. Earshot never uses the logits. On first use it
downloads the graph from Hugging Face at a pinned commit, checks its sha256, and
adds the transformer's patch tokens as extra graph outputs. Those tokens form a
time x frequency grid (4 mel bands x 62 frames per 10 s), which is what the drift
scorer compares. Nothing else is downloaded and no audio leaves the machine.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

MODELS = {
    "ced-base": ("mispeech/ced-base", "db3e14a8db4c21b56b165261c39649741a900e7f",
                 "1cb33c4300b6c52ae099a5af72058982e673ec79862855961b8b8c10eeaba74c", 768),
    "ced-small": ("mispeech/ced-small", "06bb40c5ec089e96867ebc5246be02441f4a71e4",
                  "18fa4fa30c1872c322c6b08f2824c9dd6f7fe149b8aa21320ddceb77007cff75", 384),
    "ced-tiny": ("mispeech/ced-tiny", "ace276d29dd0bb3f3517b0fa8cf300738c409019",
                 "3890f227660be71886ff48633d11a731993e0206db8a486781412c28a0873435", 192),
}
DEFAULT = os.environ.get("EARSHOT_ENCODER", "ced-base")
LAYERS = {"L5": "layer_norm_10", "L8": "layer_norm_16", "L12": "layer_norm_24"}
N_BANDS = 4
CLIP = 160_000  # 10 s at 16 kHz, the encoder's native window
BAND_LABELS = ["0-0.6 kHz", "0.6-1.8 kHz", "1.8-3.9 kHz", "3.9-8 kHz"]  # 64 HTK mel bands split 4 ways


def _fetch(name: str, models_dir: Path) -> Path:
    repo, rev, sha, _ = MODELS[name]
    raw = models_dir / f"{name}.onnx"
    if not raw.exists() or hashlib.sha256(raw.read_bytes()).hexdigest() != sha:
        url = f"https://huggingface.co/{repo}/resolve/{rev}/model.onnx"
        tmp = raw.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL, pinned revision
        got = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if got != sha:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{name}: sha256 mismatch ({got}), refusing to load")
        tmp.replace(raw)
    tok = models_dir / f"{name}-tokens.onnx"
    if not tok.exists():
        import onnx
        m = onnx.load(str(raw))
        for out in LAYERS.values():
            m.graph.output.append(onnx.helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, None))
        onnx.save(m, str(tok))
    return tok


class Encoder:
    def __init__(self, name: str = DEFAULT, models_dir: Path | None = None):
        import onnxruntime as ort
        if models_dir is None:
            from .store import data_dir
            models_dir = data_dir() / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 4) - 1))
        self.sess = ort.InferenceSession(str(_fetch(name, models_dir)), so, providers=["CPUExecutionProvider"])

    def grid(self, x: np.ndarray, layer: str = "L8") -> np.ndarray:
        """16 kHz mono clip (3-10 s; longer is trimmed) -> (T', 4, D) patch-token grid."""
        x = x[:CLIP]
        h = self.sess.run([LAYERS[layer]], {"waveform": x[None].astype(np.float32)})[0][0]
        tp = h.shape[0] // N_BANDS
        return h.reshape(N_BANDS, tp, -1).transpose(1, 0, 2)


_CACHE: dict[str, Encoder] = {}


def get(name: str = DEFAULT) -> Encoder:
    if name not in _CACHE:
        _CACHE[name] = Encoder(name)
    return _CACHE[name]


# ---------------------------------------------------------------- mn10 (EfficientAT, via Injini)
# EfficientAT mn10_as (MIT, Schmid et al.), a MobileNetV3 distilled from a transformer
# ensemble on AudioSet, exported by the Injini project with its Kaldi mel filterbank
# inside the graph. Input: STFT power (1, 513, T) at 32 kHz. Output: (1, T', 960).
MN10_PATH = Path(__file__).resolve().parent / "models" / "mn10_frames.onnx"
MN10_SHA = "172b2a6f14e145d39ef047430b409c0fd969bc12e926457218567dc396e8ff45"  # scripts/export_mn10.py
_SR32, _NFFT, _HOP, _WIN = 32_000, 1024, 320, 800
_hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(_WIN) / (_WIN - 1))
_WINDOW = np.zeros(_NFFT)
_WINDOW[(_NFFT - _WIN) // 2:(_NFFT - _WIN) // 2 + _WIN] = _hann


def _power32(x16: np.ndarray) -> np.ndarray:
    n = int(round(len(x16) * 2))
    y = np.interp(np.linspace(0, len(x16), n, endpoint=False), np.arange(len(x16)), x16)
    y = y[1:] - 0.97 * y[:-1]
    yp = np.pad(y, (_NFFT // 2, _NFFT // 2), mode="reflect")
    nf = 1 + (len(yp) - _NFFT) // _HOP
    idx = np.arange(_NFFT)[:, None] + _HOP * np.arange(nf)[None, :]
    spec = np.fft.rfft(yp[idx] * _WINDOW[:, None], n=_NFFT, axis=0)
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float32)[None]


class Mn10:
    name = "mn10"

    def __init__(self):
        import onnxruntime as ort
        if hashlib.sha256(MN10_PATH.read_bytes()).hexdigest() != MN10_SHA:
            raise RuntimeError("mn10_frames.onnx does not match its pinned sha256; reinstall the plugin")
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, min(8, (os.cpu_count() or 4) - 1))
        self.sess = ort.InferenceSession(str(MN10_PATH), so, providers=["CPUExecutionProvider"])

    def frames(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {"power": _power32(x[:CLIP].astype(np.float64))})[0][0]


def get_mn10() -> Mn10:
    if "mn10" not in _CACHE:
        _CACHE["mn10"] = Mn10()
    return _CACHE["mn10"]
