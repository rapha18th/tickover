"""Frame-level feature extractors for the few-shot benchmark. Each returns (T, D)."""
from __future__ import annotations

import os
import sys

import numpy as np

INJINI = r"C:\Users\USER\Projects\injini"


def _logmel16(x, n_mels=128):
    import librosa
    m = librosa.feature.melspectrogram(y=x, sr=16000, n_fft=1024, hop_length=256, n_mels=n_mels, power=2.0)
    return np.log(m + 1e-8).T  # (T, n_mels)


def logmel():
    return lambda x: _logmel16(x)


def mn10():
    import torch
    sys.path.insert(0, INJINI)
    from src.embedder import _load_mn
    from src.features import logmel as inj_logmel, resample_linear
    feats = _load_mn("mn10_as").features

    def fn(x):
        y = resample_linear(x, 16000, 32000)
        mel = inj_logmel(y)  # (128, T)
        t = torch.from_numpy(np.asarray(mel, dtype=np.float32))[None, None]
        with torch.no_grad():
            fm = feats(t)  # (1, C, F', T')
        return fm.mean(2)[0].T.numpy()  # (T', C)
    return fn


def dasheng(size="base"):
    import torch
    import dasheng as ds
    model = {"base": ds.dasheng_base, "06B": ds.dasheng_06B}[size]().eval()

    def fn(x):
        with torch.no_grad():
            return model(torch.from_numpy(x)[None]).squeeze(0).numpy()  # (T, 768)
    return fn


def hf_audio(repo):
    """Generic HF transformers audio encoder with trust_remote_code (EAT, CED, BEATs ports)."""
    import torch
    from transformers import AutoModel, AutoFeatureExtractor
    model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()
    try:
        fe = AutoFeatureExtractor.from_pretrained(repo, trust_remote_code=True)
    except Exception:
        fe = None

    def fn(x):
        with torch.no_grad():
            if fe is not None:
                inp = fe(x, sampling_rate=16000, return_tensors="pt")
                out = model(**inp)
            else:
                out = model(torch.from_numpy(x)[None])
            h = getattr(out, "last_hidden_state", out[0] if isinstance(out, (tuple, list)) else out)
            return h.squeeze(0).numpy()
    return fn


REGISTRY = {
    "logmel": logmel,
    "mn10": mn10,
    "dasheng": lambda: dasheng("base"),
}


def load(name):
    if name in REGISTRY:
        return REGISTRY[name]()
    if name.startswith("hf:"):
        return hf_audio(name[3:])
    raise KeyError(name)


CED_DIR = r"C:\Users\USER\Projects\earshot-data\models\ced"


def ced(size="base"):
    """CED ONNX (Apache-2.0) with exposed token outputs. Returns {Lk: (T', 4*D)} grids, 4 mel bands."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 2)
    s = ort.InferenceSession(os.path.join(CED_DIR, f"ced-{size}", "tok.onnx"), so, providers=["CPUExecutionProvider"])
    names = ["layer_norm_10", "layer_norm_16", "layer_norm_24"]
    tags = ["L5", "L8", "L12"]

    def fn(x):
        outs = s.run(names, {"waveform": x[None].astype(np.float32)})
        res = {}
        for t, h in zip(tags, outs):
            h = h[0]                              # (4*T', D), f-major
            Tp = h.shape[0] // 4
            g = h.reshape(4, Tp, -1).transpose(1, 0, 2)   # (T', 4, D)
            res[t] = g.reshape(Tp, -1).astype(np.float16)
        return res
    return fn


REGISTRY["ced-base"] = lambda: ced("base")
REGISTRY["ced-small"] = lambda: ced("small")
REGISTRY["ced-tiny"] = lambda: ced("tiny")
