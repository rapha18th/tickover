"""Rebuild tickover_core/models/mn10_frames.onnx from source (provenance for the shipped graph).

Needs a checkout of github.com/rapha18th/injini (MIT), which vendors EfficientAT (MIT,
Schmid et al.) and downloads the official mn10_as AudioSet weights from the EfficientAT
GitHub release. The graph takes the 32 kHz STFT power (1, 513, T) and applies Injini's
Kaldi mel filterbank, log and normalisation, then the mn10 feature extractor, and
returns per-frame embeddings averaged over frequency: (1, T', 960).

    python scripts/export_mn10.py /path/to/injini
"""
import sys

import numpy as np
import torch
import torch.nn as nn

injini = sys.argv[1]
sys.path.insert(0, injini)
from src.embedder import _load_mn  # noqa: E402
from src.features import _preemphasis, _stft_power, mel_matrix, resample_linear  # noqa: E402


class Mn10Frames(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = _load_mn("mn10_as").features
        self.register_buffer("mel", torch.from_numpy(mel_matrix().astype(np.float32)))

    def forward(self, power):
        m = torch.matmul(self.mel, power[0])
        m = (torch.log(m + 1e-5) + 4.5) / 5.0
        return self.f(m[None, None]).mean(2).transpose(1, 2)


x = (np.random.default_rng(0).standard_normal(160000) * 0.05).astype(np.float32)
p = torch.from_numpy(_stft_power(_preemphasis(resample_linear(x, 16000, 32000).astype(np.float64))).astype(np.float32))[None]
torch.onnx.export(Mn10Frames().eval(), (p,), "tickover_core/models/mn10_frames.onnx", input_names=["power"],
                  output_names=["frames"], dynamic_axes={"power": {2: "T"}, "frames": {1: "Tp"}},
                  opset_version=17, dynamo=False)
print("wrote tickover_core/models/mn10_frames.onnx")
