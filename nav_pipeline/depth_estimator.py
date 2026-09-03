"""Monocular metric depth from RGB via Depth Anything V2.

Used when no depth sensor is available (the real rover is RGB-only).
In Isaac Sim, prefer the simulated depth camera and skip this module.
"""

import os

import numpy as np
import torch

from .depth_anything.depth_anything_v2.dpt import DepthAnythingV2

_CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints")

# encoder -> (features, out_channels, checkpoint filename) -- the standard
# Depth Anything V2 per-size configs. vitb is meaningfully more accurate
# than the vits default, at roughly 2x the inference cost; worth it since
# depth error here feeds directly into the STOP distance decision. vitl/
# vitg configs aren't listed since no checkpoint for them is downloaded --
# add both here and to scripts/download_models.py if that changes.
_ENCODER_CONFIGS = {
    "vits": {"features": 64, "out_channels": [48, 96, 192, 384],
             "checkpoint": "depth_anything_v2_metric_hypersim_vits.pth"},
    "vitb": {"features": 128, "out_channels": [96, 192, 384, 768],
             "checkpoint": "depth_anything_v2_metric_hypersim_vitb.pth"},
}


class MetricDepthEstimator:
    def __init__(self, encoder: str = "vits", checkpoint: str = None,
                 max_depth: float = 20.0, device: str = "cuda:0"):
        if encoder not in _ENCODER_CONFIGS:
            raise ValueError(f"unknown depth encoder {encoder!r}, expected one of {list(_ENCODER_CONFIGS)}")
        econf = _ENCODER_CONFIGS[encoder]
        checkpoint = checkpoint or os.path.join(_CKPT_DIR, econf["checkpoint"])
        self.device = device
        self.encoder = encoder
        self.model = DepthAnythingV2(
            encoder=encoder, features=econf["features"], out_channels=econf["out_channels"],
            max_depth=max_depth,
        )
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 RGB -> HxW float32 depth in meters."""
        bgr = rgb[:, :, ::-1].copy()  # infer_image expects BGR (cv2 convention)
        depth = self.model.infer_image(bgr)
        return depth.astype(np.float32)
