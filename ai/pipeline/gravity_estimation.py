"""Per-image camera gravity + focal estimation via GeoCalib.

BoxerNet lifts 2D->3D in a gravity-aligned world frame and needs to know the
camera's orientation relative to gravity. For monocular web photos we have no
IMU/pose, so we estimate it from the single image with GeoCalib
(Veicht et al. 2024), which the Boxer paper itself suggests for this case.

`estimate()` returns the gravity (down) direction in camera coordinates and the
predicted focal length (px) at the original image scale.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ai.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraGravity:
    gravity_down_cam: np.ndarray  # (3,) unit vector, gravity (down) in camera frame
    focal_px: float               # estimated focal length at original image scale
    roll_deg: float
    pitch_deg: float


class CameraGravityEstimator:
    """Single-image gravity + focal estimator (GeoCalib)."""

    def __init__(self, device: Optional[str] = None):
        from geocalib import GeoCalib

        self.device = device or Config.get_default_device()
        logger.info(f"Loading GeoCalib on {self.device}")
        # GeoCalib runs its own resizing; keep it on the requested device.
        self.model = GeoCalib().to(self.device)

    @torch.inference_mode()
    def estimate(self, image: Image.Image) -> CameraGravity:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).to(self.device)  # (3,H,W) in [0,1]
        res = self.model.calibrate(img_t)

        # GeoCalib's gravity.vec3d points "up" (~[0,-1,0] for a level cam in a
        # y-down camera frame); gravity (down) is its negation.
        up = res["gravity"].vec3d.detach().cpu().numpy().reshape(-1)[:3]
        gravity_down = -up
        gravity_down = gravity_down / (np.linalg.norm(gravity_down) + 1e-9)

        focal = float(res["camera"].f.detach().cpu().numpy().reshape(-1)[0])
        rp = res["gravity"].rp.detach().cpu().numpy().reshape(-1)
        roll_deg = float(np.degrees(rp[0]))
        pitch_deg = float(np.degrees(rp[1]))
        return CameraGravity(
            gravity_down_cam=gravity_down.astype(np.float32),
            focal_px=focal,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
        )
