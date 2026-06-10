"""Stage 4: Boxer 3D OBB lifting.

Wraps Meta's facebookresearch/boxer `BoxerNet`. Boxer expects:
- img0     : RGB tensor (1, 3, H, W), values in [0, 1]
- cam0     : CameraTW   (pinhole/fisheye intrinsics)
- T_world_rig0 : PoseTW (camera pose, world<-rig)
- sdp_w    : (1, N, 3)  semi-dense 3D points (world frame)
- bb2d     : (1, M, 4)  2D boxes in BOXER ORDER (x1, x2, y1, y2)
- rotated0 : bool tensor

Boxer's output ObbTW yields **absolute metric** dimensions and volume directly,
so no relative→absolute conversion is required downstream.

Setup (run from repo root so ./boxer matches BOXER_REPO_PATH's default):
    git clone https://github.com/facebookresearch/boxer.git
    cd boxer && bash scripts/download_ckpts.sh && cd ..
    ls boxer/ckpts/*.ckpt   # filename carries a config hash; confirm it
    # BOXER_CHECKPOINT is required (no default); point it at the actual .ckpt:
    export BOXER_CHECKPOINT="$PWD/boxer/ckpts/boxernet_hw960in4x6d768-3e37cfc4.ckpt"
    # BOXER_REPO_PATH defaults to ./boxer; only set it if cloned elsewhere.
"""

import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from ai.config import Config
from ai.gpu.precision import autocast_label, select_autocast_dtype

logger = logging.getLogger(__name__)


@dataclass
class BoxerObb:
    """Single 3D OBB result with absolute metric dimensions."""

    width_m: float
    depth_m: float
    height_m: float
    volume_m3: float
    center_world: List[float]
    confidence: float
    label: str
    input_index: int  # index of the source 2D detection in the lift() input


class BoxerLifter:
    """Lift YOLOE 2D detections + depth to 3D OBBs via BoxerNet."""

    def __init__(
        self,
        repo_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        device: Optional[str] = None,
        conf_threshold: float = 0.2,
        sdp_source: Optional[str] = None,
        sdp_interp: Optional[str] = None,
        sdp_target_points: Optional[int] = None,
    ):
        self.repo_path = repo_path or Config.BOXER_REPO_PATH
        self.ckpt_path = ckpt_path or Config.BOXER_CHECKPOINT
        self.device = device or Config.get_default_device()
        self.conf_threshold = conf_threshold
        # sdp sampling strategy (see Config for semantics). Constructor args win
        # over env so the eval harness can sweep variants in-process.
        self.sdp_source = (sdp_source or Config.SDP_SOURCE).lower()
        self.sdp_interp = (sdp_interp or Config.SDP_INTERP).lower()
        self.sdp_target_points = int(sdp_target_points or Config.SDP_TARGET_POINTS)
        self.net = None
        self.CameraTW = None
        self.PoseTW = None
        # Resolved in _load() once the device is pinned: bf16 on Ampere+/L4,
        # fp16 on Turing/T4, None (fp32) elsewhere. None until loaded.
        self.autocast_dtype: Optional[torch.dtype] = None
        self._load()

    @staticmethod
    def _normalize_boxer_device(device: str) -> str:
        """BoxerNet.load_from_checkpoint only accepts 'cuda' / 'mps' / 'cpu'."""
        if device.startswith("cuda"):
            return "cuda"
        if device == "mps":
            return "mps"
        return "cpu"

    def _load(self) -> None:
        if not self.ckpt_path or not os.path.exists(self.ckpt_path):
            logger.warning(
                f"BoxerNet checkpoint not found: {self.ckpt_path}. "
                "Set BOXER_CHECKPOINT to the BoxerNet checkpoint (.ckpt) file."
            )
            return
        if not os.path.isdir(self.repo_path):
            logger.warning(
                f"Boxer repo not found: {self.repo_path}. "
                "Clone https://github.com/facebookresearch/boxer.git and "
                "set BOXER_REPO_PATH."
            )
            return

        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        from boxernet.boxernet import BoxerNet  # type: ignore
        from utils.tw.camera import CameraTW  # type: ignore
        from utils.tw.pose import PoseTW  # type: ignore

        self.CameraTW = CameraTW
        self.PoseTW = PoseTW

        # Pin the right CUDA device before generic "cuda" load.
        if self.device.startswith("cuda:"):
            torch.cuda.set_device(int(self.device.split(":")[1]))

        boxer_device = self._normalize_boxer_device(self.device)
        logger.info(f"Loading BoxerNet on {self.device} (boxer arg='{boxer_device}'): {self.ckpt_path}")
        self.net = BoxerNet.load_from_checkpoint(self.ckpt_path, device=boxer_device)
        # Re-target to the specific cuda:N if applicable.
        if self.device.startswith("cuda:"):
            self.net = self.net.to(self.device)
            self.net.device = self.device

        # Pick the inference precision for THIS GPU (T4 -> fp16, L4 -> bf16, ...).
        self.autocast_dtype = select_autocast_dtype(self.device, Config.BOXER_AUTOCAST)
        if self.device.startswith("cuda"):
            try:
                gpu_name = torch.cuda.get_device_name(self.device)
            except Exception:
                gpu_name = self.device
            logger.info(
                "BoxerNet inference precision on %s: %s autocast",
                gpu_name,
                autocast_label(self.autocast_dtype),
            )

    def lift(
        self,
        image: Image.Image,
        bboxes_xyxy: np.ndarray,
        labels: List[str],
        depth: np.ndarray,
        focal_length_px: Optional[float] = None,
        gravity_down_cam: Optional[np.ndarray] = None,
    ) -> List[BoxerObb]:
        """Run BoxerNet on a single image with N 2D detections + dense depth.

        BoxerNet expects square `hw x hw` inputs (default 960). Image, depth,
        bboxes, and intrinsics are all rescaled to match. fx / fy are scaled
        independently to absorb any aspect-ratio change.

        Args:
            image: PIL image (H, W).
            bboxes_xyxy: (N, 4) detection boxes in (x1, y1, x2, y2) at image scale.
            labels: N label strings.
            depth: (H, W) metric depth map (meters).
            focal_length_px: predicted intrinsic focal length (Depth Pro, at
                original image scale). None → 75° FOV pinhole synthetic.
        """
        if self.net is None or len(bboxes_xyxy) == 0:
            return []
        datum, _cam, _pose, _img_t = self._prepare_datum(
            image, bboxes_xyxy, depth, focal_length_px, gravity_down_cam
        )
        output = self._forward(datum)
        obb_w = output["obbs_pr_w"].cpu()[0]  # (M, 165)
        return self._to_results(obb_w, labels)

    def lift_obbs(
        self,
        image: Image.Image,
        bboxes_xyxy: np.ndarray,
        depth: np.ndarray,
        focal_length_px: Optional[float] = None,
        gravity_down_cam: Optional[np.ndarray] = None,
    ):
        """Like :meth:`lift` but return the raw ``ObbTW`` plus the exact camera,
        pose, and resized image tensor used for inference.

        This lets callers re-project the 3D OBB wireframes back onto the image
        (visualization / debugging) using the *same* intrinsics and gravity
        frame the network saw. Returns ``(obb_w, cam, pose, img_t)`` with every
        tensor on CPU (ready for ``utils.image.draw_bb3s``), or ``None`` when the
        net is unavailable or there are no detections.
        """
        if self.net is None or len(bboxes_xyxy) == 0:
            return None
        datum, cam, pose, img_t = self._prepare_datum(
            image, bboxes_xyxy, depth, focal_length_px, gravity_down_cam
        )
        output = self._forward(datum)
        obb_w = output["obbs_pr_w"].cpu()[0]  # (M, 165)
        return obb_w, cam.to("cpu"), pose.to("cpu"), img_t.cpu()

    # ------------------------------------------------------------------
    # Inference internals (shared by lift / lift_obbs)
    # ------------------------------------------------------------------
    def _prepare_datum(
        self,
        image: Image.Image,
        bboxes_xyxy: np.ndarray,
        depth: np.ndarray,
        focal_length_px: Optional[float],
        gravity_down_cam: Optional[np.ndarray],
    ):
        """Build the BoxerNet input ``datum`` + ``CameraTW`` + ``PoseTW``.

        Centralizes the square-resize, focal sanity check, intrinsic scaling,
        gravity-aligned world rotation, and semi-dense point back-projection so
        that :meth:`lift` and :meth:`lift_obbs` cannot drift apart. Returns
        ``(datum, cam, pose, img_t)``.
        """
        target = int(self.net.hw)  # boxer expects square hw x hw
        W0, H0 = image.width, image.height
        sx, sy = target / W0, target / H0

        # Resize everything to (target, target)
        image_r = image.resize((target, target), Image.BILINEAR)
        bbox_r = bboxes_xyxy.astype(np.float32).copy()
        bbox_r[:, [0, 2]] *= sx
        bbox_r[:, [1, 3]] *= sy

        # Sanity-check the predicted focal. Depth Pro occasionally returns an
        # implausible focal (very wide/narrow FOV) which wrecks the metric 3D
        # scale; fall back to a default pinhole in that case.
        if focal_length_px:
            ratio = float(focal_length_px) / W0
            if not (0.5 <= ratio <= 2.5):
                logger.warning(
                    "focal_px=%.0f implausible (focal/width=%.2f); using default FOV",
                    float(focal_length_px),
                    ratio,
                )
                focal_length_px = None

        # Scale intrinsics: fx/fy each follow their axis scale.
        if focal_length_px:
            fx = float(focal_length_px) * sx
            fy = float(focal_length_px) * sy
        else:
            fx = fy = target * 0.75
        cx, cy = target / 2.0, target / 2.0

        # World<-cam rotation: from estimated gravity (GeoCalib) when available,
        # else the level monocular default. Used for BOTH the pose and the sdp
        # so points and pose share one gravity-aligned world frame.
        if gravity_down_cam is not None:
            R_world_cam = self._R_world_cam_from_gravity(gravity_down_cam)
        else:
            R_world_cam = self._R_WORLD_FROM_CAM

        img_t = self._image_to_tensor(image_r)
        cam = self._build_camera(target, target, fx, fy, cx, cy)
        pose = self._pose_from_R(R_world_cam)
        sdp = self._build_sdp(depth, W0, H0, target, fx, fy, R_world_cam)
        bb2d = self._to_boxer_order(bbox_r)

        datum = {
            "img0": img_t.to(self.device),
            "cam0": cam,
            "T_world_rig0": pose,
            "rotated0": torch.tensor([False], device=self.device),
            "sdp_w": torch.from_numpy(sdp).to(self.device).float(),
            "bb2d": torch.from_numpy(bb2d).to(self.device).float(),
        }
        return datum, cam, pose, img_t

    def _forward(self, datum: dict):
        """Run ``BoxerNet.forward`` under the GPU-appropriate autocast precision.

        ``self.autocast_dtype`` is resolved once at load from the GPU's compute
        capability (bf16 on Ampere+/L4, fp16 on Turing/T4, None=fp32 on
        MPS/CPU/pre-Volta); see ``ai.gpu.precision``. Only CUDA yields a
        non-None dtype, so ``device_type="cuda"`` is always correct here.
        """
        if self.autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
                return self.net.forward(datum)
        return self.net.forward(datum)

    # ------------------------------------------------------------------
    # Input helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _image_to_tensor(image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

    @staticmethod
    def _resize_depth(depth: np.ndarray, target: int, interp: str = "bilinear") -> np.ndarray:
        try:
            import cv2

            flag = cv2.INTER_NEAREST if interp == "nearest" else cv2.INTER_LINEAR
            return cv2.resize(depth, (target, target), interpolation=flag).astype(np.float32)
        except ImportError:
            resample = Image.NEAREST if interp == "nearest" else Image.BILINEAR
            depth_pil = Image.fromarray(depth).resize((target, target), resample)
            return np.asarray(depth_pil, dtype=np.float32)

    def _build_sdp(
        self,
        depth: np.ndarray,
        W0: int,
        H0: int,
        target: int,
        fx: float,
        fy: float,
        R_world_cam: Optional[np.ndarray],
    ) -> np.ndarray:
        """Build the sdp point cloud per the configured source / density.

        "native" back-projects the original-resolution depth map with the
        original intrinsics (fx/fy un-scaled from the square-resize). The 3D
        geometry is identical to the legacy "resized" path, but no depth values
        are interpolated across object boundaries, so no flying points are
        synthesized at occlusion edges. Falls back to "resized" when the depth
        map does not match the image size.

        The sampling stride is derived from `sdp_target_points`; 14400 on the
        legacy path reproduces the historical stride-8 grid exactly.
        """
        n_target = max(1, self.sdp_target_points)
        if self.sdp_source == "native":
            if depth.shape == (H0, W0):
                stride = max(1, int(round(math.sqrt(W0 * H0 / n_target))))
                # Undo the per-axis square-resize scaling (fx = f_native * sx).
                fx0 = fx * W0 / target
                fy0 = fy * H0 / target
                return self._depth_to_sdp(
                    depth, cam_fx=fx0, cam_fy=fy0, cx=W0 / 2.0, cy=H0 / 2.0,
                    stride=stride, R_world_cam=R_world_cam,
                )
            logger.warning(
                "sdp native source: depth shape %s != image (%d, %d); "
                "falling back to resized path",
                depth.shape, H0, W0,
            )
        depth_r = self._resize_depth(depth, target, self.sdp_interp)
        stride = max(1, int(round(target / math.sqrt(n_target))))
        return self._depth_to_sdp(
            depth_r, cam_fx=fx, cam_fy=fy, cx=target / 2.0, cy=target / 2.0,
            stride=stride, R_world_cam=R_world_cam,
        )

    def _build_camera(self, W: int, H: int, fx: float, fy: float, cx: float, cy: float):
        """Pinhole CameraTW with provided intrinsics."""
        width = torch.tensor([float(W)], dtype=torch.float32)
        height = torch.tensor([float(H)], dtype=torch.float32)
        params = torch.tensor([[fx, fy, cx, cy]], dtype=torch.float32)
        return self.CameraTW.from_surreal(
            width=width,
            height=height,
            type_str="Pinhole",
            params=params,
        ).to(self.device)

    # Camera "Y-down" → world "Z-down" (gravity along world -Z). This is
    # BoxerNet's world convention and the Omni3D/SUN-RGBD monocular default used
    # when no per-image gravity/pose is available. Passing identity instead made
    # BoxerNet assume the camera looks ALONG gravity (top-down), wrecking box
    # orientation for normal eye-level room photos.
    _R_WORLD_FROM_CAM = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float32
    )

    def _pose_from_R(self, R_world_cam: np.ndarray):
        """Build a world<-rig PoseTW (t=0) from a 3x3 world<-cam rotation."""
        R = torch.from_numpy(np.ascontiguousarray(R_world_cam, dtype=np.float32)).reshape(-1)
        trans = torch.zeros(3)
        data = torch.cat([R, trans], dim=0).unsqueeze(0)  # (1, 12) row-major R + t
        return self.PoseTW(data).to(self.device)

    @staticmethod
    def _R_world_cam_from_gravity(gravity_down_cam: np.ndarray) -> np.ndarray:
        """World<-cam rotation from the gravity (down) direction in camera coords.

        World is gravity-aligned: Z up, Y forward (horizontal), X right.
        Reduces exactly to `_R_WORLD_FROM_CAM` when the camera is level
        (gravity = camera +Y). Heading/yaw is fixed by projecting the camera's
        optical axis (+Z) onto the horizontal plane.
        """
        g = np.asarray(gravity_down_cam, dtype=np.float32)
        g = g / (np.linalg.norm(g) + 1e-9)
        up = -g  # world up, in camera coords
        fwd = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # camera optical axis
        f = fwd - np.dot(fwd, up) * up  # horizontal forward
        if np.linalg.norm(f) < 1e-4:  # looking near-vertically; pick any horizontal
            f = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            f = f - np.dot(f, up) * up
        f = f / (np.linalg.norm(f) + 1e-9)
        r = np.cross(f, up)  # world right (right-handed: X = Y x Z)
        r = r / (np.linalg.norm(r) + 1e-9)
        # Columns are world X, Y, Z expressed in camera coords -> R_cam_world.
        R_cam_world = np.stack([r, f, up], axis=1).astype(np.float32)
        return R_cam_world.T  # R_world_cam

    @staticmethod
    def _depth_to_sdp(
        depth: np.ndarray,
        cam_fx: float,
        cam_fy: float,
        cx: float,
        cy: float,
        stride: int = 8,
        R_world_cam: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Dense depth → (N, 3) 3D points in the world frame.

        Boxer's `prepare_inputs` expects (N, 3) and unsqueezes the batch dim itself.
        Points are back-projected in the camera frame, then rotated by
        `R_world_cam` into BoxerNet's gravity-aligned world (matching the pose
        passed as `T_world_rig`). Without this rotation the points would be in
        the camera frame while the pose claims a Z-down world — inconsistent.
        """
        h, w = depth.shape
        vs = np.arange(0, h, stride, dtype=np.float32)
        us = np.arange(0, w, stride, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)
        zz = depth[vv.astype(int), uu.astype(int)]
        valid = zz > 1e-4
        uu, vv, zz = uu[valid], vv[valid], zz[valid]
        xx = (uu - cx) * zz / cam_fx
        yy = (vv - cy) * zz / cam_fy
        pts = np.stack([xx, yy, zz], axis=-1).astype(np.float32)
        if R_world_cam is not None and len(pts):
            pts = (pts @ R_world_cam.T).astype(np.float32)
        if len(pts) == 0:
            pts = np.zeros((1, 3), dtype=np.float32)
        return pts  # (N, 3) in world frame

    @staticmethod
    def _to_boxer_order(bboxes_xyxy: np.ndarray) -> np.ndarray:
        """xyxy → boxer's (x1, x2, y1, y2) order."""
        b = np.asarray(bboxes_xyxy, dtype=np.float32)
        return b[:, [0, 2, 1, 3]][np.newaxis, ...]  # (1, M, 4)

    # ------------------------------------------------------------------
    # Output conversion
    # ------------------------------------------------------------------
    def _to_results(self, obb_w, labels: List[str]) -> List[BoxerObb]:
        """Convert ObbTW rows into `BoxerObb`, filtered by confidence."""
        probs = obb_w.prob.squeeze(-1).numpy()  # (M,)
        diag = obb_w.bb3_diagonal.numpy()  # (M, 3): (footprint_a, footprint_b, vertical)
        volumes = obb_w.bb3_volumes.squeeze(-1).numpy()  # (M,)
        centers = obb_w.bb3_center_world.numpy()  # (M, 3)

        out: List[BoxerObb] = []
        for i, label in enumerate(labels):
            if i >= len(probs):
                break
            if probs[i] < self.conf_threshold:
                continue
            # Boxer's yaw rotates about gravity, so the 3rd bb3_diagonal component
            # is the true vertical extent (height); the first two are the
            # horizontal footprint pair that interchange under the 90-degree yaw
            # symmetry (cf. boxer align_boxes_r90). Mislabeling these would feed
            # the wrong axes into the per-class dimension bounds downstream.
            foot_a, foot_b, vert = diag[i].tolist()
            out.append(
                BoxerObb(
                    width_m=float(foot_a),
                    depth_m=float(foot_b),
                    height_m=float(vert),
                    volume_m3=float(volumes[i]),
                    center_world=centers[i].tolist(),
                    confidence=float(probs[i]),
                    label=label,
                    input_index=i,
                )
            )
        return out
