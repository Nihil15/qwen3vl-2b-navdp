"""Goal computation + input preprocessing for the DINO -> NavDP pipeline.

- preprocess_rgb / preprocess_depth: canonical NavDP preprocessing
  (aspect-preserving resize + pad to 224x224; RGB /255; depth meters
  clipped to [0.1, 5.0], invalid -> 0).
- goal_point_from_detection: bbox + depth + camera intrinsics -> 3D point
  goal [x fwd, y left, z up] in the robot/camera frame.
- goal_pixel_from_detection: bbox -> normalized [0,1] pixel goal.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


def preprocess_rgb(rgb: np.ndarray, image_size: int = 224) -> np.ndarray:
    """HxWx3 uint8 -> image_size x image_size x 3 float32 in [0,1]."""
    H, W = rgb.shape[:2]
    prop = image_size / max(H, W)
    resized = cv2.resize(rgb, None, fx=prop, fy=prop)
    pad_w = max((image_size - resized.shape[1]) // 2, 0)
    pad_h = max((image_size - resized.shape[0]) // 2, 0)
    padded = np.pad(resized, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")
    out = cv2.resize(padded, (image_size, image_size))
    return out.astype(np.float32) / 255.0


def preprocess_depth(depth_m: np.ndarray, image_size: int = 224) -> np.ndarray:
    """HxW float meters -> image_size x image_size x 1 float32, clipped."""
    depth = depth_m.astype(np.float32).copy()
    depth[~np.isfinite(depth)] = 0.0
    H, W = depth.shape[:2]
    prop = image_size / max(H, W)
    resized = cv2.resize(depth, None, fx=prop, fy=prop, interpolation=cv2.INTER_NEAREST)
    pad_w = max((image_size - resized.shape[1]) // 2, 0)
    pad_h = max((image_size - resized.shape[0]) // 2, 0)
    padded = np.pad(resized, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out = cv2.resize(padded, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    out[out > 5.0] = 0.0
    out[out < 0.1] = 0.0
    return out[:, :, np.newaxis]


def bbox_median_depth(depth_m: np.ndarray, box: np.ndarray, shrink: float = 0.25) -> Optional[float]:
    """Median of valid depth inside a centrally shrunk bbox (meters)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0s, x1s = int(x0 + w * shrink), int(x1 - w * shrink)
    y0s, y1s = int(y0 + h * shrink), int(y1 - h * shrink)
    x0s, y0s = max(x0s, 0), max(y0s, 0)
    x1s = min(max(x1s, x0s + 1), depth_m.shape[1])
    y1s = min(max(y1s, y0s + 1), depth_m.shape[0])
    patch = depth_m[y0s:y1s, x0s:x1s]
    valid = patch[np.isfinite(patch) & (patch > 0.1)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def pixel_depth_to_point(
    u: float,
    v: float,
    d: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stop_distance: float = 0.0,
) -> np.ndarray:
    """Pixel (u,v) at depth d (m) -> [x fwd, y left, z up] robot-frame goal.

    Camera optical frame (z fwd, x right, y down) is converted to the robot
    frame NavDP expects (x fwd, y left, z up). stop_distance shortens the
    goal along the ray so the rover stops in front of the object.
    """
    x_cam = (u - cx) / fx * d  # right
    y_cam = (v - cy) / fy * d  # down
    goal = np.array([d, -x_cam, -y_cam], dtype=np.float32)
    dist = np.linalg.norm(goal)
    if stop_distance > 0.0 and dist > stop_distance:
        goal *= (dist - stop_distance) / dist
    return goal


def goal_point_from_detection(
    box: np.ndarray,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    stop_distance: float = 0.0,
) -> Optional[np.ndarray]:
    """bbox + metric depth + pinhole intrinsics -> [x fwd, y left, z up] goal."""
    d = bbox_median_depth(depth_m, box)
    if d is None:
        return None
    u, v = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return pixel_depth_to_point(u, v, d, fx, fy, cx, cy, stop_distance)


def goal_pixel_from_detection(box: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """bbox -> normalized [x, y] in [0,1] (center of box) of the raw image."""
    H, W = image_shape[:2]
    u = (box[0] + box[2]) / 2.0 / W
    v = (box[1] + box[3]) / 2.0 / H
    return np.array([u, v], dtype=np.float32)


def intrinsics_from_fov(width: int, height: int, horizontal_fov_deg: float):
    """Approximate pinhole intrinsics from image size + horizontal FOV."""
    fx = width / (2.0 * np.tan(np.radians(horizontal_fov_deg) / 2.0))
    fy = fx
    return fx, fy, width / 2.0, height / 2.0
