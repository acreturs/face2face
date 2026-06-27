import json
import os

import cv2
import numpy as np


class PandoraFrame:
    """One Pandora frame containing RGB, depth, and annotations."""

    def __init__(self, root: str, index: int) -> None:
        stem = f"{index:06d}"

        rgb_path = os.path.join(root, "RGB", f"{stem}_RGB.png")
        rgba = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise FileNotFoundError(f"Could not load RGB image: {rgb_path}")

        if rgba.ndim == 3 and rgba.shape[2] == 4:
            self.rgb = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGB)
        elif rgba.ndim == 3 and rgba.shape[2] == 3:
            self.rgb = cv2.cvtColor(rgba, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unexpected RGB image shape: {rgba.shape}")

        depth_path = os.path.join(root, "DEPTH", f"{stem}_DEPTH.png")
        self.depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if self.depth_mm is None:
            raise FileNotFoundError(f"Could not load depth image: {depth_path}")

        annotation_path = os.path.join(root, "data.json")
        with open(annotation_path, "r", encoding="utf-8") as file:
            annotations = json.load(file)

        record = next(
            (item for item in annotations if item["frame_num"] == index),
            None,
        )
        if record is None:
            raise ValueError(f"No annotation found for frame {index}")

        self.joints2d = np.asarray(record["joints"], dtype=np.float64)
        self.joints3d = np.asarray(record["joints3D"], dtype=np.float64)


def rgb_at_depth_resolution(frame: PandoraFrame) -> np.ndarray:
    """Resize RGB to the depth-map resolution.

    This is a diagnostic approximation. Proper RGB-depth registration should
    eventually use the camera calibration supplied by the dataset.
    """
    height, width = frame.depth_mm.shape[:2]
    return cv2.resize(
        frame.rgb,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def backproject(depth_mm: np.ndarray, K: dict) -> np.ndarray:
    """Back-project a depth image into camera-frame 3D points in metres."""
    required = ("fx", "fy", "cx", "cy")
    if not all(name in K for name in required):
        raise ValueError(f"K must contain {required}")

    rows, cols = np.where(depth_mm > 0)
    z = depth_mm[rows, cols].astype(np.float64) / 1000.0
    x = (cols - float(K["cx"])) / float(K["fx"]) * z
    y = -(rows - float(K["cy"])) / float(K["fy"]) * z

    return np.stack([x, y, z], axis=1)


def crop_head(
    points: np.ndarray,
    head_center: np.ndarray,
    radius: float = 0.13,
) -> np.ndarray:
    """Keep points within a radius of the estimated head centre."""
    distances = np.linalg.norm(points - head_center.reshape(1, 3), axis=1)
    cropped = points[distances < radius]
    if cropped.size == 0:
        raise RuntimeError("Head crop is empty. Check the head-centre estimate.")
    return cropped


def keep_front(points: np.ndarray, slab: float = 0.10) -> np.ndarray:
    """Keep the front-most depth slab of the cropped head point cloud."""
    if len(points) == 0:
        raise ValueError("Cannot keep front surface of an empty point cloud")

    nearest_z = points[:, 2].min()
    front = points[points[:, 2] < nearest_z + slab]
    if front.size == 0:
        raise RuntimeError("Front-surface crop is empty.")
    return front


def head_center(frame: PandoraFrame) -> np.ndarray:
    """Estimate the head centre from the two highest 3D skeleton joints."""
    joints = frame.joints3d
    if joints.ndim != 2 or joints.shape[1] != 3 or len(joints) < 2:
        raise ValueError(f"Unexpected joints3D shape: {joints.shape}")

    highest = joints[np.argsort(joints[:, 1])[-2:]]
    center = highest.mean(axis=0)
    center[1] -= 0.04
    return center
