# loader for the Biwi Kinect head pose dataset (sequences live in data/biwi/NN/)
#
# rgb and depth come from the same Kinect and each sequence has a calibrated
# transform between the two cameras (rgb.cal / depth.cal), so the sparse
# rgb-landmark term and the dense depth term share one camera system
#
# conventions (checked visually on seq 01, see python/check_biwi.py):
#   - depth .bin is 640x480 uint16 mm, run-length encoded (see read_depth_bin)
#     background is zeroed so only the person is left (~18% of pixels)
#   - the depth camera is the world frame (depth.cal has identity R, zero t)
#   - rgb.cal R, t map depth-cam points into the rgb camera:
#         p_rgb = R @ p_depth + t     (t in mm)
#     using the inverse instead lands about 20 px off
#   - axes are OpenCV style, +X right, +Y down, +Z forward, same as the C++ side
#     everything is in mm
#   - frame_XXXXX_pose.txt gives the ground-truth head rotation (3x3) and head
#     centre (mm, depth cam), handy as a free crop centre and for pose checks
import glob
import os
import re
import struct

import cv2
import numpy as np


def read_depth_bin(path: str) -> np.ndarray:
    """decode one Biwi run-length depth file into a (480, 640) uint16 mm array

    layout is int32 width, int32 height, then repeated blocks of
    (int32 num_empty, int32 num_full, num_full int16 depth values)
    the num_empty pixels are background zeros, the num_full carry values, row-major
    """
    with open(path, "rb") as f:
        buf = f.read()
    width, height = struct.unpack_from("<ii", buf, 0)
    offset = 8
    depth = np.zeros(width * height, np.uint16)
    p = 0
    while p < width * height and offset < len(buf):
        num_empty, num_full = struct.unpack_from("<ii", buf, offset)
        offset += 8
        p += num_empty
        depth[p:p + num_full] = np.frombuffer(buf, np.int16, num_full, offset)
        p += num_full
        offset += 2 * num_full
    return depth.reshape(height, width)


def read_cal(path: str):
    """parse a Biwi .cal file into (K 3x3, dist 4, R 3x3, t 3, (width, height))"""
    values = [float(x) for x in open(path).read().split()]
    K = np.array(values[0:9]).reshape(3, 3)
    dist = np.array(values[9:13])           # all zero in this dataset
    R = np.array(values[13:22]).reshape(3, 3)
    t = np.array(values[22:25])             # mm
    size = (int(values[25]), int(values[26]))
    return K, dist, R, t, size


def read_pose(path: str):
    """parse frame_XXXXX_pose.txt into (head rotation 3x3, head centre mm)"""
    values = [float(x) for x in open(path).read().split()]
    return np.array(values[:9]).reshape(3, 3), np.array(values[9:12])


def list_frames(seq_dir: str) -> list:
    """frame numbers present in a sequence dir, Biwi numbering starts at 3"""
    frames = []
    for p in glob.glob(os.path.join(seq_dir, "frame_*_rgb.png")):
        m = re.search(r"frame_(\d+)_rgb\.png$", p)
        if m:
            frames.append(int(m.group(1)))
    return sorted(frames)


class BiwiFrame:
    """one Biwi frame, holds rgb, depth (mm) and the ground-truth head pose"""

    def __init__(self, root: str, seq: str, frame: int) -> None:
        seq_dir = os.path.join(root, seq)
        stem = os.path.join(seq_dir, f"frame_{frame:05d}")

        bgr = cv2.imread(f"{stem}_rgb.png")
        if bgr is None:
            raise FileNotFoundError(f"Could not load RGB image: {stem}_rgb.png")
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # 480x640x3

        self.depth_mm = read_depth_bin(f"{stem}_depth.bin")      # 480x640 uint16
        self.head_rotation, self.head_center_mm = read_pose(f"{stem}_pose.txt")

        self.K_depth, _, _, _, _ = read_cal(os.path.join(seq_dir, "depth.cal"))
        self.K_rgb, _, self.R_rgb, self.t_rgb_mm, _ = read_cal(
            os.path.join(seq_dir, "rgb.cal"))


def backproject_mm(depth_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    """depth image into Nx3 points in mm, depth-cam frame (+Y down)"""
    rows, cols = np.nonzero(depth_mm)
    z = depth_mm[rows, cols].astype(np.float64)
    x = (cols - K[0, 2]) / K[0, 0] * z
    y = (rows - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], axis=1)


def depth_to_rgb_cam(points_mm: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """move depth-cam points (mm) into the rgb camera, p' = R p + t"""
    return points_mm @ R.T + t


def project(points_mm: np.ndarray, K: np.ndarray) -> np.ndarray:
    """pinhole projection of Nx3 camera-frame points into Nx2 pixels"""
    uv = (points_mm / points_mm[:, 2:]) @ K.T
    return uv[:, :2]


def crop_head_mm(points_mm: np.ndarray, center_mm: np.ndarray,
                 radius: float = 90.0, front_slab: float = 80.0) -> np.ndarray:
    """keep the head only, mirrors the C++ cropHead with a tight radius"""
    keep = np.linalg.norm(points_mm - center_mm.reshape(1, 3), axis=1) < radius
    points = points_mm[keep]
    if points.size == 0:
        raise RuntimeError("Head crop is empty — check the pose head centre.")
    front = points[:, 2] < points[:, 2].min() + front_slab
    return points[front]
