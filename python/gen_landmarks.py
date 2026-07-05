"""Detect 2D face landmarks in an image and write the C++ solver input file
(one line per landmark: `bfm_vertex_index u v`). Works for any RGB image, so the
same tool feeds both the iPhone and the Biwi sparse fit.

Two landmark sets, chosen per scenario:
  --set small  (default) 9 reliable points — for the iPhone / sparse-only fit
  --set dense            25 points          — for the Biwi / dense scenario

Add --contour to also emit the dlib jawline points as CONTOUR observations,
written with vertex index -1 (the C++ solver assigns each to the nearest model
silhouette vertex dynamically, since the BFM has no named jaw landmarks). These
constrain the face width/outline, which the interior points cannot.

Usage (from project root):
  python3 python/gen_landmarks.py <image.png> <out_landmarks.txt> [debug.png]
  python3 python/gen_landmarks.py --set small --contour data/iphone/default/RGB/000000_RGB.png \
          data/iphone/default/landmarks_000000.txt
"""
import os
import sys

import cv2

from bfm import BFM
from landmarks import (
    LBF68_TO_BFM_DENSE,
    LBF68_TO_BFM_SMALL,
    bfm_correspondences,
    create_landmark_detector,
    detect_landmarks,
    draw_landmark_labels,
    landmark_correspondence_pairs,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BFM_PATH = os.path.join(ROOT, "data", "bfm", "model2017-1_bfm_nomouth.h5")


# dlib/LBF 68-pt jawline is indices 0..16 (0 = right ear side, 8 = chin,
# 16 = left ear side). We emit the sides (skip the ear-adjacent endpoints and
# the chin, which the named set already covers) as contour observations.
JAW_CONTOUR_LBF = [1, 3, 5, 7, 9, 11, 13, 15]


def main() -> None:
    args = sys.argv[1:]
    mapping = LBF68_TO_BFM_SMALL
    emit_contour = False
    # additive flags in any order before the positional args
    while args and args[0].startswith("--"):
        if args[0] == "--set":
            mapping = {"small": LBF68_TO_BFM_SMALL,
                       "dense": LBF68_TO_BFM_DENSE}[args[1]]
            args = args[2:]
        elif args[0] == "--contour":
            emit_contour = True
            args = args[1:]
        else:
            raise SystemExit(f"unknown flag: {args[0]}")
    if len(args) < 2:
        raise SystemExit(
            "usage: gen_landmarks.py [--set small|dense] [--contour] "
            "<image.png> <out.txt> [debug.png]")
    image_path, out_path = args[0], args[1]
    debug_path = args[2] if len(args) > 2 else None

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    faces = detect_landmarks(rgb, create_landmark_detector(), rgb=True)
    if not faces:
        raise RuntimeError(f"No face detected in {image_path}")

    landmarks = faces[0]                       # (68, 2)
    bfm = BFM(BFM_PATH)
    correspondence = bfm_correspondences(bfm, mapping)
    pts2d, vertex_indices = landmark_correspondence_pairs(landmarks, correspondence)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for vi, (u, v) in zip(vertex_indices, pts2d):
            f.write(f"{vi} {u:.6f} {v:.6f}\n")
        n_contour = 0
        if emit_contour:
            for idx in JAW_CONTOUR_LBF:
                u, v = landmarks[idx]
                f.write(f"-1 {u:.6f} {v:.6f}\n")   # -1 = contour, matched in C++
                n_contour += 1
    print(f"[landmarks] {len(pts2d)} fixed + {n_contour if emit_contour else 0} "
          f"contour points -> {out_path}")

    if debug_path:
        labeled = draw_landmark_labels(rgb, landmarks, correspondence, rgb=True)
        cv2.imwrite(debug_path, cv2.cvtColor(labeled, cv2.COLOR_RGB2BGR))
        print(f"[landmarks] labeled debug -> {debug_path}")


if __name__ == "__main__":
    main()
