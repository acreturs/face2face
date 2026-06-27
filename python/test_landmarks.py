import os

import cv2
import numpy as np

from bfm import BFM
from landmarks import (
    bfm_correspondences,
    create_landmark_detector,
    detect_landmarks,
    draw_landmark_labels,
    draw_landmarks,
)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    image_path = os.path.join(
        root,
        "data",
        "iphone",
        "default",
        "RGB",
        "000000_RGB.png",
    )
    bfm_path = os.path.join(
        root,
        "data",
        "bfm",
        "model2017-1_bfm_nomouth.h5",
    )
    output_dir = os.path.join(root, "data", "out")
    os.makedirs(output_dir, exist_ok=True)

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detector = create_landmark_detector()
    faces = detect_landmarks(image_rgb, detector, rgb=True)
    if not faces:
        raise RuntimeError("No face was detected")

    landmarks = faces[0]
    if landmarks.shape != (68, 2):
        raise AssertionError(
            f"Expected (68, 2), got {landmarks.shape}"
        )

    height, width = image_rgb.shape[:2]
    if not (
        np.all(landmarks[:, 0] >= 0)
        and np.all(landmarks[:, 0] < width)
        and np.all(landmarks[:, 1] >= 0)
        and np.all(landmarks[:, 1] < height)
    ):
        raise AssertionError("Some landmarks lie outside the image")

    bfm = BFM(bfm_path)
    correspondence = bfm_correspondences(bfm)
    if len(correspondence) < 6:
        raise AssertionError(
            f"Only {len(correspondence)} BFM correspondences resolved"
        )

    plain = draw_landmarks(image_rgb, landmarks, rgb=True)
    labeled = draw_landmark_labels(
        image_rgb,
        landmarks,
        correspondence,
        rgb=True,
    )

    cv2.imwrite(
        os.path.join(output_dir, "landmarks_debug.png"),
        cv2.cvtColor(plain, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(output_dir, "landmarks_debug_labels.png"),
        cv2.cvtColor(labeled, cv2.COLOR_RGB2BGR),
    )

    print("[test] landmark detection passed")
    print(f"[test] detected shape: {landmarks.shape}")
    print(f"[test] correspondences: {len(correspondence)}")


if __name__ == "__main__":
    main()
