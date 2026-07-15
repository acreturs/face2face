"""MediaPipe Face Mesh landmark pre-pass for the C++ solver.

Writes one `landmarks_mp_XXXXX.txt` per image (same `bfm_vertex_index u v`
format as gen_landmarks.py; `-1` = jaw-contour point matched dynamically by the
solver). The C++ pipeline reads these when run with `--detector mediapipe`.

Why a pre-pass: MediaPipe is Bazel-built and cannot link into the Makefile C++
pipeline, but its 468-point mesh is denser and more pose-robust than LBF, and
it provides INNER-FACE vertical mouth points (upper/lower lip) that the 5-point
YuNet set lacks.

Usage (from project root):
  # one image
  python3 python/gen_landmarks_mediapipe.py <image.png> <out.txt>
  # a Biwi sequence folder (frame_XXXXX_rgb.png -> landmarks_mp_XXXXX.txt)
  python3 python/gen_landmarks_mediapipe.py --biwi-dir data/BK-1/01 [--max N]
"""
import glob
import os
import re
import sys

import cv2

try:
    import mediapipe as mp
except ImportError:
    sys.exit(
        "mediapipe is not installed.\n"
        "  It is baked into the devcontainer image — the usual fix is to REBUILD\n"
        "  the container (Dev Containers: Rebuild Container).\n"
        "  For a quick one-off install in the current container instead:\n"
        "    pip3 install --break-system-packages --no-deps mediapipe==0.10.18\n"
        "    pip3 install --break-system-packages 'protobuf>=4.25.3,<5' "
        "attrs flatbuffers absl-py")

# MediaPipe canonical-mesh index -> BFM vertex index (subject-anatomical on
# both sides; verified against the BFM named-landmark table).
# 468/473 are the iris centres (refine_landmarks=True).
#
# 21 interior points, not 8: the original set (irises, nose tip, 4 mouth
# points, chin) is nearly COPLANAR, which leaves pitch/yaw poorly conditioned —
# small landmark noise tips the whole pose, worst on side views. Eye corners,
# eyelids, brows and the subnasale add depth variation and lateral spread; the
# solver's Huber loss handles the ones MediaPipe regresses while occluded.
MP_TO_BFM = [
    (468,  4540),   # right.eye.pupil.center
    (473, 11681),   # left.eye.pupil.center
    (1,    8156),   # center.nose.tip
    (2,    8168),   # center.nose.attachement_to_philtrum (subnasale)
    (61,   5779),   # right.lips.corner
    (291, 10598),   # left.lips.corner
    (0,    8181),   # center.lips.upper.outer  (vertical mouth signal)
    (17,   8199),   # center.lips.lower.outer
    (152, 47844),   # center.chin.tip
    (33,   2736),   # right.eye.corner_outer
    (133,  6219),   # right.eye.corner_inner
    (362,  9892),   # left.eye.corner_inner
    (263, 13360),   # left.eye.corner_outer
    (159,  4663),   # right.eye.top
    (145,  4417),   # right.eye.bottom
    (386, 11546),   # left.eye.top
    (374, 11687),   # left.eye.bottom
    (105, 39560),   # right.eyebrow.bend.upper
    (107, 40088),   # right.eyebrow.inner_upper
    (334, 41360),   # left.eyebrow.bend.upper
    (336, 40827),   # left.eyebrow.inner_upper
]

# Jawline (from FACEMESH_FACE_OVAL, sides only) -> contour observations (-1).
# 7 per side (was 4): the extra ear-adjacent (132/361) and chin-adjacent
# (176/148, 400/378) points tighten the silhouette that constrains face
# width/length — the identity signal the interior landmarks cannot provide.
MP_JAW = [132, 58, 172, 136, 150, 176, 148,      # subject-right side
          361, 288, 397, 365, 379, 400, 378]     # subject-left side

_mesh = None


def _face_mesh():
    global _mesh
    if _mesh is None:
        _mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True,           # adds the iris centres 468/473
            min_detection_confidence=0.5)
    return _mesh


def detect_to_file(image_path: str, out_path: str) -> bool:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"[mp] could not read {image_path}")
        return False
    res = _face_mesh().process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        print(f"[mp] no face in {image_path}")
        return False
    lm = res.multi_face_landmarks[0].landmark
    h, w = bgr.shape[:2]
    with open(out_path, "w", encoding="utf-8") as f:
        for mp_idx, bfm_idx in MP_TO_BFM:
            f.write(f"{bfm_idx} {lm[mp_idx].x * w:.6f} {lm[mp_idx].y * h:.6f}\n")
        for mp_idx in MP_JAW:
            f.write(f"-1 {lm[mp_idx].x * w:.6f} {lm[mp_idx].y * h:.6f}\n")
    return True


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--biwi-dir":
        seq, rest = args[1], args[2:]
        limit = int(rest[1]) if rest[:1] == ["--max"] else None
        images = sorted(glob.glob(os.path.join(seq, "frame_*_rgb.png")))
        if limit:
            images = images[:limit]
        n = 0
        for img in images:
            stem = re.search(r"frame_(\d+)_rgb", img).group(1)
            n += detect_to_file(img, os.path.join(seq, f"landmarks_mp_{stem}.txt"))
        print(f"[mp] {n}/{len(images)} frames -> {seq}/landmarks_mp_*.txt")
    elif len(args) == 2:
        ok = detect_to_file(args[0], args[1])
        print(f"[mp] {'wrote' if ok else 'FAILED'} {args[1]}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
