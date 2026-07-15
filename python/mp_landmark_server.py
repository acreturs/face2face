"""mediapipe landmark coprocess for --mode live-cpu

the c++ live loop cannot link mediapipe (bazel-built) and the offline pre-pass
cannot run on a camera stream, so this server bridges the gap: main.cpp spawns
it once then per frame

  c++  -> stdin : 8-byte header (int32 le width, int32 le height) + raw bgr bytes
  here -> stdout: "N\n" then N lines "bfm_vertex_index u v"   (-1 = jaw contour)

it uses the same landmark set as gen_landmarks_mediapipe.py (21 interior + 14
jaw) so live gets the same pose and identity conditioning as the offline modes

prints "READY" once the slow mediapipe import and graph init is done, the c++
side waits for that line before sending the first frame and falls back to yunet
if it never comes

run manually for debugging:  python3 python/mp_landmark_server.py < /dev/null
"""
import struct
import sys

import numpy as np

import mediapipe as mp

# same mapping as the offline pre-pass, import instead of duplicating
from gen_landmarks_mediapipe import MP_TO_BFM, MP_JAW


def main() -> None:
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,          # video mode: tracks between frames
        max_num_faces=1,
        refine_landmarks=True,            # iris centres 468/473
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    inp = sys.stdin.buffer
    out = sys.stdout
    out.write("READY\n")
    out.flush()

    while True:
        hdr = inp.read(8)
        if len(hdr) < 8:                  # parent closed the pipe so exit
            return
        w, h = struct.unpack("<ii", hdr)
        need = w * h * 3
        buf = b""
        while len(buf) < need:
            chunk = inp.read(need - len(buf))
            if not chunk:
                return
            buf += chunk

        bgr = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        res = mesh.process(np.ascontiguousarray(bgr[:, :, ::-1]))  # bgr to rgb

        lines = []
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            for mp_idx, bfm_idx in MP_TO_BFM:
                lines.append(f"{bfm_idx} {lm[mp_idx].x * w:.3f} {lm[mp_idx].y * h:.3f}")
            for mp_idx in MP_JAW:
                lines.append(f"-1 {lm[mp_idx].x * w:.3f} {lm[mp_idx].y * h:.3f}")

        out.write(f"{len(lines)}\n")
        if lines:
            out.write("\n".join(lines) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
