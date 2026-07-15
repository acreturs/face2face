# runs the whole proof-of-concept pipeline end to end:
#   load BFM -> load a Pandora frame -> fit the face params over N iters ->
#   render the result -> save an input-vs-reconstruction image
#
# run from the project root:  python3 python/main.py
# output goes to:             data/out/fit_result.png
import os

import matplotlib
matplotlib.use("Agg")              # no display needed, works inside a container
import matplotlib.pyplot as plt
import numpy as np
import yaml

import cv2

from bfm import BFM
from dataset import PandoraFrame, backproject, crop_head, keep_front, head_center, rgb_at_depth_resolution
from fit import Fitter
from landmarks import (
    create_landmark_detector,
    detect_landmarks,
    bfm_correspondences,
    landmark_correspondence_pairs,
    draw_landmark_labels,
)
from render import render, colorize_depth

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_config():
    with open(os.path.join(HERE, "config.yaml")) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    K = cfg["dataset"]["intrinsics"]
    frame_idx = cfg["dataset"]["frame_index"]

    # 1. load the BFM (the generator)
    bfm = BFM(os.path.join(ROOT, cfg["bfm"]["model_path"]))

    # 2. load one Pandora frame
    frame = PandoraFrame(os.path.join(ROOT, cfg["dataset"]["root"]), frame_idx)

    # 3. depth into a 3D point cloud then crop the face, backprojection is in
    #    metres but the fitter wants mm (BFM units) so we convert below
    cloud = backproject(frame.depth_mm, K)
    center = head_center(frame)
    face = keep_front(crop_head(cloud, center, radius=0.13), slab=0.10)
    if len(face) > 2500:                       # take fewer points for a fast fit
        face = face[np.random.RandomState(0).choice(len(face), 2500, replace=False)]
    face_mm = face * 1000.0
    print(f"[data] face cloud: {len(face)} points around {np.round(center, 3)} m")

    facemark = create_landmark_detector()
    landmark_image = rgb_at_depth_resolution(frame)

    faces = detect_landmarks(landmark_image,facemark,rgb=True,)   
    if len(faces) == 0:
        print("[landmarks] no face landmarks detected; continuing without sparse term")
        landmark_2d = None
        landmark_vertex_indices = None
        correspondences = {}
    else:
        landmarks2d = faces[0]
        correspondences = bfm_correspondences(bfm)
        landmark_2d, landmark_vertex_indices = landmark_correspondence_pairs(
            landmarks2d, correspondences)
        print(f"[landmarks] using {len(landmark_2d)} sparse landmark constraints")

    if len(faces) > 0 and not correspondences:
        raise RuntimeError(
            "No BFM landmark correspondences were resolved. "
            "Check the BFM landmark metadata parser and names.")

    out_dir = os.path.join(ROOT, cfg["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)

    # 4. fit the parameters (the analysis-by-synthesis loop)
    fitter = Fitter(bfm,
                    n_shape=cfg["fit"]["n_shape_coeffs"],
                    reg=cfg["fit"]["regularization"],
                    sparse_landmark_weight=cfg["fit"].get("sparse_landmark_weight", 1.0))
    p = fitter.fit(face_mm, K,
                 landmark_2d=landmark_2d,
                 landmark_vertex_indices=landmark_vertex_indices,
                 n_iters=cfg["fit"]["n_iters"],use_dense=cfg["fit"].get("use_dense", True),)

    if len(faces) > 0 and len(landmark_2d) > 0:
        label_vis = draw_landmark_labels(landmark_image, landmarks2d, correspondences, rgb=True)
        label_path = os.path.join(out_dir, "landmarks_detected.png")
        cv2.imwrite(label_path, cv2.cvtColor(label_vis, cv2.COLOR_RGB2BGR))
        print(f"[landmarks] wrote labeled detection overlay to {label_path}")

        # reproject the fitted BFM landmark points back into the same rgb-scaled image
        lm3d = bfm.shape(p[7:])[landmark_vertex_indices]
        lm3d = fitter.transform_points(lm3d, p)
        uv = fitter.project(lm3d, K)
        reproj_vis = cv2.cvtColor(landmark_image.copy(), cv2.COLOR_RGB2BGR)
        for gt, pr in zip(landmark_2d, uv):
            gt_xy = (int(round(gt[0])), int(round(gt[1])))
            pr_xy = (int(round(pr[0])), int(round(pr[1])))
            cv2.line(reproj_vis, gt_xy, pr_xy, (0, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.circle(reproj_vis, gt_xy, 3, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(reproj_vis, pr_xy, 3, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        reproj_path = os.path.join(out_dir, "landmark_reprojection.png")
        cv2.imwrite(reproj_path, reproj_vis)
        print(f"[landmarks] wrote reprojection overlay to {reproj_path}")

    # 5. render the fitted full-resolution mesh with its albedo
    #    same transform the fitter used but now on all 53k vertices (mm)
    from fit import axis_angle_to_R
    alpha = p[7:]
    V = bfm.shape(alpha) - fitter.c0
    Rm = axis_angle_to_R(p[:3])
    s = np.exp(p[6])
    world = (s * (Rm @ (fitter.R_init @ V.T))).T + p[3:6]
    H, W = frame.depth_mm.shape
    recon = render(world, bfm.albedo(), K, height=H, width=W)

    # zoom both depth and reconstruction to the fitted face's bounding box so the
    # input-vs-reconstruction comparison is actually readable
    from render import project
    u, v, z = project(world, K)
    m = z > 1e-3
    pad = 30
    c0, c1 = max(0, int(u[m].min()) - pad), min(W, int(u[m].max()) + pad)
    r0, r1 = max(0, int(v[m].min()) - pad), min(H, int(v[m].max()) + pad)
    depth_vis = colorize_depth(frame.depth_mm)

    # 6. save the input depth face next to the reconstruction, plus rgb for context
    # this whole block is only about drawing the images
    out_dir = os.path.join(ROOT, cfg["output"]["dir"])
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(frame.rgb)
    ax[0].set_title("RGB (context)")
    ax[1].imshow(depth_vis[r0:r1, c0:c1])
    ax[1].set_title("Input depth (face)")
    ax[2].imshow(recon[r0:r1, c0:c1])
    ax[2].set_title("Reconstructed BFM fit")
    for a in ax:
        a.axis("off")
    out_path = os.path.join(out_dir, "fit_result.png")
    fig.tight_layout(); fig.savefig(out_path, dpi=120)
    print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
