"""Save a BFM mesh to data/snapshots/<name>.obj so the viewer can load it.

Two modes:

    # mean BFM face, vertex colours = albedo
    python3 viewer/save_snapshot.py --name mean_face

    # mean BFM face, vertex colours = normals-as-RGB ((n+1)/2)
    python3 viewer/save_snapshot.py --name mean_normals --color-mode normals

The .obj written here is exactly the format the C++ writeObj() produces, so the
viewer can load anything either side dumps.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import h5py


def _load_bfm(path: str):
    with h5py.File(path, "r") as f:
        shape_mean = f["shape/model/mean"][:].astype(np.float32)
        color_mean = f["color/model/mean"][:].astype(np.float32)
        tris = f["shape/representer/cells"][:].astype(np.int32).T
    return shape_mean.reshape(-1, 3), color_mean.reshape(-1, 3), tris


def _vertex_normals(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    v0, v1, v2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    face = np.cross(v1 - v0, v2 - v0)
    n = np.zeros_like(V)
    np.add.at(n, T[:, 0], face)
    np.add.at(n, T[:, 1], face)
    np.add.at(n, T[:, 2], face)
    mag = np.linalg.norm(n, axis=1, keepdims=True)
    mag[mag < 1e-12] = 1.0
    return n / mag


def _write_obj(path: Path, V: np.ndarray, C: np.ndarray, T: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v, c in zip(V, C):
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} "
                    f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        for t in T:
            f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfm",  default="data/bfm/model2017-1_bfm_nomouth.h5")
    ap.add_argument("--name", required=True, help="snapshot filename (no extension)")
    ap.add_argument("--color-mode", choices=["albedo", "normals"], default="albedo")
    ap.add_argument("--out",  default="data/snapshots")
    args = ap.parse_args()

    V, albedo, T = _load_bfm(args.bfm)

    if args.color_mode == "albedo":
        C = albedo / 255.0 if albedo.max() > 1.5 else albedo
        C = np.clip(C, 0.0, 1.0)
    else:  # normals
        N = _vertex_normals(V, T)
        C = 0.5 * (N + 1.0)   # (-1,1) → (0,1) RGB

    out_path = Path(args.out) / f"{args.name}.obj"
    _write_obj(out_path, V, C, T)
    print(f"wrote {out_path}  ({V.shape[0]} vertices, {T.shape[0]} triangles, "
          f"colours={args.color_mode})")


if __name__ == "__main__":
    main()
