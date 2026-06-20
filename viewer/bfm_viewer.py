"""Tiny Flask-based BFM viewer.

Loads the BFM .h5 once, serves a single-page three.js viewer with sliders for
the first few identity (α), expression (δ), and albedo (β) PCA components.

Run:
    python3 viewer/bfm_viewer.py
    # then open http://localhost:5000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory


# ── BFM loading ──────────────────────────────────────────────────────────────

class BFM:
    def __init__(self, path: str):
        with h5py.File(path, "r") as f:
            self.shape_mean  = f["shape/model/mean"][:].astype(np.float32)
            self.shape_basis = f["shape/model/pcaBasis"][:].astype(np.float32)
            self.shape_std   = np.sqrt(f["shape/model/pcaVariance"][:].astype(np.float32))

            self.expr_mean  = f["expression/model/mean"][:].astype(np.float32)
            self.expr_basis = f["expression/model/pcaBasis"][:].astype(np.float32)
            self.expr_std   = np.sqrt(f["expression/model/pcaVariance"][:].astype(np.float32))

            self.color_mean  = f["color/model/mean"][:].astype(np.float32)
            self.color_basis = f["color/model/pcaBasis"][:].astype(np.float32)
            self.color_std   = np.sqrt(f["color/model/pcaVariance"][:].astype(np.float32))

            self.triangles = f["shape/representer/cells"][:].astype(np.int32).T  # (M,3)

        self.n_vertices = self.shape_mean.size // 3
        self.k_id   = self.shape_basis.shape[1]
        self.k_exp  = self.expr_basis.shape[1]
        self.k_alb  = self.color_basis.shape[1]

        # Pre-scale bases by σ so we can do a single matmul (B·c) per update.
        self._scaled_shape = self.shape_basis * self.shape_std[None, :]
        self._scaled_expr  = self.expr_basis  * self.expr_std[None, :]
        self._scaled_color = self.color_basis * self.color_std[None, :]

        # Whether the color model is stored in 0..255; decide once.
        self._color_scale = (1.0 / 255.0) if self.color_mean.max() > 1.5 else 1.0

    def vertices_bytes(self, alpha: np.ndarray, delta: np.ndarray) -> bytes:
        ka, kd = alpha.size, delta.size
        v = self.shape_mean.copy()
        if ka:
            v += self._scaled_shape[:, :ka] @ alpha
        v += self.expr_mean
        if kd:
            v += self._scaled_expr[:, :kd] @ delta
        return v.astype(np.float32, copy=False).tobytes()

    def colors_bytes(self, beta: np.ndarray) -> bytes:
        kb = beta.size
        c = self.color_mean.copy()
        if kb:
            c += self._scaled_color[:, :kb] @ beta
        c = c * self._color_scale
        np.clip(c, 0.0, 1.0, out=c)
        return c.astype(np.float32, copy=False).tobytes()



# ── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=None)
bfm: BFM | None = None

# Number of sliders exposed per parameter group (keeps UI manageable)
N_ALPHA_SLIDERS = 20
N_DELTA_SLIDERS = 10
N_BETA_SLIDERS  = 20

# Binary mimetype for vertex / color payloads
OCTET = "application/octet-stream"


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.route("/api/init")
def init():
    """One-shot payload: topology + slider counts."""
    return jsonify({
        "n_vertices":   bfm.n_vertices,
        "k_id":         bfm.k_id,
        "k_exp":        bfm.k_exp,
        "k_alb":        bfm.k_alb,
        "n_alpha":      N_ALPHA_SLIDERS,
        "n_delta":      N_DELTA_SLIDERS,
        "n_beta":       N_BETA_SLIDERS,
        "triangles":    bfm.triangles.flatten().tolist(),
        "n_triangles":  int(bfm.triangles.shape[0]),
    })


def _coef(name: str) -> np.ndarray:
    """Read a coefficient array from request JSON, defaulting to empty."""
    data = request.get_json(force=True, silent=True) or {}
    return np.asarray(data.get(name, []), dtype=np.float32)


@app.route("/api/vertices", methods=["POST"])
def vertices():
    """Slider change in α / δ → fresh vertex positions (binary float32)."""
    alpha = _coef("alpha")
    delta = _coef("delta")
    return Response(bfm.vertices_bytes(alpha, delta), mimetype=OCTET)


@app.route("/api/colors", methods=["POST"])
def colors():
    """Slider change in β → fresh per-vertex colors (binary float32)."""
    beta = _coef("beta")
    return Response(bfm.colors_bytes(beta), mimetype=OCTET)


# ── Snapshot loader ──────────────────────────────────────────────────────────
# A "snapshot" is a saved mesh on disk: a .obj file with vertex colours, in the
# extended format `v x y z r g b` + `f i0 i1 i2` (1-indexed). The C++ pipeline
# already writes these via writeObj(). Anything in data/snapshots/ (or data/out/)
# is offered to the viewer as an alternate mesh source via the dropdown.

SNAPSHOT_DIRS = [Path("data/snapshots"), Path("data/out")]


def _load_obj(path: Path) -> dict:
    """Parse extended .obj — supports vertex colours, normals, and faces with
    optional normal indices.

        v  x y z [r g b]
        vn nx ny nz
        f  v|v//vn|v/vt|v/vt/vn (×3)   (1-indexed)
    """
    verts, cols, norms, faces = [], [], [], []
    with open(path) as f:
        for line in f:
            if len(line) < 2:
                continue
            if line[0] == "v" and line[1].isspace():
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(parts) >= 7:
                    cols.append([float(parts[4]), float(parts[5]), float(parts[6])])
            elif line[0] == "v" and line[1] == "n":
                parts = line.split()
                norms.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line[0] == "f" and line[1].isspace():
                ids = [int(p.split("/")[0]) - 1 for p in line.split()[1:4]]
                faces.append(ids)

    V = np.asarray(verts, dtype=np.float32)
    F = np.asarray(faces, dtype=np.int32)
    C = (np.asarray(cols, dtype=np.float32)
         if cols else np.full_like(V, 0.7))
    N = np.asarray(norms, dtype=np.float32) if norms else None
    return {
        "vertices":  V,
        "triangles": F,
        "colors":    np.clip(C, 0.0, 1.0),
        "normals":   N,   # may be None
    }


def _list_snapshots() -> list[dict]:
    out = []
    for d in SNAPSHOT_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.obj")):
            try:
                size = p.stat().st_size
                out.append({"name": p.name, "path": str(p), "bytes": size})
            except OSError:
                pass
    return out


def _find_snapshot(name: str) -> Path | None:
    for s in _list_snapshots():
        if s["name"] == name:
            return Path(s["path"])
    return None


@app.route("/api/snapshots")
def snapshots():
    """List available .obj files in data/snapshots/ and data/out/."""
    return jsonify(_list_snapshots())


@app.route("/api/snapshot")
def snapshot():
    """One-shot payload for a named snapshot: triangles + flags."""
    name = request.args.get("name", "")
    p = _find_snapshot(name)
    if p is None:
        return jsonify({"error": f"snapshot not found: {name}"}), 404
    m = _load_obj(p)
    return jsonify({
        "n_vertices":  int(m["vertices"].shape[0]),
        "n_triangles": int(m["triangles"].shape[0]),
        "triangles":   m["triangles"].flatten().tolist(),
        "has_normals": m["normals"] is not None,
    })


@app.route("/api/snapshot/vertices")
def snapshot_vertices():
    """Binary float32 vertices for a named snapshot."""
    name = request.args.get("name", "")
    p = _find_snapshot(name)
    if p is None:
        return Response(b"", status=404, mimetype=OCTET)
    return Response(_load_obj(p)["vertices"].tobytes(), mimetype=OCTET)


@app.route("/api/snapshot/colors")
def snapshot_colors():
    """Binary float32 per-vertex colors for a named snapshot."""
    name = request.args.get("name", "")
    p = _find_snapshot(name)
    if p is None:
        return Response(b"", status=404, mimetype=OCTET)
    return Response(_load_obj(p)["colors"].tobytes(), mimetype=OCTET)


@app.route("/api/snapshot/normals")
def snapshot_normals():
    """Binary float32 per-vertex normals (when present in the .obj)."""
    name = request.args.get("name", "")
    p = _find_snapshot(name)
    if p is None:
        return Response(b"", status=404, mimetype=OCTET)
    n = _load_obj(p)["normals"]
    if n is None:
        return Response(b"", status=404, mimetype=OCTET)
    return Response(n.tobytes(), mimetype=OCTET)


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfm", default="data/bfm/model2017-1_bfm_nomouth.h5",
                    help="Path to BFM .h5 (default: %(default)s)")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    global bfm
    print(f"Loading BFM from {args.bfm} ...")
    bfm = BFM(args.bfm)
    print(f"  {bfm.n_vertices} vertices, {bfm.triangles.shape[0]} triangles, "
          f"K=(id={bfm.k_id}, exp={bfm.k_exp}, alb={bfm.k_alb})")
    print(f"Open http://localhost:{args.port}")
    # threaded=True so a slow request doesn't block the next one
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
