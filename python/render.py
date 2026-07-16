# small renderer, just a proof of concept, mirrors the future C++ Renderer
# it is a simple z-buffered point splat, not the differentiable triangle
# rasteriser the paper uses, just enough to see the fitted face
import cv2
import numpy as np


def project(points, K):
    """3D camera-frame points (metres) into pixel coords, returns (u, v, z)"""
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    u = K["fx"] * X / Z + K["cx"]
    v = -K["fy"] * Y / Z + K["cy"]
    return u, v, Z


def render(points, colors, K, height, width):
    """draw coloured 3D points into an (H, W, 3) image using a z-buffer

    nearest point per pixel wins and a small close-up fills the gaps between splats
    colors are rgb in [0, 1]
    """
    u, v, z = project(points, K)
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)

    inside = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height) & (z > 1e-3)
    ui, vi, z = ui[inside], vi[inside], z[inside]
    col = (colors[inside] * 255).astype(np.uint8)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    zbuf = np.full((height, width), np.inf)

    order = np.argsort(-z)                 # far to near, so near ones overwrite
    img[vi[order], ui[order]] = col[order]
    zbuf[vi[order], ui[order]] = z[order]

    # fill the small holes between projected points
    mask = (zbuf == np.inf).astype(np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return img


def colorize_depth(depth_mm):
    """turn the almost-black 16-bit depth into something you can actually look at
    by normalising over its valid range"""
    valid = depth_mm > 0
    out = np.zeros_like(depth_mm, dtype=np.uint8)
    if valid.any():
        lo, hi = depth_mm[valid].min(), depth_mm[valid].max()
        norm = np.zeros_like(depth_mm, dtype=np.float32)
        norm[valid] = (depth_mm[valid] - lo) / max(hi - lo, 1)
        out = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(out, cv2.COLORMAP_TURBO)[:, :, ::-1]  # BGR->RGB
