"""HEIC → PNG conversion + Pandora-shaped session layout.

Builds:

    data/iphone/<session>/
        RGB/000000_RGB.png ...
        intrinsics.json
"""
from __future__ import annotations

import json
from pathlib import Path

from .intrinsics import read_intrinsics, Intrinsics


def build_session(raw_paths: list[Path],
                  out_root: Path,
                  session: str) -> tuple[Path, Intrinsics] | tuple[None, None]:
    """Convert raw iPhone photos into a session directory.

    Returns (session_dir, intrinsics_of_first_photo) on success,
    or (None, None) if nothing could be imported.
    """
    out_dir = out_root / session
    rgb_dir = out_dir / "RGB"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    first: Intrinsics | None = None

    for i, src in enumerate(sorted(raw_paths)):
        if not src.exists():
            print(f"  ✗ {src} not found, skipping")
            continue
        try:
            img, K = read_intrinsics(src)
        except Exception as e:
            print(f"  ✗ {src.name}: {e}")
            continue

        dst = rgb_dir / f"{i:06d}_RGB.png"
        img.convert("RGB").save(dst, "PNG")
        print(f"  ✓ {src.name} → {dst.name}  "
              f"({K.width}×{K.height}, fx={K.fx:.1f})")

        if first is None:
            first = K
        elif (K.width, K.height) != (first.width, first.height):
            print(f"    ⚠ resolution differs from first photo "
                  f"({K.width}×{K.height} vs {first.width}×{first.height}) "
                  "— intrinsics.json keeps the first one")

    if first is None:
        print("nothing imported.")
        return None, None

    # Persist intrinsics next to the RGB folder
    with open(out_dir / "intrinsics.json", "w") as f:
        json.dump({**first.to_dict(), "K_matrix": first.K_matrix}, f, indent=2)

    return out_dir, first
