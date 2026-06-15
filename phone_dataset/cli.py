"""CLI: turn iPhone photos in data/iphone/raw/ into a usable session.

Default workflow (zero args — picks up everything in raw/):

    python3 -m phone_dataset.cli

Pick specific files:

    python3 -m phone_dataset.cli data/iphone/raw/leon_0.HEIC --session leon

Use any other location:

    python3 -m phone_dataset.cli ~/Pictures/IMG_*.HEIC --root data/iphone --session demo
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .convert import build_session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photos", nargs="*", type=Path,
                    help="iPhone photo paths (HEIC/HEIF/JPG/PNG). "
                         "Default: all files in <root>/raw/")
    ap.add_argument("--root",    default="data/iphone", type=Path,
                    help="dataset root (default: data/iphone)")
    ap.add_argument("--session", default="default",
                    help="session subfolder name (default: default)")
    args = ap.parse_args()

    if not args.photos:
        raw_dir = args.root / "raw"
        if not raw_dir.exists():
            print(f"no photos given and {raw_dir} does not exist — "
                  f"drop HEIC files there or pass paths explicitly.")
            return
        args.photos = sorted([p for p in raw_dir.iterdir()
                              if p.suffix.lower() in
                                 {".heic", ".heif", ".jpg", ".jpeg", ".png"}])
        if not args.photos:
            print(f"{raw_dir} has no images.")
            return
        print(f"picking up {len(args.photos)} photo(s) from {raw_dir}")

    out_dir, K = build_session(args.photos, args.root, args.session)
    if K is None:
        return

    print(f"\nsession → {out_dir}")
    print(f"  fx={K.fx:.2f}  fy={K.fy:.2f}  "
          f"cx={K.cx:.1f}  cy={K.cy:.1f}")
    print(f"  device={K.device}  "
          f"(35mm-eq focal: {K.focal_length_35mm_eq} mm)")


if __name__ == "__main__":
    main()
