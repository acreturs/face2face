"""EXIF → camera intrinsics.

iPhone (and most modern phone) photos embed `FocalLengthIn35mmFilm`. That
gives a pixel focal length without any calibration:

    fx = fy ≈ focal_35mm × (image_width_px / 36.0)
    cx ≈ width / 2,   cy ≈ height / 2

Accuracy is ~1-2% because Apple ships per-device lens calibration baked into
the metadata. Good enough as a stand-in for a checkerboard calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ExifTags

# HEIC is the default iPhone format; opt-in support via pillow-heif.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False


_EXIF_BY_NAME = {v: k for k, v in ExifTags.TAGS.items()}

# Numeric tag IDs we need from the EXIF sub-IFD (HEIC stores them there, not
# at the top level — that's why FocalLengthIn35mmFilm appeared "missing").
_TAG_FOCAL_LENGTH        = 0x920A  # FocalLength (real mm)
_TAG_FOCAL_LENGTH_35MM   = 0xA405  # FocalLengthIn35mmFilm
_EXIF_IFD_POINTER        = 0x8769  # location of the EXIF sub-IFD


def _merged_exif(img: Image.Image) -> dict:
    """Return a flat dict of all EXIF tags (top-level + sub-IFDs).

    PIL exposes camera-side tags via getexif().get_ifd(0x8769); the top-level
    dict only has image-side tags like Make / Model / Orientation. Merge both
    so callers can look up any tag by numeric ID in one place.
    """
    base = img.getexif()
    merged = dict(base)
    try:
        merged.update(base.get_ifd(_EXIF_IFD_POINTER))
    except Exception:
        pass
    return merged


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width:  int
    height: int
    focal_length_mm:      float | None
    focal_length_35mm_eq: float
    device: str
    source: str = "EXIF FocalLengthIn35mmFilm → pixel focal length"

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d

    @property
    def K_matrix(self) -> list[list[float]]:
        """3x3 intrinsic matrix in standard pinhole form."""
        return [[self.fx, 0.0,    self.cx],
                [0.0,     self.fy, self.cy],
                [0.0,     0.0,     1.0    ]]


def read_intrinsics(img_path: Path) -> tuple[Image.Image, Intrinsics]:
    """Open an image and derive its intrinsics from EXIF.

    Raises RuntimeError if EXIF is missing the needed field (e.g. screenshot,
    edited photo) or if HEIC is requested without pillow-heif installed.
    """
    if img_path.suffix.lower() in {".heic", ".heif"} and not HEIC_OK:
        raise RuntimeError(
            f"{img_path.name} is HEIC but pillow-heif is missing — "
            "`pip install pillow-heif` or rebuild the dev container."
        )

    img  = Image.open(img_path)
    W, H = img.size
    exif = _merged_exif(img)

    focal_35   = exif.get(_TAG_FOCAL_LENGTH_35MM)
    focal_real = exif.get(_TAG_FOCAL_LENGTH)
    make       = exif.get(_EXIF_BY_NAME["Make"],  "?")
    model      = exif.get(_EXIF_BY_NAME["Model"], "?")

    if focal_35 is None:
        # Last-ditch: dump everything we found so the user can see what's there.
        tag_names = {v: k for k, v in exif.items()}
        known = ", ".join(sorted(ExifTags.TAGS.get(k, hex(k)) for k in exif))[:300]
        raise RuntimeError(
            f"{img_path.name}: no FocalLengthIn35mmFilm in EXIF.\n"
            f"  Tags present: {known or '(none)'}\n"
            f"  If FocalLength is here but FocalLengthIn35mmFilm is not, "
            f"pass --hfov manually."
        )

    f = float(focal_35) * (W / 36.0)
    return img, Intrinsics(
        fx=f, fy=f, cx=W / 2.0, cy=H / 2.0,
        width=W, height=H,
        focal_length_mm=float(focal_real) if focal_real else None,
        focal_length_35mm_eq=float(focal_35),
        device=f"{make} {model}".strip(),
    )
