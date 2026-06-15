"""phone_dataset — turn iPhone photos into a Pandora-shaped PoC dataset.

Public entry points:
    from phone_dataset.intrinsics import read_intrinsics
    from phone_dataset.convert    import build_session
"""
from .intrinsics import read_intrinsics
from .convert    import build_session

__all__ = ["read_intrinsics", "build_session"]
