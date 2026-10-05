"""Reuse the ISOQ ideal-gas wall-surface postprocessor for Sharp-B."""

from __future__ import annotations

import importlib.util
import os

_POSTPROCESS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "isoq2d_idealgas",
    "postprocess_surfacequantities.py",
)
_SPEC = importlib.util.spec_from_file_location("sharpb2_shared_surfacepost", _POSTPROCESS_PATH)
_POST = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_POST)

postprocess_surfacequantities = _POST.postprocess_surfacequantities
