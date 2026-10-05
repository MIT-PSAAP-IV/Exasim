"""Sharp-B uses the shared axisymmetric ideal-gas mesh-adaptivity model."""

from __future__ import annotations

import importlib.util
import os

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "isoq2d_idealgas", "pdemodel.py",
)
_SPEC = importlib.util.spec_from_file_location("sharpb2_shared_pdemodel", _MODEL_PATH)
_MODEL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODEL)

mass = _MODEL.mass
flux = _MODEL.flux
source = _MODEL.source
fbou = _MODEL.fbou
ubou = _MODEL.ubou
fbouhdg = _MODEL.fbouhdg
initu = _MODEL.initu
initv = _MODEL.initv
avfield = _MODEL.avfield
visscalars = _MODEL.visscalars
visvectors = _MODEL.visvectors
surfacequantities = _MODEL.surfacequantities
