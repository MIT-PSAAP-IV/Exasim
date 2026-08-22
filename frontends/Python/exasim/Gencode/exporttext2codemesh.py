import os

import numpy as np

from .exporttext2code import _empty, _flat, _mesh_get, _shape_as_double, _writebin


def exporttext2codemesh(mesh, dest, suffix=""):
    """Export Text2Code mesh-related binary files.

    The binary layout matches the MATLAB ``exporttext2codemesh`` helper:

    ``grid<suffix>.bin = [shape(p), shape(t), p(:), t(:)]``

    Optional fields are written when present and nonempty:
    ``dgnodes -> xdg``, ``udg -> udg``, ``vdg -> vdg``, and ``wdg -> wdg``.
    Arrays are flattened in column-major order to match MATLAB.
    """
    dest = os.path.abspath(os.fspath(dest))
    suffix = "" if suffix is None else str(suffix)
    os.makedirs(dest, exist_ok=True)

    p = _mesh_get(mesh, "p")
    t = _mesh_get(mesh, "t")
    if p is None or _empty(p):
        raise ValueError("exporttext2codemesh: mesh must contain a nonempty p array.")
    if t is None or _empty(t):
        raise ValueError("exporttext2codemesh: mesh must contain a nonempty t array.")

    _writebin(
        os.path.join(dest, f"grid{suffix}.bin"),
        np.concatenate((_shape_as_double(p), _shape_as_double(t), _flat(p), _flat(t))),
    )

    optional = [
        ("dgnodes", "xdg"),
        ("udg", "udg"),
        ("vdg", "vdg"),
        ("wdg", "wdg"),
    ]
    for mesh_key, filename_base in optional:
        value = _mesh_get(mesh, mesh_key)
        if value is not None and not _empty(value):
            _writebin(
                os.path.join(dest, f"{filename_base}{suffix}.bin"),
                np.concatenate((_shape_as_double(value), _flat(value))),
            )

    return dest
