"""Exasim Python frontend.

Typical use:

    import exasim
    pde, mesh = exasim.initializeexasim()
    ...define the model, mesh, and parameters...
    sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]

Runtime data (datain/, dataout/) is written under the working directory
(override with pde['datapath']); generated code and the solver build live in
the hidden pde['builddir'] (default <cwd>/.exasim).
"""
from . import config
from . import Materials
try:
    from . import Preprocessing
    from . import Mesh
    from . import Gencode
    from . import Postprocessing

    from .Preprocessing import initializeexasim, preprocessing
    from .Gencode import exportapp, exporttext2code, exporttext2codemesh
    from .Postprocessing import exasim, vis, fetchsolution
except ModuleNotFoundError as exc:
    if exc.name != "sympy":
        raise

    def _missing_sympy(*args, **kwargs):
        raise ModuleNotFoundError(
            "The Exasim preprocessing/code-generation frontend requires sympy. "
            "Install sympy to use initializeexasim, preprocessing, exasim, or "
            "code-generation workflows. exasim.Materials remains available."
        ) from exc

    Preprocessing = None
    Mesh = None
    Gencode = None
    Postprocessing = None
    initializeexasim = _missing_sympy
    preprocessing = _missing_sympy
    exportapp = _missing_sympy
    exporttext2code = _missing_sympy
    exporttext2codemesh = _missing_sympy
    exasim = _missing_sympy
    vis = _missing_sympy
    fetchsolution = _missing_sympy

__version__ = "1.0.0"
