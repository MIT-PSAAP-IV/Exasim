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
from . import Preprocessing
from . import Mesh
from . import Gencode
from . import Postprocessing

from .Preprocessing import initializeexasim, preprocessing
from .Postprocessing import exasim, vis, fetchsolution

__version__ = "1.0.0"
