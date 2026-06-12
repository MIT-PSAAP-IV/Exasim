"""
Exasim Julia frontend.

Typical use:

    using Exasim
    pde, mesh = Exasim.initializeexasim()
    ...define the model, mesh, and parameters...
    sol, pde, mesh = Exasim.exasim(pde, mesh)

Runtime data (datain/, dataout/) is written under the working directory
(override with pde.datapath); generated code and the solver build live in the
hidden pde.builddir (default <cwd>/.exasim).
"""
module Exasim

include("config.jl")

include("Preprocessing/Types.jl")
include("Preprocessing/Master.jl")
include("Gencode/Gencode.jl")
include("Preprocessing/Preprocessing.jl")
include("Mesh/Mesh.jl")
include("Postprocessing/Postprocessing.jl")

using .Preprocessing: initializeexasim, preprocessing
using .Postprocessing: exasim, vis, fetchsolution
using .Mesh
using .Gencode

export initializeexasim, preprocessing, exasim, vis, fetchsolution,
       Preprocessing, Gencode, Mesh, Postprocessing

end
