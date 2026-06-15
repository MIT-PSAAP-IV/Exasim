# Data-transfer app example (Poisson 2D, HDG) -- Julia.
#
# Identical to the standard Poisson2D example except for one line:
#
#     pde.exportapp = joinpath(pwd(), "poisson2d-bundle")
#
# When set, exasim() additionally packages a self-contained, relocatable
# "data-transfer app" bundle (datain/, kernels/, a generated pdemodel.txt, a
# relocatable CMakeLists.txt + main.cpp, run.sh, manifest). Copy it to any
# machine with an Exasim install and build + run it with no frontend:
#
#     EXASIM_ROOT=/path/to/exasim/install ./run.sh
#
# It is arch-independent -- retarget the build machine's variant with, e.g.,
#     EXASIM_ROOT=/path/to/install EXASIM_VARIANT=gpu ./run.sh
#
# See ../README.md for details.
using Exasim

pde, mesh = Exasim.initializeexasim();

pde.model = "ModelD";
include("pdemodel.jl");

pde.porder = 3;
pde.physicsparam = [1.0 0.0];    # thermal conductivity and boundary value
pde.tau = [1.0];                 # DG stabilization parameter
pde.mpiprocs = 1;                # serial; bundle stays variant "cpu"
pde.hybrid = 1;                  # 0 -> LDG, 1 -> HDG

# >>> The only line that distinguishes this from the plain Poisson2D example:
pde.exportapp = joinpath(pwd(), "poisson2d-bundle");

mesh.p, mesh.t = Exasim.Mesh.SquareMesh(16,16,1);
mesh.boundaryexpr = [p -> (p[2,:] .< 1e-3), p -> (p[1,:] .> 1-1e-3), p -> (p[2,:] .> 1-1e-3), p -> (p[1,:] .< 1e-3)];
mesh.boundarycondition = [1 1 1 1];

# Generate + run the solver AND export the data-transfer app bundle.
sol, pde, mesh = Exasim.exasim(pde,mesh)[1:3];

print("Done! Data-transfer app bundle: ", pde.exportapp, "\n");
