# Frontend end-to-end test: Poisson 2D on the unit square (serial, HDG).
# Generates kernels with the Exasim.jl package, builds the solver against the
# installed Exasim via the external-model path, runs it, and gates the QoI
# (Domain_QoI1 = integral of (u - uexact)^2) against QOI_TOL.
using Exasim

# create pde structure and mesh structure
pde, mesh = Exasim.initializeexasim();

pde.model = "ModelD";
pde.modelfile = "Main";          # PDE model functions are defined in Main
include("pdemodel.jl");

pde.mpiprocs = 1;
pde.hybrid = 1;
pde.porder = 3;
pde.physicsparam = [1.0 0.0];    # thermal conductivity and boundary value
pde.tau = [1.0];                 # DG stabilization parameter

mesh.p, mesh.t = Exasim.Mesh.SquareMesh(16,16,1);
mesh.boundaryexpr = [p -> (p[2,:] .< 1e-3), p -> (p[1,:] .> 1-1e-3),
                     p -> (p[2,:] .> 1-1e-3), p -> (p[1,:] .< 1e-3)];
mesh.boundarycondition = [1 1 1 1];

sol, pde, mesh = Exasim.exasim(pde, mesh);

if isdefined(pde, :exportapp) && !isempty(pde.exportapp)
    # export mode: exasim() packages the bundle without running the simulation
    # locally, so there is no main-dir QoI to gate here (the harness validates
    # the exported bundle by relocating and running it).
    println("FRONTEND TEST (export mode): skipped in-app QoI gate");
else
    qoifile = joinpath(pde.datapath, "dataout", "outqoi.txt");
    last_row = split(strip(readlines(qoifile)[end]));
    err2 = parse(Float64, last_row[2]);
    tol = parse(Float64, get(ENV, "QOI_TOL", "1e-8"));
    println("L2 error^2 = $err2 (tol $tol)");
    err2 < tol || error("QoI gate failed: $err2 >= $tol");
    println("FRONTEND TEST PASSED");
end
