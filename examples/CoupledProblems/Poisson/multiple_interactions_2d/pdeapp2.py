# Domain 2: x in [0.25, 0.5].
pde[1], mesh[1] = exasim.initializeexasim()

# All four domains use the same Poisson model.
pde[1].update(
    model="ModelD",
    modelfile="pdemodel",
    Cxxpreprocessing=1,
    platform="cpu",
    mpiprocs=domainranks[1],
    hybrid=1,
    porder=porder,
    pgauss=2*porder,
    physicsparam=np.array([1.0, 0.0]),  # diffusion; trace interface equation
    tau=np.array([1.0]),
    NLiter=12,
    NLtol=1e-12,
    linearsolvertol=1e-12,
    matvectol=1e-12,
    GMRESrestart=250,
    linearsolveriter=600,
    datapath=str(run_directory),
    builddir=str(run_directory / ".exasim"),
    buildpath=str(run_directory / ".exasim"),
)

# Adjacent domains use different tangential resolutions.
nt = N if 2 % 2 == 1 else 2*N
mesh[1]["p"], mesh[1]["t"] = exasim.Mesh.SquareMesh(N, nt, 1)
mesh[1]["p"][0, :] = 0.25 + 0.25*mesh[1]["p"][0, :]

# Boundary order: y=0, right, y=1, left.
mesh[1]["boundaryexpr"] = [
    lambda x: np.abs(x[1]) < 1e-8,
    lambda x: np.abs(x[0] - 0.5) < 1e-8,
    lambda x: np.abs(x[1] - 1) < 1e-8,
    lambda x: np.abs(x[0] - 0.25) < 1e-8,
]
mesh[1]["boundarycondition"] = np.array([1, 3, 1, 2])
mesh[1]["interfacecondition"] = np.array([], dtype=int)
