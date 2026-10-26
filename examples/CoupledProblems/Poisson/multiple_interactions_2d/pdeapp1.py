# Domain 1: x in [0, 0.25].
pde[0], mesh[0] = exasim.initializeexasim()

# All four domains use the same Poisson model.
pde[0].update(
    model="ModelD",
    modelfile="pdemodel",
    Cxxpreprocessing=1,
    platform="cpu",
    mpiprocs=domainranks[0],
    hybrid=1,
    porder=porder,
    pgauss=2*porder,
    physicsparam=np.array([1.0, 1.0]),  # diffusion; numerical-flux interface equation
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
nt = N if 1 % 2 == 1 else 2*N
mesh[0]["p"], mesh[0]["t"] = exasim.Mesh.SquareMesh(N, nt, 1)
mesh[0]["p"][0, :] = 0 + 0.25*mesh[0]["p"][0, :]

# Boundary order: y=0, right, y=1, left.
mesh[0]["boundaryexpr"] = [
    lambda x: np.abs(x[1]) < 1e-8,
    lambda x: np.abs(x[0] - 0.25) < 1e-8,
    lambda x: np.abs(x[1] - 1) < 1e-8,
    lambda x: np.abs(x[0] - 0) < 1e-8,
]
mesh[0]["boundarycondition"] = np.array([1, 3, 1, 1])
mesh[0]["interfacecondition"] = np.array([], dtype=int)
