# Domain 3: x in [0.5, 0.75].
pde[2], mesh[2] = exasim.initializeexasim()

# All four domains use the same Poisson model.
pde[2].update(
    model="ModelD",
    modelfile="pdemodel",
    Cxxpreprocessing=1,
    platform="cpu",
    mpiprocs=domainranks[2],
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
nt = N if 3 % 2 == 1 else 2*N
mesh[2]["p"], mesh[2]["t"] = exasim.Mesh.cubemesh(N, nt, nt, families[2])[:2]
mesh[2]["p"][0, :] = 0.5 + 0.25*mesh[2]["p"][0, :]

# Boundary order: y=0, right, y=1, left, z=0, z=1.
mesh[2]["boundaryexpr"] = [
    lambda x: np.abs(x[1]) < 1e-8,
    lambda x: np.abs(x[0] - 0.75) < 1e-8,
    lambda x: np.abs(x[1] - 1) < 1e-8,
    lambda x: np.abs(x[0] - 0.5) < 1e-8,
    lambda x: np.abs(x[2]) < 1e-8,
    lambda x: np.abs(x[2] - 1) < 1e-8,
]
mesh[2]["boundarycondition"] = np.array([1, 3, 1, 2, 1, 1])
mesh[2]["interfacecondition"] = np.array([], dtype=int)
