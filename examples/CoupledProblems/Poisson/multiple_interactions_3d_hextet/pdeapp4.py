# Domain 4: x in [0.75, 1].
pde[3], mesh[3] = exasim.initializeexasim()

# All four domains use the same Poisson model.
pde[3].update(
    model="ModelD",
    modelfile="pdemodel",
    Cxxpreprocessing=1,
    platform="cpu",
    mpiprocs=domainranks[3],
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
nt = N if 4 % 2 == 1 else 2*N
mesh[3]["p"], mesh[3]["t"] = exasim.Mesh.cubemesh(N, nt, nt, families[3])[:2]
mesh[3]["p"][0, :] = 0.75 + 0.25*mesh[3]["p"][0, :]

# Boundary order: y=0, right, y=1, left, z=0, z=1.
mesh[3]["boundaryexpr"] = [
    lambda x: np.abs(x[1]) < 1e-8,
    lambda x: np.abs(x[0] - 1) < 1e-8,
    lambda x: np.abs(x[1] - 1) < 1e-8,
    lambda x: np.abs(x[0] - 0.75) < 1e-8,
    lambda x: np.abs(x[2]) < 1e-8,
    lambda x: np.abs(x[2] - 1) < 1e-8,
]
mesh[3]["boundarycondition"] = np.array([1, 1, 1, 2, 1, 1])
mesh[3]["interfacecondition"] = np.array([], dtype=int)
