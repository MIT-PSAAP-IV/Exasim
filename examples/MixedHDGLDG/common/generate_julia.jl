using Exasim

"""
    generate_julia(caseDir, buildDir, nd, ncu, nparam)

Generate native Julia/SymPy kernels. Run one case per Julia process, as in
the ordinary examples that include callbacks into Main. All five models use
ModelD, no ODG/WDG or generated visualization/QoI callbacks, and one tau.
The runtime configuration supplies parameter values and the rank schemes.
"""
function generate_julia(caseDir, buildDir, nd, ncu, nparam)
    caseDir = realpath(caseDir)
    model = joinpath(caseDir, "pdemodel.jl")
    isfile(model) || error("missing native Julia pdemodel.jl")
    isabspath(buildDir) && !ispath(buildDir) ||
        error("model build directory must be new and absolute")
    normalized = normpath(buildDir)
    (startswith(normalized, "/tmp/") || startswith(normalized, "/private/tmp/")) ||
        error("model build directory must be under /tmp")
    nd in (2,3) && ncu > 0 && nparam >= 0 || error("invalid ModelD dimensions")
    pde, _ = Exasim.initializeexasim()
    pde.model = "ModelD"; pde.modelfile = "Main"
    pde.hybrid = 1; pde.modelnumber = 0
    pde.nd = nd; pde.ncx = nd; pde.ncu = ncu; pde.nch = ncu
    pde.ncq = nd*ncu; pde.nc = (nd+1)*ncu
    pde.nco = 0; pde.ncw = 0; pde.nce = 0
    pde.nsca = 0; pde.nvec = 0; pde.nten = 0
    pde.nbqoi = 0; pde.nvqoi = 0; pde.nsurfq = 0; pde.nmaterialstate = 0
    pde.physicsparam = zeros(1,nparam); pde.uinf = zeros(1,0); pde.tau = [1.0]
    pde.builddir = buildDir; pde.buildpath = buildDir; pde.datapath = buildDir
    Base.include(Main, model)
    Base.invokelatest(Exasim.Gencode.gencode, pde)  # model was included above
    return joinpath(buildDir, "kernels")
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 5 || error("usage: generate_julia.jl CASE_DIR NEW_BUILD nd ncu nparam")
    println(generate_julia(ARGS[1], ARGS[2], parse.(Int, ARGS[3:5])...))
end
