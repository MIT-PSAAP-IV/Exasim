"""
    hdgldgsolve(pde, mesh, labels, nhdg=1) -> sol, pde, mesh

Solve a steady CPU ModelD problem with homogeneous HDG/LDG rank groups.
`labels[e]` is 1 for HDG and 0 for LDG in the original mesh element order.
Alternatively use `(seed_boundaries=[1], layers=2)` to select face-neighbour
layers from boundary-condition codes through the shared CPU preprocessor.
The first `nhdg` world ranks are HDG; the remaining ranks are LDG. Native
Julia model generation and compilation are used. The shared CPU preprocessor
constructs both regions; no Text2Code compiler or Python workflow is invoked.

Return full `(u,q)` in `sol[npe,nc,ne]`, reordered to the original mesh.
`mesh.hdgldg[e]` is 1 for HDG and 0 for LDG in that same element order.
Generated setup/output live in a new directory under `pde.datapath`; native
model compilation reuses `pde.builddir` in the ordinary way.
"""
function hdgldgsolve(pde, mesh, labels, nhdg::Integer=1)
    pde.platform == "cpu" && pde.model == "ModelD" && pde.tdep == 0 ||
        error("hdgldgsolve currently supports steady CPU ModelD problems")
    pde.nco == 0 && pde.ncw == 0 || error("hdgldgsolve currently requires nco=ncw=0")
    all(getproperty(pde,key) == 0 for key in
        (:nsca,:nvec,:nten,:nbqoi,:nvqoi,:nsurfq,:nmaterialstate)) ||
        error("hdgldgsolve currently returns fields without visualization/QoI callbacks")
    1 <= nhdg < pde.mpiprocs || error("require 1 <= nhdg < pde.mpiprocs")
    ne = size(mesh.t,2)
    boundary_layers = labels isa NamedTuple
    if boundary_layers
        haskey(labels,:seed_boundaries) && haskey(labels,:layers) ||
            error("provide seed_boundaries and layers")
        seeds = labels.seed_boundaries
        !isempty(seeds) && all(v isa Integer && v >= 1 for v in seeds) &&
            labels.layers isa Integer && labels.layers >= 1 ||
            error("provide positive integer seed_boundaries and layers")
    else
        length(labels) == ne && all(value in (0,1) for value in labels) ||
            error("labels must contain one 0=L or 1=H per original element")
        flags = Int.(vec(labels))
        0 < sum(flags) < ne || error("both HDG and LDG elements are required")
    end
    prefix = install_prefix()
    preprocessor = joinpath(prefix,"bin","exasim_hdgldg_preprocess")
    isfile(preprocessor) || error("installed exasim_hdgldg_preprocess is missing: $preprocessor")

    pde.nd = pde.ncx = size(mesh.p,1)
    pde.ncu > 0 || error("pde.ncu must be positive")
    pde.ncq = pde.nd*pde.ncu; pde.nc = (pde.nd+1)*pde.ncu; pde.nch = pde.ncu
    pde.hybrid = 1                 # Generate both native kernel families.
    pde.modelnumber = 0; pde.modelid = 100; pde.builtinmodelID = 100
    pde.saveSolOpt = 1             # Return the recovered gradients as well.
    # Straight nodes are useful for plotting. Curved nodes are constructed by
    # the shared backend from the boundary expressions; do not approximate
    # them here or require a second frontend geometry implementation.
    if isempty(mesh.dgnodes) && !any(mesh.curvedboundary .> 0)
        mesh.dgnodes = Preprocessing.createdgnodes(mesh.p,mesh.t,zeros(Int,0,0),
            mesh.curvedboundary,mesh.curvedboundaryexpr,pde.porder)
    end
    mkpath(pde.datapath)
    work = mktempdir(pde.datapath; prefix="hdgldg-",cleanup=false)
    println("HDG/LDG setup and output: $work")
    # Enter the latest world so freshly included Main model callbacks work
    # like the ordinary Julia examples, even from an interactive session.
    Base.invokelatest(Gencode.gencode,pde)
    Gencode.cmakecompile(pde)
    # Reuse the public serializer for its config/mesh files only. Its unused
    # pdemodel.txt is not parsed: --native-metadata consumes explicit sizes.
    exported = joinpath(work,"input")
    exportmesh = deepcopy(mesh)
    # initializeexasim's [0 0] sentinel means no periodic pairs. The existing
    # serializer accepts the equivalent empty table, not a zero-valued pair.
    if all(iszero,exportmesh.periodicboundary)
        exportmesh.periodicboundary = zeros(Int,0,0)
    end
    Base.invokelatest(Gencode.exporttext2code,deepcopy(pde),exportmesh,exported)
    config = joinpath(exported,"pdeapp.txt")
    text = read(config,String)
    open(config,"a") do io
        for key in ("nsca","nvec","nten","nsurf","nvqoi","nmaterialstate","nsurfq")
            occursin(Regex("(?m)^\\s*"*key*"\\s*="),text) || println(io,key," = 0;")
        end
    end
    manifest = joinpath(work,"selection.txt")
    open(manifest,"w") do io
        println(io,"version = 1\nworld_size = ",pde.mpiprocs)
        println(io,"h_ranks = [",join(0:nhdg-1,","),"]")
        println(io,"l_ranks = [",join(nhdg:pde.mpiprocs-1,","),"]")
        if boundary_layers
            println(io,"selection = boundary_layers\nseed_boundaries = [",join(labels.seed_boundaries,","),"]")
            println(io,"layers = ",labels.layers)
        else
            println(io,"selection = labels\nlabels = [",join(flags,","),"]")
        end
    end
    prepared = joinpath(work,"prepared")
    Preprocessing.hdgldgpreprocessing(config,manifest,prepared,preprocessor)
    build = model_builddir(pde)
    exe = joinpath(build,"build","exasimapp")
    launcher = pde.mpirun
    launcher_file = joinpath(build,"build","mpiexec.txt")
    isfile(launcher_file) && (launcher = strip(read(launcher_file,String)))
    isempty(launcher) && error("MPI launcher is not configured")
    output = joinpath(work,"output")
    run(Cmd([launcher,"-np",string(pde.mpiprocs),exe,"--hdgldg",prepared,output]))
    pde.dataoutpath = output

    sol = nothing
    element_schemes = zeros(eltype(mesh.t),ne)
    seen = falses(ne)
    for (side,nranks) in (("H",nhdg),("L",pde.mpiprocs-nhdg)), rank in 0:nranks-1
        npe,nc,owned,nsteps,values = read_rank(joinpath(output,side,"outudg_np$rank.bin"))
        if sol === nothing
            sol = zeros(Float64,npe,pde.nc,ne)
        end
        (npe,nc,nsteps) == (size(sol,1),pde.nc,1) || error("unexpected native solution dimensions")
        suffix = nranks == 1 ? "" : string(rank+1)
        open(joinpath(prepared,side,"hdgldg"*suffix*".txt"),"r") do io
            strip(readline(io)) == "EXASIM_HDGLDG 1" || error("invalid region sidecar")
            descriptor = parse.(Int,split(readline(io)))
            total = parse(Int,readline(io))
            owned <= total || error("native solution has more owned elements than metadata")
            for local_element in 1:owned
                row = parse.(Int,split(readline(io)))
                id = row[2]+1   # Backend IDs are zero-based original input columns.
                row[1] == local_element-1 && row[3] == descriptor[2] || error("invalid owned element metadata")
                1 <= id <= ne && !seen[id] || error("invalid or duplicate original element ID")
                sol[:,:,id] .= values[:,:,local_element,1]
                element_schemes[id] = side == "H" ? 1 : 0
                seen[id] = true
            end
        end
    end
    all(seen) && all(isfinite,sol) || error("missing or nonfinite original-element solution")
    mesh.hdgldg = element_schemes
    return sol,pde,mesh
end
