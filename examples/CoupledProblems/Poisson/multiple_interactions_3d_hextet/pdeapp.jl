"""Interactive and positional-CLI entry point for the four-domain 3D hex-tet example."""

function repository_root()
    candidates=String[]
    haskey(ENV,"EXASIM_ROOT") && push!(candidates,ENV["EXASIM_ROOT"])
    directory=@__DIR__
    while true
        push!(candidates,directory)
        parent=dirname(directory)
        parent==directory && break
        directory=parent
    end
    for candidate in candidates
        root=normpath(abspath(expanduser(candidate)))
        isdir(joinpath(root,"frontends","Julia","Exasim")) && return root
    end
    error("Cannot locate Exasim; set EXASIM_ROOT to the repository root")
end

const REPOSITORY=repository_root()
const _EXECUTED=abspath(PROGRAM_FILE)==@__FILE__
const _CLI_OUTDIR=_EXECUTED ? (length(ARGS)>=4 ? expanduser(ARGS[4]) :
                               mktempdir(prefix="exasim-multiple_interactions_3d_hextet-")) : nothing
const _CLI_OUTSIDE=_EXECUTED && isabspath(_CLI_OUTDIR) && let
    relative=relpath(normpath(abspath(_CLI_OUTDIR)),REPOSITORY)
    !(relative=="." || (!startswith(relative,"..$(Base.Filesystem.path_separator)") && relative!=".."))
end
if _CLI_OUTSIDE
    mkpath(_CLI_OUTDIR)
    depot=joinpath(_CLI_OUTDIR,".julia_depot")
    mkpath(depot)
    depot in DEPOT_PATH || pushfirst!(DEPOT_PATH,depot)
end
frontend=joinpath(REPOSITORY,"frontends","Julia","Exasim")
frontend in LOAD_PATH || pushfirst!(LOAD_PATH,frontend)

using Exasim
include(joinpath(@__DIR__,"example_driver.jl"))

function main(args=ARGS)
    porder=length(args)>=1 ? parse(Int,args[1]) : DEFAULT_PORDER
    N=length(args)>=2 ? parse(Int,args[2]) : DEFAULT_N
    ranks=length(args)>=3 ? parse_ranks(args[3]) : DEFAULT_RANKS
    outdir=length(args)>=4 ? args[4] : _CLI_OUTDIR
    active_edges=length(args)>=5 ? parse_active_edges(args[5]) : [1,1,1]
    result=run_example(porder,N,ranks,outdir,active_edges)
    println("workdir=$(result.case.outdir)")
    println("global final residual: ",result.final_residual)
    println("maximum nodal error: ",result.max_error)
    result
end

_EXECUTED && main()
