#!/usr/bin/env julia
"""Callable and positional-CLI runner for the four-domain mixed 3D example."""

function repository_root()
    candidates=haskey(ENV,"EXASIM_ROOT") ? [ENV["EXASIM_ROOT"]] : String[]
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
const _RUNNER_EXECUTED=abspath(PROGRAM_FILE)==@__FILE__
const _RUNNER_OUTDIR=_RUNNER_EXECUTED && length(ARGS)>=4 ?
    expanduser(ARGS[4]) : nothing
const _RUNNER_OUTSIDE=!isnothing(_RUNNER_OUTDIR) && isabspath(_RUNNER_OUTDIR) && let
    relative=relpath(normpath(abspath(_RUNNER_OUTDIR)),REPOSITORY)
    !(relative=="." || (!startswith(relative,"..$(Base.Filesystem.path_separator)") && relative!=".."))
end
if _RUNNER_OUTSIDE
    mkpath(_RUNNER_OUTDIR)
    depot=joinpath(_RUNNER_OUTDIR,".julia_depot")
    mkpath(depot)
    depot in DEPOT_PATH || pushfirst!(DEPOT_PATH,depot)
end
frontend=joinpath(REPOSITORY,"frontends","Julia","Exasim")
frontend in LOAD_PATH || pushfirst!(LOAD_PATH,frontend)

using Exasim
include(joinpath(@__DIR__,"example_driver.jl"))

run_multiple_interactions(porder,N,domain_ranks,outdir,
                          active_edges=[1,1,1]) =
    run_example(porder,N,domain_ranks,outdir,active_edges)

function main(args=ARGS)
    length(args)>=4 || throw(ArgumentError("porder, N, ranks, and outdir are required"))
    porder=parse(Int,args[1])
    N=parse(Int,args[2])
    ranks=parse_ranks(args[3])
    outdir=args[4]
    active_edges=length(args)>=5 ? parse_active_edges(args[5]) : [1,1,1]
    result=run_multiple_interactions(porder,N,ranks,outdir,active_edges)
    println("Global final residual: ",result.final_residual)
    println("Maximum nodal error: ",result.max_error)
    println("EXASIM_OUTDIR=$(result.case.outdir)")
    result
end

_RUNNER_EXECUTED && main()
