"""Callable driver and artifact writer for the four-domain example."""

const CASE="multiple_interactions_3d_hextet"
const DIMENSION=3
const FAMILIES=[1,0,1,0]
const DEFAULT_PORDER=1
const DEFAULT_N=2
const DEFAULT_RANKS=[1,1,1,1]
const FULL_INTERACTION_TABLE=Int[
    1,4,3,
    11,0,1,1,2,1,1,0,4,2,1,0,
    11,1,2,1,2,2,1,0,4,1,1,0,
    11,2,3,1,2,1,1,0,4,2,1,0,
]
const SOLVER_CONTROLS=Dict{String,Any}(
    "NLiter"=>12,"NLtol"=>1e-12,"linearsolvertol"=>1e-12,
    "matvectol"=>1e-12,"GMRESrestart"=>250,"linearsolveriter"=>600,
)

positive_integer(value,name)=value isa Integer && value>0 ? Int(value) :
    throw(ArgumentError("$name must be a positive integer"))

function parse_ranks(value)
    fields=value isa AbstractString ? split(value,",") : value
    ranks=[positive_integer(item isa AbstractString ? parse(Int,item) : item,
                            "each domain rank count") for item in fields]
    length(ranks)==4 || throw(ArgumentError("ranks must contain exactly four positive integers"))
    ranks
end

function parse_active_edges(value=[1,1,1])
    fields=value isa AbstractString ? split(value,",") : value
    flags=[item isa AbstractString ? parse(Int,item) : Int(item) for item in fields]
    length(flags)==3 && all(item->item in (0,1),flags) ||
        throw(ArgumentError("active_edges must contain exactly three 0/1 values"))
    any(==(1),flags) || throw(ArgumentError("at least one interaction edge must be active"))
    flags
end

function prepare_output_directory(outdir=nothing)
    path=isnothing(outdir) ? mktempdir(prefix="exasim-$CASE-") : String(outdir)
    isabspath(path) || throw(ArgumentError("outdir must be an absolute path"))
    path=normpath(abspath(path))
    relative=relpath(path,REPOSITORY)
    (relative=="." || (!startswith(relative,"..$(Base.Filesystem.path_separator)") && relative!="..")) &&
        throw(ArgumentError("outdir must be outside the Exasim repository"))
    mkpath(path)
    path
end

function expected_interaction_table(active_edges)
    values=Int[1,4,sum(active_edges)]
    for edge=1:3
        if active_edges[edge]==1
            start=4+12*(edge-1)
            append!(values,FULL_INTERACTION_TABLE[start:start+11])
        end
    end
    Float64.(values)
end

function build_case(porder=DEFAULT_PORDER,N=DEFAULT_N,domain_ranks=DEFAULT_RANKS,
                    outdir=nothing,active_edges=[1,1,1])
    porder=positive_integer(porder,"porder")
    N=positive_integer(N,"N")
    domain_ranks=parse_ranks(domain_ranks)
    active_edges=parse_active_edges(active_edges)
    run_directory=prepare_output_directory(outdir)
    isdefined(Main,:pdemodel) || Base.include(Main,joinpath(@__DIR__,"pdemodel.jl"))

    builder=Module(gensym(:MultipleInteractionsCase))
    Core.eval(builder,:(using Exasim))
    Core.eval(builder,:(porder=$porder))
    Core.eval(builder,:(N=$N))
    Core.eval(builder,:(domainranks=$domain_ranks))
    Core.eval(builder,:(families=$FAMILIES))
    Core.eval(builder,:(run_directory=$run_directory))
    pde=Vector{Any}(undef,4)
    mesh=Vector{Any}(undef,4)
    Core.eval(builder,:(pde=$pde))
    Core.eval(builder,:(mesh=$mesh))
    for filename in ("pdeapp1.jl","pdeapp2.jl","pdeapp3.jl","pdeapp4.jl","interaction.jl")
        Base.invokelatest(Base.include,builder,joinpath(@__DIR__,filename))
    end
    all_interactions=Base.invokelatest(Core.eval,builder,:(interaction()))
    interactions=[item for (item,active) in zip(all_interactions,active_edges) if active==1]
    expected=expected_interaction_table(active_edges)
    for domain=1:4
        expected_vertices=FAMILIES[domain]==1 ? 8 : 4
        size(mesh[domain].t,1)==expected_vertices || error("domain $domain has the wrong element family")
        app=pde[domain]
        controls=(app.porder,app.pgauss,app.mpiprocs,app.NLiter,app.NLtol,
                  app.linearsolvertol,app.matvectol,app.GMRESrestart,app.linearsolveriter)
        required=(porder,2*porder,domain_ranks[domain],12,1e-12,1e-12,1e-12,250,600)
        controls==required || error("domain $domain solver controls differ from the canonical case")
        for field in (:datapath,:builddir,:buildpath)
            path=normpath(abspath(getproperty(app,field)))
            relative=relpath(path,run_directory)
            (relative=="." || (!startswith(relative,"..$(Base.Filesystem.path_separator)") && relative!="..")) ||
                error("domain $domain $field escapes outdir")
        end
    end
    (pde=pde,mesh=mesh,interactions=interactions,table=expected,selectors=[Int[] for _=1:4],
     porder=porder,N=N,domain_ranks=domain_ranks,active_edges=active_edges,outdir=run_directory)
end

function global_dgnodes(grid,family,porder)
    faces=Base.invokelatest(Exasim.Preprocessing.facenumbering,
        grid.p,grid.t,family,grid.boundaryexpr,Any[])[1]
    Exasim.Preprocessing.createdgnodes(grid.p,grid.t,faces,Int[],Any[],porder)
end

function run_example(porder=DEFAULT_PORDER,N=DEFAULT_N,domain_ranks=DEFAULT_RANKS,
                     outdir=nothing,active_edges=[1,1,1])
    case=build_case(porder,N,domain_ranks,outdir,active_edges)
    result=open(joinpath(case.outdir,"solver.log"),"w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                Base.invokelatest(Exasim.exasim,case.pde,case.mesh,case.interactions)
            end
        end
    end
    sol,pde,mesh,master,dmd=result[1:5]
    logtext=read(joinpath(case.outdir,"solver.log"),String)
    residuals=[parse(Float64,m.captures[1]) for m in
               eachmatch(r"Updated Norm:\s*([-+0-9.eE]+)",logtext)]
    !isempty(residuals) && isfinite(residuals[end]) ||
        error("solver log contains no finite global Newton residual")
    residuals[end]<=pde[1].NLtol || error("global nonlinear solve did not converge")
    off_face=[parse(Int,m.captures[1]) for m in
              eachmatch(r"points off donor face = ([0-9]+)",logtext)]
    length(off_face)>=6 && all(iszero,off_face) || error("donor-search validation failed")
    max_error=0.0
    for domain=1:4
        values=Array(sol[domain])
        ndims(values)==4 && (values=values[:,:,:,end])
        all(isfinite,values) || error("domain $domain solution is non-finite")
        coordinates=global_dgnodes(mesh[domain],FAMILIES[domain],case.porder)
        exact=dropdims(prod(sin.(pi .* coordinates),dims=2),dims=2)
        max_error=max(max_error,maximum(abs.(values[:,1,:] .- exact)))
    end
    (case=case,sol=sol,pde=pde,mesh=mesh,master=master,dmd=dmd,
     final_residual=residuals[end],max_error=max_error)
end
