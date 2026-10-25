# Phase 1 Julia validator and golden-table driver (no code generation or solve).
root=abspath(ARGS[1]); output=abspath(ARGS[2])
using Exasim
module phase1_model
fint(args...)=Float64[0]
end
module phase1_no_fint
flux(args...)=Float64[0]
end

controls=(temporalscheme=0,torder=1,nstage=1,dt=[0.0],NLiter=20,NLtol=1e-8,
    linearsolver=0,linearsolvertol=1e-8,GMRESrestart=25,GMRESortho=0,
    linearsolveriter=200,preconditioner=0)
pde=[merge(controls,(ncu=1,elemtype=1,modelfile="phase1_model")) for _=1:3]
p=[0.0 1.0 1.0 0.0;0.0 0.0 1.0 1.0]; t=reshape([1,2,3,4],4,1)
expr=[x->abs.(x[2,:]).<1e-12,x->abs.(x[1,:].-1).<1e-12,
      x->abs.(x[2,:].-1).<1e-12,x->abs.(x[1,:]).<1e-12]
mesh=[(p=p,t=t,boundaryexpr=expr,periodicexpr=Any[],periodicboundary=Int[],
       boundarycondition=[1,2,3,4],interfacecondition=Int[]) for _=1:3]
edge1=(domains=[1,2],type=1,interface1=2,condition1=1,fluxmap1=[1],interface2=4,condition2=1,fluxmap2=[1])
edge2=(domains=[2,3],type=1,interface1=2,condition1=1,fluxmap1=[1],interface2=4,condition2=1,fluxmap2=[1])
interaction=[edge1,edge2]
golden=Float64[1,3,2,11,0,1,1,2,1,1,0,4,1,1,0,11,1,2,1,2,1,1,0,4,1,1,0]
table,_,_,_=Exasim.Preprocessing.interactiongraph(pde,mesh,interaction)
@assert table==golden
open(output,"w") do io; write(io,table); end

trimesh=copy(mesh); trip=hcat(p,[0.5,0.5]); trit=[1 2 3 4;2 3 4 1;5 5 5 5]
for d=1:3; trimesh[d]=merge(mesh[d],(p=trip,t=trit)); end
@assert Exasim.Preprocessing.interactiongraph(pde,trimesh,interaction)[1]==golden
tetmesh=copy(mesh); tetp=[0.0 1.0 0.0 0.0;0.0 0.0 1.0 0.0;0.0 0.0 0.0 1.0]; tett=reshape([1,2,3,4],4,1)
tetexpr=[x->abs.(x[1,:]).<1e-12,x->abs.(x[2,:]).<1e-12,
         x->abs.(x[3,:]).<1e-12,x->abs.(sum(x,dims=1).-1).<1e-12]
for d=1:3; tetmesh[d]=merge(mesh[d],(p=tetp,t=tett,boundaryexpr=tetexpr)); end
@assert Exasim.Preprocessing.interactiongraph(pde,tetmesh,interaction)[1]==golden

function reject(label,expected; pdes=pde,meshes=mesh,edges=interaction)
    try
        Exasim.Preprocessing.interactiongraph(pdes,meshes,edges)
    catch exc
        occursin(expected,sprint(showerror,exc)) || error("$label: wrong diagnostic: $(sprint(showerror,exc))")
        return
    end
    error("validator accepted rule mutation: $label")
end
reject("rule1","exactly the eight",edges=[merge(edge1,(extra=1,)),edge2])
reject("rule2","domains is invalid",edges=[merge(edge1,(domains=[1,1],)),edge2])
reject("rule3","only type=1",edges=[merge(edge1,(type=0,)),edge2])
badmesh=copy(mesh); badmesh[1]=merge(mesh[1],(boundaryexpr=vcat(expr,[x->falses(size(x,2))]),boundarycondition=collect(1:5))); reject("rule4","selects no boundary face",meshes=badmesh,edges=[merge(edge1,(interface1=5,)),edge2])
badmesh=copy(mesh); badmesh[1]=merge(mesh[1],(boundarycondition=[1,2,3,2],)); reject("rule5","nonunique boundary condition",meshes=badmesh)
reject("rule6","overlaps another interaction",edges=[edge1,merge(edge2,(interface1=4,))])
reject("rule7","positive integer",edges=[merge(edge1,(condition1=0,)),edge2])
reject("rule8-empty","fluxmap1 is invalid",edges=[merge(edge1,(fluxmap1=Int[],)),edge2])
reject("rule8-range","fluxmap1 is invalid",edges=[merge(edge1,(fluxmap1=[2],)),edge2])
badpde=copy(pde); badpde[1]=merge(pde[1],(modelfile="phase1_no_fint",)); reject("rule9","does not define fint",pdes=badpde)
badmesh=copy(mesh); badmesh[1]=merge(mesh[1],(interfacecondition=[0,1,0,0],)); reject("rule10","cannot be combined",meshes=badmesh)
badpde=copy(pde); badpde[2]=merge(pde[2],(torder=2,)); reject("rule11-torder","solver control torder",pdes=badpde)
badpde=copy(pde); badpde[2]=merge(pde[2],(nstage=2,)); reject("rule11-nstage","solver control nstage",pdes=badpde)
badmesh=copy(mesh); badmesh[1]=merge(mesh[1],(periodicboundary=[2],)); reject("periodic","selects a periodic boundary",meshes=badmesh)
badpde=[merge(item,(ncu=2,)) for item in pde]
reject("width","one fint output width",pdes=badpde,edges=[edge1,merge(edge2,(fluxmap2=[1,2],))])

# Phase 2 selector-union partition contract. A one-selector vector is
# identical to the legacy scalar and the union contains unique elements.
pt=[1 2 3;2 3 4]; pt2t=zeros(Int,2,3); pf=[1 2 1;2 3 4;3 4 2;4 1 3]
legacy=[Exasim.Types.DMDStruct()]
single=[Exasim.Types.DMDStruct()]
unionpart=[Exasim.Types.DMDStruct()]
empty=[Exasim.Types.DMDStruct()]
Exasim.Preprocessing.elementpartitionhdg(legacy,pt,pt2t,1,0,Int[],pf,2)
Exasim.Preprocessing.elementpartitionhdg(single,pt,pt2t,1,0,Int[],pf,[2])
Exasim.Preprocessing.elementpartitionhdg(unionpart,pt,pt2t,1,0,Int[],pf,[2,4])
Exasim.Preprocessing.elementpartitionhdg(empty,pt,pt2t,1,0,Int[],pf,Int[])
@assert single[1].elempart==legacy[1].elempart
@assert single[1].elempartpts==legacy[1].elempartpts
@assert vec(unionpart[1].elempart)==[1,2,3]
@assert vec(unionpart[1].elempartpts)==[0,3,0]
@assert vec(empty[1].elempart)==[1,2,3]
println("PASS: Julia interaction validator rules and golden table")
