function interactiongraph(pde, mesh, interaction)
    n = length(pde)
    n >= 2 && length(mesh) == n || error("interaction graph requires matching pde/mesh vectors with at least two domains.")
    interaction isa AbstractVector && !isempty(interaction) || error("interaction must be a nonempty vector of named tuples.")
    all(m->!hasproperty(m,:interfacecondition) || !any(m.interfacecondition.>0),mesh) || error("generalized interactions cannot be combined with mesh.interfacecondition.")
    expected = Set((:domains,:type,:interface1,:condition1,:fluxmap1,:interface2,:condition2,:fluxmap2))
    selectors = [Int[] for _=1:n]; widths=zeros(Int,n); nfint=ones(Int,n); maxcondition=zeros(Int,n); values=Int[1,n,length(interaction)]
    face_labels = map(1:n) do d
        nd=size(mesh[d].p,1); nve=size(mesh[d].t,1)
        elemtype=((nd==2 && nve==4) || (nd==3 && nve==8)) ? 1 : 0
        facenumbering(mesh[d].p,mesh[d].t,elemtype,
            mesh[d].boundaryexpr,mesh[d].periodicexpr)[1]
    end
    for (k, entry) in enumerate(interaction)
        entry isa NamedTuple && Set(keys(entry)) == expected || error("interaction entry $k must contain exactly the eight documented fields.")
        domains = collect(entry.domains)
        length(domains)==2 && all(x->x isa Integer,domains) && all(x->1<=x<=n,domains) && domains[1]!=domains[2] || error("interaction entry $k field domains is invalid.")
        entry.type isa Integer && entry.type==1 || error("interaction entry $k: generalized interactions support only type=1.")
        endpoints=Vector{Vector{Int}}(undef,2); maps=Vector{Vector{Int}}(undef,2)
        for side=1:2
            d=domains[side]; s=getproperty(entry,Symbol("interface$side")); c=getproperty(entry,Symbol("condition$side")); map=collect(getproperty(entry,Symbol("fluxmap$side")))
            s isa Integer && 1<=s<=length(mesh[d].boundaryexpr) || error("interaction entry $k field interface$side is invalid.")
            !(s in vec(mesh[d].periodicboundary)) || error("interaction entry $k field interface$side selects a periodic boundary.")
            any(face_labels[d].==s) || error("interaction entry $k field interface$side selects no boundary face.")
            bc=vec(mesh[d].boundarycondition); s<=length(bc) && bc[s]>0 && count(==(bc[s]),bc)==1 || error("interaction entry $k field interface$side has a nonunique boundary condition.")
            !(s in selectors[d]) || error("interaction entry $k field interface$side overlaps another interaction.")
            c isa Integer && c>=1 || error("interaction entry $k field condition$side must be a positive integer.")
            !isempty(map) && all(x->x isa Integer && 1<=x<=pde[d].ncu,map) || error("interaction entry $k field fluxmap$side is invalid.")
            modelsym=Symbol(pde[d].modelfile)
            isdefined(Main,modelsym) || error("interaction entry $k domain $d does not define model module $(pde[d].modelfile).")
            model=getfield(Main,modelsym)
            isdefined(model,:fint) || error("interaction entry $k domain $d does not define fint.")
            push!(selectors[d],s); maxcondition[d]=max(maxcondition[d],c); maps[side]=Int.(map); endpoints[side]=vcat(s,c,length(map),Int.(map).-1)
        end
        for side=1:2
            d=domains[side]; width=length(maps[3-side]); widths[d] in (0,width) || error("interaction entry $k violates the one fint output width per domain rule."); widths[d]=width; nfint[d]=width
        end
        push!(values,9+length(maps[1])+length(maps[2]),domains[1]-1,domains[2]-1,1); append!(values,endpoints[1]); append!(values,endpoints[2])
    end
    controls=(:temporalscheme,:torder,:nstage,:dt,:NLiter,:NLtol,:linearsolver,:linearsolvertol,:GMRESrestart,:GMRESortho,:linearsolveriter,:preconditioner)
    for d=2:n, key in controls
        isequal(getproperty(pde[1],key),getproperty(pde[d],key)) || error("interaction graph requires identical solver control $key on every domain.")
    end
    maximum(values)<2^31 || error("interaction table value exceeds the supported 32-bit range.")
    return Float64.(values), selectors, nfint, maxcondition
end
