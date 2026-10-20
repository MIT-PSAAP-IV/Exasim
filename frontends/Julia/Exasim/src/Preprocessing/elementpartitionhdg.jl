function elementpartitionhdg(dmd,t,t2t,nproc,metis,elem2cpu=Int[],f=zeros(Int,0,0),coupledinterface=0)

nve,ne = size(t);

# Elements owning a face on the coupled interface boundary (sorted).
coupledelements() = sort(unique([I[2] for I in findall(f .== coupledinterface)]));

if nproc==1
    i = 1;
    dmd[i].elempart = reshape(1:ne,1,ne);
    dmd[i].elempartpts = reshape([ne],1,1);
    # dmd[i].nbsd = reshape([0],1,1);
    # #dmd[i].elem2cpu = reshape([0],1,1);
    # dmd[i].elemrecv = reshape([0],1,1);
    # dmd[i].elemsend = reshape([0],1,1);
    # dmd[i].elemrecvpts = reshape([0],1,1);
    # dmd[i].elemsendpts = reshape([0],1,1);

    dmd[i].nbsd = reshape([],0,0);
    #dmd[i].elem2cpu = reshape([0],1,1);
    dmd[i].elemrecv = reshape([],0,0);
    dmd[i].elemsend = reshape([],0,0);
    dmd[i].elemrecvpts = reshape([],0,0);
    dmd[i].elemsendpts = reshape([],0,0);
    if coupledinterface > 0
        # Interface elements last: [interior, interface] (MATLAB elementpartitionhdg).
        inte = coupledelements();
        tmp = [setdiff(1:ne, inte); inte];
        dmd[i].elempart = reshape(tmp,1,ne);
        dmd[i].elempartpts = [ne-length(inte) length(inte) 0];
        dmd[i].intepartpts = [ne-length(inte) length(inte) 0 0];
    end
    return dmd;
end

if isempty(elem2cpu)
    elem2cpu,~ = partition(t,ne,nproc,metis);
else
    elem2cpu = elem2cpu[:];
    if length(elem2cpu) != ne
        error("elementpartitionhdg: elem2cpu must contain one entry per element.");
    end
end
if coupledinterface > 0
    inte = coupledelements();
end

for i = 1:nproc
    intelem = findall(elem2cpu[:] .== (i-1)); # elements in subdomain i
    elem = neighboringelements(t2t, intelem); # all elements connected to elements in subdomain i
    extelem = sort(setdiff(elem,intelem)); # exterior elements

    elem = neighboringelements(t2t, extelem); # all elements connected to exterior elements
    bndelem = intersect(elem,intelem);  # boundary elements in subdmain i

    tmp = [setdiff(intelem,bndelem); bndelem; extelem];
    dmd[i].elempart = reshape(tmp,1,length(tmp)); # partitions of elements
    dmd[i].elempartpts = [length(intelem)-length(bndelem) length(bndelem) length(extelem)];
    if coupledinterface > 0
        # Order owned elements as [interior, interface-only, interface and
        # subdomain-boundary, subdomain-boundary-only], then exterior.
        interface = intersect(intelem, inte);
        part0 = union(interface, bndelem);
        part1 = sort(setdiff(intelem, part0));
        part2 = sort(setdiff(interface, bndelem));
        part3 = sort(intersect(interface, bndelem));
        part4 = sort(setdiff(bndelem, part3));
        tmp = [part1; part2; part3; part4; extelem];
        dmd[i].elempart = reshape(tmp,1,length(tmp));
        dmd[i].intepartpts = [length(part1) length(interface) length(intelem)-length(part1)-length(interface) length(extelem)];
    end
    #dmd[i].elem2cpu = elem2cpu[dmd[i].elempart];
    nelem = dmd[i].elempartpts;

    recvelem = extelem[:]; # elements received from neighbors
    tmp = hcat(elem2cpu[recvelem].+1, Vector((sum(nelem[1:2])+1):sum(nelem)));
    dmd[i].elemrecv = hcat(tmp, recvelem);
    ind = sortperm(dmd[i].elemrecv[:,1]);
    dmd[i].elemrecv = dmd[i].elemrecv[ind,:];
    dmd[i].nbsd = (sort(unique(dmd[i].elemrecv[:,1])))'; # neighboring subdomains
end

# store elements sent to neighboring subdomains to assemble the linear system
for k = 1:nproc
    dmd[k].elemsend = zeros(Int,0,1);
end
for i = 1:nproc
    for j = 1:length(dmd[i].nbsd)
        # cpu k sends information to cpu i
        k = dmd[i].nbsd[j];
        ii = findall(dmd[i].elemrecv[:,1] .== k);
        tm = dmd[i].elemrecv[ii,:];
        tm[:,1] .= i;
        tm[:,2] = indexin(tm[:,3],dmd[k].elempart[:]);
        if length(dmd[k].elemsend)==0
            dmd[k].elemsend = tm;
        else
            dmd[k].elemsend = [dmd[k].elemsend; tm];
        end
    end
end

for i = 1:nproc
    dmd[i].elemsendpts = zeros(Int,1, length(dmd[i].nbsd));
    dmd[i].elemrecvpts = zeros(Int,1, length(dmd[i].nbsd));
    for j = 1:length(dmd[i].nbsd)
        dmd[i].elemsendpts[j] = length(findall(dmd[i].elemsend[:,1] .== dmd[i].nbsd[j]));
        dmd[i].elemrecvpts[j] = length(findall(dmd[i].elemrecv[:,1] .== dmd[i].nbsd[j]));
    end
    #dmd[i].nbsend = dmd[i].elemsend[:,1];
    #dmd[i].nbrecv = dmd[i].elemrecv[:,1];        
    dmd[i].elemsend = reshape(dmd[i].elemsend[:,2],1,size(dmd[i].elemsend,1));
    dmd[i].elemrecv = reshape(dmd[i].elemrecv[:,2],1,size(dmd[i].elemrecv,1));
end

return dmd; # elem2cpu; #elempart,elempartall

end
