#pragma once

#include <algorithm>
#include <cmath>
#include <array>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace exasim::hdgldg {

// One-time CPU schedules only. All numerical packets borrow tmp.tempn;
// lambda stays in sol.uh and the positive L face flux stays in res.Rh.
struct Endpoint {
    Int face = -1, peer = -1;
    std::vector<Int> nodeOfH;
    bool contributesFlux = true; // Only the owned L endpoint returns a flux.
};
// CPU-only matrix schedule. Numerical derivative rows stay in native L
// scratch; no matrix is owned by this descriptor. nodeOfH maps canonical
// H nodes to this endpoint's element-face nodes.
struct TraceFace {
    std::array<Int,4> key{}; // H element/face, L element/face, persistent IDs.
    Int hRank=-1, lRank=-1, element=-1, face=-1, boundarySlot=-1;
    std::vector<Int> nodeOfH;
    std::vector<Int> columns; // Other HL faces of this same L parent.
};
struct Interface {
    Int scheme = -1, boundary = 0;
    std::vector<Int> originalIDs; // Existing sidecar element IDs, including ghosts.
    std::vector<Endpoint> faces; // Common sidecar order on both endpoints.
    std::vector<TraceFace> traceFaces; // Unique faces, common global order.
};

template<class T, class I>
Interface initializeInterface(CDiscretizationT<T,I>& d,
                              const std::string& directory, Int backend)
{
#ifndef HAVE_MPI
    error("HDG/LDG coupling requires MPI");
#else
    auto& c=d.common;
    int worldSize=0, worldRank=0, localSize=0, localRank=0;
    MPI_Comm_size(EXASIM_COMM_WORLD,&worldSize);
    MPI_Comm_rank(EXASIM_COMM_WORLD,&worldRank);
    MPI_Comm_size(EXASIM_COMM_LOCAL,&localSize);
    MPI_Comm_rank(EXASIM_COMM_LOCAL,&localRank);
    if(c.mpiProcs!=localSize || c.mpiRank!=localRank || localSize>=worldSize)
        error("HDG/LDG requires separate native H and L rank groups");
    if(c.components.ncw || c.timeparams.wave || c.physicsparams.ncAV)
        error("HDG/LDG runtime checkpoint does not support auxiliary, wave or AV fields");
    const std::string suffix=localSize>1?std::to_string(localRank+1):"";
    std::ifstream in(directory+"/hdgldg"+suffix+".txt");
    std::string magic; int version=0, rank=0, local=0, selector=0;
    Interface result;
    in>>magic>>version>>result.scheme>>rank>>local>>selector>>result.boundary;
    if(!in || magic!="EXASIM_HDGLDG" || version!=1 || rank!=worldRank ||
       local!=localRank || result.scheme!=c.spatialScheme || result.boundary<=0)
        error("invalid HDG/LDG endpoint metadata");
    Int groupMin=0,groupMax=0;
    MPI_Allreduce(&result.scheme,&groupMin,1,mpi_type<Int>(),MPI_MIN,EXASIM_COMM_LOCAL);
    MPI_Allreduce(&result.scheme,&groupMax,1,mpi_type<Int>(),MPI_MAX,EXASIM_COMM_LOCAL);
    if(groupMin!=groupMax) error("HDG/LDG native rank groups must be scheme-homogeneous");
    Int ne=0; in>>ne;
    if(ne!=c.meshsizes.ne) error("HDG/LDG element metadata extent differs from native mesh");
    result.originalIDs.resize(ne);
    for(Int e=0; e<ne; ++e) { int index,id,owner; in>>index>>id>>owner;
        if(!in || index!=e || owner<0 || owner>=worldSize ||
           (e<c.meshsizes.ne1 && owner!=worldRank))
            error("invalid HDG/LDG element ownership metadata");
        result.originalIDs[e]=id;
    }
    const Int npf=c.grid.npf, npe=c.grid.npe, ncx=c.components.ncx;
    const Int ncu=c.components.ncu, nc=c.components.nc, ncw=c.components.ncw;
    const Int ngf=c.grid.ngf;
    if(npf<=0 || ncu<=0 || static_cast<long long>(npf)*ncu>std::numeric_limits<int>::max())
        error("HDG/LDG face packet exceeds MPI count range");
    const int dimensions[]={npf,ncu,ncx,c.grid.nd,c.grid.elemtype,npe,
        c.meshsizes.nfe,c.grid.porder,c.grid.nodetype};
    constexpr int dimensionCount=sizeof(dimensions)/sizeof(dimensions[0]);
    int minimum[dimensionCount],maximum[dimensionCount];
    MPI_Allreduce(dimensions,minimum,dimensionCount,MPI_INT,MPI_MIN,EXASIM_COMM_WORLD);
    MPI_Allreduce(dimensions,maximum,dimensionCount,MPI_INT,MPI_MAX,EXASIM_COMM_WORLD);
    for(int a=0;a<dimensionCount;++a) if(minimum[a]!=maximum[a])
        error("HDG/LDG conformal interface requires matching face and component dimensions");
    if(d.tmp.sztempn < npf*(ncu+nc+ncw) ||
       d.tmp.sztempg < ngf*(ncu+2*nc+2*ncw+ncu*c.grid.nd))
        error("native HDG/LDG scratch cannot hold a one-face packet and flux");
    for(Int b=0; b<c.meshsizes.nbf; ++b) if(c.fblks[3*b+2]==result.boundary) {
        const Int faces=c.fblks[3*b+1]-c.fblks[3*b]+1;
        if(static_cast<long long>(ngf)*faces*(ncu+2*nc+2*ncw+ncu*c.grid.nd)>d.tmp.sztempg)
            error("native HDG/LDG scratch cannot hold the HL face block flux");
    }
    int* tagUpper=nullptr, tagPresent=0;
    MPI_Comm_get_attr(EXASIM_COMM_WORLD,MPI_TAG_UB,&tagUpper,&tagPresent);
    if(!tagPresent || *tagUpper<3) error("HDG/LDG message tags are unavailable");
    // Native halos use the distinct split LOCAL communicator. WORLD tags
    // 1/2/3 are lambda / positive L flux / trace derivative; calls
    // complete before reusing the single borrowed packet.
    std::vector<I> elemcon(d.mesh.szelemcon), nativePerm(d.mesh.szperm);
    TemplateCopytoHost(elemcon.data(),d.mesh.elemcon,d.mesh.szelemcon,backend);
    TemplateCopytoHost(nativePerm.data(),d.mesh.perm,d.mesh.szperm,backend);
    using Key=std::array<Int,4>; // Original H/L element IDs and local faces.
    // Periodic vertex identification was already done on the complete mesh.
    // Reuse those ordered vertices, not physical-coordinate proximity, for
    // native face orientation on both ordinary and periodic HL faces.
    std::ifstream region(directory+"/hdgldg.meta");
    Int regionScheme,regionRanks,regionSelector,regionBoundary,nvf;
    region>>magic>>version>>regionScheme>>regionRanks>>regionSelector>>regionBoundary>>nvf;
    const Int expectedNvf=c.grid.nd==3?3+c.grid.elemtype:c.grid.nd;
    if(!region || magic!="EXASIM_HDGLDG_REGION" || version!=1 ||
       regionScheme!=result.scheme || regionRanks!=localSize ||
       regionSelector!=selector || regionBoundary!=result.boundary || nvf!=expectedNvf)
        error("invalid HDG/LDG regional interface metadata");
    for(Int r=0;r<regionRanks;++r) { Int world; region>>world; }
    Int globalFaces=0; region>>globalFaces;
    if(globalFaces<0) error("invalid HDG/LDG regional face count");
    std::map<Key,std::vector<int>> vertices;
    std::map<Key,std::array<Int,2>> owners;
    for(Int f=0;f<globalFaces;++f) {
        Key key; Int hRank,lRank;
        region>>key[0]>>key[1]>>key[2]>>key[3]>>hRank>>lRank;
        if(key[1]<0 || key[1]>=c.meshsizes.nfe || key[3]<0 || key[3]>=c.meshsizes.nfe ||
           hRank<0 || hRank>=worldSize || lRank<0 || lRank>=worldSize)
            error("invalid HDG/LDG regional face endpoint");
        auto& ordered=vertices[key]; ordered.resize(2*nvf);
        owners[key]={hRank,lRank};
        for(auto& vertex:ordered) region>>vertex;
    }
    if(!region) error("truncated HDG/LDG regional interface metadata");
    const Int nfe=c.meshsizes.nfe, nd=c.grid.nd, elemtype=c.grid.elemtype;
    const Int nve=elemtype==0?nd+1:(1<<nd);
    std::vector<int> localfaces(nvf*nfe), perm(nativePerm.begin(),nativePerm.end());
    getelemface(localfaces.data(),nd,elemtype);
    std::vector<T> reference(npf*std::max(Int(1),nd-1));
    TemplateCopytoHost(reference.data(),d.master.xpf,npf*(nd-1),backend);
    int* permutations=nullptr;
    permindex<T>(permutations,reference.data(),npf,nd,elemtype);
    std::map<Key,Int> hFaces, ownedElements;
    // Nine integer fields per L trace subscriber. These setup descriptors
    // are not numerical runtime buffers. Outer u-only ghosts are omitted.
    std::vector<Int> subscribers;
    Int count=0; in>>count;
    if(!in || count<0) error("invalid HDG/LDG interface count");
    for(Int i=0; i<count; ++i) {
        Int hID,hFace,lID,lFace,hRank,lRank,element,owned;
        in>>hID>>hFace>>lID>>lFace>>hRank>>lRank>>element>>owned;
        if(!in || element<0 || element>=ne || hRank==lRank ||
           hRank<0 || hRank>=worldSize || lRank<0 || lRank>=worldSize ||
           (owned!=0 && owned!=1) ||
           (owned && (result.scheme?hRank:lRank)!=worldRank) ||
           static_cast<bool>(owned)!=(element<c.meshsizes.ne1))
            error("invalid HDG/LDG interface endpoint");
        if((result.scheme && !owned) || (!result.scheme && element>=c.meshsizes.ne2)) continue;
        Int lf=result.scheme?hFace:lFace;
        if(lf<0 || lf>=c.meshsizes.nfe) error("invalid HDG/LDG element face");
        const Int encoded=elemcon[npf*(lf+c.meshsizes.nfe*element)];
        if(encoded<0) error("HDG/LDG interface has no native element-face map");
        const Int face=encoded/npf;
        if(face<0 || face>=c.meshsizes.nf)
            error("HDG/LDG interface has no native face");
        if(result.scheme) hFaces[{hID,hFace,lID,lFace}]=face;
        else subscribers.insert(subscribers.end(),{hID,hFace,lID,lFace,hRank,lRank,worldRank,face,owned});
        if(owned) ownedElements[{hID,hFace,lID,lFace}]=element;
    }
    if(!in) error("truncated HDG/LDG endpoint metadata");
    if(subscribers.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))
        error("HDG/LDG setup descriptors exceed MPI count range");
    const int localCount=static_cast<int>(subscribers.size());
    std::vector<int> counts(worldSize),offsets(worldSize);
    MPI_Allgather(&localCount,1,MPI_INT,counts.data(),1,MPI_INT,EXASIM_COMM_WORLD);
    long long total=0;
    for(int r=0;r<worldSize;++r) {
        if(total+counts[r]>std::numeric_limits<int>::max())
            error("HDG/LDG setup descriptors exceed MPI displacement range");
        offsets[r]=static_cast<int>(total); total+=counts[r];
    }
    std::vector<Int> gathered(static_cast<std::size_t>(total));
    MPI_Allgatherv(subscribers.data(),localCount,mpi_type<Int>(),gathered.data(),
        counts.data(),offsets.data(),mpi_type<Int>(),EXASIM_COMM_WORLD);
    std::vector<std::array<Int,9>> schedule;
    for(std::size_t i=0;i<gathered.size();i+=9) {
        std::array<Int,9> record;
        std::copy_n(gathered.data()+i,9,record.begin()); schedule.push_back(record);
    }
    std::sort(schedule.begin(),schedule.end());
    // Every interface must have exactly one owned L flux contributor, even
    // when several ranks hold its first-ghost trace. Catch missing/duplicate
    // ownership before starting endpoint messages.
    for(std::size_t begin=0; begin<schedule.size();) {
        std::size_t end=begin; int contributors=0;
        while(end<schedule.size() && std::equal(schedule[begin].begin(),schedule[begin].begin()+4,schedule[end].begin())) {
            const auto& record=schedule[end];
            if(record[4]!=schedule[begin][4] || record[5]!=schedule[begin][5] ||
               (record[8] && record[6]!=record[5]) ||
               (end>begin && record[6]==schedule[end-1][6]))
                error("inconsistent HDG/LDG interface ownership");
            contributors+=record[8]!=0; ++end;
        }
        if(contributors!=1) error("HDG/LDG interface requires one owned L flux contributor");
        begin=end;
    }
    for(const auto& [key,face]:hFaces) {
        const bool present=std::any_of(schedule.begin(),schedule.end(),[&](const auto& record) {
            return std::equal(key.begin(),key.end(),record.begin()); });
        if(!present) error("HDG/LDG H interface has no L trace subscriber");
    }
    // All ranks see the same total order. Only endpoints participate in each
    // message; runtime traffic remains sparse, including zero-interface ranks.
    for(const auto& record:schedule) {
        const Int hRank=record[4], consumer=record[6];
        if(worldRank!=hRank && worldRank!=consumer) continue;
        Endpoint endpoint;
        endpoint.contributesFlux=record[8]!=0;
        endpoint.nodeOfH.resize(npf);
        if(result.scheme) {
            const auto found=hFaces.find({record[0],record[1],record[2],record[3]});
            if(found==hFaces.end()) error("HDG/LDG trace subscriber has no H owner");
            endpoint.face=found->second; endpoint.peer=consumer;
        } else { endpoint.face=record[7]; endpoint.peer=hRank; }
        if(result.scheme) {
            std::iota(endpoint.nodeOfH.begin(),endpoint.nodeOfH.end(),0);
        } else {
            const auto found=vertices.find({record[0],record[1],record[2],record[3]});
            if(found==vertices.end()) error("missing HDG/LDG interface vertices");
            // Ask the existing builder for this single H/L face. Only the
            // specified face vertices are read; other element slots are unused.
            std::vector<int> t(2*nve,-1), elementFaces(2*npf*nfe,-1), faceNodes(2*npf);
            for(Int j=0;j<nvf;++j) {
                t[localfaces[j+nvf*record[1]]]=found->second[j];
                t[nve+localfaces[j+nvf*record[3]]]=found->second[nvf+j];
            }
            const int f2t[4]={0,record[1],1,record[3]};
            build_connectivity<T>(elementFaces.data(),faceNodes.data(),f2t,
                localfaces.data(),t.data(),perm.data(),permutations,nd,elemtype,npf,nfe,npe,2,1);
            for(Int j=0;j<npf;++j) {
                const int node=faceNodes[2*j+1]-npe;
                auto first=perm.begin()+npf*record[3];
                const auto match=std::find(first,first+npf,node);
                if(match==first+npf) error("invalid native HDG/LDG face permutation");
                endpoint.nodeOfH[j]=static_cast<Int>(match-first);
            }
        }
        result.faces.push_back(std::move(endpoint));
    }
    // Reuse the native boundary-face list and already established endpoint
    // orientation. Only the owner needs a local element and derivative row.
    if(!result.scheme && !c.nboufaces)
        BuildElementBlockBoundaryFaces(c,d.mesh,backend);
    std::vector<I> boundaryFaces(d.mesh.szboufaces);
    TemplateCopytoHost(boundaryFaces.data(),d.mesh.boufaces,d.mesh.szboufaces,backend);
    for(const auto& [key,ranks]:owners) {
        TraceFace row; row.key=key; row.hRank=ranks[0]; row.lRank=ranks[1];
        const auto owned=ownedElements.find(key);
        if(owned!=ownedElements.end()) {
            row.element=owned->second;
            const Int localFace=result.scheme?key[1]:key[3];
            row.face=elemcon[npf*(localFace+nfe*row.element)]/npf;
            if(result.scheme) {
                row.nodeOfH.resize(npf);
                for(Int j=0;j<npf;++j) {
                    const Int canonical=elemcon[j+npf*(localFace+nfe*row.element)]-npf*row.face;
                    if(canonical<0 || canonical>=npf) error("invalid HDG/LDG patch face map");
                    row.nodeOfH[canonical]=j;
                }
            } else {
                const auto endpoint=std::find_if(result.faces.begin(),result.faces.end(),[&](const Endpoint& e) {
                    return e.face==row.face && e.contributesFlux;
                });
                if(endpoint==result.faces.end()) error("HDG/LDG matrix row has no owned flux endpoint");
                row.nodeOfH=endpoint->nodeOfH;
                for(Int b=0;b<c.meshsizes.nbe;++b) {
                    const Int first=c.eblks[3*b]-1, last=c.eblks[3*b+1];
                    if(row.element<first || row.element>=last) continue;
                    const Int group=result.boundary-1+c.meshsizes.maxnbc*b;
                    const Int begin=c.nboufaces[group],end=c.nboufaces[group+1];
                    // Native boufaces is element-block-relative, not global.
                    const auto slot=std::find(boundaryFaces.begin()+begin,boundaryFaces.begin()+end,
                        localFace+nfe*(row.element-first));
                    if(slot!=boundaryFaces.begin()+end)
                        row.boundarySlot=static_cast<Int>(slot-boundaryFaces.begin());
                    break;
                }
                if(row.boundarySlot<0) error("HDG/LDG derivative row has no native boundary-face slot");
            }
        }
        result.traceFaces.push_back(std::move(row));
    }
    std::map<Int,std::vector<Int>> columnsOfElement;
    for(Int f=0;f<static_cast<Int>(result.traceFaces.size());++f)
        columnsOfElement[result.traceFaces[f].key[2]].push_back(f);
    for(auto& row:result.traceFaces) row.columns=columnsOfElement[row.key[2]];
    TemplateFree(permutations,0);
    c.hdgldgBoundary=result.boundary;
    if(!result.scheme && c.hasFaceBlocks())
        (void)PutFaceGatherMapGet(d.mesh.facecon,npf,npe,c.firstFace(),c.lastFace());
    return result;
#endif
    return {};
}

template<class T, class I>
void exchangeTrace(CDiscretizationT<T,I>& d, const Interface& interface, Int)
{
#ifdef HAVE_MPI
    const Int npf=d.common.grid.npf, ncu=d.common.components.ncu;
    for(const auto& endpoint:interface.faces) {
        T* packet=d.tmp.tempn;
        if(interface.scheme) {
            GetFaceNodesHDG(packet,d.sol.uh,npf,ncu,0,ncu,endpoint.face,endpoint.face+1);
            Kokkos::fence("HDG/LDG lambda before MPI");
            MPI_Send(packet,npf*ncu,mpi_type<T>(),endpoint.peer,1,EXASIM_COMM_WORLD);
        } else {
            Kokkos::fence("HDG/LDG packet before receive");
            MPI_Recv(packet,npf*ncu,mpi_type<T>(),endpoint.peer,1,EXASIM_COMM_WORLD,MPI_STATUS_IGNORE);
            // CPU indices only schedule existing device copies: no host field reads.
            for(Int j=0; j<npf; ++j) for(Int c=0; c<ncu; ++c)
                ArrayCopy(d.sol.uh+endpoint.nodeOfH[j]+npf*(c+ncu*endpoint.face),packet+j+npf*c,1);
            Kokkos::fence("HDG/LDG packet consumed");
        }
    }
#endif
}

template<class T, class I>
void returnFlux(CDiscretizationT<T,I>& d, const Interface& interface, T* hRhs, Int)
{
#ifdef HAVE_MPI
    const Int npf=d.common.grid.npf, ncu=d.common.components.ncu;
    if(interface.scheme && !hRhs) error("HDG/LDG H right-hand side is missing");
    for(const auto& endpoint:interface.faces) {
        if(!endpoint.contributesFlux) continue;
        T* packet=d.tmp.tempn;
        if(!interface.scheme) {
            for(Int j=0; j<npf; ++j) for(Int c=0; c<ncu; ++c)
                ArrayCopy(packet+j+npf*c,d.res.Rh+endpoint.nodeOfH[j]+npf*(c+ncu*endpoint.face),1);
            Kokkos::fence("HDG/LDG positive flux before MPI");
            MPI_Send(packet,npf*ncu,mpi_type<T>(),endpoint.peer,2,EXASIM_COMM_WORLD);
        } else {
            Kokkos::fence("HDG/LDG packet before receive");
            MPI_Recv(packet,npf*ncu,mpi_type<T>(),endpoint.peer,2,EXASIM_COMM_WORLD,MPI_STATUS_IGNORE);
            for(Int j=0; j<npf; ++j) for(Int c=0; c<ncu; ++c)
                ArrayAXPBY(hRhs+c+ncu*(j+npf*endpoint.face),
                    hRhs+c+ncu*(j+npf*endpoint.face),packet+j+npf*c,T(1),T(-1),1);
            Kokkos::fence("HDG/LDG packet consumed");
        }
    }
#endif
}

} // namespace exasim::hdgldg
