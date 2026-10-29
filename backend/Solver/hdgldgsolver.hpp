#pragma once

#include "../Discretization/hdgldgmatrix.hpp"
#include "../Discretization/elementhalo.hpp"
#include "../Solution/timestepcoeff.cpp"
#include "../Solution/previoussolutions.cpp"
#include "../Solution/updatesource.cpp"
#include "../Solution/updatesolution.cpp"

#ifdef HAVE_MPI
namespace exasim::hdgldg {

struct OwnedTraceFace {
    Int face=0;
    std::vector<Int> nodes; // Canonical node -> native face node.
};
struct TraceHalo {
    Int face=0,peer=0;
    bool sends=false;
    std::vector<Int> nodes;
};

// A non-owning view of the existing solver, not another solver/memory owner.
// Only the integer communication schedules below own storage.
template<class M>
struct System {
    CDiscretization& d;
    CResidual<M>& residual;
    CAssembler<M>& assembler;
    CPreconditioner<M>& preconditioner;
    CSolver<M>& solver;
    const Interface& interface;
    Int backend=0,N=0;
    std::vector<OwnedTraceFace> owned;
    std::vector<TraceHalo> halo;
};

template<class M> void packTrace(System<M>&,dstype*,const dstype*);
template<class M> void expandTrace(System<M>&,dstype*,const dstype*);

template<class M>
System<M> initializeSystem(CDiscretization& d,CResidual<M>& residual,
        CAssembler<M>& assembler,CPreconditioner<M>& preconditioner,
        CSolver<M>& solver,const Interface& interface,Int backend)
{
    auto& c=d.common;
    if(!c.components.ncq || c.components.ncw || c.timeparams.wave ||
       c.physicsparams.ncAV || c.ppdegree>1 || c.solverparams.RBdim ||
       c.solverparams.preconditioner!=1 || c.couplingparams.ndofuhatinterface ||
       c.solverparams.matvecOrder!=1 || !(c.solverparams.matvecTol>0) ||
       !is_finite_bitwise(c.solverparams.matvecTol))
        error("HDG/LDG coupled solve requires gradient recovery, option 1, and no auxiliary/AV/RB/polynomial coupling");
    // One coupled iteration must take identical branches on every rank.
    // Check scalar controls once; numerical buffers remain in native storage.
    const auto& p=c.solverparams;
    const Int integers[]={p.linearSolver,p.linearSolverMaxIter,p.nonlinearSolverMaxIter,
        p.gmresRestart,p.gmresOrthogMethod,p.matvecOrder};
    Int minimum[6],maximum[6];
    MPI_Allreduce(integers,minimum,6,mpi_type<Int>(),MPI_MIN,EXASIM_COMM_WORLD);
    MPI_Allreduce(integers,maximum,6,mpi_type<Int>(),MPI_MAX,EXASIM_COMM_WORLD);
    for(Int k=0;k<6;++k) if(minimum[k]!=maximum[k])
        error("HDG/LDG coupled solver controls must agree on all ranks");
    const dstype tolerances[]={p.linearSolverTol,p.nonlinearSolverTol,p.matvecTol};
    dstype lower[3],upper[3];
    for(const dstype value:tolerances) if(!is_finite_bitwise(value) || value<=0)
        error("HDG/LDG coupled solver tolerances must be finite and positive");
    MPI_Allreduce(tolerances,lower,3,mpi_type<dstype>(),MPI_MIN,EXASIM_COMM_WORLD);
    MPI_Allreduce(tolerances,upper,3,mpi_type<dstype>(),MPI_MAX,EXASIM_COMM_WORLD);
    for(Int k=0;k<3;++k) if(lower[k]!=upper[k])
        error("HDG/LDG coupled solver tolerances must agree on all ranks");
    if(p.linearSolver!=0 || p.linearSolverMaxIter<1 || p.nonlinearSolverMaxIter<1 || p.gmresRestart<1)
        error("HDG/LDG coupled solve requires GMRES with positive iteration limits");
    System<M> s{d,residual,assembler,preconditioner,solver,interface,backend};
    residual.initializeSolution();
    const Int npf=c.grid.npf,ncu=c.components.ncu,nfe=c.meshsizes.nfe;
    if(interface.scheme) {
        // Initialize allocating geometry operators ONCE, before mass work is
        // repurposed. Subsequent restores use the allocation-free q recovery.
        residual.recoverInitialState(backend,false);
        std::vector<Int> f2e(d.mesh.szf2e),elemcon(d.mesh.szelemcon);
        TemplateCopytoHost(f2e.data(),d.mesh.f2e,d.mesh.szf2e,backend);
        TemplateCopytoHost(elemcon.data(),d.mesh.elemcon,d.mesh.szelemcon,backend);
        using Key=std::array<Int,4>;
        std::map<Key,OwnedTraceFace> local;
        std::vector<Int> records;
        int rank=0,size=0;
        MPI_Comm_rank(EXASIM_COMM_LOCAL,&rank);MPI_Comm_size(EXASIM_COMM_LOCAL,&size);
        for(Int f=0;f<c.meshsizes.nf;++f) {
            Int e=f2e[4*f],lf=f2e[4*f+1],other=f2e[4*f+2],of=f2e[4*f+3];
            Key a{interface.originalIDs[e],lf,-1,-1};
            if(other>=0) {
                std::array<Int,2> left{a[0],lf},right{interface.originalIDs[other],of};
                if(right<left) { a={right[0],right[1],left[0],left[1]};e=other;lf=of; }
                else a={left[0],left[1],right[0],right[1]};
            }
            OwnedTraceFace face;face.face=f;face.nodes.resize(npf);
            for(Int j=0;j<npf;++j) {
                face.nodes[j]=elemcon[j+npf*(lf+nfe*e)]-npf*f;
                if(face.nodes[j]<0 || face.nodes[j]>=npf)
                    error("HDG/LDG owned trace has invalid native orientation");
            }
            if(!local.emplace(a,std::move(face)).second)
                error("HDG/LDG owned trace has a duplicate face key");
            records.insert(records.end(),{a[0],a[1],a[2],a[3],rank});
        }
        if(records.size()>std::size_t(std::numeric_limits<int>::max()))
            error("HDG/LDG trace ownership exceeds MPI count limits");
        const int count=static_cast<int>(records.size());
        std::vector<int> counts(size),offsets(size);
        MPI_Allgather(&count,1,MPI_INT,counts.data(),1,MPI_INT,EXASIM_COMM_LOCAL);
        std::size_t total=0;
        for(int r=0;r<size;++r) {
            if(total+counts[r]>std::size_t(std::numeric_limits<int>::max()))
                error("HDG/LDG trace ownership exceeds MPI count limits");
            offsets[r]=static_cast<int>(total);total+=counts[r];
        }
        std::vector<Int> gathered(total);
        MPI_Allgatherv(records.data(),count,mpi_type<Int>(),gathered.data(),counts.data(),offsets.data(),
            mpi_type<Int>(),EXASIM_COMM_LOCAL);
        std::map<Key,std::vector<Int>> participants;
        for(std::size_t k=0;k<total;k+=5)
            participants[{gathered[k],gathered[k+1],gathered[k+2],gathered[k+3]}].push_back(gathered[k+4]);
        // Identical key order on H ranks; rank numbers here are LOCAL only.
        for(auto& [key,peers]:participants) {
            std::sort(peers.begin(),peers.end());
            const Int owner=peers.front();
            auto face=local.find(key);
            if(face!=local.end() && rank==owner) s.owned.push_back(face->second);
            for(const Int peer:peers) if(peer!=owner && (rank==owner || rank==peer)) {
                if(face==local.end()) error("HDG/LDG trace halo endpoint is missing");
                s.halo.push_back({face->second.face,rank==owner?peer:owner,rank==owner,face->second.nodes});
            }
        }
        std::sort(s.owned.begin(),s.owned.end(),[](const auto& a,const auto& b){return a.face<b.face;});
        s.N=npf*ncu*static_cast<Int>(s.owned.size());
    } else s.N=c.sizes.ndof1;
    const std::size_t baseVolume=std::size_t(c.grid.npe)*ncu*c.meshsizes.ne1;
    std::size_t flux=0;
    for(const auto& endpoint:interface.faces) if(endpoint.contributesFlux) flux+=npf*ncu;
    if(!d.res.Mass2 || !d.res.Minv2 ||
       (interface.scheme && (baseVolume>std::size_t(d.res.szMass2) || baseVolume>std::size_t(d.res.szMinv2))) ||
       (!interface.scheme && flux>std::size_t(d.res.szMass2)) ||
       (interface.scheme && (c.sizes.ndofuhat>solver.sys.szu ||
            c.sizes.ndofuhat>solver.sys.szb || c.sizes.ndofuhat>solver.sys.szr)) ||
       s.N>solver.sys.szu || s.N>solver.sys.szx || s.N>solver.sys.szb || s.N>solver.sys.szr)
        error("HDG/LDG frozen state exceeds existing native mass/solver work; no extra allocation is allowed");
    if(interface.scheme) {
        packTrace(s,solver.sys.r,d.sol.uh);
        expandTrace(s,d.sol.uh,solver.sys.r);
        residual.evalQ(backend);
    }
    exchangeTrace(d,interface,backend);
    if(!interface.scheme) {
        // A transient stage source has not been prepared yet. Recover q only
        // rather than evaluating an equation with uninitialized sdg/sdgg.
        if(c.timeparams.tdep) residual.evalQ(backend);
        else residual.evalResidual(backend);
    }
    return s;
}

template<class M>
void packTrace(System<M>& s,dstype* out,const dstype* native)
{
    const Int npf=s.d.common.grid.npf,ncu=s.d.common.components.ncu;
    Int offset=0;
    for(const auto& face:s.owned) {
        for(Int j=0;j<npf;++j)
            ArrayCopy(s.d.tmp.tempn+ncu*j,native+ncu*(face.nodes[j]+npf*face.face),ncu);
        // Face order is ascending: compacting in place cannot overwrite a
        // later face. The packet also protects a reversed within-face map.
        ArrayCopy(out+offset,s.d.tmp.tempn,npf*ncu);
        offset+=npf*ncu;
    }
}

template<class M>
void expandTrace(System<M>& s,dstype* native,const dstype* packed)
{
    auto& d=s.d;const Int npf=d.common.grid.npf,ncu=d.common.components.ncu;
    ArraySetValue(native,dstype(0),d.common.sizes.ndofuhat);
    Int offset=0;
    for(const auto& face:s.owned) {
        for(Int j=0;j<npf;++j)
            ArrayCopy(native+ncu*(face.nodes[j]+npf*face.face),packed+offset+ncu*j,ncu);
        offset+=npf*ncu;
    }
    for(const auto& halo:s.halo) {
        dstype* packet=d.tmp.tempn;
        if(halo.sends) {
            for(Int j=0;j<npf;++j)
                ArrayCopy(packet+ncu*j,native+ncu*(halo.nodes[j]+npf*halo.face),ncu);
            Kokkos::fence("H trace replica before MPI");
            MPI_Send(packet,npf*ncu,mpi_type<dstype>(),halo.peer,0,EXASIM_COMM_LOCAL);
        } else {
            Kokkos::fence("H trace replica before receive");
            MPI_Recv(packet,npf*ncu,mpi_type<dstype>(),halo.peer,0,EXASIM_COMM_LOCAL,MPI_STATUS_IGNORE);
            for(Int j=0;j<npf;++j)
                ArrayCopy(native+ncu*(halo.nodes[j]+npf*halo.face),packet+ncu*j,ncu);
            Kokkos::fence("H trace replica consumed");
        }
    }
}

// Cold/restart and timestep-boundary recovery. Refresh ghosts BEFORE native
// PreviousSolutions copies ne2 entries into stage history. Reuse the same
// native halo as mesh adaptivity; no staging or additional packet buffers.
template<class M>
void refreshState(System<M>& s)
{
    auto& d=s.d;auto& c=d.common;auto& sys=s.solver.sys;
    const std::size_t block=std::size_t(c.grid.npe)*c.components.nc;
    if(block*c.nelemsend>std::size_t(d.tmp.szbuffsend) ||
       block*c.nelemrecv>std::size_t(d.tmp.szbuffrecv))
        error("HDG/LDG state halo exceeds native communication workspace");
    exasim_meshadapt::exchangeElementField(d,d.sol.udg,d.mesh.elemsendudg,
        d.mesh.elemrecvudg,static_cast<Int>(block));
    if(s.interface.scheme) {
        packTrace(s,sys.r,d.sol.uh);expandTrace(s,d.sol.uh,sys.r);
        s.residual.evalQ(s.backend);
        ArrayCopy(sys.u,d.sol.uh,c.sizes.ndofuhat);
    }
    exchangeTrace(d,s.interface,s.backend);
    if(!s.interface.scheme) {
        s.residual.evalQ(s.backend);
        ArrayExtract(sys.u,d.sol.udg,c.grid.npe,c.components.nc,c.meshsizes.ne,
            0,c.grid.npe,0,c.components.ncu,0,c.meshsizes.ne1);
    }
}

template<class M>
void beginStep(System<M>& s,Int step)
{
    s.d.common.timestate.currentstep=step;
    refreshState(s);
    PreviousSolutions(s.d.sol,s.solver.sys,s.d.common,s.backend);
}

template<class M>
void prepareStage(System<M>& s,Int stage,dstype stepStart)
{
    auto& d=s.d;auto& c=d.common;
    c.timestate.currentstage=stage;
    c.timestate.time=stepStart+c.dt[c.timestate.currentstep]*c.DIRKcoeff_t[stage];
    UpdateSource(d.sol,s.solver.sys,d.app,d.driver_abi,d.res,c,s.backend);
    // The native steady driver performs this projection before Newton.
    // Freeze these source arrays and dtfactor for the complete stage solve.
    for(Int b=0;b<c.meshsizes.nbe;++b) {
        const Int first=c.eblks[3*b]-1,last=c.eblks[3*b+1];
        GetElemNodes(d.tmp.tempn,d.sol.sdg,c.grid.npe,c.components.ncs,
            0,c.components.ncs,first,last);
        Node2Gauss(c.cublasHandle,d.sol.sdgg+c.grid.nge*c.components.ncs*first,
            d.tmp.tempn,d.master.shapegt,c.grid.nge,c.grid.npe,
            (last-first)*c.components.ncs,s.backend);
    }
}

template<class M>
void commitStage(System<M>& s)
{
    // The full native overload uses res.Rq, NOT the H sys.u trace cache,
    // as its temporary volume gather. DIRK22/33 are stiffly accurate.
    auto& d=s.d;
    UpdateSolution<M>(d.sol,s.solver.sys,d.app,d.driver_abi,d.res,d.tmp,d.common,s.backend);
}

template<class M>
void cacheFlux(System<M>& s)
{
    if(s.interface.scheme) return;
    const Int npf=s.d.common.grid.npf,ncu=s.d.common.components.ncu;
    Int offset=0;
    for(const auto& e:s.interface.faces) if(e.contributesFlux) {
        for(Int j=0;j<npf;++j) for(Int c=0;c<ncu;++c)
            ArrayCopy(s.d.res.Mass2+offset+j+npf*c,s.d.res.Rh+e.nodeOfH[j]+npf*(c+ncu*e.face),1);
        offset+=npf*ncu;
    }
}

// Positive flux differences ADD to the H product (unlike a Newton RHS).
template<class M>
void addFluxDifference(System<M>& s,dstype* hProduct,dstype scale)
{
    auto& d=s.d;const Int npf=d.common.grid.npf,ncu=d.common.components.ncu;
    Int offset=0;
    for(const auto& e:s.interface.faces) if(e.contributesFlux) {
        dstype* packet=d.tmp.tempn;
        if(!s.interface.scheme) {
            for(Int j=0;j<npf;++j) for(Int c=0;c<ncu;++c)
                ArrayAXPBY(packet+j+npf*c,d.res.Rh+e.nodeOfH[j]+npf*(c+ncu*e.face),
                    d.res.Mass2+offset+j+npf*c,scale,-scale,1);
            Kokkos::fence("L flux difference before MPI");
            MPI_Send(packet,npf*ncu,mpi_type<dstype>(),e.peer,2,EXASIM_COMM_WORLD);
        } else {
            Kokkos::fence("L flux difference before receive");
            MPI_Recv(packet,npf*ncu,mpi_type<dstype>(),e.peer,2,EXASIM_COMM_WORLD,MPI_STATUS_IGNORE);
            for(Int j=0;j<npf;++j) for(Int c=0;c<ncu;++c) {
                dstype* row=hProduct+c+ncu*(j+npf*e.face);
                ArrayAXPBY(row,row,packet+j+npf*c,dstype(1),dstype(1),1);
            }
            Kokkos::fence("L flux difference consumed");
        }
        offset+=npf*ncu;
    }
}

template<class M>
void restore(System<M>& s)
{
    auto& d=s.d;auto& c=d.common;auto& sys=s.solver.sys;
    if(s.interface.scheme) {
        ArrayCopy(d.sol.uh,sys.u,c.sizes.ndofuhat);
        ArrayInsert(d.sol.udg,d.res.Mass2,c.grid.npe,c.components.nc,c.meshsizes.ne,
            0,c.grid.npe,0,c.components.ncu,0,c.meshsizes.ne1);
        s.residual.evalQ(s.backend);
    } else
        ArrayInsert(d.sol.udg,sys.u,c.grid.npe,c.components.nc,c.meshsizes.ne,
            0,c.grid.npe,0,c.components.ncu,0,c.meshsizes.ne1);
    exchangeTrace(d,s.interface,s.backend);
    if(!s.interface.scheme) s.residual.evalResidual(s.backend);
}

template<class M>
void linearize(System<M>& s)
{
    auto& d=s.d;auto& c=d.common;auto& sys=s.solver.sys;
    if(s.interface.scheme) {
        packTrace(s,sys.r,d.sol.uh);
        expandTrace(s,d.sol.uh,sys.r);
        s.residual.evalQ(s.backend);
        ArrayCopy(sys.u,d.sol.uh,c.sizes.ndofuhat);
        ArrayExtract(d.res.Mass2,d.sol.udg,c.grid.npe,c.components.nc,c.meshsizes.ne,
            0,c.grid.npe,0,c.components.ncu,0,c.meshsizes.ne1);
        s.assembler.hdgAssembleLinearSystem(sys.b,s.backend);
    } else
        ArrayExtract(sys.u,d.sol.udg,c.grid.npe,c.components.nc,c.meshsizes.ne,
            0,c.grid.npe,0,c.components.ncu,0,c.meshsizes.ne1);
    buildPreconditioner<M>(d,s.interface,s.interface.scheme?nullptr:sys.u,s.backend);
    // Native BJ used Rh as work: evaluate/cache the base residual AFTER it.
    if(!s.interface.scheme) {
        s.residual.evalResidual(s.backend);cacheFlux(s);
        ArrayAXPB(sys.b,d.res.Ru,dstype(-1),dstype(0),s.N);
    }
    returnFlux(d,s.interface,s.interface.scheme?sys.b:nullptr,s.backend);
    if(s.interface.scheme) packTrace(s,sys.b,sys.b);
}

template<class M>
void matvec(System<M>& s,dstype* out,const dstype* direction)
{
    auto& d=s.d;auto& c=d.common;auto& sys=s.solver.sys;
    const dstype norm=PNORM(c.cublasHandle,s.N,0,const_cast<dstype*>(direction),s.backend);
    if(norm==0) {ArraySetValue(out,dstype(0),s.N);return;}
    if(!is_finite_bitwise(norm)) {ArraySetNaN(out,s.N);return;}
    const dstype step=c.solverparams.matvecTol/norm;
    if(!is_finite_bitwise(step) || step<=0 || !is_finite_bitwise(dstype(1)/step)) {
        ArraySetNaN(out,s.N);return;
    }
    if(s.interface.scheme) {
        expandTrace(s,d.sol.uh,direction);
        hdgMatVec(sys.r,d.res.H,d.sol.uh,d.res.Rh,d.res.Rq,d.res,d.app,d.mesh,c,d.tmp,
            c.cublasHandle,s.backend,d.getNonmatchingInteractions());
        ArrayAXPBY(d.sol.uh,sys.u,d.sol.uh,dstype(1),step,c.sizes.ndofuhat);
    } else {
        // Finish reading direction before any possibly aliased output write.
        ArrayAXPBY(d.res.Ru,sys.u,direction,dstype(1),step,s.N);
        ArrayInsert(d.sol.udg,d.res.Ru,c.grid.npe,c.components.nc,c.meshsizes.ne,
            0,c.grid.npe,0,c.components.ncu,0,c.meshsizes.ne1);
    }
    exchangeTrace(d,s.interface,s.backend);
    if(!s.interface.scheme) s.residual.evalResidual(s.backend);
    addFluxDifference(s,s.interface.scheme?sys.r:nullptr,dstype(1)/step);
    if(s.interface.scheme) packTrace(s,out,sys.r);
    else ArrayAXPBY(out,d.res.Ru,sys.b,dstype(1)/step,dstype(1)/step,s.N);
    restore(s);
}

template<class M>
void apply(System<M>& s,dstype* x)
{
    if(s.interface.scheme) {
        expandTrace(s,s.d.sol.uh,x);
        applyNativePreconditioner(s.preconditioner,s.solver.sys,s.d,s.interface,s.d.sol.uh,s.backend);
        packTrace(s,x,s.d.sol.uh);
        ArrayCopy(s.d.sol.uh,s.solver.sys.u,s.d.common.sizes.ndofuhat);
    } else applyNativePreconditioner(s.preconditioner,s.solver.sys,s.d,s.interface,x,s.backend);
}

// Nonlinear acceptance uses the RAW H volume and assembled trace equations,
// not the smaller condensed RHS alone. All reductions count owned rows once.
template<class M>
dstype nonlinearNorm(System<M>& s)
{
    auto& d=s.d;auto& c=d.common;auto& sys=s.solver.sys;
    exchangeTrace(d,s.interface,s.backend);
    dstype local=0,volume=0,trace=0,total=0;
    if(s.interface.scheme) {
        s.residual.evalQ(s.backend);
        s.assembler.hdgAssembleResidual(sys.r,s.backend);
        const Int n=c.grid.npe*c.components.ncu*c.meshsizes.ne1;
        LDOT(c.cublasHandle,n,d.res.Ru,1,d.res.Ru,1,&volume,s.backend);
    } else {
        s.residual.evalResidual(s.backend);
        LDOT(c.cublasHandle,s.N,d.res.Ru,1,d.res.Ru,1,&volume,s.backend);
    }
    returnFlux(d,s.interface,s.interface.scheme?sys.r:nullptr,s.backend);
    if(s.interface.scheme) {
        packTrace(s,sys.r,sys.r);
        LDOT(c.cublasHandle,s.N,sys.r,1,sys.r,1,&trace,s.backend);
    }
    local=volume+trace;
    MPI_Allreduce(&local,&total,1,mpi_type<dstype>(),MPI_SUM,EXASIM_COMM_WORLD);
    return std::sqrt(total);
}

template<class M>
void trial(System<M>& s,dstype alpha)
{
    auto& d=s.d;auto& c=d.common;auto& sys=s.solver.sys;
    const Int n=c.grid.npe*c.components.ncu*c.meshsizes.ne1;
    if(s.interface.scheme) {
        expandTrace(s,d.sol.uh,sys.x);
        ArrayAXPBY(d.sol.uh,sys.u,d.sol.uh,dstype(1),alpha,c.sizes.ndofuhat);
        ArrayAXPBY(d.res.Ru,d.res.Mass2,d.res.Minv2,dstype(1),alpha,n);
    } else ArrayAXPBY(d.res.Ru,sys.u,sys.x,dstype(1),alpha,n);
    ArrayInsert(d.sol.udg,d.res.Ru,c.grid.npe,c.components.nc,c.meshsizes.ne,
        0,c.grid.npe,0,c.components.ncu,0,c.meshsizes.ne1);
}

template<class M>
bool backtrack(System<M>& s,dstype baseNorm,dstype& updated,dstype& alpha)
{
    alpha=1;updated=baseNorm;
    while(alpha>=std::sqrt(std::numeric_limits<dstype>::epsilon())) {
        trial(s,alpha);updated=nonlinearNorm(s);
        if(is_finite_bitwise(updated) &&
           (updated<baseNorm || updated<=s.d.common.solverparams.nonlinearSolverTol)) return true;
        alpha*=dstype(.5);
    }
    restore(s);
    return false;
}

// Coupled orchestration only: native assembly, option-1 preconditioning,
// GMRES and recovery do the numerical work. There is one WORLD acceptance
// decision and every trial starts from the saved base, never the last trial.
template<class M>
bool solve(System<M>& s)
{
    auto& d=s.d;auto& c=d.common;auto& sys=s.solver.sys;
    int worldRank=0;MPI_Comm_rank(EXASIM_COMM_WORLD,&worldRank);
    const dstype tolerance=c.solverparams.nonlinearSolverTol;
    dstype norm=nonlinearNorm(s);
    if(!worldRank) std::cout<<"HDGLDG initial residual "<<norm<<std::endl;
    for(Int iteration=0;iteration<c.solverparams.nonlinearSolverMaxIter;++iteration) {
        if(is_finite_bitwise(norm) && norm<=tolerance) return true;
        if(!is_finite_bitwise(norm)) return false;
        linearize(s);
        ArraySetValue(sys.x,dstype(0),s.N);
        const HdgLdgLinearActions actions{&s,
            [](void* p,dstype* out,const dstype* in){matvec(*static_cast<System<M>*>(p),out,in);},
            [](void* p,dstype* x){apply(*static_cast<System<M>*>(p),x);}};
        const Int iterations=s.solver.gmres(s.assembler,d,s.preconditioner,s.N,c.spatialScheme,s.backend,&actions);
        if(!is_finite_bitwise(s.solver.state.linearSolverRelError)) {restore(s);return false;}
        if(s.interface.scheme) {
            const Int n=c.grid.npe*c.components.ncu*c.meshsizes.ne1;
            ArrayCopy(d.res.Minv2,d.res.Ru,n);
            expandTrace(s,d.sol.uh,sys.x);
            hdgGetDUDG(d.res.Minv2,d.res.F,d.sol.uh,d.res.Rh,d.mesh,c,s.backend);
            ArrayCopy(d.sol.uh,sys.u,c.sizes.ndofuhat);
        }
        dstype alpha=1,updated=norm;
        const bool accepted=backtrack(s,norm,updated,alpha);
        if(!worldRank) std::cout<<"HDGLDG Newton "<<iteration+1<<" GMRES "<<iterations
            <<" relative "<<s.solver.state.linearSolverRelError<<" alpha "<<alpha
            <<" residual "<<updated<<" accepted "<<accepted<<std::endl;
        if(!accepted) return false;
        // Publish accepted independent state, so output/restart and a later
        // restore do not see the preceding Newton base in sys.u.
        if(s.interface.scheme) {
            ArrayCopy(sys.u,d.sol.uh,c.sizes.ndofuhat);
            ArrayExtract(d.res.Mass2,d.sol.udg,c.grid.npe,c.components.nc,c.meshsizes.ne,
                0,c.grid.npe,0,c.components.ncu,0,c.meshsizes.ne1);
        } else {
            ArrayExtract(sys.u,d.sol.udg,c.grid.npe,c.components.nc,c.meshsizes.ne,
                0,c.grid.npe,0,c.components.ncu,0,c.meshsizes.ne1);
            cacheFlux(s);
        }
        norm=updated;
    }
    return is_finite_bitwise(norm) && norm<=tolerance;
}

} // namespace exasim::hdgldg
#endif
