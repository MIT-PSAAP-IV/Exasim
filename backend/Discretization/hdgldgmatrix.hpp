#pragma once

#include "hdgldginterface.hpp"
#include "hdgldgpreconditioner.hpp"

namespace exasim::hdgldg {

// Call after native H assembly, before GMRES borrows the L K-arena tail.
// Native K owns the inverse blocks. F/H on L are dead assembly views; H on
// H ranks remains the unmodified Schur operator for subsequent products.
template<class M>
void buildPreconditioner(CDiscretization& d, const Interface& interface,
                         dstype* uL, Int backend)
{
#ifdef HAVE_MPI
    auto& c=d.common;
    const Int npf=c.grid.npf,ncu=c.components.ncu,nfe=c.meshsizes.nfe;
    const Int ncf=npf*ncu,m=ncf*nfe;
    const std::size_t packetSize=std::size_t(ncf)*ncf;
    if(c.solverparams.preconditioner!=1)
        error("HDG/LDG coupling requires native option-1 preconditioning");
    if(packetSize>std::size_t(d.tmp.sztempn) || packetSize>std::size_t(std::numeric_limits<int>::max()))
        error("native HDG/LDG scratch cannot hold a trace derivative block");
    int worldRank=0;
    MPI_Comm_rank(EXASIM_COMM_WORLD,&worldRank);

    exchangeTrace(d,interface,backend);
    if(interface.scheme) {
        // Exactly the native Schwarz construction, with inversion postponed
        // until all in-patch L contributions have been added.
        ArrayCopy(d.res.K,d.res.H,m*m*c.meshsizes.ne);
        ElementalAdditiveSchwarz(d.res.K,d.res.H,d.mesh.f2e,d.mesh.elemcon,
            npf,nfe,ncu,c.meshsizes.nf);
    } else {
        if(!uL) error("HDG/LDG native LDG preconditioner requires its volume input");
        if(c.mpiProcs>1) {
            // Native mpiBlockJacobianLDG reuses res.ipiv as an npf*nfb
            // interior-face map, but native LDG setup sizes it npe*ncu*neb.
            // Refuse a layout that would write past it (no new storage).
            Int needed=0;
            for(Int j=0;j<c.meshsizes.nbf;++j)if(c.fblks[3*j+2]==0)
                needed=std::max(needed,npf*(c.fblks[3*j+1]-c.fblks[3*j]+1));
            if(needed>d.res.szipiv)
                error("HDG/LDG L block-Jacobi face map exceeds native ipiv capacity");
        }
        if(c.mpiProcs>1)
            mpiBlockJacobianLDG(d.res.K,uL,d.sol,d.res,d.app,d.driver_abi,
                d.master,d.mesh,d.tmp,c,c.cublasHandle,backend);
        else
            BlockJacobianLDG(d.res.K,uL,d.sol,d.res,d.app,d.driver_abi,
                d.master,d.mesh,d.tmp,c,c.cublasHandle,backend);
    }

    // One unique owned L row at a time, including every HL column of its
    // element. The common metadata order makes blocking sparse traffic safe.
    // Cross-patch blocks are transmitted too; only the Schwarz restriction is
    // stored. The full later operator uses residual differences, not this K.
    for(const auto& row:interface.traceFaces) {
        dstype* derivative=nullptr;
        if(worldRank==row.lRank)
            derivative=buildTraceRow<M>(d,row.face,row.boundarySlot,row.element,backend);
        for(const Int columnIndex:row.columns) {
            const auto& column=interface.traceFaces[columnIndex];
            dstype* packet=d.tmp.tempn;
            if(worldRank==row.lRank) {
                if(column.element!=row.element || column.nodeOfH.size()!=std::size_t(npf))
                    error("inconsistent HDG/LDG trace derivative column");
                for(Int j=0;j<npf;++j) for(Int b=0;b<ncu;++b)
                    for(Int i=0;i<npf;++i) {
                        const Int source=ncu*row.nodeOfH[i]+ncf*(b+ncu*(column.nodeOfH[j]+npf*column.key[3]));
                        const Int target=ncu*i+ncf*(b+ncu*j);
                        ArrayCopy(packet+target,derivative+source,ncu);
                    }
                Kokkos::fence("HDG/LDG trace derivative before MPI");
                MPI_Send(packet,static_cast<int>(packetSize),mpi_type<dstype>(),row.hRank,3,EXASIM_COMM_WORLD);
            } else if(worldRank==row.hRank) {
                Kokkos::fence("HDG/LDG derivative packet before receive");
                MPI_Recv(packet,static_cast<int>(packetSize),mpi_type<dstype>(),row.lRank,3,
                    EXASIM_COMM_WORLD,MPI_STATUS_IGNORE);
                if(row.key[0]==column.key[0]) {
                    if(row.element<0 || column.element!=row.element)
                        error("HDG/LDG Schwarz patch has no owned H element");
                    for(Int j=0;j<npf;++j) for(Int b=0;b<ncu;++b)
                        for(Int i=0;i<npf;++i) {
                            const Int r=ncu*(row.nodeOfH[i]+npf*row.key[1]);
                            const Int s=b+ncu*(column.nodeOfH[j]+npf*column.key[1]);
                            dstype* target=d.res.K+r+m*(s+m*row.element);
                            ArrayAXPBY(target,target,packet+ncu*i+ncf*(b+ncu*j),dstype(1),dstype(1),ncu);
                        }
                }
                Kokkos::fence("HDG/LDG derivative packet consumed");
            }
        }
    }
    if(interface.scheme) for(Int b=0;b<c.meshsizes.nbe;++b) {
        const Int first=c.eblks[3*b]-1,last=c.eblks[3*b+1];
        InverseWithWorkspace(c.cublasHandle,d.res.K+m*m*first,d.tmp.tempn,d.res.ipiv,
            m,last-first,backend,d.tmp.tempg,std::size_t(d.tmp.sztempg)*sizeof(dstype),
            d.tmp.sztempn,d.res.szipiv);
    }
#else
    error("HDG/LDG coupling requires MPI");
#endif
}

// x uses this rank's NATIVE vector layout here. Phase 4 packs/restricts owned
// H entries outside this adapter; it must never pass a packed vector directly
// to a native H kernel. The overload distinction matters: the explicit-scheme
// native overload uses only mass inverse for L, while the other applies BJ.
template<class M>
void applyNativePreconditioner(CPreconditioner<M>& preconditioner,sysstruct& sys,
        CDiscretization& d,const Interface& interface,dstype* x,Int backend)
{
    if(d.common.solverparams.preconditioner!=1)
        error("HDG/LDG coupling requires native option-1 preconditioning");
    if(interface.scheme)
        preconditioner.ApplyPreconditioner(x,sys,d,1,backend);
    else
        preconditioner.ApplyPreconditioner(x,sys,d,backend);
}

} // namespace exasim::hdgldg
