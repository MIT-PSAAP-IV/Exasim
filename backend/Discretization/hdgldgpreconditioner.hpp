#pragma once

#include <exasim/drivers.hpp>
#include <exasim/detail/driver_dispatch.hpp>
#include <algorithm>
#include <cstddef>
#include <limits>

namespace exasim::hdgldg {

// One positive L-outward flux row against ALL element-face trace columns,
// with u_L fixed. Native recovery is q = C*u - E*uh, hence D = H_uh - G_q*E.
//
// Output res.H is column-major [npf*ncu, npf*nfe*ncu], with component first
// inside each node. res.F holds G_q temporarily. Both are existing LDG K-arena
// views: call after native block-Jacobi construction and finish consuming the
// row before constructing another row or starting GMRES. The inverse blocks
// in K[0:szP) and the current solution/recovery arrays are not modified.
// boufaceSlot is an EXISTING device-readable mesh.boufaces entry; its element
// index is block-relative, but the native assembly helpers use only its local
// face number. No host numerical array access or numerical allocation occurs.
template<class M, class T, class I>
T* buildTraceRow(CDiscretizationT<T,I>& d, Int nativeFace, Int boufaceSlot,
                Int element, Int backend)
{
    using dstype=T;
    auto& common=d.common;
    auto& mesh=d.mesh;
    auto& master=d.master;
    auto& sol=d.sol;
    auto& res=d.res;
    auto& tmp=d.tmp;
    const Int npf=common.grid.npf, ngf=common.grid.ngf, npe=common.grid.npe;
    const Int nd=common.grid.nd, nfe=common.meshsizes.nfe;
    const Int ncu=common.components.ncu, nc=common.components.nc;
    const Int ncq=common.components.ncq, nco=common.components.nco;
    const Int ncx=common.components.ncx;
    if(common.spatialScheme!=0 || common.components.ncw ||
       common.timeparams.wave || common.physicsparams.ncAV ||
       (ncq!=0 && ncq!=ncu*nd) || nc!=ncu+ncq)
        error("HDG/LDG trace derivative requires native L storage and supported gradient-only recovery");
    if(element<0 || element>=common.meshsizes.ne1 ||
       nativeFace<0 || nativeFace>=common.meshsizes.nf ||
       boufaceSlot<0 || boufaceSlot>=mesh.szboufaces)
        error("HDG/LDG trace derivative has invalid owned face metadata");

    // The F/H size markers belong to ordinary HDG allocations, not the LDG
    // aliases. Use the actual native arena view boundaries instead.
    const std::size_t gSize=std::size_t(npf)*npe*ncu*ncq;
    const std::size_t hSize=std::size_t(npf)*npf*nfe*ncu*ncu;
    const std::size_t nodalSize=std::max(std::size_t(npf)*(ncu+nc+nco),
        std::size_t(npf)*npf*ncu*std::max(ncu,ncq));
    const std::size_t gaussSize=std::size_t(ngf)*
        (ncu+nc+nco+ncu*nd+ncu*nd*nc+ncu*ncu+ncx+nd+1);
    if(hSize>std::size_t(std::numeric_limits<Int>::max()) ||
       gSize>std::size_t(std::numeric_limits<Int>::max()) ||
       nodalSize>std::size_t(tmp.sztempn) || gaussSize>std::size_t(tmp.sztempg) ||
       !res.fhAliasesK || !res.K || !res.F || !res.G || !res.H ||
       res.F<res.K+res.szP || res.G<res.F || res.H<res.G ||
       std::size_t(res.G-res.F)<gSize ||
       std::size_t(res.H-res.K)>res.kArenaCursor ||
       res.kArenaCursor>res.szK ||
       hSize>res.kArenaCursor-std::size_t(res.H-res.K) ||
       (ncq && (!res.E || std::size_t(res.szE)<
        std::size_t(npe)*npf*nfe*common.meshsizes.ne*nd)))
        error("HDG/LDG trace derivative exceeds existing native assembly scratch");

    // One face's fields and full VECTOR flux derivatives. Fhat contracts the
    // latter in place; reserving only their contracted sizes is not sufficient.
    T* uhg=tmp.tempg;
    T* udg=uhg+ngf*ncu;
    T* odg=udg+ngf*nc;
    T* fh=odg+ngf*nco;
    T* fh_uq=fh+ngf*ncu*nd;
    T* fh_uh=fh_uq+ngf*ncu*nd*nc;
    T* xg=fh_uh+ngf*ncu*ncu;
    T* nlg=xg+ngf*ncx;
    T* jac=nlg+ngf*nd;

    GetElemNodes(tmp.tempn,sol.uh,npf,ncu,0,ncu,nativeFace,nativeFace+1);
    // findudg1 is packed by WHOLE face blocks; an arbitrary one-face slice
    // is not contiguous in that map. The native facecon gather is face-local.
    GetFaceNodes(tmp.tempn+npf*ncu,sol.udg,mesh.facecon,
                 npf,nc,npe,nc,nativeFace,nativeFace+1,1);
    if(nco)
        GetFaceNodes(tmp.tempn+npf*(ncu+nc),sol.odg,mesh.facecon,
                     npf,nco,npe,nco,nativeFace,nativeFace+1,1);
    Node2Gauss(common.cublasHandle,uhg,tmp.tempn,master.shapfgt,
               ngf,npf,ncu+nc+nco,backend);

    // Native geometry is SOA WITHIN EACH face block, not over all faces.
    Int block=-1;
    for(Int b=0;b<common.meshsizes.nbf;++b)
        if(nativeFace>=common.fblks[3*b]-1 && nativeFace<common.fblks[3*b+1]) {
            block=b; break;
        }
    if(block<0) error("HDG/LDG trace derivative face has no native geometry block");
    const Int first=common.fblks[3*block]-1;
    const Int count=common.fblks[3*block+1]-first;
    const T* geometry=sol.faceg+ngf*first*(ncx+nd+1)+ngf*(nativeFace-first);
    for(Int c=0;c<ncx;++c) ArrayCopy(xg+ngf*c,geometry+ngf*count*c,ngf);
    for(Int c=0;c<nd;++c) ArrayCopy(nlg+ngf*c,geometry+ngf*count*(ncx+c),ngf);
    ArrayCopy(jac,geometry+ngf*count*(ncx+nd),ngf);

    EXASIM_DRIVER_CALL(FhatDriver,fh,fh_uq,static_cast<T*>(nullptr),fh_uh,
        xg,udg,odg,static_cast<const T*>(nullptr),uhg,nlg,
        mesh,master,d.app,sol,tmp,common,ngf,backend);

    ArraySetValue(res.H,T(0),static_cast<Int>(hSize));
    columnwiseMultiply(fh_uh,fh_uh,jac,ngf,ncu*ncu);
    Gauss2Node(common.cublasHandle,tmp.tempn,fh_uh,master.shapfgwdotshapfg,
               ngf,npf*npf,ncu*ncu,backend);
    assembleMatrixHint(res.H,tmp.tempn,mesh.boufaces+boufaceSlot,
                       npe,npf,nfe,ncu,ncu,1);
    if(ncq) {
        ArraySetValue(res.F,T(0),static_cast<Int>(gSize));
        T* fq=fh_uq+ngf*ncu*ncu;
        columnwiseMultiply(fq,fq,jac,ngf,ncu*ncq);
        Gauss2Node(common.cublasHandle,tmp.tempn,fq,master.shapfgwdotshapfg,
                   ngf,npf*npf,ncu*ncq,backend);
        assembleMatrixGint(res.F,tmp.tempn,mesh.boufaces+boufaceSlot,mesh.perm,
                           npe,npf,nfe,ncu,ncq,1);
        for(Int dir=0;dir<nd;++dir) {
            const T* e=res.E+std::size_t(npe)*npf*nfe*
                (element+dir*common.meshsizes.ne);
            const T* g=res.F+std::size_t(npf)*npe*ncu*ncu*dir;
            schurMatrixGintMinvE(res.H,g,e,T(1),npe,ncu,ncu,npf,nfe,1);
        }
    }
    return res.H;
}

} // namespace exasim::hdgldg
