/*
    residualstages.hpp -- the LDG residual as a sequence of replaceable stages.

    RuResidualStaged<M, Stages> runs exactly the orchestration of RuResidual (one rank) and RuResidualMPI (several
    ranks: u halo exchange overlapped with the interior element blocks), followed by the Residual tail, but every
    stage goes through a Stages policy:

        uhat(pass)                 GetUhat over all face blocks
        q(nbe1, nbe2, pass)        GetQ on element blocks [nbe1, nbe2)
        w(nbe1, nbe2)              GetW on element blocks [nbe1, nbe2)
        elem(nbe1, nbe2)           RuElem on element blocks [nbe1, nbe2)
        face()                     RuFace over all face blocks
        tail()                     face-to-element scatter (PutFaceNodes), then Ru *= -1 and, if tdep, Ru *= 1/dtfactor

    pass is 0 on one rank; with MPI it is 1 before the halo exchange completes and 2 after it. A stage may use it to
    restrict work to what the exchange can change (e.g. only faces adjacent to ghost elements in pass 2); the
    production stages ignore it.

    Policies:
      ProductionResidualStages<M>   the existing functions (RuResidual's calls, verbatim)
      RegisteredResidualStages<M>   per stage: the registered ResidualStageTable entry if non-null, else production
      residual_stages<M>            customization point; defaults to RegisteredResidualStages<M>. A compile-time model
                                    can specialize it to supply its own stages with no runtime table.

    Residual<M> takes the staged path only when a table is registered (SetResidualStages), so without one it is the
    unchanged production code. Stage tables are case-side code (e.g. generated fused kernels for one model); they
    receive the discretization structs through ResidualStageContext.
*/
#ifndef __RESIDUALSTAGES
#define __RESIDUALSTAGES

template <class T=dstype, class I=Int>
struct ResidualStageContextT {
    solstructT<T,I>& sol; resstructT<T,I>& res; appstructT<T,I>& app; masterstructT<T,I>& master;
    meshstructT<T,I>& mesh; tempstructT<T,I>& tmp; commonstructT<T,I>& common; cublasHandle_t handle; Int backend;
};
using ResidualStageContext = ResidualStageContextT<dstype, Int>;

// Runtime stage table (default precision only). Every entry is optional; nullptr selects the production stage.
struct ResidualStageTable {
    int version = 1;
    const char* name = "";
    void (*uhat)(ResidualStageContext&, int pass) = nullptr;
    void (*q)(ResidualStageContext&, Int nbe1, Int nbe2, int pass) = nullptr;
    void (*w)(ResidualStageContext&, Int nbe1, Int nbe2) = nullptr;
    void (*elem)(ResidualStageContext&, Int nbe1, Int nbe2) = nullptr;
    void (*face)(ResidualStageContext&) = nullptr;
    void (*tail)(ResidualStageContext&) = nullptr;
};

inline const ResidualStageTable*& ResidualStageRegistry() { static const ResidualStageTable* t = nullptr; return t; }
// Register (or, with nullptr, clear) the stage table the staged residual path uses. The table must outlive its use.
inline void SetResidualStages(const ResidualStageTable* t) { ResidualStageRegistry() = t; }

template <class M, class T=dstype, class I=Int>
struct ProductionResidualStages {
    using C = ResidualStageContextT<T,I>;
    static void uhat(C& c, int) {
        GetUhat<M>(c.sol, c.res, c.app, c.master, c.mesh, c.tmp, c.common, c.handle, 0, c.common.meshsizes.nbf, c.backend); }
    static void q(C& c, Int nbe1, Int nbe2, int) {
        if (c.common.components.ncq>0)
            GetQ(c.sol, c.res, c.app, c.master, c.mesh, c.tmp, c.common, c.handle, nbe1, nbe2, 0, c.common.meshsizes.nbf, c.backend); }
    static void w(C& c, Int nbe1, Int nbe2) {
        if (c.common.components.ncw>0)
            GetW<M>(c.sol, c.res, c.app, c.master, c.mesh, c.tmp, c.common, c.handle, nbe1, nbe2, 0, c.common.meshsizes.nbf, c.backend); }
    static void elem(C& c, Int nbe1, Int nbe2) {
        RuElem<M>(c.sol, c.res, c.app, c.master, c.mesh, c.tmp, c.common, c.handle, nbe1, nbe2, c.backend); }
    static void face(C& c) {
        RuFace<M>(c.sol, c.res, c.app, c.master, c.mesh, c.tmp, c.common, c.handle, 0, c.common.meshsizes.nbf, c.backend); }
    static void tail(C& c) {
        auto& common = c.common;
        if (common.mpiProcs>1) {                     // RuResidualMPI
            if (common.hasFaceBlocks())
                PutFaceNodes(c.res.Ru, c.res.Rh, c.mesh.facecon, common.grid.npf, common.components.ncu, common.grid.npe,
                             common.components.ncu, common.firstFace(), common.lastFace());
        }
        else {                                       // RuResidual
            Int f1 = common.fblks[0]-1;
            Int f2 = common.fblks[3*(common.meshsizes.nbf-1)+1];
            PutFaceNodes(c.res.Ru, &c.res.Rh[common.grid.npf*common.components.ncu*f1], c.mesh.facecon, common.grid.npf,
                         common.components.ncu, common.grid.npe, common.components.ncu, f1, f2);
        }
        ArrayMultiplyScalar(c.res.Ru, minusone, common.sizes.ndof1);                // Residual
        if (common.timeparams.tdep==1)
            ArrayMultiplyScalar(c.res.Ru, one/common.timestate.dtfactor, common.sizes.ndof1);
    }
};

template <class M, class T=dstype, class I=Int>
struct RegisteredResidualStages {
    using C = ResidualStageContextT<T,I>;
    using P = ProductionResidualStages<M,T,I>;
    static const ResidualStageTable* table() {
        if constexpr (std::is_same_v<T, dstype> && std::is_same_v<I, Int>) return ResidualStageRegistry();
        else return nullptr;
    }
    static void uhat(C& c, int pass) {
        if constexpr (std::is_same_v<C, ResidualStageContext>) { auto t = table(); if (t && t->uhat) { t->uhat(c, pass); return; } }
        P::uhat(c, pass); }
    static void q(C& c, Int b1, Int b2, int pass) {
        if constexpr (std::is_same_v<C, ResidualStageContext>) { auto t = table(); if (t && t->q) { t->q(c, b1, b2, pass); return; } }
        P::q(c, b1, b2, pass); }
    static void w(C& c, Int b1, Int b2) {
        if constexpr (std::is_same_v<C, ResidualStageContext>) { auto t = table(); if (t && t->w) { t->w(c, b1, b2); return; } }
        P::w(c, b1, b2); }
    static void elem(C& c, Int b1, Int b2) {
        if constexpr (std::is_same_v<C, ResidualStageContext>) { auto t = table(); if (t && t->elem) { t->elem(c, b1, b2); return; } }
        P::elem(c, b1, b2); }
    static void face(C& c) {
        if constexpr (std::is_same_v<C, ResidualStageContext>) { auto t = table(); if (t && t->face) { t->face(c); return; } }
        P::face(c); }
    static void tail(C& c) {
        if constexpr (std::is_same_v<C, ResidualStageContext>) { auto t = table(); if (t && t->tail) { t->tail(c); return; } }
        P::tail(c); }
};

// customization point: residual_stages<M>::type<T,I> is the policy Residual<M> uses on the staged path
template <class M>
struct residual_stages {
    template <class T, class I> using type = RegisteredResidualStages<M,T,I>;
};

template <class M, class S, class T=dstype, class I=Int>
inline void RuResidualStaged(ResidualStageContextT<T,I>& c)
{
    using dstype=T;
    auto& common = c.common;
    const bool frozen = common.physicsparams.frozenAVflag == 1;
    const bool unfrozenAV = common.physicsparams.ncAV>0 && common.physicsparams.frozenAVflag == 0;
    if (common.mpiProcs<=1) {                        // RuResidual
        S::uhat(c, 0);
        S::q(c, 0, common.meshsizes.nbe, 0);
        S::w(c, 0, common.meshsizes.nbe);
        if (unfrozenAV) GetAv<M>(c.sol, c.res, c.app, c.master, c.mesh, c.tmp, common, c.handle, c.backend);
        S::elem(c, 0, common.meshsizes.nbe);
        S::face(c);
        S::tail(c);
        return;
    }
#ifdef HAVE_MPI                                      // RuResidualMPI
    auto& sol = c.sol; auto& mesh = c.mesh; auto& tmp = c.tmp;
    Int bsz = common.grid.npe*common.components.ncu;
    Int n;
    INIT_TIMING;
    START_TIMING;
    /* copy some portion of u to buffsend */
    GetArrayAtIndex(tmp.buffsend, sol.udg, mesh.elemsendind, bsz*common.nelemsend);
#ifdef HAVE_CUDA
    cudaDeviceSynchronize();
#endif
#ifdef HAVE_HIP
    hipDeviceSynchronize();
#endif
    END_TIMING(13);
    START_TIMING;
    /* non-blocking send */
    Int neighbor, nsend, psend = 0, request_counter = 0;
    for (n=0; n<common.nnbsd; n++) {
        neighbor = common.nbsd[n];
        nsend = common.elemsendpts[n]*bsz;
        if (nsend>0) {
            MPI_Isend(&tmp.buffsend[psend], nsend, mpi_type<dstype>(), neighbor, 0,
                   EXASIM_COMM_LOCAL, &common.requests[request_counter]);
            psend += nsend;
            request_counter += 1;
        }
    }
    /* non-blocking receive */
    Int nrecv, precv = 0;
    for (n=0; n<common.nnbsd; n++) {
        neighbor = common.nbsd[n];
        nrecv = common.elemrecvpts[n]*bsz;
        if (nrecv>0) {
            MPI_Irecv(&tmp.buffrecv[precv], nrecv, mpi_type<dstype>(), neighbor, 0,
                   EXASIM_COMM_LOCAL, &common.requests[request_counter]);
            precv += nrecv;
            request_counter += 1;
        }
    }
    END_TIMING(6);
    START_TIMING;
    S::uhat(c, 1);
    END_TIMING(7);
    START_TIMING;
    S::q(c, 0, common.meshsizes.nbe0, 1);
    S::w(c, 0, common.meshsizes.nbe0);
    END_TIMING(8);
    START_TIMING;
    if (frozen) S::elem(c, 0, common.meshsizes.nbe0);
    END_TIMING(9);
    START_TIMING;
    /* wait until all send and receive operations are completely done */
    MPI_Waitall(request_counter, common.requests, common.statuses);
    END_TIMING(10);
    START_TIMING;
    /* copy buffrecv to udg */
    PutArrayAtIndex(sol.udg, tmp.buffrecv, mesh.elemrecvind, bsz*common.nelemrecv);
    END_TIMING(14);
    START_TIMING;
    S::uhat(c, 2);
    END_TIMING(7);
    START_TIMING;
    S::q(c, common.meshsizes.nbe0, common.meshsizes.nbe2, 2);
    S::w(c, common.meshsizes.nbe0, common.meshsizes.nbe2);
    if (unfrozenAV) GetAv<M>(sol, c.res, c.app, c.master, mesh, tmp, common, c.handle, c.backend);
    if (frozen) S::elem(c, common.meshsizes.nbe0, common.meshsizes.nbe1);   // interface elements
    else S::elem(c, 0, common.meshsizes.nbe1);                              // unfrozen AV: all owned elements now
    END_TIMING(11);
    START_TIMING;
    S::face(c);
    END_TIMING(12);
    S::tail(c);
#endif
}

#endif
