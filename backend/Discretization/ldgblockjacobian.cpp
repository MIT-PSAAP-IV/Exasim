/*
  ldg_block_jacobian.cpp

  Element-local LDG diagonal-block Jacobian helpers.

  The routines in this file implement the algebraic blocks described in
  docs/03-internals/ldg_linearized_weak_form_in_exasim.tex:

      alpha M Q + C U - E Uhat = 0
      B Q + D U + F Uhat      = R_u
      G U - Uhat              = 0

  Only diagonal element blocks are assembled. Neighbor-side derivatives from
  the LDG numerical flux are intentionally excluded.
*/
#ifndef __LDG_BLOCK_JACOBIAN
#define __LDG_BLOCK_JACOBIAN

#include <chrono>

struct LDGSchurBenchmarkTimes {
    double total = 0.0;
    double layoutD = 0.0;
    double layoutF = 0.0;
    double bMinvC = 0.0;
    double bMinvE = 0.0;
    double fg = 0.0;
};

struct LDGBenchmarkTimes {
    double total = 0.0;
    double insert = 0.0;
    double communication = 0.0;
    double uhat = 0.0;
    double q = 0.0;
    double w = 0.0;
    double av = 0.0;
    double elem = 0.0;
    double face = 0.0;
    double trace = 0.0;
    double schur = 0.0;
    double copy = 0.0;
    double cross = 0.0;
    double inverse = 0.0;
    LDGSchurBenchmarkTimes schurDetail;
};

struct LDGRuFaceCrossBenchmarkTimes {
    double total = 0.0;
    double prep = 0.0;
    double flux_m = 0.0;
    double flux_p = 0.0;
    double projection_m = 0.0;
    double projection_p = 0.0;
    double assemble_m = 0.0;
    double assemble_p = 0.0;
};

static inline void LDGBenchmarkFence(const Int backend)
{
    Kokkos::fence();
#ifdef HAVE_CUDA
    if (backend == 2)
        CHECK(cudaDeviceSynchronize());
#endif
#ifdef HAVE_HIP
    if (backend == 3)
        CHECK(hipDeviceSynchronize());
#endif
}

static inline double LDGBenchmarkTime()
{
    return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

static inline double LDGBenchmarkStart(const Int backend)
{
    LDGBenchmarkFence(backend);
    return LDGBenchmarkTime();
}

static inline double LDGBenchmarkStop(const double start, const Int backend)
{
    LDGBenchmarkFence(backend);
    return LDGBenchmarkTime() - start;
}

static inline void LDGPrintBenchmark(const char* label,
        const LDGBenchmarkTimes& tm, const commonstruct& common)
{
    if (common.mpiRank != 0)
        return;

    std::cout << "==> LDG BlockJacobianLDG benchmark: " << label << std::endl;
    std::cout << "  total BlockJacobianLDG              : " << tm.total << " ms" << std::endl;
    std::cout << "  insert primal state                 : " << tm.insert << " ms" << std::endl;
    if (common.mpiProcs > 1)
        std::cout << "  MPI exchange                        : " << tm.communication << " ms" << std::endl;
    std::cout << "  GetUhat                             : " << tm.uhat << " ms" << std::endl;
    std::cout << "  GetQ                                : " << tm.q << " ms" << std::endl;
    std::cout << "  GetW                                : " << tm.w << " ms" << std::endl;
    std::cout << "  GetAv                               : " << tm.av << " ms" << std::endl;
    std::cout << "  uEquationElemBlock                  : " << tm.elem << " ms" << std::endl;
    std::cout << "  uEquationElemFaceBlockLDG           : " << tm.face << " ms" << std::endl;
    std::cout << "  uhatEquationElemFaceBlockLDG        : " << tm.trace << " ms" << std::endl;
    std::cout << "  uEquationSchurBlockLDG              : " << tm.schur << " ms" << std::endl;
    if (tm.schurDetail.total > 0.0) {
        std::cout << "    ==> uEquationSchurBlockLDG benchmark" << std::endl;
        std::cout << "      total                           : " << tm.schurDetail.total << " ms" << std::endl;
        std::cout << "      layoutD                         : " << tm.schurDetail.layoutD << " ms" << std::endl;
        std::cout << "      layoutF                         : " << tm.schurDetail.layoutF << " ms" << std::endl;
        std::cout << "      BMinvC                          : " << tm.schurDetail.bMinvC << " ms" << std::endl;
        std::cout << "      BMinvE                          : " << tm.schurDetail.bMinvE << " ms" << std::endl;
        std::cout << "      F times G                       : " << tm.schurDetail.fg << " ms" << std::endl;
    }
    std::cout << "  copy local blocks                   : " << tm.copy << " ms" << std::endl;
    std::cout << "  RuFaceCrossDeriv                    : " << tm.cross << " ms" << std::endl;
    std::cout << "  Inverse                             : " << tm.inverse << " ms" << std::endl;
}

static inline void LDGPrintRuFaceCrossBenchmark(
        const LDGRuFaceCrossBenchmarkTimes& tm, const commonstruct& common)
{
    if (common.mpiRank != 0)
        return;

    double flux = tm.flux_m + tm.flux_p;
    double projection = tm.projection_m + tm.projection_p;
    double assemble = tm.assemble_m + tm.assemble_p;

    std::cout << "==> RuFaceCrossDeriv benchmark" << std::endl;
    std::cout << "  total                         : " << tm.total << " ms" << std::endl;
    std::cout << "  face data preparation         : " << tm.prep << " ms" << std::endl;
    std::cout << "  FluxDriver side minus         : " << tm.flux_m << " ms" << std::endl;
    std::cout << "  FluxDriver side plus          : " << tm.flux_p << " ms" << std::endl;
    std::cout << "  derivative projection minus   : " << tm.projection_m << " ms" << std::endl;
    std::cout << "  derivative projection plus    : " << tm.projection_p << " ms" << std::endl;
    std::cout << "  cross assembly minus          : " << tm.assemble_m << " ms" << std::endl;
    std::cout << "  cross assembly plus           : " << tm.assemble_p << " ms" << std::endl;
    std::cout << "  grouped FluxDriver total      : " << flux << " ms" << std::endl;
    std::cout << "  grouped projection total      : " << projection << " ms" << std::endl;
    std::cout << "  grouped cross assembly total  : " << assemble << " ms" << std::endl;
}

inline void LDGPutInteriorTraceMatrix(dstype* G, const Int* elemcon,
                                      const Int* f2e, const Int* perm, const Int e1, const Int npe,
                                      const Int npf, const Int nfe, const Int ne, const Int ncu)
{
    Int ndf = npf*nfe;
    Int nlocu = npe*ncu;
    Int nlocuh = ndf*ncu;
    Int N = ne*nfe*npf*ncu;
    Kokkos::parallel_for("LDGPutInteriorTraceMatrix", N, KOKKOS_LAMBDA(const size_t idx) {
        Int c = idx % ncu;
        Int t = idx / ncu;
        Int a = t % npf;
        t = t / npf;
        Int lf = t % nfe;
        Int e = t / nfe;
        Int eg = e + e1;
        Int i = a + npf*lf;

        Int face = elemcon[i + ndf*eg] / npf;
        Int isboundary = (f2e[4*face + 2] < 0) ? 1 : 0;
        if (isboundary == 0) {
            Int row = i + ndf*c;
            Int col = perm[i] + npe*c;
            G[row + nlocuh*col + nlocuh*nlocu*e] = 0.5;
        }
    });
}

inline void LDGFluxDerivativeDotNormal(dstype* fh, const dstype* fg,
                                       const dstype* nl, const dstype factor, const Int nga, const Int ncu,
                                       const Int nd, const Int ncol)
{
    Int N = nga*ncu*ncol;
    Kokkos::parallel_for("LDGFluxDerivativeDotNormal", N, KOKKOS_LAMBDA(const size_t idx) {
        Int i = idx % nga;
        Int t = idx / nga;
        Int m = t % ncu;
        Int n = t / ncu;

        dstype value = 0.0;
        for (Int d = 0; d < nd; d++) {
            value += fg[i + nga*m + nga*ncu*d + nga*ncu*nd*n] *
                nl[i + nga*d];
        }
        fh[i + nga*m + nga*ncu*n] = factor*value;
    });
}

inline void LDGAddTraceStabilizationDerivatives(dstype* fh_uq, dstype* fh_uh,
                                                const dstype* tau, const Int ntau, const Int nga, const Int ncu,
                                                const Int nc)
{
    Int N = nga*ncu*ncu;
    Kokkos::parallel_for("LDGAddTraceStabilizationDerivatives", N, KOKKOS_LAMBDA(const size_t idx) {
        Int i = idx % nga;
        Int t = idx / nga;
        Int m = t % ncu;
        Int n = t / ncu;

        dstype value = 0.0;
        if (ntau == ncu*ncu) {
            value = tau[n*ncu + m];
        }
        else if (ntau == ncu) {
            value = (m == n) ? tau[m] : 0.0;
        }
        else {
            value = (m == n) ? tau[0] : 0.0;
        }

        Int uqidx = i + nga*m + nga*ncu*n;
        Int uhidx = i + nga*m + nga*ncu*n;
        fh_uq[uqidx] += 2.0*value;
        fh_uh[uhidx] -= 2.0*value;
    });
}

inline void LDGPutBoundaryTraceMatrixNodal(dstype* G, const dstype* ub_u,
        const Int* boufaces, const Int* perm, const Int npe, const Int npf,
        const Int nfe, const Int ne, const Int ncu, const Int nfaces)
{
    Int ndf = npf*nfe;
    Int nlocu = npe*ncu;
    Int nlocuh = ndf*ncu;
    Int N = npf*nfaces*ncu*ncu;
    Kokkos::parallel_for("LDGPutBoundaryTraceMatrixNodal", N, KOKKOS_LAMBDA(const size_t idx) {
        Int i = idx % npf;
        Int t = idx / npf;
        Int f = t % nfaces;
        t = t / nfaces;
        Int m = t % ncu;
        Int n = t / ncu;

        Int lfne = boufaces[f];
        Int lf = lfne % nfe;
        Int e = lfne / nfe;

        Int row = i + npf*lf + ndf*m;
        Int col = perm[i + npf*lf] + npe*n;
        G[row + nlocuh*col + nlocuh*nlocu*e] =
            ub_u[i + npf*f + npf*nfaces*(m + ncu*n)];
    });
}

static void LDGFluxQDerivativeDotNormal(dstype* fq, const dstype* fudg,
                                        const dstype* nl, const dstype factor, const Int nga,
                                        const Int ncu, const Int nd, const Int ncq, const Int nc)
{
    Int N = nga*ncu*ncq;
    Kokkos::parallel_for("LDGFluxQDerivativeDotNormal", N, KOKKOS_LAMBDA(const size_t idx) {
        Int i = idx % nga;
        Int t = idx / nga;
        Int m = t % ncu;
        Int q = t / ncu;
        Int c = ncu + q;

        dstype value = 0.0;
        for (Int d = 0; d < nd; d++) {
            value += fudg[i + nga*m + nga*ncu*d + nga*ncu*nd*c]*
                nl[i + nga*d];
        }
        fq[i + nga*m + nga*ncu*q] = factor*value;
    });
}

static void LDGFluxQDerivativeDotNormalJac(dstype* fq, const dstype* fudg,
                                           const dstype* nl, const dstype* jac,
                                           const dstype factor, const Int ngf,
                                           const Int nfb, const Int ncu,
                                           const Int nd, const Int nc)
{
    Int nga = ngf*nfb;
    Int N = ngf*nd*ncu*ncu*nfb;

    Kokkos::parallel_for("LDGFluxQDerivativeDotNormalJac", N, KOKKOS_LAMBDA(const size_t idx) {
        Int i = idx % ngf;
        Int t = idx / ngf;
        Int qdir = t % nd;
        t = t / nd;
        Int m = t % ncu;
        t = t / ncu;
        Int c = t % ncu;
        Int f = t / ncu;

        Int ig = i + ngf*f;
        Int qcomp = c + ncu*qdir;
        Int udgcomp = ncu + qcomp;

        dstype value = 0.0;
        for (Int d = 0; d < nd; d++) {
            value += fudg[ig + nga*m + nga*ncu*d + nga*ncu*nd*udgcomp]*
                     nl[ig + nga*d];
        }

        fq[i + ngf*(qdir + nd*(m + ncu*(c + ncu*f)))] =
            factor*jac[ig]*value;
    });
}

static void LDGBuildFaceEForCrossBlock(dstype* Ef, const dstype* E,
                                       const Int* facecon, const Int* f2e,
                                       const Int* elemcon, const Int sideQ,
                                       const Int f1, const Int nfb,
                                       const Int npe, const Int npf,
                                       const Int nfe, const Int nd,
                                       const Int neStride)
{
    Int ndf = npf*nfe;
    Int sideQOffset = sideQ - 1;
    Int N = npf*nd*npf*nfb;

    Kokkos::parallel_for("LDGBuildFaceEForCrossBlock", N, KOKKOS_LAMBDA(const size_t idx) {
        Int b = idx % npf;
        Int t = idx / npf;
        Int d = t % nd;
        t = t / nd;
        Int tnode = t % npf;
        Int flocal = t / npf;
        Int f = f1 + flocal;

        dstype value = 0.0;

        if (f2e[4*f + 2] >= 0) {
            Int lfQ = (sideQ == 1) ? f2e[4*f + 1] : f2e[4*f + 3];
            Int eq = (sideQ == 1) ? f2e[4*f + 0] : f2e[4*f + 2];

            Int faceSlotQ = -1;
            Int globalTraceNode = f*npf + tnode;
            for (Int j = 0; j < npf; j++) {
                Int slot = j + npf*lfQ;
                if (elemcon[slot + ndf*eq] == globalTraceNode)
                    faceSlotQ = slot;
            }

            if ((faceSlotQ >= 0) && (eq >= 0) && (eq < neStride)) {
                Int mq = b + npf*f;
                Int kq = facecon[2*mq + sideQOffset];
                Int qnode = kq % npe;
                Int qelem = (kq - qnode) / npe;

                if ((qelem >= 0) && (qelem < neStride))
                    value = E[qnode + npe*faceSlotQ + npe*ndf*qelem +
                              npe*ndf*neStride*d];
            }
        }

        Ef[b + npf*(d + nd*(tnode + npf*flocal))] = value;
    });
}

static void LDGBuildFaceSlotQMap(Int* faceSlotQMap, const Int* f2e,
                                 const Int* elemcon, const Int sideQ,
                                 const Int f1, const Int nfb,
                                 const Int npf, const Int nfe)
{
    Int ndf = npf*nfe;
    Int N = npf*nfb;

    Kokkos::parallel_for("LDGBuildFaceSlotQMap", N, KOKKOS_LAMBDA(const size_t idx) {
        Int tnode = idx % npf;
        Int flocal = idx / npf;
        Int f = f1 + flocal;
        Int faceSlotQ = -1;

        if (f2e[4*f + 2] >= 0) {
            Int lfQ = (sideQ == 1) ? f2e[4*f + 1] : f2e[4*f + 3];
            Int eq = (sideQ == 1) ? f2e[4*f + 0] : f2e[4*f + 2];
            Int globalTraceNode = f*npf + tnode;

            for (Int j = 0; j < npf; j++) {
                Int slot = j + npf*lfQ;
                if (elemcon[slot + ndf*eq] == globalTraceNode)
                    faceSlotQ = slot;
            }
        }

        faceSlotQMap[idx] = faceSlotQ;
    });
}

static void LDGBuildFaceEForCrossBlockOptimized(dstype* Ef, const dstype* E,
                                                const Int* f2e,
                                                const Int* perm,
                                                const Int* faceSlotQMap,
                                                const Int sideQ,
                                                const Int f1,
                                                const Int nfb,
                                                const Int npe,
                                                const Int npf,
                                                const Int nfe,
                                                const Int nd,
                                                const Int neStride)
{
    Int ndf = npf*nfe;
    Int N = npf*nd*npf*nfb;

    Kokkos::parallel_for("LDGBuildFaceEForCrossBlockOptimized", N, KOKKOS_LAMBDA(const size_t idx) {
        Int b = idx % npf;
        Int t = idx / npf;
        Int d = t % nd;
        t = t / nd;
        Int tnode = t % npf;
        Int flocal = t / npf;
        Int f = f1 + flocal;

        dstype value = 0.0;

        if (f2e[4*f + 2] >= 0) {
            Int lfQ = (sideQ == 1) ? f2e[4*f + 1] : f2e[4*f + 3];
            Int eq = (sideQ == 1) ? f2e[4*f + 0] : f2e[4*f + 2];

            if ((eq >= 0) && (eq < neStride)) {
                Int qFaceSlotQ = faceSlotQMap[b + npf*flocal];
                Int faceSlotQ = faceSlotQMap[tnode + npf*flocal];
                if ((qFaceSlotQ >= 0) && (faceSlotQ >= 0)) {
                    Int qnode = perm[qFaceSlotQ];
                    value = E[qnode + npe*faceSlotQ + npe*ndf*eq +
                              npe*ndf*neStride*d];
                }
            }
        }

        Ef[b + npf*(d + nd*(tnode + npf*flocal))] = value;
    });
}

static void LDGValidateBuildFaceEForCrossBlock(dstype* EfRef,
                                               dstype* EfOpt,
                                               dstype* diff,
                                               const dstype* E,
                                               const Int* facecon,
                                               const Int* f2e,
                                               const Int* elemcon,
                                               const Int* perm,
                                               Int* faceSlotQMap,
                                               const Int sideQ,
                                               const Int f1,
                                               const Int nfb,
                                               const Int npe,
                                               const Int npf,
                                               const Int nfe,
                                               const Int nd,
                                               const Int neStride,
                                               cublasHandle_t handle,
                                               const Int backend,
                                               const Int rank)
{
    Int sz = npf*nd*npf*nfb;

    LDGBuildFaceEForCrossBlock(EfRef, E, facecon, f2e, elemcon, sideQ,
            f1, nfb, npe, npf, nfe, nd, neStride);
    LDGBuildFaceSlotQMap(faceSlotQMap, f2e, elemcon, sideQ, f1, nfb,
            npf, nfe);
    LDGBuildFaceEForCrossBlockOptimized(EfOpt, E, f2e, perm, faceSlotQMap, sideQ,
            f1, nfb, npe, npf, nfe, nd, neStride);

    ArrayAXPBY(diff, EfOpt, EfRef, one, minusone, sz);
    dstype normRef = NORM(handle, sz, EfRef, backend);
    dstype err = NORM(handle, sz, diff, backend);

    if (rank == 0) {
        std::cout << "LDGBuildFaceEForCrossBlockOptimized validation: "
                  << "sideQ = " << sideQ
                  << ", f1 = " << f1
                  << ", nfb = " << nfb
                  << ", abs = " << err
                  << ", rel = " << err/(normRef + 1.0e-14)
                  << std::endl;
    }
}

static void LDGAssembleFaceQToElementCrossBlock(dstype* A, const dstype* Rf_q,
                                                const dstype* E, const Int* facecon, const Int* f2e,
                                                const Int* elemcon, const dstype scalar, const Int sideResidual,
                                                const Int sideQ, const dstype sign, const Int f1, const Int nfb,
                                                const Int npe, const Int npf, const Int nfe, const Int ncu,
                                                const Int ncq, const Int ne, const Int neStride, const Int rank)
{
    Int ndf = npf*nfe;
    Int nlocu = npe*ncu;
    Int sideResidualOffset = sideResidual - 1;
    Int sideQOffset = sideQ - 1;
    Int N = npf*npf*npf*nfb*ncu*ncq;

    //if (rank==0) print2darray(&A[npe*npe*4], npe, npe);

    Kokkos::parallel_for("LDGAssembleFaceQToElementCrossBlock", N, KOKKOS_LAMBDA(const size_t idx) {
        Int tnode = idx % npf;
        Int t = idx / npf;
        Int qcomp = t % ncq;
        t = t / ncq;
        Int m = t % ncu;
        t = t / ncu;
        Int flocal = t % nfb;
        t = t / nfb;
        Int b = t % npf;
        Int a = t / npf;
        Int f = f1 + flocal;

        if (f2e[4*f + 2] < 0)
            return;

        Int lfQ = (sideQ == 1) ? f2e[4*f + 1] : f2e[4*f + 3];
        Int eq = (sideQ == 1) ? f2e[4*f + 0] : f2e[4*f + 2];

        Int faceSlotQ = -1;
        Int globalTraceNode = f*npf + tnode;
        for (Int j = 0; j < npf; j++) {
            Int slot = j + npf*lfQ;
            if (elemcon[slot + ndf*eq] == globalTraceNode)
                faceSlotQ = slot;
        }
        if (faceSlotQ < 0)
            return;

        Int mr = a + npf*f;
        Int mq = b + npf*f;
        Int mt = tnode + npf*f;
        Int kr = facecon[2*mr + sideResidualOffset];
        Int kq = facecon[2*mq + sideQOffset];
        Int kt = facecon[2*mt + sideResidualOffset];

        Int rownode = kr % npe;
        Int rowelem = (kr - rownode) / npe;
        Int qnode = kq % npe;
        Int qelem = (kq - qnode) / npe;
        Int unode = kt % npe;
        Int uelem = (kt - unode) / npe;

        if ((rowelem < 0) || (rowelem >= ne) || (rowelem != uelem))
            return;
        if ((qelem < 0) || (qelem >= neStride))
            return;

        Int d = qcomp / ncu;
        Int c = qcomp - d*ncu;
        const dstype* Ed = &E[npe*ndf*neStride*d];
        dstype qu = -0.5*scalar*
            Ed[qnode + npe*faceSlotQ + npe*ndf*qelem];

        Int row = rownode + npe*m;
        Int ucol = unode + npe*c;
        dstype rf = Rf_q[a + npf*b + npf*npf*(flocal + nfb*(m + ncu*qcomp))];

        // if (rank == 0 && f==10 && rowelem == 4) {
        //     printf("%d %d %d %d %d %d %d %g %g %g\n", sideQ, sideResidual, rowelem, qelem, uelem, row, ucol, sign, rf, qu);
        // }

        Kokkos::atomic_add(&A[row + nlocu*ucol + nlocu*nlocu*rowelem], sign*rf*qu);
    });
}

static void LDGPackFaceQForCrossGEMM(dstype* B, const dstype* bufq,
                                     const Int npf, const Int ncu,
                                     const Int nd, const Int nfb)
{
    Int nrow = npf*ncu*ncu;
    Int nmid = npf*nd;
    Int N = nrow*nmid*nfb;

    Kokkos::parallel_for("LDGPackFaceQForCrossGEMM", N, KOKKOS_LAMBDA(const size_t idx) {
        Int row = idx % nrow;
        Int t = idx / nrow;
        Int mid = t % nmid;
        Int f = t / nmid;

        Int a = row % npf;
        Int s = row / npf;
        Int m = s % ncu;
        Int c = s / ncu;
        Int b = mid % npf;
        Int d = mid / npf;

        B[idx] = bufq[a + npf*b +
                      npf*npf*(d + nd*(m + ncu*(c + ncu*f)))];
    });
}

static void LDGScatterCrossFaceGEMMBlock(dstype* A, const dstype* Af,
                                         const Int* facecon,
                                         const Int* f2e,
                                         const Int sideResidual,
                                         const Int f1,
                                         const Int nfb,
                                         const Int npe,
                                         const Int npf,
                                         const Int ncu,
                                         const Int ne)
{
    Int nrow = npf*ncu*ncu;
    Int nlocu = npe*ncu;
    Int sideResidualOffset = sideResidual - 1;
    Int N = nrow*npf*nfb;

    Kokkos::parallel_for("LDGScatterCrossFaceGEMMBlock", N, KOKKOS_LAMBDA(const size_t idx) {
        Int rowPacked = idx % nrow;
        Int t = idx / nrow;
        Int tnode = t % npf;
        Int flocal = t / npf;
        Int f = f1 + flocal;

        if (f2e[4*f + 2] < 0)
            return;

        Int a = rowPacked % npf;
        Int s = rowPacked / npf;
        Int m = s % ncu;
        Int c = s / ncu;

        Int mr = a + npf*f;
        Int mt = tnode + npf*f;
        Int kr = facecon[2*mr + sideResidualOffset];
        Int kt = facecon[2*mt + sideResidualOffset];

        Int rownode = kr % npe;
        Int rowelem = (kr - rownode) / npe;
        Int unode = kt % npe;
        Int uelem = (kt - unode) / npe;

        if ((rowelem < 0) || (rowelem >= ne) || (rowelem != uelem))
            return;

        Int row = rownode + npe*m;
        Int col = unode + npe*c;
        dstype value = Af[rowPacked + nrow*tnode + nrow*npf*flocal];
        Kokkos::atomic_add(&A[row + nlocu*col + nlocu*nlocu*rowelem], value);
    });
}


static void LDGSchurMatrixF(dstype* F, const dstype* Ftmp, const Int npe,
                            const Int ncu, const Int npf, const Int nfe, const Int ne)
{
    Int ndf = npf*nfe;
    Int n = npe*ncu;
    Int m = ndf*ncu;
    Int M = npe*ndf;
    Int L = M*ne;
    Int N = n*m*ne;

    Kokkos::parallel_for("LDGSchurMatrixF", N, KOKKOS_LAMBDA(const size_t idx) {
        Int i = idx % npe;
        Int t = idx / npe;
        Int r = t % ncu;
        t = t / ncu;
        Int j = t % ndf;
        t = t / ndf;
        Int s = t % ncu;
        Int e = t / ncu;

        F[idx] = Ftmp[i + npe*j + M*e + L*(r + ncu*s)];
    });
}

static void LDGSchurMatrixBMinvE(dstype* F, const dstype* B,
                                 const dstype* MinvE, const dstype scalar, const Int npe,
                                 const Int ncu, const Int npf, const Int nfe, const Int ne)
{
    Int ndf = npf*nfe;
    Int n = npe*ncu;
    Int m = ndf*ncu;
    Int M = npe*npe;
    Int L = M*ne;
    Int ME = npe*ndf;
    Int N = n*m*ne;

    Kokkos::parallel_for("LDGSchurMatrixBMinvE", N, KOKKOS_LAMBDA(const size_t idx) {
        Int i = idx % npe;
        Int t = idx / npe;
        Int r = t % ncu;
        t = t / ncu;
        Int j = t % ndf;
        t = t / ndf;
        Int s = t % ncu;
        Int e = t / ncu;

        dstype sum = 0.0;
        for (Int k = 0; k < npe; k++) {
            sum += B[i + npe*k + M*e + L*(r + ncu*s)]* MinvE[k + npe*j + ME*e];
        }
        F[idx] -= scalar*sum;
    });
}

static void LDGScatterBMinvCToD(dstype* D, const dstype* work,
                                const Int npe, const Int ncu, const Int ne,
                                const Int mcomp, const Int ncomp)
{
    Int nlocu = npe*ncu;
    Int M = npe*npe;
    Int N = M*ne;

    Kokkos::parallel_for("LDGScatterBMinvCToD", N, KOKKOS_LAMBDA(const size_t idx) {
        Int i = idx % npe;
        Int t = idx / npe;
        Int j = t % npe;
        Int e = t / npe;

        Int row = i + npe*mcomp;
        Int col = j + npe*ncomp;

        D[row + nlocu*col + nlocu*nlocu*e] += work[i + npe*j + M*e];
    });
}

static void LDGSchurMatrixBMinvC_GEMM(cublasHandle_t handle,
                                      dstype* D, const dstype* B,
                                      const dstype* MinvC, dstype* work,
                                      const dstype scalar, const Int npe,
                                      const Int ncu, const Int ne,
                                      const Int backend)
{
    Int M = npe*npe;
    Int L = M*ne;
    Int K = L*ncu;

    for (Int ncomp = 0; ncomp < ncu; ncomp++) {
        for (Int mcomp = 0; mcomp < ncu; mcomp++) {
            const dstype* Bmn = &B[L*mcomp + K*ncomp];

            PGEMNMStridedBached(handle, npe, npe, npe, scalar,
                    const_cast<dstype*>(Bmn), npe,
                    const_cast<dstype*>(MinvC), npe,
                    0.0, work, npe, ne, backend);

            LDGScatterBMinvCToD(D, work, npe, ncu, ne, mcomp, ncomp);
        }
    }
}

static void LDGScatterBMinvEToF(dstype* F, const dstype* work,
                                const Int npe, const Int ncu, const Int npf,
                                const Int nfe, const Int ne,
                                const Int mcomp, const Int ncomp)
{
    Int ndf = npf*nfe;
    Int nlocu = npe*ncu;
    Int m = ndf*ncu;
    Int M = npe*ndf;
    Int N = M*ne;

    Kokkos::parallel_for("LDGScatterBMinvEToF", N, KOKKOS_LAMBDA(const size_t idx) {
        Int i = idx % npe;
        Int t = idx / npe;
        Int j = t % ndf;
        Int e = t / ndf;

        Int row = i + npe*mcomp;
        Int col = j + ndf*ncomp;

        F[row + nlocu*col + nlocu*m*e] -= work[i + npe*j + M*e];
    });
}

static void LDGSchurMatrixBMinvE_GEMM(cublasHandle_t handle,
                                      dstype* F, const dstype* B,
                                      const dstype* MinvE, dstype* work,
                                      const dstype scalar, const Int npe,
                                      const Int ncu, const Int npf,
                                      const Int nfe, const Int ne,
                                      const Int backend)
{
    Int M = npe*npe;
    Int L = M*ne;
    Int K = L*ncu;
    Int ndf = npf*nfe;

    for (Int ncomp = 0; ncomp < ncu; ncomp++) {
        for (Int mcomp = 0; mcomp < ncu; mcomp++) {
            const dstype* Bmn = &B[L*mcomp + K*ncomp];

            PGEMNMStridedBached(handle, npe, ndf, npe, scalar,
                    const_cast<dstype*>(Bmn), npe,
                    const_cast<dstype*>(MinvE), npe,
                    0.0, work, npe, ne, backend);

            LDGScatterBMinvEToF(F, work, npe, ncu, npf, nfe, ne,
                    mcomp, ncomp);
        }
    }
}

void uhatEquationElemFaceBlockLDG(solstruct &sol, resstruct &res, appstruct &app,
                           ExasimDriverABI& driver_abi, masterstruct &master, meshstruct &mesh,
                           tempstruct &tmp, commonstruct &common, cublasHandle_t handle,
                           Int jth, Int backend)
{
    Int e1 = common.eblks[3*jth]-1;
    Int e2 = common.eblks[3*jth+1];
    Int npe = common.npe;
    Int npf = common.npf;
    Int nfe = common.nfe;
    Int ncu = common.ncu;
    Int nc = common.nc;
    Int nco = common.nco;
    Int ncw = common.ncw;
    Int ncx = common.ncx;
    Int nd = common.nd;
    Int ne = e2 - e1;
    Int ndf = npf*nfe;
    Int nlocu = npe*ncu;
    Int nlocuh = ndf*ncu;

    ArraySetValue(res.G, 0.0, nlocuh*nlocu*ne);
    LDGPutInteriorTraceMatrix(res.G, mesh.elemcon, mesh.f2e, mesh.perm,
            e1, npe, npf, nfe, ne, ncu);

    Int nf = nfe*ne;
    Int nn = npf*nf;
    
    GetElementFaceNodes(tmp.tempn, sol.uh, mesh.elemcon, npf*nfe, ncu, e1, e2, 0);
    GetElementFaceNodes(&tmp.tempn[nn*ncu], sol.udg, mesh.perm, npf*nfe, nc, npe, nc, e1, e2);
    if (nco > 0)
        GetElementFaceNodes(&tmp.tempn[nn*(ncu+nc)], sol.odg, mesh.perm, npf*nfe, nco, npe, nco, e1, e2);
    if (ncw > 0)
        GetElementFaceNodes(&tmp.tempn[nn*(ncu+nc+nco)], sol.wdg, mesh.perm, npf*nfe, ncw, npe, ncw, e1, e2);

    Int nfn = npf*nf;
    dstype *xfnode = tmp.tempg;
    dstype *xfn = &tmp.tempg[nfn*ncx];
    dstype *nlfn = &tmp.tempg[nfn*2*ncx];
    dstype *jacfn = &tmp.tempg[nfn*(2*ncx + nd)];
    dstype *Jfn = &tmp.tempg[nfn*(2*ncx + nd + 1)];
    Int n9 = nfn*(2*ncx + nd + 1 + 2*nd);

    GetElementFaceNodes(xfnode, sol.xdg, mesh.perm, npf*nfe, ncx, npe, ncx, e1, e2);
    Node2Gauss(handle, xfn, xfnode, master.shapfnt, npf, npf, nf*ncx, backend);
    if (nd == 1) {
        FaceGeom1D(jacfn, nlfn, Jfn, nfn);
        FixNormal1D(nlfn, nfn);
    }
    else if (nd == 2) {
        Node2Gauss(handle, Jfn, xfnode, &master.shapfnt[npf*npf],
                npf, npf, nf*nd, backend);
        FaceGeom2D(jacfn, nlfn, Jfn, nfn);
    }
    else if (nd == 3) {
        Node2Gauss(handle, Jfn, xfnode, &master.shapfnt[npf*npf],
                npf, npf, nf*nd, backend);
        Node2Gauss(handle, &Jfn[nfn*nd], xfnode, &master.shapfnt[2*npf*npf],
                npf, npf, nf*nd, backend);
        FaceGeom3D(jacfn, nlfn, Jfn, nfn);
    }
    
    for (Int ibc = 0; ibc < common.maxnbc; ibc++) {
        Int n = ibc + common.maxnbc*jth;
        Int start = common.nboufaces[n];
        Int nfaces = common.nboufaces[n + 1] - start;
        if (nfaces > 0) {
            Int nnb = nfaces*npf;
            Int ubwSize = max((Int) 1, nnb*ncu*ncw);
            dstype *xgb = &tmp.tempg[n9];
            dstype *ugb = &tmp.tempg[n9 + nnb*ncx];
            dstype *ogb = &tmp.tempg[n9 + nnb*ncx + nnb*nc];
            dstype *wgb = &tmp.tempg[n9 + nnb*ncx + nnb*nc + nnb*nco];
            dstype *uhb = &tmp.tempg[n9 + nnb*ncx + nnb*nc + nnb*nco + nnb*ncw];
            dstype *nlb = &tmp.tempg[n9 + nnb*ncx + nnb*nc + nnb*nco + nnb*ncw + nnb*ncu];
            dstype *ub = &tmp.tempg[n9 + nnb*ncx + nnb*nc + nnb*nco + nnb*ncw + nnb*ncu + nnb*nd];
            dstype *ub_u = &tmp.tempg[n9 + nnb*ncx + nnb*nc + nnb*nco + nnb*ncw + nnb*ncu + nnb*nd + nnb*ncu];
            dstype *ub_w = &tmp.tempg[n9 + nnb*ncx + nnb*nc + nnb*nco + nnb*ncw + nnb*ncu + nnb*nd + nnb*ncu + nnb*ncu*nc];
            dstype *ub_uh = &tmp.tempg[n9 + nnb*ncx + nnb*nc + nnb*nco + nnb*ncw + nnb*ncu + nnb*nd + nnb*ncu + nnb*ncu*nc + ubwSize];
    
            GetBoundaryNodes(xgb, xfn, &mesh.boufaces[start], npf, nfe, ne, ncx, nfaces);
            GetBoundaryNodes(nlb, nlfn, &mesh.boufaces[start], npf, nfe, ne, nd, nfaces);
            GetBoundaryNodes(uhb, &tmp.tempn[0], &mesh.boufaces[start], npf, nfe, ne, ncu, nfaces);
            GetBoundaryNodes(ugb, &tmp.tempn[nn*ncu], &mesh.boufaces[start], npf, nfe, ne, nc, nfaces);
            GetBoundaryNodes(ogb, &tmp.tempn[nn*(ncu+nc)], &mesh.boufaces[start], npf, nfe, ne, nco, nfaces);
            GetBoundaryNodes(wgb, &tmp.tempn[nn*(ncu+nc+nco)], &mesh.boufaces[start], npf, nfe, ne, ncw, nfaces);
    
            ArraySetValue(ub, 0.0, nnb*ncu);
            ArraySetValue(ub_u, 0.0, nnb*ncu*nc);
            if (ncw > 0)
                ArraySetValue(ub_w, 0.0, nnb*ncu*ncw);
            ArraySetValue(ub_uh, 0.0, nnb*ncu*ncu);
    
            UbouJacDriver(ub, ub_u, ub_w, ub_uh, xgb, ugb, ogb, wgb, uhb,
                    nlb, driver_abi, mesh, master, app, sol, tmp, common,
                    nnb, ibc+1, backend);
    
            LDGPutBoundaryTraceMatrixNodal(res.G, ub_u, &mesh.boufaces[start],
                    mesh.perm, npe, npf, nfe, ne, ncu, nfaces);
        }
    }
}

void uEquationElemFaceBlockLDG(solstruct &sol, resstruct &res, appstruct &app,
        ExasimDriverABI& driver_abi, masterstruct &master, meshstruct &mesh,
        tempstruct &tmp, commonstruct &common, cublasHandle_t handle,
        Int jth, Int backend)
{
    Int nc = common.nc;
    Int ncu = common.ncu;
    Int ncq = common.ncq;
    Int nco = common.nco;
    Int ncx = common.ncx;
    Int ncw = common.ncw;
    Int nd = common.nd;
    Int npe = common.npe;
    Int npf = common.npf;
    Int ngf = common.ngf;
    Int nfe = common.nfe;

    Int e1 = common.eblks[3*jth]-1;
    Int e2 = common.eblks[3*jth+1];
    Int ne = e2-e1;
    Int nf = nfe*ne;
    Int nn = npf*nf;
    Int nga = ngf*nf;

    Int n4 = nga*ncu;
    Int n5 = nga*(ncu+nc);
    Int n6 = nga*(ncu+nc+nco);
    Int n7 = nga*(ncu+nc+nco+ncw);
    Int n8 = nga*(ncu+nc+nco+ncw+ncw);
    Int nm = ngf*nfe*e1*(ncx+nd+1);

    dstype *xg = &sol.elemfaceg[nm];
    dstype *nlg = &sol.elemfaceg[nm + nga*ncx];
    dstype *jac = &sol.elemfaceg[nm + nga*(ncx+nd)];

    dstype *uhg = &tmp.tempg[0];
    dstype *udg = &tmp.tempg[n4];
    dstype *odg = &tmp.tempg[n5];
    dstype *wsrc = &tmp.tempg[n6];
    dstype *wdg = &tmp.tempg[n7];

    dstype *fg     = &tmp.tempg[n8];
    dstype *fg_uq  = &tmp.tempg[n8 + nga*ncu*nd];
    dstype *fh_uh  = &tmp.tempg[n8 + nga*ncu*nd + nga*ncu*nd*nc];
    dstype *fg_w   = &tmp.tempg[n8 + nga*ncu*nd + nga*ncu*nd*nc + nga*ncu*ncu];
    dstype *wdg_uq = &tmp.tempg[n8 + nga*ncu*nd + nga*ncu*nd*nc + nga*ncu*ncu + nga*ncu*nd*ncw];

    GetElementFaceNodes(tmp.tempn, sol.uh, mesh.elemcon, npf*nfe, ncu, e1, e2, 0);
    GetElementFaceNodes(&tmp.tempn[nn*ncu], sol.udg, mesh.perm, npf*nfe, nc, npe, nc, e1, e2);
    if (nco > 0)
        GetElementFaceNodes(&tmp.tempn[nn*(ncu+nc)], sol.odg, mesh.perm, npf*nfe, nco, npe, nco, e1, e2);
    if ((ncw > 0) && (common.wave == 0)) {
        GetElementFaceNodes(&tmp.tempn[nn*(ncu+nc+nco)], sol.wsrc, mesh.perm, npf*nfe, ncw, npe, ncw, e1, e2);
        GetElementFaceNodes(&tmp.tempn[nn*(ncu+nc+nco+ncw)], sol.wdg, mesh.perm, npf*nfe, ncw, npe, ncw, e1, e2);
    }

    Node2Gauss(handle, tmp.tempg, tmp.tempn, master.shapfgt,
            ngf, npf, nf*(ncu+nc+nco+ncw+ncw), backend);

    if ((ncw > 0) && (common.wave == 0)) {
        wEquation(wdg, wdg_uq, xg, udg, odg, wsrc, tmp.tempn, app,
                driver_abi, common, nga, backend);
    }

    ArraySetValue(fg, 0.0, nga*ncu*nd);
    ArraySetValue(fg_uq, 0.0, nga*ncu*nd*nc);
    if (ncw > 0)
        ArraySetValue(fg_w, 0.0, nga*ncu*nd*ncw);

    FluxDriver(fg, fg_uq, fg_w, xg, udg, odg, wdg, driver_abi,
            mesh, master, app, sol, tmp, common, ngf*nfe, e1, e2, backend);

    ArraySetValue(fh_uh, 0.0, nga*ncu*ncu);
    LDGFluxDerivativeDotNormal(tmp.tempn, fg_uq, nlg, 0.5, nga, ncu, nd, nc);
    ArrayCopy(fg_uq, tmp.tempn, nga*ncu*nc);

    if ((ncw > 0) && (common.wave == 0)) {
        LDGFluxDerivativeDotNormal(tmp.tempn, fg_w, nlg, 0.5, nga, ncu, nd, ncw);
        ArrayCopy(fg_w, tmp.tempn, nga*ncu*ncw);
        ArrayGemmBatch2(fg_uq, fg_w, wdg_uq, one, ncu, nc, ncw, nga);
    }

    LDGAddTraceStabilizationDerivatives(fg_uq, fh_uh, app.tau, common.ntau,
            nga, ncu, nc);

    columnwiseMultiply(fg_uq, fg_uq, jac, nga, ncu*nc);
    columnwiseMultiply(fh_uh, fh_uh, jac, nga, ncu*ncu);

    dstype *Dtmp = &res.H[0];
    dstype *Btmp = &res.H[npf*npf*nf*ncu*ncu];
    dstype *Ftmp = &res.H[npf*npf*nf*ncu*ncu + npf*npf*nf*ncu*ncq];

    Gauss2Node(handle, Dtmp, fg_uq, master.shapfgwdotshapfg,
            ngf, npf*npf, nf*ncu*ncu, backend);

    if (ncq > 0) {
        Gauss2Node(handle, Btmp, &fg_uq[nga*ncu*ncu],
                master.shapfgwdotshapfg, ngf, npf*npf, nf*ncu*ncq, backend);
    }

    Gauss2Node(handle, Ftmp, fh_uh, master.shapfgwdotshapfg,
            ngf, npf*npf, nf*ncu*ncu, backend);

    for (Int ibc = 0; ibc < common.maxnbc; ibc++) {
        Int n = ibc + common.maxnbc*jth;
        Int start = common.nboufaces[n];
        Int nfaces = common.nboufaces[n + 1] - start;
        if (nfaces > 0) {
            Int ngb = nfaces*ngf;
            dstype *xgb = &tmp.tempg[n8];
            dstype *ugb = &tmp.tempg[n8 + ngb*ncx];
            dstype *ogb = &tmp.tempg[n8 + ngb*ncx + ngb*nc];
            dstype *wgb = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco];
            dstype *uhb = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw];
            dstype *nlb = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu];
            dstype *wsb = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu + ngb*nd];
            dstype *fhb = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu + ngb*nd + ngb*ncw];
            dstype *fhb_uq = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu + ngb*nd + ngb*ncw + ngb*ncu];
            dstype *fhb_w = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu + ngb*nd + ngb*ncw + ngb*ncu + ngb*ncu*nc];
            dstype *fhb_uh = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu + ngb*nd + ngb*ncw + ngb*ncu + ngb*ncu*nc + ngb*ncu*ncw];
            dstype *wgb_uq = &tmp.tempg[n8 + ngb*ncx + ngb*nc + ngb*nco + ngb*ncw + ngb*ncu + ngb*nd + ngb*ncw + ngb*ncu + ngb*ncu*nc + ngb*ncu*ncw + ngb*ncu*ncu];

            GetBoundaryNodes(xgb, xg, &mesh.boufaces[start], ngf, nfe, ne, ncx, nfaces);
            GetBoundaryNodes(ugb, udg, &mesh.boufaces[start], ngf, nfe, ne, nc, nfaces);
            GetBoundaryNodes(ogb, odg, &mesh.boufaces[start], ngf, nfe, ne, nco, nfaces);
            GetBoundaryNodes(wgb, wdg, &mesh.boufaces[start], ngf, nfe, ne, ncw, nfaces);
            GetBoundaryNodes(wsb, wsrc, &mesh.boufaces[start], ngf, nfe, ne, ncw, nfaces);
            GetBoundaryNodes(uhb, uhg, &mesh.boufaces[start], ngf, nfe, ne, ncu, nfaces);
            GetBoundaryNodes(nlb, nlg, &mesh.boufaces[start], ngf, nfe, ne, nd, nfaces);

            if ((ncw > 0) && (common.wave == 0)) {
                ArrayCopy(res.F, ugb, ngb*nc);
                ArrayCopy(res.F, uhb, ngb*ncu);
                wEquation(wgb, wgb_uq, xgb, res.F, ogb, wsb, &res.F[ngb*nc],
                        app, driver_abi, common, ngb, backend);
            }

            ArraySetValue(fhb, 0.0, ngb*ncu);
            ArraySetValue(fhb_uq, 0.0, ngb*ncu*nc);
            ArraySetValue(fhb_uh, 0.0, ngb*ncu*ncu);
            if (ncw > 0) ArraySetValue(fhb_w, 0.0, ngb*ncu*ncw);

            FbouJacDriver(fhb, fhb_uq, fhb_w, fhb_uh, xgb, ugb, ogb,
                          wgb, uhb, nlb, driver_abi, mesh, master, app,
                          sol, tmp, common, ngb, ibc+1, backend);

            if ((ncw > 0) && (common.wave == 0)) {
                ArrayGemmBatch2(fhb_uh, fhb_w, wgb_uq, one, ncu, ncu, ncw, ngb);
                ArraySetValue(wgb_uq, 0.0, ngb*ncw*ncu);
                ArrayGemmBatch2(fhb_uq, fhb_w, wgb_uq, one, ncu, nc, ncw, ngb);
            }

            dstype *jacb = &tmp.tempg[n8];
            GetBoundaryNodes(jacb, jac, &mesh.boufaces[start], ngf, nfe, ne, 1, nfaces);
            columnwiseMultiply(fhb_uq, fhb_uq, jacb, ngb, ncu*nc);
            columnwiseMultiply(fhb_uh, fhb_uh, jacb, ngb, ncu*ncu);

            dstype *Rb = res.F;
            Gauss2Node(handle, Rb, fhb_uq, master.shapfgwdotshapfg,
                    ngf, npf*npf, nfaces*ncu*ncu, backend);
            PutBoundaryNodes(Dtmp, Rb, &mesh.boufaces[start],
                    npf*npf, nfe, ne, ncu*ncu, nfaces);

            if (ncq > 0) {
                Gauss2Node(handle, Rb, &fhb_uq[ngb*ncu*ncu],
                        master.shapfgwdotshapfg, ngf, npf*npf,
                        nfaces*ncu*ncq, backend);
                PutBoundaryNodes(Btmp, Rb, &mesh.boufaces[start],
                        npf*npf, nfe, ne, ncu*ncq, nfaces);
            }

            Gauss2Node(handle, Rb, fhb_uh, master.shapfgwdotshapfg,
                    ngf, npf*npf, nfaces*ncu*ncu, backend);
            PutBoundaryNodes(Ftmp, Rb, &mesh.boufaces[start],
                    npf*npf, nfe, ne, ncu*ncu, nfaces);
        }
    }    

    assembleMatrixBD(res.D, Dtmp, mesh.perm, npe, npf, nfe, ne*ncu*ncu);
    if (ncq > 0)
        assembleMatrixBD(res.B, Btmp, mesh.perm, npe, npf, nfe, ne*ncu*ncq);

    ArraySetValue(res.F, 0.0, npe*npf*nfe*ne*ncu*ncu);
    assembleMatrixF(res.F, Ftmp, mesh.perm, npe, npf, nfe, ne*ncu*ncu);
}

void uEquationSchurBlockLDG(solstruct &sol, resstruct &res, appstruct &app,
        ExasimDriverABI& driver_abi, masterstruct &master, meshstruct &mesh,
        tempstruct &tmp, commonstruct &common, cublasHandle_t handle,
        Int jth, Int backend, LDGSchurBenchmarkTimes* benchmark = nullptr)
{
    (void)sol;
    (void)app;
    (void)driver_abi;
    (void)master;
    (void)mesh;

    Int ncu = common.ncu;
    Int nd = common.nd;
    Int npe = common.npe;
    Int npf = common.npf;
    Int nfe = common.nfe;

    Int e1 = common.eblks[3*jth]-1;
    Int e2 = common.eblks[3*jth+1];
    Int ne = e2-e1;

    Int n = npe*ncu;
    Int m = npf*nfe*ncu;

    dstype *D = res.D;
    dstype *F = res.F;
    dstype *G = res.G;
    dstype *workC = res.H;

    double tTotal = 0.0;
    double t0 = 0.0;
    if (benchmark != nullptr)
        tTotal = LDGBenchmarkStart(backend);

    if (benchmark != nullptr)
        t0 = LDGBenchmarkStart(backend);
    schurMatrixD(res.H, res.D, npe, ncu, ne);
    ArrayCopy(D, res.H, n*n*ne);
    if (benchmark != nullptr)
        benchmark->layoutD += LDGBenchmarkStop(t0, backend);

    if (benchmark != nullptr)
        t0 = LDGBenchmarkStart(backend);
    LDGSchurMatrixF(res.H, F, npe, ncu, npf, nfe, ne);
    ArrayCopy(F, res.H, n*m*ne);
    if (benchmark != nullptr)
        benchmark->layoutF += LDGBenchmarkStop(t0, backend);

    dstype scalar = 1.0;
    if (common.wave == 1)
        scalar = 1.0/common.dtfactor;

    if (common.ncq > 0) {
        if (nd == 1) {
            if (benchmark != nullptr)
                t0 = LDGBenchmarkStart(backend);
            LDGSchurMatrixBMinvC_GEMM(handle, D, res.B,
                    &res.C[npe*npe*e1], workC, scalar, npe, ncu, ne,
                    backend);
            if (benchmark != nullptr)
                benchmark->bMinvC += LDGBenchmarkStop(t0, backend);

            if (benchmark != nullptr)
                t0 = LDGBenchmarkStart(backend);
            LDGSchurMatrixBMinvE_GEMM(handle, F, res.B,
                    &res.E[npe*npf*nfe*e1], workC, scalar, npe, ncu,
                    npf, nfe, ne, backend);
            if (benchmark != nullptr)
                benchmark->bMinvE += LDGBenchmarkStop(t0, backend);
        }
        else if (nd == 2) {
            dstype *Cx = &res.C[npe*npe*e1];
            dstype *Cy = &res.C[npe*npe*common.ne + npe*npe*e1];
            dstype *Ex = &res.E[npe*npf*nfe*e1];
            dstype *Ey = &res.E[npe*npf*nfe*common.ne + npe*npf*nfe*e1];
            dstype *Bx = res.B;
            dstype *By = &res.B[npe*npe*ncu*ncu*ne];

            if (benchmark != nullptr)
                t0 = LDGBenchmarkStart(backend);
            LDGSchurMatrixBMinvC_GEMM(handle, D, Bx, Cx, workC,
                    scalar, npe, ncu, ne, backend);
            LDGSchurMatrixBMinvC_GEMM(handle, D, By, Cy, workC,
                    scalar, npe, ncu, ne, backend);
            if (benchmark != nullptr)
                benchmark->bMinvC += LDGBenchmarkStop(t0, backend);

            if (benchmark != nullptr)
                t0 = LDGBenchmarkStart(backend);
            LDGSchurMatrixBMinvE_GEMM(handle, F, Bx, Ex, workC,
                    scalar, npe, ncu, npf, nfe, ne, backend);
            LDGSchurMatrixBMinvE_GEMM(handle, F, By, Ey, workC,
                    scalar, npe, ncu, npf, nfe, ne, backend);
            if (benchmark != nullptr)
                benchmark->bMinvE += LDGBenchmarkStop(t0, backend);
        }
        else if (nd == 3) {
            dstype *Cx = &res.C[npe*npe*e1];
            dstype *Cy = &res.C[npe*npe*common.ne + npe*npe*e1];
            dstype *Cz = &res.C[npe*npe*common.ne*2 + npe*npe*e1];
            dstype *Ex = &res.E[npe*npf*nfe*e1];
            dstype *Ey = &res.E[npe*npf*nfe*common.ne + npe*npf*nfe*e1];
            dstype *Ez = &res.E[npe*npf*nfe*common.ne*2 + npe*npf*nfe*e1];
            dstype *Bx = res.B;
            dstype *By = &res.B[npe*npe*ncu*ncu*ne];
            dstype *Bz = &res.B[npe*npe*ncu*ncu*ne*2];

            if (benchmark != nullptr)
                t0 = LDGBenchmarkStart(backend);
            LDGSchurMatrixBMinvC_GEMM(handle, D, Bx, Cx, workC,
                    scalar, npe, ncu, ne, backend);
            LDGSchurMatrixBMinvC_GEMM(handle, D, By, Cy, workC,
                    scalar, npe, ncu, ne, backend);
            LDGSchurMatrixBMinvC_GEMM(handle, D, Bz, Cz, workC,
                    scalar, npe, ncu, ne, backend);
            if (benchmark != nullptr)
                benchmark->bMinvC += LDGBenchmarkStop(t0, backend);

            if (benchmark != nullptr)
                t0 = LDGBenchmarkStart(backend);
            LDGSchurMatrixBMinvE_GEMM(handle, F, Bx, Ex, workC,
                    scalar, npe, ncu, npf, nfe, ne, backend);
            LDGSchurMatrixBMinvE_GEMM(handle, F, By, Ey, workC,
                    scalar, npe, ncu, npf, nfe, ne, backend);
            LDGSchurMatrixBMinvE_GEMM(handle, F, Bz, Ez, workC,
                    scalar, npe, ncu, npf, nfe, ne, backend);
            if (benchmark != nullptr)
                benchmark->bMinvE += LDGBenchmarkStop(t0, backend);
        }
    }

    if (benchmark != nullptr)
        t0 = LDGBenchmarkStart(backend);
    PGEMNMStridedBached(handle, n, n, m, one, F, n, G, m, one, D, n, ne, backend);
    if (benchmark != nullptr) {
        benchmark->fg += LDGBenchmarkStop(t0, backend);
        benchmark->total += LDGBenchmarkStop(tTotal, backend);
    }
}

void RuFaceCrossDeriv(dstype* A, solstruct &sol,
        resstruct &res, appstruct &app, ExasimDriverABI& driver_abi,
        masterstruct &master, meshstruct &mesh, tempstruct &tmp,
        commonstruct &common)
{
    if (common.ncq <= 0)
        return;

    Int backend = common.backend;
    Int npe = common.npe;
    Int npf = common.npf;
    Int ngf = common.ngf;
    Int nfe = common.nfe;
    Int ncu = common.ncu;
    Int ncq = common.ncq;
    Int nc = common.nc;
    Int nco = common.nco;
    Int ncw = common.ncw;
    Int ncx = common.ncx;
    Int nd = common.nd;
    Int ne = common.ne1;
    LDGRuFaceCrossBenchmarkTimes tm;
    double tTotal = LDGBenchmarkStart(backend);
    double t0;

    //cout<<common.mpiRank<<", "<<ne<<", "<<common.ne<<endl;

    dstype scalar = 1.0;
    if (common.wave == 1)
        scalar = 1.0/common.dtfactor;
    
    //print2iarray(common.fblks, 3, common.nbf, "fblks", EXASIM_COMM_WORLD);

    //if (common.mpiRank==0) print3darray(A, npe, npe, ne);

    for (Int jblk = 0; jblk < common.nbf; jblk++) {
        Int f1 = common.fblks[3*jblk]-1;
        Int f2 = common.fblks[3*jblk+1];
        Int ib = common.fblks[3*jblk+2];
        if (ib != 0)
            continue;

        Int nfb = f2 - f1;
        Int nn = npf*nfb;
        Int nga = ngf*nfb;
        Int M = nga*ncu;
        Int N = nga*ncu*nd;
        Int nm = ngf*f1*(ncx+nd+1);

        Int fncols = ncu + 2*nc + 2*ncw;
        dstype *fn = tmp.tempn;
        dstype *fg = tmp.tempg;
        Int fluxSize = N;
        Int fluxWSize = max((Int) 1, fluxSize*ncw);
        dstype *flux = &fg[nga*fncols];
        dstype *flux_udg = &flux[fluxSize];
        dstype *fw = &flux_udg[fluxSize*nc];
        dstype *fq = tmp.tempn;
        dstype *bufq = &tmp.tempn[M*ncq];

        t0 = LDGBenchmarkStart(backend);
        GetElemNodes(fn, sol.uh, npf, ncu, 0, ncu, f1, f2);
        GetArrayAtIndex(&fn[nn*ncu], sol.udg, &mesh.findudg1[npf*nc*f1], nn*nc);
        if (ncw > 0)
            GetFaceNodes(&fn[nn*(ncu+nc)], sol.wdg, mesh.facecon, npf, ncw, npe, ncw, f1, f2, 1);
        GetArrayAtIndex(&fn[nn*(ncu+nc+ncw)], sol.udg, &mesh.findudg2[npf*nc*f1], nn*nc);
        if (ncw > 0)
            GetFaceNodes(&fn[nn*(ncu+2*nc+ncw)], sol.wdg, mesh.facecon, npf, ncw, npe, ncw, f1, f2, 2);

        Node2Gauss(common.cublasHandle, fg, fn, master.shapfgt, ngf, npf, nfb*fncols, backend);
        tm.prep += LDGBenchmarkStop(t0, backend);

        dstype *ug1 = &fg[nga*ncu];
        dstype *wg1 = (ncw > 0) ? &fg[nga*(ncu+nc)] : nullptr;
        dstype *ug2 = &fg[nga*(ncu+nc+ncw)];
        dstype *wg2 = (ncw > 0) ? &fg[nga*(ncu+2*nc+ncw)] : nullptr;
        dstype *xg = &sol.faceg[nm];
        dstype *nlg = &sol.faceg[nm + nga*ncx];
        dstype *jac = &sol.faceg[nm + nga*(ncx+nd)];
        dstype *og1 = (nco > 0) ? &sol.og1[ngf*nco*f1] : nullptr;
        dstype *og2 = (nco > 0) ? &sol.og2[ngf*nco*f1] : nullptr;

        //print3darray(xg, ngf, nfb, ncx, "xg", EXASIM_COMM_WORLD);
        //print3darray(xg, ngf, nfb, ncx);

        t0 = LDGBenchmarkStart(backend);
        ArraySetValue(fw, 0.0, fluxWSize);
        FluxDriver(flux, flux_udg, fw, xg, ug1, og1, wg1, driver_abi, mesh, 
                   master, app, sol, tmp, common, ngf, f1, f2, backend);
        tm.flux_m += LDGBenchmarkStop(t0, backend);

        t0 = LDGBenchmarkStart(backend);
        LDGFluxQDerivativeDotNormal(fq, flux_udg, nlg, 0.5, nga, ncu, nd, ncq, nc);
        columnwiseMultiply(fq, fq, jac, nga, ncu*ncq);
        Gauss2Node(common.cublasHandle, bufq, fq, master.shapfgwdotshapfg, 
                   ngf, npf*npf, nfb*ncu*ncq, backend);
        tm.projection_m += LDGBenchmarkStop(t0, backend);

        t0 = LDGBenchmarkStart(backend);
        LDGAssembleFaceQToElementCrossBlock(A, bufq, res.E,
                mesh.facecon, mesh.f2e, mesh.elemcon, scalar, 2, 1,
                minusone, f1, nfb, npe, npf, nfe, ncu, ncq, ne, common.ne, common.mpiRank);
        tm.assemble_m += LDGBenchmarkStop(t0, backend);

        //print3darray(bufq, npf*npf, nfb, ncu*ncq, "Rf_qm", EXASIM_COMM_WORLD);
        //print3darray(bufq, npf*npf, nfb, ncu*ncq);
        // {
        //     Int szEf = npf*nd*npf*nfb;
        //     dstype *EfRef = &tmp.tempn[M*ncq + npf*npf*nfb*ncu*ncq];
        //     dstype *EfOpt = &EfRef[szEf];
        //     dstype *EfDiff = &EfOpt[szEf];
        //     LDGValidateBuildFaceEForCrossBlock(EfRef, EfOpt, EfDiff, res.E,
        //                                        mesh.facecon, mesh.f2e, mesh.elemcon,
        //                                        mesh.perm, res.ipiv, 1,
        //                                        f1, nfb, npe, npf, nfe, nd, common.ne,
        //                                        common.cublasHandle, backend, common.mpiRank);
        // }
         
        t0 = LDGBenchmarkStart(backend);
        ArraySetValue(fw, 0.0, fluxWSize);
        FluxDriver(flux, flux_udg, fw, xg, ug2, og2, wg2, driver_abi, mesh, 
                   master, app, sol, tmp, common, ngf, f1, f2, backend);
        tm.flux_p += LDGBenchmarkStop(t0, backend);

        t0 = LDGBenchmarkStart(backend);
        LDGFluxQDerivativeDotNormal(fq, flux_udg, nlg, 0.5, nga, ncu, nd, ncq, nc);
        columnwiseMultiply(fq, fq, jac, nga, ncu*ncq);
        Gauss2Node(common.cublasHandle, bufq, fq, master.shapfgwdotshapfg, 
                   ngf, npf*npf, nfb*ncu*ncq, backend);
        tm.projection_p += LDGBenchmarkStop(t0, backend);

        t0 = LDGBenchmarkStart(backend);
        LDGAssembleFaceQToElementCrossBlock(A, bufq, res.E,
                mesh.facecon, mesh.f2e, mesh.elemcon, scalar, 1, 2, one,
                f1, nfb, npe, npf, nfe, ncu, ncq, ne, common.ne, common.mpiRank);
        tm.assemble_p += LDGBenchmarkStop(t0, backend);

        //print3darray(bufq, npf*npf, nfb, ncu*ncq, "Rf_qp", EXASIM_COMM_WORLD);
        //print3darray(bufq, npf*npf, nfb, ncu*ncq);

        // {
        //     Int szEf = npf*nd*npf*nfb;
        //     dstype *EfRef = &tmp.tempn[M*ncq + npf*npf*nfb*ncu*ncq];
        //     dstype *EfOpt = &EfRef[szEf];
        //     dstype *EfDiff = &EfOpt[szEf];
        //     LDGValidateBuildFaceEForCrossBlock(EfRef, EfOpt, EfDiff, res.E,
        //             mesh.facecon, mesh.f2e, mesh.elemcon, mesh.perm, res.ipiv, 2,
        //             f1, nfb, npe, npf, nfe, nd, common.ne,
        //             common.cublasHandle, backend, common.mpiRank);
        // }
    }

    tm.total = LDGBenchmarkStop(tTotal, backend);
    //LDGPrintRuFaceCrossBenchmark(tm, common);

    // if (common.mpiRank==0) {
    //     cout<<"+++++++++++++++++++++\n";
    //     print3darray(A, npe, npe, ne);
    // }
}

void RuFaceCrossDerivOptimized(dstype* A, solstruct &sol,
        resstruct &res, appstruct &app, ExasimDriverABI& driver_abi,
        masterstruct &master, meshstruct &mesh, tempstruct &tmp,
        commonstruct &common)
{
    if (common.ncq <= 0)
        return;

    Int backend = common.backend;
    Int npe = common.npe;
    Int npf = common.npf;
    Int ngf = common.ngf;
    Int ncu = common.ncu;
    //Int ncq = common.ncq;
    Int nc = common.nc;
    Int nco = common.nco;
    Int ncw = common.ncw;
    Int ncx = common.ncx;
    Int nd = common.nd;
    Int ne = common.ne1;

    dstype scalar = 1.0;
    if (common.wave == 1)
        scalar = 1.0/common.dtfactor;

    for (Int jblk = 0; jblk < common.nbf; jblk++) {
        Int f1 = common.fblks[3*jblk]-1;
        Int f2 = common.fblks[3*jblk+1];
        Int ib = common.fblks[3*jblk+2];
        if (ib != 0)
            continue;

        Int nfb = f2 - f1;
        Int nn = npf*nfb;
        Int nga = ngf*nfb;
        Int M = nga*ncu;
        Int N = nga*ncu*nd;
        Int nm = ngf*f1*(ncx+nd+1);

        Int fncols = ncu + 2*nc + 2*ncw;
        dstype *fn = tmp.tempn;
        dstype *fg = tmp.tempg;
        Int fluxSize = N;
        Int fluxWSize = max((Int) 1, fluxSize*ncw);
        dstype *flux = &fg[nga*fncols];
        dstype *flux_udg = &flux[fluxSize];
        dstype *fw = &flux_udg[fluxSize*nc];

        Int szFq = ngf*nd*ncu*ncu*nfb;
        Int szBufq = npf*npf*nd*ncu*ncu*nfb;
        Int szEf = npf*nd*npf*nfb;
        //Int szAf = npf*ncu*ncu*npf*nfb;

        dstype *fq = res.H;
        dstype *B = fq;
        dstype *bufq = &res.H[max(szFq, szBufq)];
        dstype *Ef = bufq;
        dstype *Af = &bufq[max(szBufq, szEf)];
        //(void)szAf;

        GetElemNodes(fn, sol.uh, npf, ncu, 0, ncu, f1, f2);
        GetArrayAtIndex(&fn[nn*ncu], sol.udg, &mesh.findudg1[npf*nc*f1], nn*nc);
        if (ncw > 0)
            GetFaceNodes(&fn[nn*(ncu+nc)], sol.wdg, mesh.facecon,
                    npf, ncw, npe, ncw, f1, f2, 1);
        GetArrayAtIndex(&fn[nn*(ncu+nc+ncw)], sol.udg,
                &mesh.findudg2[npf*nc*f1], nn*nc);
        if (ncw > 0)
            GetFaceNodes(&fn[nn*(ncu+2*nc+ncw)], sol.wdg, mesh.facecon,
                    npf, ncw, npe, ncw, f1, f2, 2);

        Node2Gauss(common.cublasHandle, fg, fn, master.shapfgt,
                ngf, npf, nfb*fncols, backend);

        dstype *ug1 = &fg[nga*ncu];
        dstype *wg1 = (ncw > 0) ? &fg[nga*(ncu+nc)] : nullptr;
        dstype *ug2 = &fg[nga*(ncu+nc+ncw)];
        dstype *wg2 = (ncw > 0) ? &fg[nga*(ncu+2*nc+ncw)] : nullptr;
        dstype *xg = &sol.faceg[nm];
        dstype *nlg = &sol.faceg[nm + nga*ncx];
        dstype *jac = &sol.faceg[nm + nga*(ncx+nd)];
        dstype *og1 = (nco > 0) ? &sol.og1[ngf*nco*f1] : nullptr;
        dstype *og2 = (nco > 0) ? &sol.og2[ngf*nco*f1] : nullptr;

        ArraySetValue(fw, 0.0, fluxWSize);
        FluxDriver(flux, flux_udg, fw, xg, ug1, og1, wg1, driver_abi,
                   mesh, master, app, sol, tmp, common, ngf, f1, f2, backend);
        LDGFluxQDerivativeDotNormalJac(fq, flux_udg, nlg, jac, 0.5,
                ngf, nfb, ncu, nd, nc);
        Gauss2Node(common.cublasHandle, bufq, fq, master.shapfgwdotshapfg,
                ngf, npf*npf, nfb*nd*ncu*ncu, backend);
        LDGPackFaceQForCrossGEMM(B, bufq, npf, ncu, nd, nfb);
        LDGBuildFaceSlotQMap(res.ipiv, mesh.f2e, mesh.elemcon, 1,
                f1, nfb, npf, common.nfe);
        LDGBuildFaceEForCrossBlockOptimized(Ef, res.E, mesh.f2e,
                mesh.perm, res.ipiv, 1, f1, nfb, npe, npf, common.nfe,
                nd, common.ne);
        ArrayMultiplyScalar(common.cublasHandle, Ef, 0.5*scalar,
                szEf, backend);
        PGEMNMStridedBached(common.cublasHandle, npf*ncu*ncu, npf,
                npf*nd, one, B, npf*ncu*ncu, Ef, npf*nd, 0.0,
                Af, npf*ncu*ncu, nfb, backend);
        LDGScatterCrossFaceGEMMBlock(A, Af, mesh.facecon, mesh.f2e,
                2, f1, nfb, npe, npf, ncu, ne);

        ArraySetValue(fw, 0.0, fluxWSize);
        FluxDriver(flux, flux_udg, fw, xg, ug2, og2, wg2, driver_abi,
                   mesh, master, app, sol, tmp, common, ngf, f1, f2, backend);
        LDGFluxQDerivativeDotNormalJac(fq, flux_udg, nlg, jac, 0.5,
                ngf, nfb, ncu, nd, nc);
        Gauss2Node(common.cublasHandle, bufq, fq, master.shapfgwdotshapfg,
                ngf, npf*npf, nfb*nd*ncu*ncu, backend);
        LDGPackFaceQForCrossGEMM(B, bufq, npf, ncu, nd, nfb);
        LDGBuildFaceSlotQMap(res.ipiv, mesh.f2e, mesh.elemcon, 2,
                f1, nfb, npf, common.nfe);
        LDGBuildFaceEForCrossBlockOptimized(Ef, res.E, mesh.f2e,
                mesh.perm, res.ipiv, 2, f1, nfb, npe, npf, common.nfe,
                nd, common.ne);
        ArrayMultiplyScalar(common.cublasHandle, Ef, 0.5*scalar*minusone,
                szEf, backend);
        PGEMNMStridedBached(common.cublasHandle, npf*ncu*ncu, npf,
                npf*nd, one, B, npf*ncu*ncu, Ef, npf*nd, 0.0,
                Af, npf*ncu*ncu, nfb, backend);
        LDGScatterCrossFaceGEMMBlock(A, Af, mesh.facecon, mesh.f2e,
                1, f1, nfb, npe, npf, ncu, ne);

        //(void)ncq;
        //(void)szAf;
    }
}

#ifdef DEBUG
static void LDGDebugCompareRuFaceCrossDeriv(dstype* K, solstruct &sol,
        resstruct &res, appstruct &app, ExasimDriverABI& driver_abi,
        masterstruct &master, meshstruct &mesh, tempstruct &tmp,
        commonstruct &common)
{
    if (common.ncq <= 0)
        return;

    Int nlocu = common.npe*common.ncu;
    Int szA = nlocu*nlocu*common.ne1;
    Int backend = common.backend;
    dstype *Aref = nullptr, *Aopt = nullptr;
    TemplateMalloc(&Aref, szA, backend);
    TemplateMalloc(&Aopt, szA, backend);

    ArrayCopy(common.cublasHandle, Aref, K, szA, backend);
    ArrayCopy(common.cublasHandle, Aopt, K, szA, backend);

    RuFaceCrossDeriv(Aref, sol, res, app, driver_abi, master, mesh, tmp, common);
    RuFaceCrossDerivOptimized(Aopt, sol, res, app, driver_abi, master, mesh, tmp, common);

    dstype *href = (dstype*) malloc(sizeof(dstype)*szA);
    dstype *hopt = (dstype*) malloc(sizeof(dstype)*szA);
    TemplateCopytoHost(href, Aref, szA, backend);
    TemplateCopytoHost(hopt, Aopt, szA, backend);

#ifdef USE_FLOAT
    const dstype atol = 1.0e-5f;
    const dstype rtol = 1.0e-5f;
#else
    const dstype atol = 1.0e-10;
    const dstype rtol = 1.0e-10;
#endif
    for (Int idx = 0; idx < szA; idx++) {
        dstype ref = href[idx];
        dstype opt = hopt[idx];
        dstype diff = opt - ref;
        if (diff < 0.0) diff = -diff;
        dstype scale = ref;
        if (scale < 0.0) scale = -scale;

        if (diff > atol + rtol*scale) {
            Int row = idx % nlocu;
            Int t = idx / nlocu;
            Int col = t % nlocu;
            Int e = t / nlocu;
            Int rowNode = row % common.npe;
            Int rowComp = row / common.npe;
            Int colNode = col % common.npe;
            Int colComp = col / common.npe;

            if (common.mpiRank == 0) {
                std::cout << "RuFaceCrossDerivOptimized mismatch"
                          << ": element = " << e
                          << ", local row = " << row
                          << " (node " << rowNode << ", comp " << rowComp << ")"
                          << ", local col = " << col
                          << " (node " << colNode << ", comp " << colComp << ")"
                          << ", global row block = " << e*nlocu + row
                          << ", global col block = " << e*nlocu + col
                          << ", reference = " << ref
                          << ", optimized = " << opt
                          << ", abs diff = " << diff
                          << std::endl;
            }

            CPUFREE(href);
            CPUFREE(hopt);
            TemplateFree(Aref, backend);
            TemplateFree(Aopt, backend);
            error("RuFaceCrossDerivOptimized DEBUG comparison failed.");
        }
    }

    if (common.mpiRank == 0)
        std::cout << "RuFaceCrossDerivOptimized DEBUG comparison passed for "
                  << szA << " entries." << std::endl;

    CPUFREE(href);
    CPUFREE(hopt);
    TemplateFree(Aref, backend);
    TemplateFree(Aopt, backend);
}
#endif

void BlockJacobianLDG(dstype* K, dstype* u, solstruct &sol, resstruct &res, appstruct &app,
                  ExasimDriverABI& driver_abi, masterstruct &master, meshstruct &mesh,
                  tempstruct &tmp, commonstruct &common, cublasHandle_t handle, Int backend)
{    
    LDGBenchmarkTimes tm;
    double tTotal = LDGBenchmarkStart(backend);
    double t0;

    // compareRuFaceDerivFromU(u, sol, res, app, driver_abi, master, mesh, tmp, common, handle, backend);
    // error("here");

    // insert u into udg
    t0 = LDGBenchmarkStart(backend);
    ArrayInsert(sol.udg, u, common.npe, common.nc, common.ne, 0, common.npe, 
                0, common.ncu, 0, common.ne);  
    tm.insert += LDGBenchmarkStop(t0, backend);

    // compute uhat
    t0 = LDGBenchmarkStart(backend);
    GetUhat(sol, res, app, driver_abi, master, mesh, tmp, common, handle, 0, common.nbf, backend);
    tm.uhat += LDGBenchmarkStop(t0, backend);

    // compute q
    if (common.ncq>0) {
        t0 = LDGBenchmarkStart(backend);
        GetQ(sol, res, app, driver_abi, master, mesh, tmp, common, handle, 0, common.nbe, 0, common.nbf, backend);                
        tm.q += LDGBenchmarkStop(t0, backend);
    }

    // compute w
    if (common.ncw>0) {
        t0 = LDGBenchmarkStart(backend);
        GetW(sol, res, app, driver_abi, master, mesh, tmp, common, handle, 0, common.nbe, 0, common.nbf, backend);                
        tm.w += LDGBenchmarkStop(t0, backend);
    }

    if (common.ncAV>0 && common.frozenAVflag == 0) {
        t0 = LDGBenchmarkStart(backend);
        GetAv(sol, res, app, driver_abi, master, mesh, tmp, common, handle, backend);
        tm.av += LDGBenchmarkStop(t0, backend);
    }

    Int n = common.npe*common.ncu;
    Int m = common.npf*common.nfe*common.ncu;
    for (Int j = 0; j < common.nbe; j++) {
        Int e1 = common.eblks[3*j]-1;
        Int e2 = common.eblks[3*j+1];
        Int ne = e2-e1;        

        t0 = LDGBenchmarkStart(backend);
        uEquationElemBlock(sol, res, app, driver_abi, master, mesh, tmp,
                common, handle, j, backend);
        tm.elem += LDGBenchmarkStop(t0, backend);

        t0 = LDGBenchmarkStart(backend);
        uEquationElemFaceBlockLDG(sol, res, app, driver_abi, master, mesh,
                tmp, common, handle, j, backend);
        tm.face += LDGBenchmarkStop(t0, backend);
        //ArraySetValue(res.F, 0.0, n*m*ne);

        t0 = LDGBenchmarkStart(backend);
        uhatEquationElemFaceBlockLDG(sol, res, app, driver_abi, master, mesh,
                tmp, common, handle, j, backend);
        tm.trace += LDGBenchmarkStop(t0, backend);

        t0 = LDGBenchmarkStart(backend);
        uEquationSchurBlockLDG(sol, res, app, driver_abi, master, mesh, tmp,
                common, handle, j, backend, &tm.schurDetail);
        tm.schur += LDGBenchmarkStop(t0, backend);
        
        t0 = LDGBenchmarkStart(backend);
        ArrayCopy(&K[n*n*e1], res.D, n*n*ne);                
        tm.copy += LDGBenchmarkStop(t0, backend);
    }

    t0 = LDGBenchmarkStart(backend);
#ifdef DEBUG
    LDGDebugCompareRuFaceCrossDeriv(K, sol, res, app, driver_abi, master, mesh, tmp, common);
#endif
    RuFaceCrossDerivOptimized(K, sol, res, app, driver_abi, master, mesh, tmp, common);
    //RuFaceCrossDeriv(K, sol, res, app, driver_abi, master, mesh, tmp, common);
    tm.cross += LDGBenchmarkStop(t0, backend);

    // if (common.tdep == 1)
    //     ArrayMultiplyScalar(handle, K, minusone/common.dtfactor, n*n*common.ne1, backend);
    
    // print3darray(sol.xdg, common.npe, common.ncx, common.ne1);
    // print3darray(K, n, n, common.ne1);

    // VerifyRuDerivFromUFiniteDifference(K, u, sol, res, app, driver_abi, master, mesh, tmp, common, 1e-6);
    // error("here");

    //common.debugMode = 1;
    // if (common.debugMode == 1) {
    //     Int szA = n*n*common.ne1;
    //     dstype *A = nullptr;
    //     TemplateMalloc(&A, szA, backend);
    // 
    //     //RuElemDeriv(A, u, sol, res, app, driver_abi, master, mesh, tmp, common);
    //     RuDerivFromU(A, u, sol, res, app, driver_abi, master, mesh, tmp, common);
    // 
    //     for (int i=0; i <common.ne1; i++) {
    //         dstype *diff = tmp.tempn;
    //         dstype residualScale = one;
    //         ArrayAXPBY(diff, &K[n*n*i], &A[n*n*i], one, -residualScale, n*n);
    //         dstype normA = residualScale*NORM(handle, n*n, &A[n*n*i], backend);
    //         dstype errMinus = NORM(handle, n*n, diff, backend);
    // 
    //         ArrayAXPBY(diff, &K[n*n*i], &A[n*n*i], one, residualScale, n*n);
    //         dstype errPlus = NORM(handle, n*n, diff, backend);
    // 
    //         cout << "Rank " << common.mpiRank << ", element " << i
    //              << ": BlockJacobianLDG comparison before inversion: "
    //              << "||K-sA|| = " << scientific << errMinus
    //              << ", rel = " << errMinus/(normA + 1.0e-14)
    //              << ", ||K+sA|| = " << errPlus
    //              << ", rel = " << errPlus/(normA + 1.0e-14)
    //              << endl;
    //     }
    // 
    //     if (A != nullptr) TemplateFree(A, backend);
    // 
    //     error("here");
    // }

    for (Int j = 0; j < common.nbe; j++) {
        Int e1 = common.eblks[3*j]-1;
        Int e2 = common.eblks[3*j+1];
        Int ne = e2-e1;
        t0 = LDGBenchmarkStart(backend);
        Inverse(handle, &K[n*n*e1], res.H, res.ipiv, n, ne, backend);
        tm.inverse += LDGBenchmarkStop(t0, backend);
    }

    tm.total = LDGBenchmarkStop(tTotal, backend);
    //LDGPrintBenchmark("serial", tm, common);
}

void mpiBlockJacobianLDG(dstype* K, dstype* u, solstruct &sol, resstruct &res, appstruct &app,
                  ExasimDriverABI& driver_abi, masterstruct &master, meshstruct &mesh,
                  tempstruct &tmp, commonstruct &common, cublasHandle_t handle, Int backend)
{
#ifdef HAVE_MPI
    LDGBenchmarkTimes tm;
    double tTotal = LDGBenchmarkStart(backend);
    double t0;

    Int bsz = common.npe*common.ncu;
    Int n;

    // Insert owned primal unknowns. Neighbor/exterior primal values are received below.
    t0 = LDGBenchmarkStart(backend);
    ArrayInsert(sol.udg, u, common.npe, common.nc, common.ne, 0, common.npe,
                0, common.ncu, 0, common.ne1);
    tm.insert += LDGBenchmarkStop(t0, backend);

    // Non-blocking exchange of owned primal unknowns, matching RuResidualMPI().
    t0 = LDGBenchmarkStart(backend);
    GetArrayAtIndex(tmp.buffsend, sol.udg, mesh.elemsendind, bsz*common.nelemsend);

#ifdef HAVE_CUDA
    cudaDeviceSynchronize();
#endif

#ifdef HAVE_HIP
    hipDeviceSynchronize();
#endif

    Int neighbor, nsend, psend = 0, request_counter = 0;
    for (n=0; n<common.nnbsd; n++) {
        neighbor = common.nbsd[n];
        nsend = common.elemsendpts[n]*bsz;
        if (nsend>0) {
            MPI_Isend(&tmp.buffsend[psend], nsend, MPI_DOUBLE, neighbor, 0,
                   EXASIM_COMM_LOCAL, &common.requests[request_counter]);
            psend += nsend;
            request_counter += 1;
        }
    }

    Int nrecv, precv = 0;
    for (n=0; n<common.nnbsd; n++) {
        neighbor = common.nbsd[n];
        nrecv = common.elemrecvpts[n]*bsz;
        if (nrecv>0) {
            MPI_Irecv(&tmp.buffrecv[precv], nrecv, MPI_DOUBLE, neighbor, 0,
                   EXASIM_COMM_LOCAL, &common.requests[request_counter]);
            precv += nrecv;
            request_counter += 1;
        }
    }
    tm.communication += LDGBenchmarkStop(t0, backend);

    // Interior q and w can be computed while neighbor values are in flight.
    t0 = LDGBenchmarkStart(backend);
    GetUhat(sol, res, app, driver_abi, master, mesh, tmp, common, handle, 0, common.nbf, backend);
    tm.uhat += LDGBenchmarkStop(t0, backend);

    if (common.ncq>0) {
        t0 = LDGBenchmarkStart(backend);
        GetQ(sol, res, app, driver_abi, master, mesh, tmp, common, handle,
             0, common.nbe0, 0, common.nbf, backend);
        tm.q += LDGBenchmarkStop(t0, backend);
    }

    if (common.ncw>0) {
        t0 = LDGBenchmarkStart(backend);
        GetW(sol, res, app, driver_abi, master, mesh, tmp, common, handle,
             0, common.nbe0, 0, common.nbf, backend);
        tm.w += LDGBenchmarkStop(t0, backend);
    }

    t0 = LDGBenchmarkStart(backend);
    MPI_Waitall(request_counter, common.requests, common.statuses);
    PutArrayAtIndex(sol.udg, tmp.buffrecv, mesh.elemrecvind, bsz*common.nelemrecv);
    tm.communication += LDGBenchmarkStop(t0, backend);

    // Recompute traces with up-to-date exterior states, then update interface/exterior q and w.
    t0 = LDGBenchmarkStart(backend);
    GetUhat(sol, res, app, driver_abi, master, mesh, tmp, common, handle, 0, common.nbf, backend);
    tm.uhat += LDGBenchmarkStop(t0, backend);

    if (common.ncq>0) {
        t0 = LDGBenchmarkStart(backend);
        GetQ(sol, res, app, driver_abi, master, mesh, tmp, common, handle,
             common.nbe0, common.nbe2, 0, common.nbf, backend);
        tm.q += LDGBenchmarkStop(t0, backend);
    }

    if (common.ncw>0) {
        t0 = LDGBenchmarkStart(backend);
        GetW(sol, res, app, driver_abi, master, mesh, tmp, common, handle,
             common.nbe0, common.nbe2, 0, common.nbf, backend);
        tm.w += LDGBenchmarkStop(t0, backend);
    }

    if (common.ncAV>0 && common.frozenAVflag == 0) {
        t0 = LDGBenchmarkStart(backend);
        GetAv(sol, res, app, driver_abi, master, mesh, tmp, common, handle, backend);
        tm.av += LDGBenchmarkStop(t0, backend);
    }

    Int nlocu = common.npe*common.ncu;
    for (Int j = 0; j < common.nbe1; j++) {
        Int e1 = common.eblks[3*j]-1;
        Int e2 = common.eblks[3*j+1];
        Int ne = e2-e1;

        t0 = LDGBenchmarkStart(backend);
        uEquationElemBlock(sol, res, app, driver_abi, master, mesh, tmp,
                common, handle, j, backend);
        tm.elem += LDGBenchmarkStop(t0, backend);

        t0 = LDGBenchmarkStart(backend);
        uEquationElemFaceBlockLDG(sol, res, app, driver_abi, master, mesh,
                tmp, common, handle, j, backend);
        tm.face += LDGBenchmarkStop(t0, backend);

        t0 = LDGBenchmarkStart(backend);
        uhatEquationElemFaceBlockLDG(sol, res, app, driver_abi, master, mesh,
                tmp, common, handle, j, backend);
        tm.trace += LDGBenchmarkStop(t0, backend);

        t0 = LDGBenchmarkStart(backend);
        uEquationSchurBlockLDG(sol, res, app, driver_abi, master, mesh, tmp,
                common, handle, j, backend, &tm.schurDetail);
        tm.schur += LDGBenchmarkStop(t0, backend);

        t0 = LDGBenchmarkStart(backend);
        ArrayCopy(&K[nlocu*nlocu*e1], res.D, nlocu*nlocu*ne);
        tm.copy += LDGBenchmarkStop(t0, backend);
    }

    t0 = LDGBenchmarkStart(backend);
#ifdef DEBUG
    LDGDebugCompareRuFaceCrossDeriv(K, sol, res, app, driver_abi, master, mesh, tmp, common);
#endif
    RuFaceCrossDerivOptimized(K, sol, res, app, driver_abi, master, mesh, tmp, common);
    tm.cross += LDGBenchmarkStop(t0, backend);

    // if (common.tdep == 1)
    //     ArrayMultiplyScalar(handle, K, one/common.dtfactor, nlocu*nlocu*common.ne1, backend);

    for (Int j = 0; j < common.nbe1; j++) {
        Int e1 = common.eblks[3*j]-1;
        Int e2 = common.eblks[3*j+1];
        Int ne = e2-e1;
        t0 = LDGBenchmarkStart(backend);
        Inverse(handle, &K[nlocu*nlocu*e1], res.H, res.ipiv, nlocu, ne, backend);
        tm.inverse += LDGBenchmarkStop(t0, backend);
    }

    tm.total = LDGBenchmarkStop(tTotal, backend);
    LDGPrintBenchmark("mpi", tm, common);
#endif
}


// void compareRuFaceDerivFromU(dstype* u, solstruct &sol, resstruct &res, appstruct &app,
//                              ExasimDriverABI& driver_abi, masterstruct &master, meshstruct &mesh,
//                              tempstruct &tmp, commonstruct &common, cublasHandle_t handle, Int backend)
// {    
//     // insert u into udg
//     ArrayInsert(sol.udg, u, common.npe, common.nc, common.ne, 0, common.npe, 
//                 0, common.ncu, 0, common.ne);  
// 
//     // compute uhat
//     GetUhat(sol, res, app, driver_abi, master, mesh, tmp, common, handle, 0, common.nbf, backend);
// 
//     // compute q
//     if (common.ncq>0)
//         GetQ(sol, res, app, driver_abi, master, mesh, tmp, common, handle, 0, common.nbe, 0, common.nbf, backend);                
// 
//     // compute w
//     if (common.ncw>0)
//         GetW(sol, res, app, driver_abi, master, mesh, tmp, common, handle, 0, common.nbe, 0, common.nbf, backend);                
// 
//     if (common.ncAV>0 && common.frozenAVflag == 0)
//         GetAv(sol, res, app, driver_abi, master, mesh, tmp, common, handle, backend);
// 
//     Int n = common.npe*common.ncu;
//     Int szA = n*n*common.ne1;
//     dstype *K = nullptr;
//     TemplateMalloc(&K, szA, backend);
// 
//     for (Int j = 0; j < common.nbe; j++) {
//         Int e1 = common.eblks[3*j]-1;
//         Int e2 = common.eblks[3*j+1];
//         Int ne = e2-e1;        
// 
//         ArraySetValue(res.D, 0.0, n*n*ne);
//         ArraySetValue(res.B, 0.0, n*common.npe*common.ncq*ne);
//         uEquationElemFaceBlockLDG(sol, res, app, driver_abi, master, mesh,
//                                   tmp, common, handle, j, backend);
// 
//         uhatEquationElemFaceBlockLDG(sol, res, app, driver_abi, master, mesh,
//                                      tmp, common, handle, j, backend);
// 
//         uEquationSchurBlockLDG(sol, res, app, driver_abi, master, mesh, tmp,
//                                common, handle, j, backend);
// 
//         ArrayCopy(&K[n*n*e1], res.D, n*n*ne);                
//     }
// 
//     CrossFaceDeriv(K, sol, res, app, driver_abi, master, mesh, tmp, common);
// 
//     dstype *A = nullptr;
//     TemplateMalloc(&A, szA, backend);    
//     RuFaceDerivFromU(A, u, sol, res, app, driver_abi, master, mesh, tmp, common);
// 
//     for (int i=0; i <common.ne1; i++) {
//         dstype *diff = tmp.tempn;
//         ArrayAXPBY(diff, &K[n*n*i], &A[n*n*i], one, minusone, n*n);
//         dstype normA = NORM(handle, n*n, &A[n*n*i], backend);
//         dstype errMinus = NORM(handle, n*n, diff, backend);
// 
//         ArrayAXPBY(diff, &K[n*n*i], &A[n*n*i], one, one, n*n);
//         dstype errPlus = NORM(handle, n*n, diff, backend);
// 
//         cout << "Rank " << common.mpiRank << ", element " << i
//             << ": BlockJacobianLDG comparison before inversion: "
//             << "||K-A|| = " << scientific << errMinus
//             << ", rel = " << errMinus/(normA + 1.0e-14)
//             << ", ||K+A|| = " << errPlus
//             << ", rel = " << errPlus/(normA + 1.0e-14)
//             << endl;
//     }
// 
//     if (A != nullptr) TemplateFree(A, backend);
//     if (K != nullptr) TemplateFree(K, backend);
// }

// void uEquationLDG(dstype* K, dstype* u, solstruct &sol, resstruct &res, appstruct &app,
//         ExasimDriverABI& driver_abi, masterstruct &master, meshstruct &mesh,
//         tempstruct &tmp, commonstruct &common, cublasHandle_t handle,
//         Int backend)
// {
//     Int n = common.npe*common.ncu;
//     Int m = common.npf*common.nfe*common.ncu;
// 
//     // common.debugMode = 1;
//     // if (common.debugMode == 1) {
//     //     VerifyUEquationElemFiniteDifference(sol.udg, sol, res, app,
//     //             driver_abi, master, mesh, tmp, common, 1e-6);
//     // 
//     //     dstype* u = nullptr;
//     //     TemplateMalloc(&u, common.npe*common.ncu*common.ne1, backend);
//     //     ArrayExtract(u, sol.udg, common.npe, common.nc, common.ne,
//     //             0, common.npe, 0, common.ncu, 0, common.ne1);
//     //     VerifyRuElemDerivFromUFiniteDifference(u, sol, res, app,
//     //             driver_abi, master, mesh, tmp, common, 1e-6);
//     //     VerifyRuFaceDerivFromUFiniteDifference(u, sol, res, app,
//     //             driver_abi, master, mesh, tmp, common, 1e-6);
//     //     VerifyRuDerivFromUFiniteDifference(u, sol, res, app,
//     //             driver_abi, master, mesh, tmp, common, 1e-6);
//     //     TemplateFree(u, backend);
//     // }
// 
//     RuDerivFromU(K, u, sol, res, app, driver_abi, master, mesh, tmp, common);    
//     Inverse(handle, K, tmp.tempn, res.ipiv, n, common.ne1, backend);    
//     //ArrayMultiplyScalar(K, minusone, n*n*common.ne1);      
// 
//     // dstype *A = nullptr;
//     // TemplateMalloc(&A, n*n*common.ne1, backend);
//     // 
//     // RuDerivFromU(A, u, sol, res, app, driver_abi, master, mesh, tmp, common);
//     // ArrayCopy(K, A, n*n*common.ne1);
//     // Inverse(handle, K, A, res.ipiv, n, common.ne1, backend);    
//     // ArrayMultiplyScalar(K, minusone, n*n*common.ne1);      
//     // 
//     // // print3darray(A, n, n, 2);
//     // // print3darray(K, n, n, 2);
//     // // error("here");
//     // 
//     // if (A != nullptr) TemplateFree(A, backend);
// 
//     // for (Int j = 0; j < common.nbe; j++) {
//     //     Int e1 = common.eblks[3*j]-1;
//     //     Int e2 = common.eblks[3*j+1];
//     //     Int ne = e2-e1;
//     // 
//     //     uEquationElemBlock(sol, res, app, driver_abi, master, mesh, tmp,
//     //             common, handle, j, backend);
//     // 
//     //     // print3darray(res.B, n, n*common.nd, ne);
//     // 
//     //     uEquationElemFaceBlockLDG0(sol, res, app, driver_abi, master, mesh,
//     //             tmp, common, handle, j, backend);
//     //     uhatEquationElemFaceBlockLDG(sol, res, app, driver_abi, master, mesh,
//     //             tmp, common, handle, j, backend);
//     //     uEquationSchurBlockLDG0(sol, res, app, driver_abi, master, mesh, tmp,
//     //             common, handle, j, backend);
//     // 
//     //     //PGEMNMStridedBached(handle, n, n, m, one, res.F, n, res.G, m, one, res.D, n, ne, backend);
//     //     // print3darray(res.D, n, n, 2);
//     // 
//     //     Inverse(handle, res.D, tmp.tempn, res.ipiv, n, ne, backend);
//     //     //ArrayMultiplyScalar(res.D, 256.0, n*n*ne);
//     //     ArrayCopy(&res.K[n*n*e1], res.D, n*n*ne);                
//     // 
//     //     // print3darray(res.D, n, n, 2);
//     //     // print3darray(res.Minv2, n, n, 2);
//     //     // 
//     //     //ArrayCopy(&res.K[n*n*e1], &res.Minv2[n*n*e1], n*n*ne);
//     // 
//     //     //error("here");
//     // 
//     //     // if (backend <= 1) {
//     //     //     for (Int e = e1; e < e2; e++) {
//     //     //         cout << "Rank " << common.mpiRank
//     //     //              << ", uEquationLDG res.K for element " << e
//     //     //              << " (" << n << " x " << n << ")" << endl;
//     //     //         print2darray(&res.K[n*n*e], n, n);
//     //     //     }
//     //     // }
//     // }
// 
// 
// }
// 
// void uEquationElemFaceBlockLDG0(solstruct &sol, resstruct &res, appstruct &app, ExasimDriverABI& driver_abi, masterstruct &master, 
//         meshstruct &mesh, tempstruct &tmp, commonstruct &common, cublasHandle_t handle, Int jth, Int backend)
// {            
//     Int nc = common.nc; // number of compoments of (u, q, p)
//     Int ncu = common.ncu;// number of compoments of (u)
//     Int ncq = common.ncq;// number of compoments of (q)
//     Int nco = common.nco;// number of compoments of (o)
//     Int ncx = common.ncx;// number of compoments of (xdg)        
//     Int ncw = common.ncw;
//     Int nd = common.nd;     // spatial dimension    
//     Int npe = common.npe; // number of nodes on master element
//     Int npf = common.npf; // number of nodes on master face           
//     Int ngf = common.ngf; // number of gauss poInts on master face              
//     Int nfe = common.nfe; // number of faces in each element
// 
//     Int e1 = common.eblks[3*jth]-1;
//     Int e2 = common.eblks[3*jth+1];            
//     Int ne = e2-e1;
//     Int nf = nfe*ne;
//     Int nn =  npf*nf; 
//     Int nga = ngf*nf;   
//     Int n0 = 0;                                 // xg
//     Int n1 = nga*ncx;                           // nlg
//     Int n2 = nga*(ncx+nd);                      // jac
//     //Int n3 = nga*(0);                           // uhg    
//     Int n4 = nga*(ncu);                         // udg
//     Int n5 = nga*(ncu+nc);                      // odg
//     Int n6 = nga*(ncu+nc+nco);                  // wsrc
//     Int n7 = nga*(ncu+nc+nco+ncw);              // wdg
//     Int n8 = nga*(ncu+nc+nco+ncw+ncw);          // fhg
//     Int nm = ngf*nfe*e1*(ncx+nd+1);
// 
//     dstype *xg = &sol.elemfaceg[nm+n0];    
//     dstype *nlg = &sol.elemfaceg[nm+n1];
//     dstype *jac = &sol.elemfaceg[nm+n2];
// 
//     dstype *uhg = &tmp.tempg[0];
//     dstype *udg = &tmp.tempg[n4];
//     dstype *odg = &tmp.tempg[n5];
//     dstype *wsrc = &tmp.tempg[n6];
//     dstype *wdg = &tmp.tempg[n7];
// 
//     dstype *fh     = &tmp.tempg[n8];
//     dstype *fh_uq  = &tmp.tempg[n8 + nga*ncu*nd];
//     dstype *fh_uh  = &tmp.tempg[n8 + nga*ncu*nd + nga*ncu*nd*nc];
//     dstype *fh_w   = &tmp.tempg[n8 + nga*ncu*nd + nga*ncu*nd*nc + nga*ncu*ncu];
//     dstype *wdg_uq = &tmp.tempg[n8 + nga*ncu*nd + nga*ncu*nd*nc + nga*ncu*ncu + nga*ncu*nd*ncw];        
// 
//     // npf * nfe * ne * ncu
//     GetElementFaceNodes(tmp.tempn, sol.uh, mesh.elemcon, npf*nfe, ncu, e1, e2, 0); // fixed bug here
// 
//     // udg = tmp.tempg[n4] at gauss points on face
//     GetElementFaceNodes(&tmp.tempn[nn*ncu], sol.udg, mesh.perm, npf*nfe, nc, npe, nc, e1, e2);
// 
//     if (nco>0) GetElementFaceNodes(&tmp.tempn[nn*(ncu+nc)], sol.odg, mesh.perm, npf*nfe, nco, npe, nco, e1, e2);      
// 
//     if ((ncw>0) & (common.wave==0)) {
//       GetElementFaceNodes(&tmp.tempn[nn*(ncu+nc+nco)], sol.wsrc, mesh.perm, npf*nfe, ncw, npe, ncw, e1, e2); 
//       GetElementFaceNodes(&tmp.tempn[nn*(ncu+nc+nco+ncw)], sol.wdg, mesh.perm, npf*nfe, ncw, npe, ncw, e1, e2); // fix bug here
//     }
// 
//     Node2Gauss(handle, tmp.tempg, tmp.tempn, master.shapfgt, ngf, npf, nfe*ne*(ncu+nc+nco+ncw+ncw), backend); // fix bug here
// 
//     if ((ncw>0) & (common.wave==0)) {
//         // copy udg to tmp.tempn
//         ArrayCopy(tmp.tempn, udg, nga*nc);
// 
//         // replace u with uhat 
//         ArrayCopy(tmp.tempn, uhg, nga*ncu);
// 
//         // solve the w equation to get wg and wg_uq
//         wEquation(wdg, wdg_uq, xg, tmp.tempn, odg, wsrc, &tmp.tempn[nga*nc], app, driver_abi, common, nga, backend);                
//     }
// 
//     ArraySetValue(fh, 0.0, nga*ncu);    
//     ArraySetValue(fh_uq, 0.0, nga*ncu*nc);        
//     ArraySetValue(fh_uh, 0.0, nga*ncu*ncu);
// 
//     if (ncw > 0) ArraySetValue(fh_w, 0.0, nga*ncu*ncw);       
// 
//     FhatDriver(fh, fh_uq, fh_w, fh_uh, xg, udg, odg, wdg, uhg, nlg,
//         driver_abi, mesh, master, app, sol, tmp, common, nga, backend);
// 
//     if ((ncw>0) & (common.wave==0)) {
//       ArrayGemmBatch2(fh_uh, fh_w, wdg_uq, one, ncu, ncu, ncw, nga); // fix bug here       
// 
//       ArraySetValue(wdg_uq, 0.0, nga*ncu*ncu);
//       ArrayGemmBatch2(fh_uq, fh_w, wdg_uq, one, ncu, nc, ncw, nga); // fix bug here       
//     }
// 
//     columnwiseMultiply(fh_uq, fh_uq, jac, nga, ncu*nc);
//     columnwiseMultiply(fh_uh, fh_uh, jac, nga, ncu*ncu);
// 
//     dstype *Dtmp  = &tmp.tempn[0];
//     dstype *Btmp  = &tmp.tempn[npf*npf*nfe*ne*ncu*ncu];
//     dstype *Ftmp  = &tmp.tempn[npf*npf*nfe*ne*ncu*ncu + npf*npf*nfe*ne*ncu*ncq];
// 
//     // npf*npf*nfe*ne*ncu*ncu
//     Gauss2Node(handle, Dtmp, fh_uq, master.shapfgwdotshapfg, ngf, npf*npf, nf*ncu*ncu, backend);            
// 
//     // npf*npf*nfe*ne*ncu*ncu -> npe*npe*ne*ncu*ncu  
//     assembleMatrixBD(res.D, Dtmp, mesh.perm, npe, npf, nfe, ne*ncu*ncu);
// 
//     if (ncq > 0) {
//       // npf*npf*nfe*ne*ncu*ncq
//       Gauss2Node(handle, Btmp, &fh_uq[nga*ncu*ncu], master.shapfgwdotshapfg, ngf, npf*npf, nf*ncu*ncq, backend);            
//       // npf*npf*nfe*ne*ncu*ncq -> npe*npe*ne*ncu*ncq
//       assembleMatrixBD(res.B, Btmp, mesh.perm, npe, npf, nfe, ne*ncu*ncq);
//     }
// 
//     // npf*npf*nfe*ne*ncu*ncu
//     Gauss2Node(handle, Ftmp, fh_uh, master.shapfgwdotshapfg, ngf, npf*npf, nf*ncu*ncu, backend);            
//     // npf*npf*nfe*ne*ncu*ncu -> npe*npf*nfe*ne*ncu*ncu
//     assembleMatrixF(&res.F[npe*npf*nfe*ncu*ncu*e1], Ftmp, mesh.perm, npe, npf, nfe, ne*ncu*ncu);
// }
// 
// void uEquationSchurBlockLDG0(solstruct &sol, resstruct &res, appstruct &app, ExasimDriverABI& driver_abi, masterstruct &master, 
//         meshstruct &mesh, tempstruct &tmp, commonstruct &common, cublasHandle_t handle, Int jth, Int backend)
// {        
//     Int ncu = common.ncu;// number of compoments of (u)
//     Int nd = common.nd;     // spatial dimension    
//     Int npe = common.npe; // number of nodes on master element
//     Int npf = common.npf; // number of nodes on master face           
//     Int nfe = common.nfe; // number of faces in each element
// 
//     Int e1 = common.eblks[3*jth]-1;
//     Int e2 = common.eblks[3*jth+1];            
//     Int ne = e2-e1;
//     Int n = npe*ncu; 
//     Int m = npf*nfe*ncu;
// 
//     dstype *D = res.D;
//     dstype *F = &res.F[n*m*e1];
// 
//     // npe * npe * ne * ncu * ncu -> npe * ncu * npe * ncu * ne
//     schurMatrixD(tmp.tempn, res.D, npe, ncu, ne);
//     ArrayCopy(D, tmp.tempn, n * n * ne);
// 
//     // npe * npf * nfe * ne * ncu * ncu -> npe * ncu * ncu * npf * nfe * ne
//     schurMatrixF(tmp.tempn, &res.F[n*m*e1], npe, ncu, npf, nfe, ne);
//     ArrayCopy(F, tmp.tempn, n * m * ne);
// 
//     dstype scalar = 1.0;
//     if (common.wave==1)
//         scalar = 1.0/common.dtfactor;    
// 
//     if (common.ncq > 0) {      
//       if (nd == 1) {
//         // D = D + B * Minv * C
//         schurMatrixBMinvC(D, res.B, &res.C[npe*npe*e1], scalar, npe, ncu, ne);
//         // F = F - B * Minv * E
//         schurMatrixBMinvE(F, res.B, &res.E[npe*npf*nfe*e1], scalar, npe, ncu, npf, nfe, ne);
//       } 
//       else if (nd == 2) {
//         dstype *Cx = &res.C[npe*npe*e1]; // fix bug here
//         dstype *Cy = &res.C[npe*npe*common.ne + npe*npe*e1]; // fix bug here
//         dstype *Ex = &res.E[npe*npf*nfe*e1]; // fix bug here
//         dstype *Ey = &res.E[npe*npf*nfe*common.ne + npe*npf*nfe*e1]; // fix bug here
//         dstype *Bx = res.B; // npe * npe * ne * ncu * ncu
//         dstype *By = &res.B[npe*npe*ncu*ncu*ne]; // npe * npe * ne * ncu * ncu
// 
//         schurMatrixBMinvC(D, Bx, Cx, scalar, npe, ncu, ne);
//         schurMatrixBMinvC(D, By, Cy, scalar, npe, ncu, ne);
// 
//         schurMatrixBMinvE(F, Bx, Ex, scalar, npe, ncu, npf, nfe, ne);
//         schurMatrixBMinvE(F, By, Ey, scalar, npe, ncu, npf, nfe, ne);
//       }
//       else if (nd == 3) {
//         dstype *Cx = &res.C[npe*npe*e1]; // fixed bug here
//         dstype *Cy = &res.C[npe*npe*common.ne + npe*npe*e1]; // fixed bug here
//         dstype *Cz = &res.C[npe*npe*common.ne*2 + npe*npe*e1]; // fixed bug here
//         dstype *Ex = &res.E[npe*npf*nfe*e1]; // fixed bug here
//         dstype *Ey = &res.E[npe*npf*nfe*common.ne + npe*npf*nfe*e1]; // fixed bug here
//         dstype *Ez = &res.E[npe*npf*nfe*common.ne*2 + npe*npf*nfe*e1]; // fixed bug here
//         dstype *Bx = res.B;
//         dstype *By = &res.B[npe*npe*ncu*ncu*ne];
//         dstype *Bz = &res.B[npe*npe*ncu*ncu*ne*2];
// 
//         schurMatrixBMinvC(D, Bx, Cx, scalar, npe, ncu, ne);
//         schurMatrixBMinvC(D, By, Cy, scalar, npe, ncu, ne);
//         schurMatrixBMinvC(D, Bz, Cz, scalar, npe, ncu, ne);
// 
//         schurMatrixBMinvE(F, Bx, Ex, scalar, npe, ncu, npf, nfe, ne);
//         schurMatrixBMinvE(F, By, Ey, scalar, npe, ncu, npf, nfe, ne);
//         schurMatrixBMinvE(F, Bz, Ez, scalar, npe, ncu, npf, nfe, ne);
//       }
//     }
// 
//     // compute the inverse of D 
//     // Inverse(handle, D, tmp.tempn, res.ipiv, n, ne, backend);        
// }

#endif
