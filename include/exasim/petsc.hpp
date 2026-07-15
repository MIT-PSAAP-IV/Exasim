// SPDX-License-Identifier: see LICENSE
//
// <exasim/petsc.hpp> — thin, optional PETSc glue for the operator-export path.
//
// PETSc drives the entire solve; this only factors the MatShell / PCShell / Vec / KSP+SNES
// boilerplate that every consumer would otherwise hand-roll (~20 lines each). It is
// backend-portable (serial / MPI / CUDA) and never touches PETSc's solver choices — the
// caller still sets KSP/SNES type, tolerances, options, or swaps the PC entirely.
//
//   exasim::petsc::Operator<MyModel> op(disc, assembler, prec, sys, comm);
//   KSP ksp = op.make_ksp();               // GMRES + PCShell(res.K) wired; tweak freely
//   Vec u  = op.make_vec(); VecSet(u, 0);
//   KSPSolve(ksp, op.rhs(), u);            // op.rhs() aliases sys.b (zero-copy)
//   op.recover(u);                         // trace -> volume (recover_volume)
//
// For a NONLINEAR steady problem (state-dependent kernels, or a coupling BC re-applied every
// eval), use make_snes_nonlinear -- it re-assembles res.H + sys.b + res.K on each Newton step:
//   SNES snes = op.make_snes_nonlinear(op.make_vec());   // F = -sys.b; caller sets snes/ksp tol
//   SNESSolve(snes, nullptr, u); op.recover(u);
//
// AVAILABILITY: this header + the Exasim::petsc CMake target are only present when PETSc was
// found at Exasim build time. Consumers request it via find_package(Exasim COMPONENTS petsc)
// and link their own PETSc; EXASIM_HAVE_PETSC then gates this header.
#pragma once

#ifndef EXASIM_HAVE_PETSC
#error "<exasim/petsc.hpp> requires the Exasim::petsc component (find_package(Exasim COMPONENTS petsc)) + a linked PETSc."
#endif

#include <petscksp.h>
#include <petscsnes.h>

#include <functional>
#include <memory>
#include <vector>

#include <exasim/operators.hpp>   // CDiscretization / CAssembler<M> / CPreconditioner<M> / sysstruct
#include <exasim/export.hpp>      // recover_volume / apply_mass_inverse / recover_q / eval_residual

namespace exasim {
namespace petsc {

// PETSc device Vec/Mat type for the active Exasim GPU backend (backend 2 = CUDA, 3 = HIP).
// Returns the device type only when PETSc actually supports that device; nullptr means the
// backend is not a supported GPU (caller must NOT wrap a device pointer as host memory).
static inline const char* petsc_device_vectype(int backend) {
#if defined(PETSC_HAVE_CUDA)
    if (backend == 2) return VECCUDA;
#endif
#if defined(PETSC_HAVE_HIP)
    if (backend == 3) return VECHIP;
#endif
    return nullptr;
}
static inline const char* petsc_device_mattype(int backend) {
#if defined(PETSC_HAVE_CUDA)
    if (backend == 2) return MATAIJCUSPARSE;
#endif
#if defined(PETSC_HAVE_HIP)
    if (backend == 3) return MATAIJHIPSPARSE;
#endif
    return MATAIJ;   // host
}

// Compile-time ABI guard: PETSc's scalar/index widths MUST match Exasim's exported precision, or the
// zero-copy Vec/Mat wrapping below reinterprets a wrong-width buffer. This turns the previously
// silent precision mismatch into a build error. Fix by matching one to the other: build PETSc
// --with-precision=single / --with-64-bit-indices, or build Exasim -DEXASIM_FLOAT / -DEXASIM_INT64.
// (Phase 2) The ABI precision guard is no longer a global static_assert on floatTy/intTy: precision
// is a template property of each exported wrapper (Operator<M,Scalar,Idx> / ShellMat<Scalar>), so the
// sizeof(PetscScalar)==sizeof(Scalar) / sizeof(PetscInt)==sizeof(Idx) checks live INSIDE those types
// and fire only for the precision you actually instantiate. Including this header no longer forces a
// TU-wide precision match.

// Wraps Exasim's exported HDG operators (Jacobian-apply res.H + preconditioner res.K) as PETSc
// MatShell + PCShell, plus a zero-copy RHS Vec aliasing the condensed b in sys.b. Templated on
// the model M only because CAssembler<M>/CPreconditioner<M> are; the PETSc side is model-agnostic.
template <class M, class Scalar = exasim::floatTy, class Idx = exasim::intTy>
class Operator {
    static_assert(sizeof(PetscScalar) == sizeof(Scalar),
                  "PetscScalar precision != exported Scalar: rebuild PETSc (--with-precision) or "
                  "Exasim (-DEXASIM_FLOAT), or instantiate Operator with the matching Scalar.");
    static_assert(sizeof(PetscInt) == sizeof(Idx),
                  "PetscInt width != exported Idx: match --with-64-bit-indices / -DEXASIM_INT64.");
    using disc_t = CDiscretizationT<Scalar, Idx>;
public:
    using scalar_type = Scalar;   // exported precision (bit-compatible with PetscScalar, see static_assert)
    using index_type  = Idx;      // exported index type  (bit-compatible with PetscInt)
    // disc/asmb/prec/sys are borrowed (must outlive this). comm is the PETSc/MPI communicator;
    // layout (serial vs MPI, host vs CUDA) is derived from it + disc.common.backend. The condensed
    // system (res.H, res.K, sys.b) must already be assembled by the caller.
    Operator(disc_t& disc, CAssembler<M, Scalar, Idx>& asmb, CPreconditioner<M, Scalar, Idx>& prec,
             sysstructT<Scalar, Idx>& sys, MPI_Comm comm, int spatialScheme = 1)
        : disc_(&disc), asmb_(&asmb), prec_(&prec), sys_(&sys), comm_(comm),
          backend_(static_cast<int>(disc.common.backend)),
          scheme_(spatialScheme), N_(disc.common.sizes.ndofuhat),
          gpu_(disc.common.backend >= 2)
    {
        int nprocs = 1; MPI_Comm_size(comm_, &nprocs);
        const bool par = nprocs > 1;
        // Zero-copy RHS: wrap sys.b (device buffer when gpu) directly as a Vec.
        // The CUDA Vec constructors only EXIST in a CUDA-enabled PETSc, so guard them
        // at compile time (the runtime gpu_ check can't rescue a missing symbol) --
        // this lets petsc.hpp build against a CPU-only PETSc (e.g. GitHub CI's apt PETSc).
        // A GPU backend whose PETSc lacks matching device support would silently wrap the
        // device pointer sys.b as HOST memory below -- abort instead of corrupting memory.
        if (gpu_ && !petsc_device_vectype(backend_))
            SETERRABORT(comm_, PETSC_ERR_SUP,
                "Exasim GPU backend has no matching PETSc device (backend 2 needs CUDA PETSc, backend 3 needs HIP PETSc)");
#if defined(PETSC_HAVE_CUDA)
        if      (backend_ == 2 && par) VecCreateMPICUDAWithArray(comm_, 1, N_, PETSC_DECIDE, sys.b, &b0_);
        else if (backend_ == 2)        VecCreateSeqCUDAWithArray(comm_, 1, N_, sys.b, &b0_);
        else
#endif
#if defined(PETSC_HAVE_HIP)
        if      (backend_ == 3 && par) VecCreateMPIHIPWithArray(comm_, 1, N_, PETSC_DECIDE, sys.b, &b0_);
        else if (backend_ == 3)        VecCreateSeqHIPWithArray(comm_, 1, N_, sys.b, &b0_);
        else
#endif
        if      (par)         VecCreateMPIWithArray(comm_, 1, N_, PETSC_DECIDE, sys.b, &b0_);
        else                  VecCreateSeqWithArray(comm_, 1, N_, sys.b, &b0_);
        // Matrix-free Jacobian.
        MatCreateShell(comm_, N_, N_, PETSC_DETERMINE, PETSC_DETERMINE, this, &J_);
        MatShellSetOperation(J_, MATOP_MULT, (void(*)(void))matmult);
        if (const char* vt = petsc_device_vectype(backend_)) MatShellSetVecType(J_, vt);
    }

    ~Operator() { if (J_) MatDestroy(&J_); if (b0_) VecDestroy(&b0_); }
    Operator(const Operator&) = delete;
    Operator& operator=(const Operator&) = delete;

    Mat mat()  const { return J_; }         // the matrix-free Jacobian (res.H apply)
    Vec rhs()  const { return b0_; }        // the condensed RHS b0 (aliases sys.b)
    Vec make_vec() const { Vec v; VecDuplicate(b0_, &v); return v; }

    // Configure an EXTERNALLY-owned PC (e.g. from PETSc TS's internally-created KSP) as our PCShell
    // applying res.K -- so a KSP/SNES/TS the caller (not this Operator) created can still drive the
    // exported preconditioner. make_ksp/make_snes below are thin wrappers over this.
    void configure_pc(PC pc)
    {
        PCSetType(pc, PCSHELL);
        PCShellSetContext(pc, this);
        PCShellSetApply(pc, pcapply);
    }

    // A KSP with our MatShell + a PCShell applying res.K. Caller sets type/tol/options.
    KSP make_ksp()
    {
        KSP ksp; KSPCreate(comm_, &ksp);
        KSPSetOperators(ksp, J_, J_);
        PC pc; KSPGetPC(ksp, &pc);
        configure_pc(pc);
        return ksp;
    }

    // A SNES for the affine condensed residual F(U) = H*U - b0 (linear steady problems solve in
    // one Newton step); the same MatShell + PCShell back it. `work` must outlive the SNES.
    SNES make_snes(Vec work)
    {
        SNES snes; SNESCreate(comm_, &snes);
        SNESSetFunction(snes, work, formfunction, this);
        SNESSetJacobian(snes, J_, J_, formjacobian, this);
        KSP ksp; SNESGetKSP(snes, &ksp);
        PC pc; KSPGetPC(ksp, &pc);
        configure_pc(pc);
        return snes;
    }

    // A SNES for the NONLINEAR condensed HDG trace residual F(U) = sign * G(U), where G is the
    // condensed residual assembled by hdgAssembleLinearSystem. Unlike make_snes (which is affine
    // -- F = H*U - b0, one Newton step, empty Jacobian), this RE-ASSEMBLES res.H + sys.b + res.K
    // on every residual/Jacobian eval, so state-dependent kernels and any coupling boundary term
    // (e.g. an interface Fext re-applied by hdgAssembleLinearSystem) track U, and it advances the
    // volume state udg from the trace increment (recover_volume) each eval -- the same static-
    // condensation recovery the native Newton does. The same MatShell (res.H) + PCShell (res.K)
    // back it; the caller sets SNES/KSP type, tolerances and options. `work` must outlive the SNES.
    // `sign` relates the condensed residual to res.H: the native convention res.H*d = -sys.b gives
    // d(sys.b)/dU = -res.H, so sign defaults to -1 => dF/dU = res.H = the MatShell.
    SNES make_snes_nonlinear(Vec work, double sign = -1.0)
    {
        nlsign_ = sign;
        SNES snes; SNESCreate(comm_, &snes);
        SNESSetFunction(snes, work, nl_formfunction, this);
        SNESSetJacobian(snes, J_, J_, nl_formjacobian, this);
        KSP ksp; SNESGetKSP(snes, &ksp);
        PC pc; KSPGetPC(ksp, &pc);
        configure_pc(pc);
        return snes;
    }

    // Recover the volume state udg from a converged trace vector U (uses sys.x as scratch).
    void recover(Vec U)
    {
        const PetscScalar* u; PetscMemType mt;
        VecGetArrayReadAndMemType(U, &u, &mt);
        exasim::recover_volume(*disc_, const_cast<Scalar*>(u), sys_->x);
        VecRestoreArrayReadAndMemType(U, const_cast<const PetscScalar**>(&u));
    }

private:
    static PetscErrorCode matmult(Mat J, Vec V, Vec JV)
    {
        Operator* c; MatShellGetContext(J, reinterpret_cast<void**>(&c));
        const PetscScalar* v; PetscScalar* jv; PetscMemType a, b;
        VecGetArrayReadAndMemType(V, &v, &a);
        VecGetArrayWriteAndMemType(JV, &jv, &b);
        c->asmb_->evalMatVec(jv, const_cast<Scalar*>(v), c->sys_->u, c->sys_->b, c->scheme_, c->backend_);
        VecRestoreArrayWriteAndMemType(JV, &jv);
        VecRestoreArrayReadAndMemType(V, &v);
        return PETSC_SUCCESS;
    }
    static PetscErrorCode pcapply(PC pc, Vec V, Vec PV)
    {
        Operator* c; PCShellGetContext(pc, reinterpret_cast<void**>(&c));
        VecCopy(V, PV);
        PetscScalar* pv; PetscMemType m;
        VecGetArrayAndMemType(PV, &pv, &m);
        c->prec_->ApplyPreconditioner(pv, *c->sys_, *c->disc_, c->scheme_, c->backend_);
        VecRestoreArrayAndMemType(PV, &pv);
        return PETSC_SUCCESS;
    }
    static PetscErrorCode formfunction(SNES, Vec U, Vec F, void* ctx)
    {
        Operator* c = static_cast<Operator*>(ctx);
        matmult(c->J_, U, F);                 // F = H*U
        VecAXPY(F, -1.0, c->b0_);             // F -= b0
        return PETSC_SUCCESS;
    }
    static PetscErrorCode formjacobian(SNES, Vec, Mat, Mat, void*) { return PETSC_SUCCESS; }

    // Nonlinear residual: advance udg from the trace U, re-assemble res.H + the condensed
    // residual sys.b (re-applying any coupling BCs), and return F = sign * sys.b -- all
    // backend-dispatched, so U/sys.b may be device pointers when backend>=2.
    static PetscErrorCode nl_formfunction(SNES, Vec U, Vec F, void* ctx)
    {
        Operator* c = static_cast<Operator*>(ctx);
        const PetscScalar* u; PetscMemType mt; VecGetArrayReadAndMemType(U, &u, &mt);
        exasim::recover_volume(*c->disc_, reinterpret_cast<const Scalar*>(u), c->sys_->x);
        c->asmb_->hdgAssembleLinearSystem(c->sys_->b, c->backend_);
        VecRestoreArrayReadAndMemType(U, const_cast<const PetscScalar**>(&u));
        PetscScalar* f; PetscMemType fm; VecGetArrayWriteAndMemType(F, &f, &fm);
        // F = sign * sys.b  (the second term is 0*sys.b, kept valid for device safety)
        ArrayAXPBY(reinterpret_cast<Scalar*>(f), c->sys_->b, c->sys_->b,
                   static_cast<Scalar>(c->nlsign_), static_cast<Scalar>(0), c->N_);
        VecRestoreArrayWriteAndMemType(F, &f);
        return PETSC_SUCCESS;
    }
    // Nonlinear Jacobian: refresh res.H at U (the MatShell operator) and res.K (the PCShell).
    static PetscErrorCode nl_formjacobian(SNES, Vec U, Mat, Mat, void* ctx)
    {
        Operator* c = static_cast<Operator*>(ctx);
        const PetscScalar* u; PetscMemType mt; VecGetArrayReadAndMemType(U, &u, &mt);
        exasim::recover_volume(*c->disc_, reinterpret_cast<const Scalar*>(u), c->sys_->x);
        c->asmb_->hdgAssembleLinearSystem(c->sys_->b, c->backend_);
        VecRestoreArrayReadAndMemType(U, const_cast<const PetscScalar**>(&u));
        c->prec_->ComputeHDGPreconditioner(*c->disc_, c->backend_);
        return PETSC_SUCCESS;
    }

    disc_t*                     disc_;
    CAssembler<M, Scalar, Idx>* asmb_;
    CPreconditioner<M, Scalar, Idx>* prec_;
    sysstructT<Scalar, Idx>*    sys_;
    MPI_Comm                    comm_;
    int                         backend_, scheme_;
    Idx                         N_;
    bool                        gpu_;
    double                      nlsign_ = -1.0;   // F = nlsign * sys.b in the nonlinear SNES path
    Mat                         J_  = nullptr;
    Vec                         b0_ = nullptr;
};

// Options for solve_steady (below). Defaults reproduce the tuned steady HDG solve the
// CHEFSI isoq2d_cht-petsc-fluid app hand-rolled: a genuine PETSc SNES over the nonlinear
// condensed trace residual, GMRES on res.H (MatShell) left-preconditioned by res.K (PCShell).
// The condensed HDG trace system is near-singular, so ksp_rtol is a deliberately loose,
// inexact tolerance -- early stopping regularizes the solve (converging it tightly pumps the
// near-unconstrained mode into the solution). Every field is overridable at runtime via
// -ksp_*/-snes_* (KSPSetFromOptions/SNESSetFromOptions are called).
struct SteadyOptions {
    double snes_rtol = 1e-9;
    double snes_atol = 1e-7;
    double snes_stol = 1e-12;
    int    snes_max_it = 0;          // 0 => disc.common.solverparams.nonlinearSolverMaxIter
    double ksp_rtol = 1e-4;          // loose on purpose (regularizes the near-singular trace)
    double ksp_atol = 1e-14;
    int    ksp_max_it = 1000;
    int    gmres_restart = 250;
    double sign = -1.0;              // F = sign * condensed residual (native convention: -1)
    bool   ksp_error_if_not_converged = false;  // accept the inexact (regularizing) solve
};

// Drive ONE steady nonlinear HDG solve on the exported operators via a genuine PETSc SNES +
// GMRES, warm-started from disc.sol.uh; on return disc.sol.udg is the converged volume state.
// This factors the ~50-line PETSc boilerplate every consumer would otherwise hand-roll (the
// pre-assembly of res.H/res.K/sys.b, the zero-copy trace vector, the SNES/KSP wiring and the
// static-condensation recover). The caller has already run any pre-solve setup (InitSolution,
// odg->Gauss interpolation). Returns the SNESConvergedReason. Requires the condensed system's
// components to be present (this assembles them). Backend-portable (serial / MPI / CUDA / HIP).
template <class M, class Scalar = exasim::floatTy, class Idx = exasim::intTy>
inline int solve_steady(CDiscretizationT<Scalar, Idx>& disc,
                        CAssembler<M, Scalar, Idx>& asmb,
                        CPreconditioner<M, Scalar, Idx>& prec,
                        sysstructT<Scalar, Idx>& sys,
                        MPI_Comm comm, const SteadyOptions& o = {})
{
    const int backend = static_cast<int>(disc.common.backend);
    const Idx N = disc.common.sizes.ndofuhat;

    // Raise explicitly on a non-HDG problem: solve_steady drives the condensed HDG trace
    // system (res.H / res.K / hdgAssembleLinearSystem), which only exists for HDG
    // (spatialScheme=1). LDG datain (spatialScheme=0) would otherwise fail cryptically (the
    // HDG connectivity build's DOF-consistency check) or silently produce garbage.
    if (disc.common.spatialScheme != 1)
        SETERRABORT(comm, PETSC_ERR_ARG_WRONG,
            "exasim::petsc::solve_steady is HDG-only, but this problem is not HDG "
            "(spatialScheme != 1; 0 = LDG). Preprocess the mesh with discretization=\"hdg\" "
            "(hybrid=1), or use the frontend exportapp path for an LDG / ExasimSolver app.");

    // Assemble the operators once at the warm-started trace: the Operator ctor requires res.H
    // (evalMatVec), res.K and sys.b to already exist. hdgGetQ recovers q first when ncq>0.
    if (disc.common.components.ncq > 0)
        hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);
    asmb.hdgAssembleLinearSystem(sys.b, backend);
    prec.ComputeHDGPreconditioner(disc, backend);

    Operator<M, Scalar, Idx> op(disc, asmb, prec, sys, comm);

    Vec U = op.make_vec();
    // Seed U <- disc.sol.uh with a memtype-aware copy: on a GPU backend both the Vec
    // storage and disc.sol.uh are device memory, so a host for-loop would touch invalid
    // memory. ArrayCopy(cublasHandle, dst, src, N, backend) is the backend-dispatched
    // device/host copy the native path uses (solution.cpp: sys.u <- sol.uh).
    { PetscScalar* u; PetscMemType mt; VecGetArrayWriteAndMemType(U, &u, &mt);
      ArrayCopy(disc.common.cublasHandle, reinterpret_cast<Scalar*>(u), disc.sol.uh, N, backend);
      VecRestoreArrayWriteAndMemType(U, &u); }

    Vec work = op.make_vec();
    SNES snes = op.make_snes_nonlinear(work, o.sign);

    KSP ksp; SNESGetKSP(snes, &ksp);
    KSPSetType(ksp, KSPGMRES);
    KSPGMRESSetRestart(ksp, o.gmres_restart);
    KSPGMRESSetOrthogonalization(ksp, KSPGMRESModifiedGramSchmidtOrthogonalization);
    KSPSetTolerances(ksp, o.ksp_rtol, o.ksp_atol, PETSC_DEFAULT, o.ksp_max_it);
    KSPSetFromOptions(ksp);
    KSPSetErrorIfNotConverged(ksp, o.ksp_error_if_not_converged ? PETSC_TRUE : PETSC_FALSE);
    SNESSetMaxLinearSolveFailures(snes, 100000);
    const int nlMax = o.snes_max_it > 0 ? o.snes_max_it
                                        : static_cast<int>(disc.common.solverparams.nonlinearSolverMaxIter);
    SNESSetTolerances(snes, o.snes_atol, o.snes_rtol, o.snes_stol, nlMax, 100 * nlMax);
    SNESSetFromOptions(snes);

    SNESSolve(snes, nullptr, U);
    SNESConvergedReason reason; SNESGetConvergedReason(snes, &reason);
    op.recover(U);   // trace -> final volume state udg

    VecDestroy(&U); VecDestroy(&work); SNESDestroy(&snes);
    return static_cast<int>(reason);
}

// Pre-Newton setup a hand-driven SNES must run that the native SteadyProblem does internally:
// seed sys.u from sol.uh (InitSolution), (re)compute the dynamic artificial-viscosity field if
// the model uses one, and interpolate the odg auxiliary field to the element/face Gauss points
// (odgg/og1/og2). The volume/face kernels read those Gauss buffers; without this they are
// uninitialized -> NaN residual. Lifted from the CHEFSI isoq2d_cht-petsc-fluid app's prepareSolve
// so consumers no longer carry it. Single-rank AV halo smoothing is a no-op at np=1.
template <class M, class Scalar = exasim::floatTy, class Idx = exasim::intTy>
inline void prepare_steady(CSolution<M>& model, int backend)
{
    auto& disc = model.disc;
    model.InitSolution(backend);   // HDG: ArrayCopy(sys.u, sol.uh) + boundary-avg zeroing

    if (disc.common.physicsparams.ncAV > 0 && disc.common.physicsparams.frozenAVflag > 0) {
        Idx nco  = disc.common.components.nco;
        Idx ncAV = disc.common.physicsparams.ncAV;
        Idx npe  = disc.common.grid.npe;
        Idx ne   = disc.common.meshsizes.ne;
        Scalar* avField = &disc.res.Rq[0];
        Scalar* utm     = &disc.res.Rq[npe * ncAV * ne];
        model.residual.evalAVfield(avField, backend);
        for (Idx iav = 0; iav < disc.common.physicsparams.AVsmoothingIter; iav++)
            disc.DG2CG2(avField, avField, utm, ncAV, ncAV, ncAV, backend);
        ArrayInsert(disc.sol.odg, avField, npe, nco, ne, 0, npe, nco - ncAV, nco, 0, ne);
    }

    if (disc.common.components.nco > 0) {
        for (Idx j = 0; j < disc.common.meshsizes.nbe; j++) {
            Idx e1 = disc.common.eblks[3 * j] - 1, e2 = disc.common.eblks[3 * j + 1];
            GetElemNodes(disc.tmp.tempn, disc.sol.odg, disc.common.grid.npe, disc.common.components.nco,
                         0, disc.common.components.nco, e1, e2);
            Node2Gauss(disc.common.cublasHandle,
                       &disc.sol.odgg[disc.common.grid.nge * disc.common.components.nco * e1],
                       disc.tmp.tempn, disc.master.shapegt, disc.common.grid.nge, disc.common.grid.npe,
                       (e2 - e1) * disc.common.components.nco, backend);
        }
        for (Idx j = 0; j < disc.common.meshsizes.nbf; j++) {
            Idx f1 = disc.common.fblks[3 * j] - 1, f2 = disc.common.fblks[3 * j + 1];
            GetFaceNodes(disc.tmp.tempn, disc.sol.odg, disc.mesh.facecon, disc.common.grid.npf,
                         disc.common.components.nco, disc.common.grid.npe, disc.common.components.nco, f1, f2, 1);
            Node2Gauss(disc.common.cublasHandle,
                       &disc.sol.og1[disc.common.grid.ngf * disc.common.components.nco * f1],
                       disc.tmp.tempn, disc.master.shapfgt, disc.common.grid.ngf, disc.common.grid.npf,
                       (f2 - f1) * disc.common.components.nco, backend);
            GetFaceNodes(disc.tmp.tempn, disc.sol.odg, disc.mesh.facecon, disc.common.grid.npf,
                         disc.common.components.nco, disc.common.grid.npe, disc.common.components.nco, f1, f2, 2);
            Node2Gauss(disc.common.cublasHandle,
                       &disc.sol.og2[disc.common.grid.ngf * disc.common.components.nco * f1],
                       disc.tmp.tempn, disc.master.shapfgt, disc.common.grid.ngf, disc.common.grid.npf,
                       (f2 - f1) * disc.common.components.nco, backend);
        }
    }
}

// Convenience: prepare + one steady nonlinear HDG solve on a CSolution<M> built from datain (the
// header-only concrete-model path). This is the whole solver a generated standalone app needs --
// prepare_steady then solve_steady over the exported operators -- so the app owns no PETSc glue.
// Warm-started from sol.uh; on return sol.udg is the converged volume state. Returns the reason.
template <class M>
inline int solve_steady(CSolution<M>& model, MPI_Comm comm, const SteadyOptions& o = {})
{
    const int backend = static_cast<int>(model.disc.common.backend);
    // Guard before prepare_steady (HDG-only pre-solve setup) so an LDG problem raises here
    // with a clear message rather than misbehaving downstream.
    if (model.disc.common.spatialScheme != 1)
        SETERRABORT(comm, PETSC_ERR_ARG_WRONG,
            "exasim::petsc::solve_steady is HDG-only, but this problem is not HDG "
            "(spatialScheme != 1; 0 = LDG). Preprocess the mesh with discretization=\"hdg\" "
            "(hybrid=1), or use the frontend exportapp path for an LDG / ExasimSolver app.");
    prepare_steady<M>(model, backend);
    return solve_steady<M>(model.disc, model.assembler, model.prec, model.solv.sys, comm, o);
}

// Wrap ANY Exasim linear-operator apply (y = A*x, backend/memtype-aware) as a PETSc MatShell.
// The extensibility primitive: the condensed Jacobian, the inverse mass, a custom block, etc.
// are one std::function each. The apply closure captures whatever Exasim state it needs; the
// caller keeps that state (and this ShellMat) alive for the Mat's lifetime.
template <class Scalar = exasim::floatTy>
class ShellMat {
    static_assert(sizeof(PetscScalar) == sizeof(Scalar),
                  "PetscScalar precision != exported Scalar (ShellMat): match PETSc / Exasim precision.");
public:
    using scalar_type = Scalar;            // exported precision (bit-compatible with PetscScalar)
    using index_type  = exasim::intTy;     // exported index type (matches PetscInt)
    using Apply = std::function<void(Scalar* y, const Scalar* x)>;
    // backend: 0/1 = host, 2 = CUDA, 3 = HIP (Exasim's disc.common.backend).
    ShellMat(MPI_Comm comm, exasim::intTy n_local, Apply apply, int backend = 0)
        : apply_(std::move(apply))
    {
        MatCreateShell(comm, n_local, n_local, PETSC_DETERMINE, PETSC_DETERMINE, this, &mat_);
        MatShellSetOperation(mat_, MATOP_MULT, (void(*)(void))mult);
        if (const char* vt = petsc_device_vectype(backend)) MatShellSetVecType(mat_, vt);
    }
    ~ShellMat() { if (mat_) MatDestroy(&mat_); }
    ShellMat(const ShellMat&) = delete;
    ShellMat& operator=(const ShellMat&) = delete;

    Mat mat() const { return mat_; }
    Vec make_vec() const { Vec v; MatCreateVecs(mat_, &v, nullptr); return v; }

private:
    static PetscErrorCode mult(Mat A, Vec X, Vec Y)
    {
        ShellMat* c; MatShellGetContext(A, reinterpret_cast<void**>(&c));
        const PetscScalar* x; PetscScalar* y; PetscMemType a, b;
        VecGetArrayReadAndMemType(X, &x, &a);
        VecGetArrayWriteAndMemType(Y, &y, &b);
        c->apply_(y, x);
        VecRestoreArrayWriteAndMemType(Y, &y);
        VecRestoreArrayReadAndMemType(X, &x);
        return PETSC_SUCCESS;
    }
    Apply apply_;
    Mat   mat_ = nullptr;
};

// A PETSc Mat applying the element block-diagonal inverse VOLUME mass M^{-1} (needs res.Minv,
// e.g. after disc.compMassInverse()). Acts on the volume space (npe*ncr*ne), not the trace.
template <class M, class Scalar = exasim::floatTy, class Idx = exasim::intTy>
inline std::unique_ptr<ShellMat<Scalar>> make_mass_inverse(CDiscretizationT<Scalar, Idx>& disc, MPI_Comm comm, int ncr = M::ncu)
{
    const Idx  nvol = disc.common.grid.npe * ncr * disc.common.meshsizes.ne1;
    return std::make_unique<ShellMat<Scalar>>(comm, nvol,
        [&disc, ncr](Scalar* y, const Scalar* x){ exasim::apply_mass_inverse<M>(disc, x, y, ncr); },
        static_cast<int>(disc.common.backend));
}

// Assemble the condensed HDG trace operator (res.H) into a real PETSc MATAIJ(CUSPARSE), so the
// full PETSc arsenal that needs entries (LU/ILU/AMG, ...) can act on it. res.H
// holds one m x m element block per element (m = ncu*npf*nfe, column-major, applied by
// hdgMatVec); the local->global trace map is j + elemcon[i + e*ndf]*ncu. This is standard FE
// scatter: A = sum_e P_e AE[e] P_e^T, the identical operator to the matrix-free op.mat().
//
// TRANSFER-FREE: the sparsity pattern (row/col index arrays, one per res.H entry, in res.H's
// column-major per-element order) is built on host from elemcon -- small integer metadata, once.
// The VALUES are then set straight from disc.res.H via MatSetValuesCOO: on GPU res.H is a device
// pointer and MATAIJCUSPARSE consumes it in place, so the numeric assembly never leaves the device.
template <class M, class Scalar = exasim::floatTy, class Idx = exasim::intTy>
inline Mat assemble_matrix(CDiscretizationT<Scalar, Idx>& disc, MPI_Comm comm)
{
    auto& c = disc.common;
    const int backend = static_cast<int>(c.backend);
    const int ncu = c.components.ncu, npf = c.grid.npf, nfe = static_cast<int>(c.meshsizes.nfe);
    const int ndf = npf * nfe, m = ncu * ndf, ne = static_cast<int>(c.meshsizes.ne1);
    const Idx N = c.sizes.ndofuhat;

    // elemcon -> host (small metadata) to build the COO (row,col) pattern.
    std::vector<int> elemcon(static_cast<size_t>(ndf) * ne);
    if (backend >= 2) TemplateCopytoHost(elemcon.data(), disc.mesh.elemcon, (Int)elemcon.size(), backend);
    else              std::copy(disc.mesh.elemcon, disc.mesh.elemcon + elemcon.size(), elemcon.begin());

    // One COO entry per res.H entry, ordered k = e*m*m + lr + m*lc (res.H's column-major layout),
    // so the values array below is exactly disc.res.H with no reordering or copy.
    const PetscCount ncoo = static_cast<PetscCount>(ne) * m * m;
    std::vector<PetscInt> ci(static_cast<size_t>(ncoo)), cj(static_cast<size_t>(ncoo));
    std::vector<PetscInt> g(static_cast<size_t>(m));
    for (int e = 0; e < ne; ++e) {
        for (int i = 0; i < ndf; ++i) {
            const int base = elemcon[i + e * ndf] * ncu;
            for (int j = 0; j < ncu; ++j) g[j + ncu * i] = base + j;
        }
        const size_t e0 = static_cast<size_t>(e) * m * m;
        for (int lc = 0; lc < m; ++lc)
            for (int lr = 0; lr < m; ++lr) {
                const size_t k = e0 + lr + static_cast<size_t>(m) * lc;
                ci[k] = g[lr]; cj[k] = g[lc];      // duplicate (row,col) across elements -> COO sums them
            }
    }

    Mat A;
    MatCreate(comm, &A);
    MatSetSizes(A, N, N, PETSC_DETERMINE, PETSC_DETERMINE);
    MatSetType(A, petsc_device_mattype(backend));   // device-resident on GPU (CUDA/HIP), host otherwise
    MatSetPreallocationCOO(A, ncoo, ci.data(), cj.data());
    MatSetValuesCOO(A, disc.res.H, ADD_VALUES);              // values straight from (device) res.H
    return A;
}

} // namespace petsc
} // namespace exasim
