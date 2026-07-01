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
// AVAILABILITY: this header + the Exasim::petsc CMake target are only present when PETSc was
// found at Exasim build time. Consumers request it via find_package(Exasim COMPONENTS petsc)
// and link their own PETSc; EXASIM_HAVE_PETSC then gates this header.
#pragma once

#ifndef EXASIM_HAVE_PETSC
#error "<exasim/petsc.hpp> requires the Exasim::petsc component (find_package(Exasim COMPONENTS petsc)) + a linked PETSc."
#endif

#include <petscksp.h>
#include <petscsnes.h>

#include <exasim/operators.hpp>   // CDiscretization / CAssembler<M> / CPreconditioner<M> / sysstruct
#include <exasim/export.hpp>      // recover_volume

namespace exasim {
namespace petsc {

// Wraps Exasim's exported HDG operators (Jacobian-apply res.H + preconditioner res.K) as PETSc
// MatShell + PCShell, plus a zero-copy RHS Vec aliasing the condensed b in sys.b. Templated on
// the model M only because CAssembler<M>/CPreconditioner<M> are; the PETSc side is model-agnostic.
template <class M>
class Operator {
public:
    // disc/asmb/prec/sys are borrowed (must outlive this). comm is the PETSc/MPI communicator;
    // layout (serial vs MPI, host vs CUDA) is derived from it + disc.common.backend. The condensed
    // system (res.H, res.K, sys.b) must already be assembled by the caller.
    Operator(CDiscretization& disc, CAssembler<M>& asmb, CPreconditioner<M>& prec,
             sysstruct& sys, MPI_Comm comm, int spatialScheme = 1)
        : disc_(&disc), asmb_(&asmb), prec_(&prec), sys_(&sys), comm_(comm),
          backend_(static_cast<int>(disc.common.backend)),
          scheme_(spatialScheme), N_(disc.common.sizes.ndofuhat),
          gpu_(disc.common.backend >= 2)
    {
        int nprocs = 1; MPI_Comm_size(comm_, &nprocs);
        const bool par = nprocs > 1;
        // Zero-copy RHS: wrap sys.b (device buffer when gpu) directly as a Vec.
        if      (gpu_ && par) VecCreateMPICUDAWithArray(comm_, 1, N_, PETSC_DECIDE, sys.b, &b0_);
        else if (gpu_)        VecCreateSeqCUDAWithArray(comm_, 1, N_, sys.b, &b0_);
        else if (par)         VecCreateMPIWithArray(comm_, 1, N_, PETSC_DECIDE, sys.b, &b0_);
        else                  VecCreateSeqWithArray(comm_, 1, N_, sys.b, &b0_);
        // Matrix-free Jacobian.
        MatCreateShell(comm_, N_, N_, PETSC_DETERMINE, PETSC_DETERMINE, this, &J_);
        MatShellSetOperation(J_, MATOP_MULT, (void(*)(void))matmult);
        if (gpu_) MatShellSetVecType(J_, VECCUDA);
    }

    ~Operator() { if (J_) MatDestroy(&J_); if (b0_) VecDestroy(&b0_); }
    Operator(const Operator&) = delete;
    Operator& operator=(const Operator&) = delete;

    Mat mat()  const { return J_; }         // the matrix-free Jacobian (res.H apply)
    Vec rhs()  const { return b0_; }        // the condensed RHS b0 (aliases sys.b)
    Vec make_vec() const { Vec v; VecDuplicate(b0_, &v); return v; }

    // A KSP with our MatShell + a PCShell applying res.K. Caller sets type/tol/options.
    KSP make_ksp()
    {
        KSP ksp; KSPCreate(comm_, &ksp);
        KSPSetOperators(ksp, J_, J_);
        PC pc; KSPGetPC(ksp, &pc);
        PCSetType(pc, PCSHELL);
        PCShellSetContext(pc, this);
        PCShellSetApply(pc, pcapply);
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
        PCSetType(pc, PCSHELL);
        PCShellSetContext(pc, this);
        PCShellSetApply(pc, pcapply);
        return snes;
    }

    // Recover the volume state udg from a converged trace vector U (uses sys.x as scratch).
    void recover(Vec U)
    {
        const PetscScalar* u; PetscMemType mt;
        VecGetArrayReadAndMemType(U, &u, &mt);
        exasim::recover_volume(*disc_, const_cast<dstype*>(u), sys_->x);
        VecRestoreArrayReadAndMemType(U, const_cast<const PetscScalar**>(&u));
    }

private:
    static PetscErrorCode matmult(Mat J, Vec V, Vec JV)
    {
        Operator* c; MatShellGetContext(J, &c);
        const PetscScalar* v; PetscScalar* jv; PetscMemType a, b;
        VecGetArrayReadAndMemType(V, &v, &a);
        VecGetArrayWriteAndMemType(JV, &jv, &b);
        c->asmb_->evalMatVec(jv, const_cast<dstype*>(v), c->sys_->u, c->sys_->b, c->scheme_, c->backend_);
        VecRestoreArrayWriteAndMemType(JV, &jv);
        VecRestoreArrayReadAndMemType(V, &v);
        return PETSC_SUCCESS;
    }
    static PetscErrorCode pcapply(PC pc, Vec V, Vec PV)
    {
        Operator* c; PCShellGetContext(pc, &c);
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

    CDiscretization*            disc_;
    CAssembler<M>*              asmb_;
    CPreconditioner<M>*         prec_;
    sysstruct*                  sys_;
    MPI_Comm                    comm_;
    int                         backend_, scheme_;
    Int                         N_;
    bool                        gpu_;
    Mat                         J_  = nullptr;
    Vec                         b0_ = nullptr;
};

} // namespace petsc
} // namespace exasim
