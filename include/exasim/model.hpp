// SPDX-License-Identifier: see LICENSE
//
// <exasim/model.hpp> — the Model contract.
//
// A user PDE model is a C++ struct that supplies pointwise math (and,
// for HDG, hand-written pointwise Jacobians) which the templated
// kernels in <exasim/kernels/*.hpp> call inside `Kokkos::parallel_for`
// to evaluate volume / surface integrals at quadrature points.
//
// There is no autodiff and no DSL. Users write `flux(f, x, uq, w, mu, t)`
// in plain C++ and, where the discretization needs derivatives, write
// `flux_jac_uq(f_uq, …)` themselves. exasim::ModelDefaults<Self> (CRTP
// base) supplies zero-fill no-op implementations of every optional
// method so users override only what their PDE actually has.
//
// Boundary identification is NOT part of the Model contract
// ---------------------------------------------------------
// The user's `M::fbou(...)`, `M::ubou(...)`, `M::qoi_boundary(...)`
// methods receive an integer `ib` boundary tag and dispatch on it:
//
//   void fbou(double fb[], int ib, …) {
//       if (ib == 1)      { /* wall */ }
//       else if (ib == 2) { /* inflow */ }
//       …
//   }
//
// The mapping from physical geometry to the integer `ib` happens at
// the runtime / preprocessing layer, not here. Users still write the
// usual `pdeapp.txt` config:
//
//   boundaryconditions  = [1, 2, 1, 3];
//   boundaryexpressions = ["abs(y)<1e-8", "x>9-1e-2", …];
//
// Preprocessing's `tinyexpr` evaluator parses the expression strings,
// tags each mesh face with the matching integer, and the kernels see
// only the resulting `int ib`. Curved-boundary expressions and
// periodic boundary matching are also preprocessing-layer concerns.
//
// Storage layout convention
// -------------------------
// The kernels see SoA buffers laid out as:
//   xdg [k*ng + i]   for k in [0, nd),       (spatial coords)
//   udg [k*ng + i]   for k in [0, Nq),       (u and ∇u, Nq = ncu*(1+nd))
//   wdg [k*ng + i]   for k in [0, ncw),      (auxiliary scalar fields)
//   param[k]         for k in [0, nparam),   (physics params; not per-i)
//
// At each quadrature point i, the kernel gathers small fixed-size
// arrays and passes them to the model's pointwise function:
//
//   double x [nd], uq[Nq], w[ncw];
//   for k in [0, nd):  x [k] = xdg[k*ng + i];
//   for k in [0, Nq):  uq[k] = udg[k*ng + i];
//   for k in [0, ncw): w [k] = wdg[k*ng + i];
//   double f_local[ncu*nd];
//   Model::flux(f_local, x, uq, w, param, t);
//   for k in [0, ncu*nd): f[k*ng + i] = f_local[k];
//
// Jacobian outputs use a row-major convention with the output index
// outer:
//   f_uq [ i * Nq  + j ] = ∂f_i / ∂uq_j
//   f_w  [ i * ncw + j ] = ∂f_i / ∂w_j
//   fb_uh[ i * ncu + j ] = ∂fb_i / ∂uh_j   (boundary trace Jacobian)
//
// Required compile-time configuration
// -----------------------------------
// Every Model must publish:
//
//   static constexpr int nd;       // spatial dimension (1, 2, or 3)
//   static constexpr int ncu;      // # primary unknowns
//   static constexpr int ncw;      // # auxiliary scalar fields (often 0)
//   static constexpr int nparam;   // # physics parameters
//
//   // Optional (default 0 from ModelDefaults<Self>):
//   static constexpr int nco;      // # "other DG" auxiliary fields
//                                  // (the `v` / `odg` arrays in
//                                  // libpdemodel and text2code's DSL)
//
//   // Derived (provided by ModelDefaults<Self>; users may override):
//   static constexpr int Nq = ncu * (1 + nd);
//
// Naming alignment with the text2code DSL (pdemodel.txt) and the
// libpdemodel.hpp ABI:
//
//   text2code DSL    libpdemodel ABI    Model contract (this file)
//   ─────────────────────────────────────────────────────────────
//   x                xdg                x[]    (spatial coords)
//   uq               udg                uq[]   ((u, ∇u) packed)
//   v                odg                — (only nco; not pointwise)
//   w                wdg                w[]    (auxiliary scalars)
//   uhat             uhg                uh[]   (HDG trace)
//   n                nlg                n[]    (face normal)
//   tau              tau                tau[]  (stabilization)
//   mu               param              mu[]   (physics params)
//   eta              uinf               uinf[] (free-stream / reference)
//   uext             uext               uext[] (Fext only; multi-domain)
//   t                time               t      (current time)
//
// Required pointwise functions
// ----------------------------
// The minimal set every Model must define (no defaults):
//
//   KOKKOS_INLINE_FUNCTION static
//   void flux  (double f [/*ncu*nd*/], const double x[/*nd*/],
//               const double uq[/*Nq*/], const double w[/*ncw*/],
//               const double mu[/*nparam*/], const double uinf[/*ncu*/],
//               double t);
//
//   KOKKOS_INLINE_FUNCTION static
//   void initu (double ui[/*ncu*/], const double x[/*nd*/],
//               const double uinf[/*ncu*/], const double mu[/*nparam*/]);
//
// Volume pointwise methods (`flux`, `source`, `sourcew`, `materialstate`,
// `tdfunc`, `avfield`, `eos`, `eos_du`, `eos_dw`) all take args
//   (out, x, uq, w, mu, uinf, t)
// after the output buffer. `uinf` is pointer-passed and may be
// nullptr — methods that need free-stream values dereference at
// their own risk.
//
// Optional pointwise functions (default = zero-fill, via ModelDefaults):
//
//   source, sourcew, materialstate, fbou, ubou, fhat, uhat, stab,
//   tdfunc, eos, eos_du, eos_dw, avfield,
//   init{q, udg, wdg, odg}, monitor, output,
//   vis_scalars, vis_vectors, vis_tensors,
//   qoi_volume, qoi_boundary
//
// HDG Jacobians (required iff Discretization::HDG is selected; default
// = zero-fill, so users get a compile error / zero-residual Jacobian
// only if they forget):
//
//   flux_jac_uq, flux_jac_w,
//   source_jac_uq, source_jac_w,
//   fbou_jac_uq, fbou_jac_uh, fbou_jac_w,
//   fhat_jac_uq, fhat_jac_uh, fhat_jac_w,
//   stab_jac_uq, stab_jac_uh,
//   eos_jac_uq, eos_jac_w
//
// All Jacobians are pointwise and `KOKKOS_INLINE_FUNCTION static`. See
// the per-kernel headers in <exasim/kernels/> for exact signatures.

#pragma once

#include <type_traits>

#include <Kokkos_Core.hpp>

#include "common.h"   // dstype + Int

namespace exasim {

// Discretization tag — published as `Self::disc` by the user model so
// kernels can `if constexpr` on it. Defaults to LDG (Local DG) since
// it requires no Jacobians.
enum class Discretization {
    LDG,   // Local DG: only flux/source/initu values needed.
    HDG    // Hybridized DG: requires hand-written pointwise Jacobians.
};

// CRTP base supplying zero-fill no-op defaults for every optional
// method on the Model contract. Users inherit and override only what
// their PDE has.
//
//   struct Poisson2D : exasim::ModelDefaults<Poisson2D> {
//       static constexpr int nd = 2, ncu = 1, ncw = 0, nparam = 1;
//       static constexpr auto disc = exasim::Discretization::HDG;
//
//       KOKKOS_INLINE_FUNCTION static
//       void flux(double f[], const double[], const double uq[],
//                 const double[], const double mu[], double) {
//           f[0] = mu[0]*uq[1];
//           f[1] = mu[0]*uq[2];
//       }
//
//       KOKKOS_INLINE_FUNCTION static
//       void flux_jac_uq(double f_uq[], const double[], const double[],
//                        const double[], const double mu[], double) {
//           constexpr int Nq = ncu*(1+nd);     // 3
//           for (int k = 0; k < ncu*nd*Nq; ++k) f_uq[k] = 0;
//           f_uq[0*Nq + 1] = mu[0];
//           f_uq[1*Nq + 2] = mu[0];
//       }
//
//       // ... source, ubou, initu — explicit overrides for what's used.
//   };
namespace detail {
// Zero-fill an output buffer of size N at compile time. A free helper (formerly a
// ModelDefaults member) so the per-concern default mixins below can all call it
// without depending on a shared base class (avoids two-phase-lookup on a dependent base).
template <int N, class T = double>
KOKKOS_INLINE_FUNCTION void zero_fill(T f[]) {
    for (int k = 0; k < N; ++k) f[k] = 0.0;
}
} // namespace detail

template <class Self, class T = double>
struct ModelConstants {
    // Default discretization tag. Override by `static constexpr auto
    // disc = exasim::Discretization::HDG;` in the derived struct.
    static constexpr Discretization disc = Discretization::LDG;

    // Default count of "other DG" auxiliary fields (the `v` / `odg`
    // arrays in libpdemodel and the text2code DSL `vectors v(nco)`).
    // Most PDEs leave this at 0; only models with an external scalar
    // field (artificial viscosity, level-set, … plumbed through `odg`)
    // override.
    static constexpr int nco = 0;

    // ---- Multi-domain HDG interface coupling (Fint / Fext) ----
    //
    // Off by default: single-domain HDG models (Poisson, Navier-Stokes,
    // …) never reach the `fint`/`fext` kernels, so the whole coupling
    // surface is disabled and their assembled residual/Jacobian is
    // byte-identical to the pre-coupling build. A coupled model turns it
    // on with `static constexpr bool has_external_coupling = true;`.
    static constexpr bool has_external_coupling = false;

    // Number of material-state components returned by materialstate().
    // Models without materialstate leave this at zero.
    static constexpr int nmaterialstate = 0;

    // Interface-residual widths — the text2code `Fint`/`Fext`
    // `output_size`, i.e. the ABI's `common.szinterfacefluxmap`
    // (`ncu12`) and `common.ncuext` — are read through the
    // `nfint_v<M>` / `nfext_v<M>` / `ncuext_v<M>` traits declared after
    // the mixin chain, which fall back to `M::ncu`. They are deliberately
    // NOT defaulted here: referencing `Self::ncu` in a `static constexpr`
    // member initializer of a CRTP base is fragile (the base is
    // instantiated while `Self` is still incomplete). So a coupled model
    // whose interface residual matches `ncu` needs no extra members, and
    // one that differs (e.g. pdemodel2's 2-component `Fint` with ncu==1)
    // just adds `static constexpr int nfint = 2;`.
};

// Per-concern default mixins (a C3-style split of the model "god-base"): each adds the
// zero/identity defaults for ONE concern, chained so the composite ModelDefaults below
// supplies everything. Each mixin can be referenced on its own to express that a kernel
// or class needs only that concern's surface.
template <class Self, class T = double>
struct VolumeDefaults : ModelConstants<Self, T> {
    // ---- Volume terms (default: zero) ----
    //
    // Volume pointwise methods take 8 args after the output buffer:
    //   (out, x[nd], uq[Nq], v[nco], w[ncw], mu[nparam], uinf[], t)
    // `v` is the auxiliary "other DG" field (pdemodel.txt's `vectors v(nco)`,
    // libpdemodel's `odg`). `uinf` is pointer-passed and may be nullptr —
    // methods that need free-stream values dereference at their own risk.

    KOKKOS_INLINE_FUNCTION static
    void source(T s[],
                const T /*x*/[],  const T /*uq*/[],
                const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                const T /*uinf*/[], T /*t*/) {
        detail::zero_fill<Self::ncu>(s);
    }

    KOKKOS_INLINE_FUNCTION static
    void sourcew(T sw[],
                 const T /*x*/[],  const T /*uq*/[],
                 const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                 const T /*uinf*/[], T /*t*/) {
        if constexpr (Self::ncw > 0) detail::zero_fill<Self::ncw>(sw);
    }

    KOKKOS_INLINE_FUNCTION static
    void materialstate(T state[],
                       const T /*x*/[],  const T /*uq*/[],
                       const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                       const T /*uinf*/[], T /*t*/) {
        if constexpr (Self::nmaterialstate > 0) detail::zero_fill<Self::nmaterialstate>(state);
    }

    KOKKOS_INLINE_FUNCTION static
    void tdfunc(T m[],
                const T /*x*/[],  const T /*uq*/[],
                const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                const T /*uinf*/[], T /*t*/) {
        // Identity mass weighting by default.
        for (int k = 0; k < Self::ncu; ++k) m[k] = 1.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void avfield(T av[],
                 const T /*x*/[],  const T /*uq*/[],
                 const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                 const T /*uinf*/[], T /*t*/) {
        detail::zero_fill<Self::ncu>(av);
    }

};

template <class Self, class T = double>
struct BoundaryDefaults : VolumeDefaults<Self, T> {
    // ---- Boundary terms (default: zero) ----

    // Boundary kernels: pointwise functions seen by every kernel call from
    // <exasim/kernels/boundary.hpp>. Args (after ub/fb output and `ib` tag):
    //   x[nd], uq[Nq], v[nco], w[ncw], uh[ncu], n[nd], tau[ncu], mu[nparam], uinf, t
    KOKKOS_INLINE_FUNCTION static
    void ubou(T ub[], int /*ib*/,
              const T /*x*/[],  const T /*uq*/[],
              const T /*v*/[],  const T /*w*/[],  const T /*uh*/[],
              const T /*n*/[],  const T /*tau*/[],
              const T /*mu*/[], const T /*uinf*/[], T /*t*/) {
        detail::zero_fill<Self::ncu>(ub);
    }

    KOKKOS_INLINE_FUNCTION static
    void fbou(T fb[], int /*ib*/,
              const T /*x*/[],  const T /*uq*/[],
              const T /*v*/[],  const T /*w*/[],  const T /*uh*/[],
              const T /*n*/[],  const T /*tau*/[],
              const T /*mu*/[], const T /*uinf*/[], T /*t*/) {
        detail::zero_fill<Self::ncu>(fb);
    }

    // HDG boundary residual — distinct from `fbou`.
    //
    // In every example PDE in apps/, `Fbou` (LDG path) and `FbouHdg`
    // (HDG path) are *different math*: `Fbou` is the boundary flux
    // contribution in a volume residual; `FbouHdg` is the trace-side
    // residual that pins the hybrid trace `uhat`. Same signature
    // shape; the HDG kernels in <exasim/kernels/boundary.hpp> route
    // through `fbou_hdg` and its three companion Jacobians:
    //   fbou_hdg_jac_uq, fbou_hdg_jac_w, fbou_hdg_jac_uh
    KOKKOS_INLINE_FUNCTION static
    void fbou_hdg(T fb[], int /*ib*/,
                  const T /*x*/[],  const T /*uq*/[],
                  const T /*v*/[],  const T /*w*/[],  const T /*uh*/[],
                  const T /*n*/[],  const T /*tau*/[],
                  const T /*mu*/[], const T /*uinf*/[], T /*t*/) {
        detail::zero_fill<Self::ncu>(fb);
    }

    // Interface kernels see both sides (uq1, uq2, v1, v2, w1, w2) plus the
    // trace uh, normal n, stabilization tau, and physics params mu. uinf is
    // pointer-passed and may be nullptr — model methods that need
    // free-stream values dereference at their own risk.
    KOKKOS_INLINE_FUNCTION static
    void fhat(T fh[],     const T /*x*/[],
              const T /*uq1*/[],  const T /*uq2*/[],
              const T /*v1*/[],   const T /*v2*/[],
              const T /*w1*/[],   const T /*w2*/[],
              const T /*uh*/[],   const T /*n*/[],   const T /*tau*/[],
              const T /*mu*/[],   const T /*uinf*/[], T /*t*/) {
        detail::zero_fill<Self::ncu>(fh);
    }

    KOKKOS_INLINE_FUNCTION static
    void uhat(T uh[],     const T /*x*/[],
              const T /*uq1*/[],  const T /*uq2*/[],
              const T /*v1*/[],   const T /*v2*/[],
              const T /*w1*/[],   const T /*w2*/[],
              const T /*trace*/[],const T /*n*/[],   const T /*tau*/[],
              const T /*mu*/[],   const T /*uinf*/[], T /*t*/) {
        detail::zero_fill<Self::ncu>(uh);
    }

    KOKKOS_INLINE_FUNCTION static
    void stab(T tau[],    const T /*x*/[],
              const T /*uq1*/[],  const T /*uq2*/[],
              const T /*v1*/[],   const T /*v2*/[],
              const T /*w1*/[],   const T /*w2*/[],
              const T /*uh*/[],   const T /*n*/[],   const T /*tau_in*/[],
              const T /*mu*/[],   const T /*uinf*/[], T /*t*/) {
        for (int k = 0; k < Self::ncu; ++k) tau[k] = 1.0;
    }

};

template <class Self, class T = double>
struct EoSDefaults : BoundaryDefaults<Self, T> {
    // ---- Equation of state (default: identity / zero) ----

    KOKKOS_INLINE_FUNCTION static
    void eos(T e[],
             const T /*x*/[],  const T /*uq*/[],
             const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
             const T /*uinf*/[], T /*t*/) {
        detail::zero_fill<Self::ncu>(e);
    }

    KOKKOS_INLINE_FUNCTION static
    void eos_du(T ed[],
                const T /*x*/[],  const T /*uq*/[],
                const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                const T /*uinf*/[], T /*t*/) {
        constexpr int Nq = Self::ncu * (1 + Self::nd);
        for (int k = 0; k < Self::ncu * Nq; ++k) ed[k] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void eos_dw(T ew[],
                const T /*x*/[],  const T /*uq*/[],
                const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                const T /*uinf*/[], T /*t*/) {
        if constexpr (Self::ncw > 0) {
            for (int k = 0; k < Self::ncu * Self::ncw; ++k) ew[k] = 0.0;
        }
    }

};

template <class Self, class T = double>
struct InitDefaults : EoSDefaults<Self, T> {
    // ---- Initial conditions (default: zero, except `initu` which is required) ----

    KOKKOS_INLINE_FUNCTION static
    void initq(T q[], const T[], const T[], const T[]) {
        detail::zero_fill<Self::ncu * Self::nd>(q);
    }

    KOKKOS_INLINE_FUNCTION static
    void initwdg(T w[], const T[], const T[], const T[]) {
        if constexpr (Self::ncw > 0) detail::zero_fill<Self::ncw>(w);
    }

    KOKKOS_INLINE_FUNCTION static
    void initudg(T udg[], const T[], const T[], const T[]) {
        constexpr int Nq = Self::ncu * (1 + Self::nd);
        detail::zero_fill<Nq>(udg);
    }

    KOKKOS_INLINE_FUNCTION static
    void initodg(T odg[], const T[], const T[], const T[]) {
        // No default; `nco` is determined by the discretization, not by Self.
        // Concrete kernels pass the correct count in.
        (void)odg;
    }

};

template <class Self, class T = double>
struct OutputDefaults : InitDefaults<Self, T> {
    // ---- Visualization, QoI, monitor/output (default: empty / zero) ----
    //
    // Volume-shape methods (`vis_*`, `qoi_volume`, `monitor`, `output`)
    // take the same args as `flux`/`source`:
    //   (out, x, uq, w, mu, uinf, t)
    // `qoi_boundary` adds an `ib` tag and the boundary-trace args
    // (uh, n, tau).

    KOKKOS_INLINE_FUNCTION static
    void vis_scalars(T[], const T /*x*/[], const T /*uq*/[],
                     const T /*v*/[], const T /*w*/[], const T /*mu*/[],
                     const T /*uinf*/[], T /*t*/) { }

    KOKKOS_INLINE_FUNCTION static
    void vis_vectors(T[], const T /*x*/[], const T /*uq*/[],
                     const T /*v*/[], const T /*w*/[], const T /*mu*/[],
                     const T /*uinf*/[], T /*t*/) { }

    KOKKOS_INLINE_FUNCTION static
    void vis_tensors(T[], const T /*x*/[], const T /*uq*/[],
                     const T /*v*/[], const T /*w*/[], const T /*mu*/[],
                     const T /*uinf*/[], T /*t*/) { }

    KOKKOS_INLINE_FUNCTION static
    void qoi_volume(T[], const T /*x*/[], const T /*uq*/[],
                    const T /*v*/[], const T /*w*/[], const T /*mu*/[],
                    const T /*uinf*/[], T /*t*/) { }

    KOKKOS_INLINE_FUNCTION static
    void qoi_boundary(T[], int /*ib*/,
                      const T /*x*/[],  const T /*uq*/[],
                      const T /*v*/[],  const T /*w*/[],  const T /*uh*/[],
                      const T /*n*/[],  const T /*tau*/[],
                      const T /*mu*/[], const T /*uinf*/[], T /*t*/) { }

    KOKKOS_INLINE_FUNCTION static
    void monitor(T[], const T /*x*/[], const T /*uq*/[],
                 const T /*v*/[], const T /*w*/[], const T /*mu*/[],
                 const T /*uinf*/[], T /*t*/) { }

    KOKKOS_INLINE_FUNCTION static
    void output(T[], const T /*x*/[], const T /*uq*/[],
                const T /*v*/[], const T /*w*/[], const T /*mu*/[],
                const T /*uinf*/[], T /*t*/) { }

};

template <class Self, class T = double>
struct HDGJacobianDefaults : OutputDefaults<Self, T> {
    // ---- HDG Jacobians (default: zero) ----
    //
    // For HDG models these MUST be overridden by the user — the
    // defaults give zero Jacobians, which makes the linearized system
    // zero and Newton fail to converge. The defaults exist only so
    // that LDG models (which don't call into the *_jac_* paths) don't
    // need to write empty stubs.

    // Volume Jacobians: same arg shape as the value methods —
    //   (out, x, uq, v, w, mu, uinf, t)
    KOKKOS_INLINE_FUNCTION static
    void flux_jac_uq(T f_uq[],
                     const T /*x*/[],  const T /*uq*/[],
                     const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                     const T /*uinf*/[], T /*t*/) {
        constexpr int Nq = Self::ncu * (1 + Self::nd);
        for (int k = 0; k < Self::ncu * Self::nd * Nq; ++k) f_uq[k] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void flux_jac_w(T f_w[],
                    const T /*x*/[],  const T /*uq*/[],
                    const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                    const T /*uinf*/[], T /*t*/) {
        if constexpr (Self::ncw > 0) {
            for (int k = 0; k < Self::ncu * Self::nd * Self::ncw; ++k) f_w[k] = 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION static
    void source_jac_uq(T s_uq[],
                       const T /*x*/[],  const T /*uq*/[],
                       const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                       const T /*uinf*/[], T /*t*/) {
        constexpr int Nq = Self::ncu * (1 + Self::nd);
        for (int k = 0; k < Self::ncu * Nq; ++k) s_uq[k] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void source_jac_w(T s_w[],
                      const T /*x*/[],  const T /*uq*/[],
                      const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                      const T /*uinf*/[], T /*t*/) {
        if constexpr (Self::ncw > 0) {
            for (int k = 0; k < Self::ncu * Self::ncw; ++k) s_w[k] = 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION static
    void materialstate_jac_uq(T state_uq[],
                              const T /*x*/[],  const T /*uq*/[],
                              const T /*v*/[],  const T /*w*/[],
                              const T /*mu*/[], const T /*uinf*/[],
                              T /*t*/) {
        constexpr int Nq = Self::ncu * (1 + Self::nd);
        for (int k = 0; k < Self::nmaterialstate * Nq; ++k) state_uq[k] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void materialstate_jac_w(T state_w[],
                             const T /*x*/[],  const T /*uq*/[],
                             const T /*v*/[],  const T /*w*/[],
                             const T /*mu*/[], const T /*uinf*/[],
                             T /*t*/) {
        if constexpr (Self::ncw > 0) {
            for (int k = 0; k < Self::nmaterialstate * Self::ncw; ++k) state_w[k] = 0.0;
        }
    }

    // ---- Auxiliary `w`-field source Jacobians (HDG w-equation) ----
    //
    // Companions to `sourcew` (the auxiliary-field forcing, output width
    // `ncw`). Consumed by `hdg_sourcew_kernel<M>` / `hdg_sourcewonly_kernel<M>`
    // in <exasim/kernels/sourcew.hpp>, mirroring the libpdemodel ABI
    // `HdgSourcew` (value + ∂sw/∂udg + ∂sw/∂w) and `HdgSourcewonly`
    // (value + ∂sw/∂w) driven by the implicit w-solve in
    // backend/Discretization/wequation.hpp. Only ever instantiated for
    // `ncw > 0` models, so single-domain / ncw==0 models keep these
    // zero-fill defaults (guarded by `if constexpr`) and pay nothing.
    //
    // Jacobian local layout is INPUT-index-outer, exactly like the volume
    // `f_udg`/`f_wdg` the generated `HdgSourcew` emits (the kernel scatters
    // it linearly and the wequation.hpp `SmallMatrixSolve` reads `nc`
    // right-hand-side columns of the `ncw × ncw` w-block):
    //   s_uq[j*ncw + o] = ∂sw[o]/∂uq[j]   (j in [0,Nq),  o in [0,ncw))
    //   s_w [j*ncw + o] = ∂sw[o]/∂w [j]   (j in [0,ncw), o in [0,ncw))
    KOKKOS_INLINE_FUNCTION static
    void sourcew_jac_uq(T sw_uq[],
                        const T /*x*/[],  const T /*uq*/[],
                        const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                        const T /*uinf*/[], T /*t*/) {
        if constexpr (Self::ncw > 0) {
            constexpr int Nq = Self::ncu * (1 + Self::nd);
            for (int k = 0; k < Self::ncw * Nq; ++k) sw_uq[k] = 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION static
    void sourcew_jac_w(T sw_w[],
                       const T /*x*/[],  const T /*uq*/[],
                       const T /*v*/[],  const T /*w*/[],  const T /*mu*/[],
                       const T /*uinf*/[], T /*t*/) {
        if constexpr (Self::ncw > 0) {
            for (int k = 0; k < Self::ncw * Self::ncw; ++k) sw_w[k] = 0.0;
        }
    }

    // LDG-path boundary Jacobians. These are not currently consumed by
    // any kernel — text2code's libpdemodel.hpp ABI has `KokkosFbou`
    // (LDG, value-only) and `HdgFbou` (HDG, value + Jacobians) but no
    // LDG-path Jacobians. Kept in the contract for completeness in
    // case a future numerical scheme needs them.
    KOKKOS_INLINE_FUNCTION static
    void fbou_jac_uq(T fb_uq[], int /*ib*/,
                     const T /*x*/[],  const T /*uq*/[],
                     const T /*v*/[],  const T /*w*/[],  const T /*uh*/[],
                     const T /*n*/[],  const T /*tau*/[],
                     const T /*mu*/[], const T /*uinf*/[], T /*t*/) {
        constexpr int Nq = Self::ncu * (1 + Self::nd);
        for (int k = 0; k < Self::ncu * Nq; ++k) fb_uq[k] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void fbou_jac_uh(T fb_uh[], int /*ib*/,
                     const T /*x*/[],  const T /*uq*/[],
                     const T /*v*/[],  const T /*w*/[],  const T /*uh*/[],
                     const T /*n*/[],  const T /*tau*/[],
                     const T /*mu*/[], const T /*uinf*/[], T /*t*/) {
        for (int k = 0; k < Self::ncu * Self::ncu; ++k) fb_uh[k] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void fbou_jac_w(T fb_w[], int /*ib*/,
                    const T /*x*/[],  const T /*uq*/[],
                    const T /*v*/[],  const T /*w*/[],  const T /*uh*/[],
                    const T /*n*/[],  const T /*tau*/[],
                    const T /*mu*/[], const T /*uinf*/[], T /*t*/) {
        if constexpr (Self::ncw > 0) {
            for (int k = 0; k < Self::ncu * Self::ncw; ++k) fb_w[k] = 0.0;
        }
    }

    // HDG-path boundary Jacobians (companions to `fbou_hdg`). These
    // ARE consumed by `hdg_fbou_kernel<M>` in
    // <exasim/kernels/boundary.hpp>: the user MUST override them to
    // get a converging HDG Newton solve. Defaults zero-fill so that
    // LDG-only models compile without writing empty stubs.
    KOKKOS_INLINE_FUNCTION static
    void fbou_hdg_jac_uq(T fb_uq[], int /*ib*/,
                         const T /*x*/[],  const T /*uq*/[],
                         const T /*v*/[],  const T /*w*/[],  const T /*uh*/[],
                         const T /*n*/[],  const T /*tau*/[],
                         const T /*mu*/[], const T /*uinf*/[], T /*t*/) {
        constexpr int Nq = Self::ncu * (1 + Self::nd);
        for (int k = 0; k < Self::ncu * Nq; ++k) fb_uq[k] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void fbou_hdg_jac_uh(T fb_uh[], int /*ib*/,
                         const T /*x*/[],  const T /*uq*/[],
                         const T /*v*/[],  const T /*w*/[],  const T /*uh*/[],
                         const T /*n*/[],  const T /*tau*/[],
                         const T /*mu*/[], const T /*uinf*/[], T /*t*/) {
        for (int k = 0; k < Self::ncu * Self::ncu; ++k) fb_uh[k] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void fbou_hdg_jac_w(T fb_w[], int /*ib*/,
                        const T /*x*/[],  const T /*uq*/[],
                        const T /*v*/[],  const T /*w*/[],  const T /*uh*/[],
                        const T /*n*/[],  const T /*tau*/[],
                        const T /*mu*/[], const T /*uinf*/[], T /*t*/) {
        if constexpr (Self::ncw > 0) {
            for (int k = 0; k < Self::ncu * Self::ncw; ++k) fb_w[k] = 0.0;
        }
    }

    // ---- Multi-domain HDG interface residuals (Fint / Fext) ----
    //
    // These mirror the libpdemodel ABI's `HdgFint` / `HdgFext` (see
    // backend/Discretization/KokkosDrivers.cpp and the text2code
    // `HdgFint.cpp` / `HdgFext.cpp`). They are only ever instantiated for
    // models with `has_external_coupling == true` (the `fint_kernel<M>` /
    // `fext_kernel<M>` calls in <exasim/kernels/boundary.hpp> are gated
    // on it), so single-domain models keep these zero-fill defaults and
    // pay nothing.
    //
    // `fint` is the HDG trace residual on an internal model interface;
    // `fext` adds the neighbour model's injected trace `uext` (SoA, like
    // `uh`) so the user writes coupling residuals such as
    // `fb[0] = uext[0] - uh[0]`. The output width is `nfint_v<Self>` /
    // `nfext_v<Self>` (defaults to `ncu`); `uext` has `ncuext_v<Self>`
    // components.
    //
    // Jacobian local layout is TRACE/INPUT-index-outer to match the ABI
    // byte layout exactly (kernel scatters it linearly):
    //   fb_uq[j*nf + o] = ∂fb[o]/∂uq[j]   (j in [0,Nq), o in [0,nf))
    //   fb_w [j*nf + o] = ∂fb[o]/∂w [j]
    //   fb_uh[j*nf + o] = ∂fb[o]/∂uh[j]
    // (This differs from `fbou_hdg`'s output-index-outer convention; the
    // Fint/Fext generated kernels emit input-outer, so we follow them.)

    KOKKOS_INLINE_FUNCTION static
    void fint(double fb[], int /*ib*/,
              const double /*x*/[],  const double /*uq*/[],
              const double /*v*/[],  const double /*w*/[],  const double /*uh*/[],
              const double /*n*/[],  const double /*tau*/[],
              const double /*mu*/[], const double /*uinf*/[], double /*t*/) {
        detail::zero_fill<Self::ncu>(fb);
    }

    KOKKOS_INLINE_FUNCTION static
    void fint_jac_uq(double fb_uq[], int /*ib*/,
                     const double /*x*/[],  const double /*uq*/[],
                     const double /*v*/[],  const double /*w*/[],  const double /*uh*/[],
                     const double /*n*/[],  const double /*tau*/[],
                     const double /*mu*/[], const double /*uinf*/[], double /*t*/) {
        constexpr int Nq = Self::ncu * (1 + Self::nd);
        for (int k = 0; k < Self::ncu * Nq; ++k) fb_uq[k] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void fint_jac_w(double fb_w[], int /*ib*/,
                    const double /*x*/[],  const double /*uq*/[],
                    const double /*v*/[],  const double /*w*/[],  const double /*uh*/[],
                    const double /*n*/[],  const double /*tau*/[],
                    const double /*mu*/[], const double /*uinf*/[], double /*t*/) {
        if constexpr (Self::ncw > 0) {
            for (int k = 0; k < Self::ncu * Self::ncw; ++k) fb_w[k] = 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION static
    void fint_jac_uh(double fb_uh[], int /*ib*/,
                     const double /*x*/[],  const double /*uq*/[],
                     const double /*v*/[],  const double /*w*/[],  const double /*uh*/[],
                     const double /*n*/[],  const double /*tau*/[],
                     const double /*mu*/[], const double /*uinf*/[], double /*t*/) {
        for (int k = 0; k < Self::ncu * Self::ncu; ++k) fb_uh[k] = 0.0;
    }

    // `fext` and its Jacobians take the extra `uext` argument (the
    // neighbour trace) between `n` and `tau`, matching the ABI
    // `HdgFext(..., nlg, uext, tau, ...)`.
    KOKKOS_INLINE_FUNCTION static
    void fext(double fb[], int /*ib*/,
              const double /*x*/[],  const double /*uq*/[],
              const double /*v*/[],  const double /*w*/[],  const double /*uh*/[],
              const double /*n*/[],  const double /*uext*/[], const double /*tau*/[],
              const double /*mu*/[], const double /*uinf*/[], double /*t*/) {
        detail::zero_fill<Self::ncu>(fb);
    }

    KOKKOS_INLINE_FUNCTION static
    void fext_jac_uq(double fb_uq[], int /*ib*/,
                     const double /*x*/[],  const double /*uq*/[],
                     const double /*v*/[],  const double /*w*/[],  const double /*uh*/[],
                     const double /*n*/[],  const double /*uext*/[], const double /*tau*/[],
                     const double /*mu*/[], const double /*uinf*/[], double /*t*/) {
        constexpr int Nq = Self::ncu * (1 + Self::nd);
        for (int k = 0; k < Self::ncu * Nq; ++k) fb_uq[k] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void fext_jac_w(double fb_w[], int /*ib*/,
                    const double /*x*/[],  const double /*uq*/[],
                    const double /*v*/[],  const double /*w*/[],  const double /*uh*/[],
                    const double /*n*/[],  const double /*uext*/[], const double /*tau*/[],
                    const double /*mu*/[], const double /*uinf*/[], double /*t*/) {
        if constexpr (Self::ncw > 0) {
            for (int k = 0; k < Self::ncu * Self::ncw; ++k) fb_w[k] = 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION static
    void fext_jac_uh(double fb_uh[], int /*ib*/,
                     const double /*x*/[],  const double /*uq*/[],
                     const double /*v*/[],  const double /*w*/[],  const double /*uh*/[],
                     const double /*n*/[],  const double /*uext*/[], const double /*tau*/[],
                     const double /*mu*/[], const double /*uinf*/[], double /*t*/) {
        for (int k = 0; k < Self::ncu * Self::ncu; ++k) fb_uh[k] = 0.0;
    }
    // (fhat_jac_*, stab_jac_*, eos_jac_* follow the same shape; added
    // when the corresponding kernel template is wired up.)
};

// Compose the per-concern mixins into the public CRTP base. Users still derive from
// ModelDefaults<Self> and inherit every default; each default now lives in the mixin for
// its concern (mirrors the commonstruct -> per-concern-struct split). A future refinement
// can have a kernel/class derive a constraint from just the mixin it needs.
template <class Self, class T = double>
struct ModelDefaults : HDGJacobianDefaults<Self, T> {};

// ---- Interface-coupling traits (read by kernels; defaults handled here) ----
//
// A model may omit `nfint` / `nfext` / `ncuext`; the trait then falls
// back to `M::ncu`. `has_external_coupling` is always present on models
// that inherit `ModelDefaults`, but the trait tolerates ones that do not.
namespace detail {

template <class M, class = void> struct nfint_trait  { static constexpr int value = M::ncu; };
template <class M> struct nfint_trait<M, std::void_t<decltype(M::nfint)>>   { static constexpr int value = M::nfint; };

template <class M, class = void> struct nfext_trait  { static constexpr int value = M::ncu; };
template <class M> struct nfext_trait<M, std::void_t<decltype(M::nfext)>>   { static constexpr int value = M::nfext; };

template <class M, class = void> struct ncuext_trait { static constexpr int value = M::ncu; };
template <class M> struct ncuext_trait<M, std::void_t<decltype(M::ncuext)>> { static constexpr int value = M::ncuext; };

template <class M, class = void> struct ext_coupling_trait { static constexpr bool value = false; };
template <class M> struct ext_coupling_trait<M, std::void_t<decltype(M::has_external_coupling)>>
    { static constexpr bool value = M::has_external_coupling; };

} // namespace detail

template <class M> inline constexpr int  nfint_v  = detail::nfint_trait<M>::value;
template <class M> inline constexpr int  nfext_v  = detail::nfext_trait<M>::value;
template <class M> inline constexpr int  ncuext_v = detail::ncuext_trait<M>::value;
template <class M> inline constexpr bool has_external_coupling_v = detail::ext_coupling_trait<M>::value;

// ===========================================================================
// Multi-domain HDG interface coupling — IMPLEMENTED (v2, array-sourced uext)
// ===========================================================================
//
// `apps/poisson/poisson2d/pdemodel2.txt` and the `HdgFint*` / `HdgFext*`
// libpdemodel.hpp ABI symbols define a multi-domain HDG path now mirrored
// by the header-only (concrete-template) port:
//
//   M::fint        (out, ib, x, uq, v, w, uh, n, tau, mu, uinf, t)
//   M::fint_jac_uq (...)
//   M::fint_jac_w  (...)
//   M::fint_jac_uh (...)
//   M::fext        (out, ib, x, uq, v, w, uh, n, uext, tau, mu, uinf, t)
//   M::fext_jac_uq (...)
//   M::fext_jac_w  (...)
//   M::fext_jac_uh (...)
//
// (The `v`/odg argument is carried through to match `fbou_hdg`'s shape;
// the ABI kernels take odg as well.)
//
// `Fint` is the HDG trace residual on internal model interfaces; `Fext`
// adds an extra input array `uext` carrying the trace from a
// neighbouring PDE model so the user can write coupling residuals like
// `fb[0] = uext[0] - uh[0]`. See the zero-fill defaults + Jacobian layout
// note in `ModelDefaults` above, the `has_external_coupling` trait, and
// the `fint_kernel<M>` / `fext_kernel<M>` templates in
// <exasim/kernels/boundary.hpp>. The kernels are gated behind
// `has_external_coupling` (default false) so single-domain models keep a
// byte-identical residual/Jacobian.
//
// Sourcing of `uext` (implemented): the ARRAY form, exactly like the ABI
// `HdgFext`. The coupler writes the neighbour trace into `disc.sol.uext`
// (SoA, `ncuext_v<M>` components per face point); the HDG boundary
// assembly reads it under `disc.common.couplingparams.FextCall` /
// `common.FextCall` and passes it to `fext_kernel<M>` as a plain array.
//
// Further extension (NOT implemented): a `fext_kernel<M, OtherModel>`
// overload that pulls the neighbour trace directly from a foreign
// `CSolution<OtherModel>` instead of a pre-populated array, which would
// let two templated models be orchestrated without the coupler staging
// `uext` into `disc.sol.uext`. The array form above already covers the
// real driver (isoq2d CHT).
//
// Also deferred to v2:
//   - `hessian` declarations from text2code's DSL: second-derivative
//     methods for Galerkin schemes that need them. None of the
//     reference apps populate `hessian …`.
//   - Multi-model dispatch via runtime `modelnumber` (text2code's
//     KokkosFlux1/KokkosFlux2 split). In the templated world a user
//     orchestrates by instantiating `CSolution<Model1>`,
//     `CSolution<Model2>` separately, so kernels see only one Model
//     at a time and `modelnumber` becomes a no-op.

// ---- Compile-time concept-style checks ----
//
// Static asserts that fire if a user struct is missing the required
// surface. Used inside kernel templates: `static_assert(is_model_v<M>, …);`

namespace detail {

template <class, class = void> struct has_required_constants : std::false_type {};
template <class M>
struct has_required_constants<M,
    std::void_t<
        decltype(M::nd), decltype(M::ncu),
        decltype(M::ncw), decltype(M::nparam)
    >> : std::true_type {};

} // namespace detail

template <class M>
inline constexpr bool is_model_v = detail::has_required_constants<M>::value;

// ---- Per-capability concept checks (one per kernel concern) ----
//
// Each kernel in <exasim/kernels/*.hpp> consumes only ONE concern's surface (see the mixin
// chain above). These narrow traits let a kernel static_assert exactly the capability it uses
// instead of the whole model, so a model missing e.g. `fbou` fails with a clear concept message
// at the boundary kernel rather than a deep member-lookup error. Because ModelDefaults supplies
// zero-fill defaults for every optional method, a model deriving ModelDefaults satisfies all of
// these — their value is precise diagnostics + self-documenting each kernel's required surface.
namespace detail {
#define EXASIM_HAS_METHOD(name) \
    template <class, class = void> struct has_##name : std::false_type {}; \
    template <class M> struct has_##name<M, std::void_t<decltype(&M::name)>> : std::true_type {}
EXASIM_HAS_METHOD(flux);        EXASIM_HAS_METHOD(source);      EXASIM_HAS_METHOD(sourcew);
EXASIM_HAS_METHOD(materialstate);
EXASIM_HAS_METHOD(tdfunc);      EXASIM_HAS_METHOD(avfield);     EXASIM_HAS_METHOD(eos);
EXASIM_HAS_METHOD(eos_du);      EXASIM_HAS_METHOD(eos_dw);      EXASIM_HAS_METHOD(fbou);
EXASIM_HAS_METHOD(ubou);        EXASIM_HAS_METHOD(fbou_hdg);    EXASIM_HAS_METHOD(fhat);
EXASIM_HAS_METHOD(uhat);        EXASIM_HAS_METHOD(stab);        EXASIM_HAS_METHOD(initu);
EXASIM_HAS_METHOD(qoi_volume);  EXASIM_HAS_METHOD(qoi_boundary);EXASIM_HAS_METHOD(monitor);
EXASIM_HAS_METHOD(output);      EXASIM_HAS_METHOD(vis_scalars); EXASIM_HAS_METHOD(vis_vectors);
EXASIM_HAS_METHOD(vis_tensors);
#undef EXASIM_HAS_METHOD
} // namespace detail

template <class M> inline constexpr bool is_flux_model_v        = is_model_v<M> && detail::has_flux<M>::value;
template <class M> inline constexpr bool is_source_model_v      = is_model_v<M> && detail::has_source<M>::value;
template <class M> inline constexpr bool is_sourcew_model_v     = is_model_v<M> && detail::has_sourcew<M>::value;
template <class M> inline constexpr bool is_materialstate_model_v = is_model_v<M> && detail::has_materialstate<M>::value;
template <class M> inline constexpr bool is_tdfunc_model_v      = is_model_v<M> && detail::has_tdfunc<M>::value;
template <class M> inline constexpr bool is_avfield_model_v     = is_model_v<M> && detail::has_avfield<M>::value;
template <class M> inline constexpr bool is_eos_model_v         = is_model_v<M> && detail::has_eos<M>::value && detail::has_eos_du<M>::value && detail::has_eos_dw<M>::value;
template <class M> inline constexpr bool is_boundary_model_v    = is_model_v<M> && detail::has_fbou<M>::value && detail::has_ubou<M>::value;
template <class M> inline constexpr bool is_hdg_boundary_model_v= is_model_v<M> && detail::has_fbou_hdg<M>::value;
template <class M> inline constexpr bool is_interface_model_v   = is_model_v<M> && detail::has_fhat<M>::value && detail::has_uhat<M>::value && detail::has_stab<M>::value;
template <class M> inline constexpr bool is_init_model_v        = is_model_v<M> && detail::has_initu<M>::value;
template <class M> inline constexpr bool is_qoi_model_v         = is_model_v<M> && detail::has_qoi_volume<M>::value && detail::has_qoi_boundary<M>::value;
template <class M> inline constexpr bool is_output_model_v      = is_model_v<M> && detail::has_monitor<M>::value && detail::has_output<M>::value;
template <class M> inline constexpr bool is_vis_model_v         = is_model_v<M> && detail::has_vis_scalars<M>::value && detail::has_vis_vectors<M>::value && detail::has_vis_tensors<M>::value;

} // namespace exasim
