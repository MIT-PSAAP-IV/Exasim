// SPDX-License-Identifier: see Exasim LICENSE
//
// poisson2d.hpp — Poisson 2D PDE model, written by hand against the
// `<exasim/model.hpp>` contract. No DSL, no autodiff, no codegen.
//
// SHARED across the Poisson2D consumers (tests/consumers/operators, tests/consumers/facade,
// tests/petsc/petsc_poisson, tests/petsc/petsc_heat) -- one canonical copy, pulled in via each
// consumer's `../../models` include path. (The builtin/external-model consumers use a different
// model and keep their own.)
//
// Solves   -∇·(μ∇u) = 2π² sin(πx) sin(πy)   on the unit square
//                u = 0                       on the boundary
//
// Exact solution:   u(x,y) = sin(πx) sin(πy)
// Physics param:    μ = mu[0]  (set to 1 in pdeapp.txt -> physicsparam = [1])
//
// The math here matches what text2code's SymEngine pipeline produces
// for apps/poisson/poisson2d/pdemodel.txt, byte-for-byte in evaluation
// order, so the templated path can reproduce the existing baseline
// numerics bit-for-bit when the runtime context is the same.

#pragma once

#include <exasim/model.hpp>

template <class T = double>
struct Poisson2DT : exasim::ModelDefaults<Poisson2DT<T>> {
    // ---- Compile-time configuration -------------------------------
    static constexpr int nd     = 2;   // spatial dimension
    static constexpr int ncu    = 1;   // # primary unknowns
    static constexpr int ncw    = 0;   // # auxiliary scalar fields
    static constexpr int nco    = 0;   // # other DG fields (`v` / `odg`)
    static constexpr int nparam = 1;   // # physics parameters

    static constexpr auto disc = exasim::Discretization::HDG;

    // Derived: Nq = ncu*(1 + nd) = 3
    static constexpr int Nq = ncu * (1 + nd);

    // The post-processing views below carry just these sizes (referenced, single source)
    // + their own methods, so eval_qoi/write_vtk template on the capability, not the model.

    // ---- Volume terms ---------------------------------------------

    // Flux f = μ ∇u
    KOKKOS_INLINE_FUNCTION static
    void flux(T f[], const T /*x*/[], const T uq[],
              const T /*v*/[], const T /*w*/[], const T mu[],
              const T /*uinf*/[], T /*t*/) {
        const T mu0  = mu[0];
        const T udg2 = uq[1];
        const T udg3 = uq[2];
        f[0] = mu0 * udg2;
        f[1] = mu0 * udg3;
    }

    // Source s = 2π² sin(πx) sin(πy)
    // The constant 1.973920880217872E+1 = 2*π² and matches text2code's
    // SymEngine output exactly.
    KOKKOS_INLINE_FUNCTION static
    void source(T s[], const T x[], const T /*uq*/[],
                const T /*v*/[], const T /*w*/[], const T /*mu*/[],
                const T /*uinf*/[], T /*t*/) {
        const T xdg1 = x[0];
        const T xdg2 = x[1];
        s[0] = Kokkos::sin(xdg1 * 3.141592653589793)
             * Kokkos::sin(xdg2 * 3.141592653589793)
             * 1.973920880217872E+1;
    }

    // tdfunc: empty body in text2code's KokkosTdfunc1 (steady problem,
    // torder=1, never called). The ModelDefaults<Poisson2D>::tdfunc
    // identity-fill default is fine — it's just not exercised at
    // runtime for this app.

    // ---- Required initial condition -------------------------------

    KOKKOS_INLINE_FUNCTION static
    void initu(T ui[], const T /*x*/[],
               const T /*uinf*/[], const T /*mu*/[]) {
        ui[0] = 0.0;
    }

    // ---- HDG path -------------------------------------------------

    // Volume-term Jacobian ∂f/∂uq:
    //
    // Layout (column-major; uq index is outer):
    //   f_uq[ j * (ncu*nd) + i ] = ∂f[i] / ∂uq[j]
    //
    // For Poisson 2D (ncu=1, nd=2, Nq=3 -> shape [ncu*nd=2 × Nq=3] = 6
    // entries):
    //   ∂f[0]/∂uq[1] = mu  ->  index = 1*2 + 0 = 2
    //   ∂f[1]/∂uq[2] = mu  ->  index = 2*2 + 1 = 5
    // All others zero.  This matches text2code's HdgFlux output exactly.
    KOKKOS_INLINE_FUNCTION static
    void flux_jac_uq(T f_uq[], const T /*x*/[],
                     const T /*uq*/[],
                     const T /*v*/[], const T /*w*/[],
                     const T mu[], const T /*uinf*/[],
                     T /*t*/) {
        for (int k = 0; k < ncu * nd * Nq; ++k) f_uq[k] = 0.0;
        f_uq[1 * (ncu * nd) + 0] = mu[0];   // ∂f[0]/∂uq[1]
        f_uq[2 * (ncu * nd) + 1] = mu[0];   // ∂f[1]/∂uq[2]
    }

    // Source has no uq dependence — defaults zero-fill from ModelDefaults.

    // ---- HDG boundary residual (FbouHdg in pdemodel.txt) ----------
    //
    // FbouHdg = τ*(0 - uhat) = -τ * uhat   (Dirichlet via stabilization)
    KOKKOS_INLINE_FUNCTION static
    void fbou_hdg(T fb[], int /*ib*/,
                  const T /*x*/[],  const T /*uq*/[],
                  const T /*v*/[],  const T /*w*/[],  const T uh[],
                  const T /*n*/[],  const T tau[],
                  const T /*mu*/[], const T /*uinf*/[],
                  T /*t*/) {
        const T tau0  = tau[0];
        const T uhat0 = uh[0];
        fb[0] = -tau0 * uhat0;
    }

    // ∂fbou_hdg/∂uq = 0 (Poisson's HDG residual doesn't depend on uq).
    // ∂fbou_hdg/∂uh = -τ (1×1 block).
    KOKKOS_INLINE_FUNCTION static
    void fbou_hdg_jac_uh(T fb_uh[], int /*ib*/,
                         const T /*x*/[],  const T /*uq*/[],
                         const T /*v*/[],  const T /*w*/[],  const T /*uh*/[],
                         const T /*n*/[],  const T tau[],
                         const T /*mu*/[], const T /*uinf*/[],
                         T /*t*/) {
        fb_uh[0] = -tau[0];
    }

    // fbou_hdg_jac_uq, fbou_hdg_jac_w: zero — defaults from ModelDefaults.

    // ---- LDG boundary flux Fbou (used by some assembly paths) -----
    //
    // Fbou = f·n + τ*(uq[0] - uhat[0])  — exact form from pdemodel.txt
    KOKKOS_INLINE_FUNCTION static
    void fbou(T fb[], int /*ib*/,
              const T x[],  const T uq[],
              const T v[],  const T w[],  const T uh[],
              const T n[],  const T tau[],
              const T mu[], const T uinf[], T t) {
        T f_local[ncu * nd];
        flux(f_local, x, uq, v, w, mu, uinf, t);
        fb[0] = f_local[0] * n[0] + f_local[1] * n[1]
              + tau[0] * (uq[0] - uh[0]);
    }

    // ---- Ubou: zero Dirichlet -------------------------------------
    KOKKOS_INLINE_FUNCTION static
    void ubou(T ub[], int /*ib*/,
              const T /*x*/[],  const T /*uq*/[],
              const T /*v*/[],  const T /*w*/[],  const T /*uh*/[],
              const T /*n*/[],  const T /*tau*/[],
              const T /*mu*/[], const T /*uinf*/[],
              T /*t*/) {
        ub[0] = 0.0;
    }

    // ---- Post-processing views (NOT part of the operator surface) -------------
    // QoI and visualization are separate capability concerns. Each is a small struct that
    // carries the shared config (Poisson2DConfig -> satisfies is_model_v) plus ONLY its own
    // methods, so eval_qoi<Poisson2DT<T>::QoI> and write_vtk<Poisson2DT<T>::Vis> template on exactly
    // the piece they need -- never the whole physics model. (The discretization is still
    // built from Poisson2D, whose flux/source/fbou are the only thing the operators touch.)

    // Volume + boundary quantities of interest (is_qoi_model_v requires both methods).
    struct QoI {
        static constexpr int nd=Poisson2DT<T>::nd, ncu=Poisson2DT<T>::ncu, ncw=Poisson2DT<T>::ncw,
                             nco=Poisson2DT<T>::nco, nparam=Poisson2DT<T>::nparam;
        // s[0] = (u - u_exact)^2 with u_exact = sin(πx) sin(πy);  s[1] = u
        KOKKOS_INLINE_FUNCTION static
        void qoi_volume(T s[], const T x[], const T uq[],
                        const T /*v*/[], const T /*w*/[], const T /*mu*/[],
                        const T /*uinf*/[], T /*t*/) {
            const T t1 = 3.141592653589793;
            const T t2 = Kokkos::sin(t1 * x[0]);
            const T t3 = Kokkos::sin(t1 * x[1]);
            const T uexact = t2 * t3;
            s[0] = (uq[0] - uexact) * (uq[0] - uexact);
            s[1] = uq[0];
        }

        // Boundary-flux QoI: reuses the physics flux (f·n + τ(u - uhat)).
        KOKKOS_INLINE_FUNCTION static
        void qoi_boundary(T fb[], int /*ib*/,
                          const T x[],  const T uq[],
                          const T v[],  const T w[],  const T uh[],
                          const T n[],  const T tau[],
                          const T mu[], const T uinf[],
                          T t) {
            T f_local[ncu * nd];
            Poisson2DT<T>::flux(f_local, x, uq, v, w, mu, uinf, t);
            fb[0] = f_local[0] * n[0] + f_local[1] * n[1]
                  + tau[0] * (uq[0] - uh[0]);
        }
    };

    // Visualization fields. (is_vis_model_v also wants vis_tensors, but write_vtk only
    // instantiates the tensor path when nten>0, which this model never sets.)
    struct Vis {
        static constexpr int nd=Poisson2DT<T>::nd, ncu=Poisson2DT<T>::ncu, ncw=Poisson2DT<T>::ncw,
                             nco=Poisson2DT<T>::nco, nparam=Poisson2DT<T>::nparam;
        KOKKOS_INLINE_FUNCTION static
        void vis_scalars(T s[], const T /*x*/[], const T uq[],
                         const T /*v*/[], const T /*w*/[], const T /*mu*/[],
                         const T /*uinf*/[], T /*t*/) {
            s[0] = uq[0];
            s[1] = uq[1] + uq[2];
        }

        KOKKOS_INLINE_FUNCTION static
        void vis_vectors(T s[], const T /*x*/[], const T uq[],
                         const T /*v*/[], const T /*w*/[], const T /*mu*/[],
                         const T /*uinf*/[], T /*t*/) {
            s[0] = uq[1];
            s[1] = uq[2];
        }

        // unused here (this model sets no tensor fields) but required by is_vis_model_v;
        // no-op, matching ModelDefaults::vis_tensors.
        KOKKOS_INLINE_FUNCTION static
        void vis_tensors(T[], const T[], const T[], const T[],
                         const T[], const T[], const T[], T) {}
    };
};

using Poisson2D = Poisson2DT<double>;
