// SPDX-License-Identifier: see Exasim LICENSE
//
// heatsolid2d.hpp — transient linear heat conduction for the isoq2d CHT SOLID side.
//
// Physics (the isoq2d CHT solid thermal domain, Summit's job in the legacy app):
//
//     rho*cp dT/dt = div(k grad T) + f
//
// with material properties from CHEFSI-apps/.../isoq2d_cht-with-kitesurf/thermal.yaml
// (rho=280, cp=2000, k=0.5). HDG discretization, one primary unknown T.
//
// Written by hand against <exasim/model.hpp> (no DSL / codegen), structurally
// following tests/models/poisson2d.hpp. Differences from Poisson2D:
//   - conductivity k = mu[0] (Poisson2D's mu is a generic diffusivity; same math),
//   - transient MASS rho*cp folded into du/dt via tdfunc (m[0] = mu[1]*mu[2]),
//   - a manufactured source so the STEADY solution is a known eigenmode phi,
//   - TWO boundary types dispatched on `ib`:
//       ib==1  Dirichlet  (T = 0 on the slab edges; phi vanishes there),
//       ib==2  interface heat-flux (Neumann): imposes a prescribed wall normal
//              flux qn = mu[3] -- the CHT COUPLING HOOK (what the fluid supplies).
//
// Parameters (physicsparam -> mu[]):
//   mu[0] = k        thermal conductivity
//   mu[1] = rho      density
//   mu[2] = cp       specific heat
//   mu[3] = qn       interface wall heat flux (Neumann BC ib==2); 0 for the MMS run
//   mu[4] = Lx       slab width  (manufactured-solution geometry)
//   mu[5] = Ly       slab height
//   mu[6] = srcScale 1 => manufactured source on (MMS run); 0 => source off (flux demo)
//
// Manufactured solution (MMS, used for verification on a rectangular slab):
//   phi(x,y) = sin(pi x/Lx) sin(pi y/Ly)             (Dirichlet eigenmode, 0 on edges)
//   gamma    = pi^2 (1/Lx^2 + 1/Ly^2),  -Lap(phi) = gamma phi
//   source   = srcScale * k * gamma * phi            => steady T = phi exactly
//   from T=0: exact transient  T = a(t) phi,  a(t) = 1 - e^{-lambda t},
//             lambda = k*gamma/(rho*cp)              (the rho*cp mass sets the RATE)

#pragma once

#include <exasim/model.hpp>

template <class T = double>
struct HeatSolid2DT : exasim::ModelDefaults<HeatSolid2DT<T>, T> {
    // ---- Compile-time configuration -------------------------------
    static constexpr int nd     = 2;   // spatial dimension
    static constexpr int ncu    = 1;   // # primary unknowns (temperature T)
    static constexpr int ncw    = 0;
    static constexpr int nco    = 0;
    static constexpr int nparam = 7;   // [k, rho, cp, qn, Lx, Ly, srcScale]

    static constexpr auto disc = exasim::Discretization::HDG;

    static constexpr int Nq = ncu * (1 + nd);   // = 3
    static constexpr T PI = 3.141592653589793;

    // ---- Volume terms ---------------------------------------------

    // Flux f = k grad T   (uq[1], uq[2] are the gradient components)
    KOKKOS_INLINE_FUNCTION static
    void flux(T f[], const T /*x*/[], const T uq[],
              const T /*v*/[], const T /*w*/[], const T mu[],
              const T /*uinf*/[], T /*t*/) {
        const T k = mu[0];
        f[0] = k * uq[1];
        f[1] = k * uq[2];
    }

    // Source s = srcScale * k * gamma * sin(pi x/Lx) sin(pi y/Ly)
    KOKKOS_INLINE_FUNCTION static
    void source(T s[], const T x[], const T /*uq*/[],
                const T /*v*/[], const T /*w*/[], const T mu[],
                const T /*uinf*/[], T /*t*/) {
        const T k  = mu[0];
        const T Lx = mu[4], Ly = mu[5], sc = mu[6];
        const T gamma = PI * PI * (1.0 / (Lx * Lx) + 1.0 / (Ly * Ly));
        s[0] = sc * k * gamma
             * Kokkos::sin(PI * x[0] / Lx) * Kokkos::sin(PI * x[1] / Ly);
    }

    // Transient mass: rho*cp dT/dt  ->  m[0] = rho*cp.
    KOKKOS_INLINE_FUNCTION static
    void tdfunc(T m[], const T /*x*/[], const T /*uq*/[],
                const T /*v*/[], const T /*w*/[], const T mu[],
                const T /*uinf*/[], T /*t*/) {
        m[0] = mu[1] * mu[2];   // rho * cp
    }

    // ---- Initial condition: cold solid ----------------------------
    KOKKOS_INLINE_FUNCTION static
    void initu(T ui[], const T /*x*/[], const T /*uinf*/[], const T /*mu*/[]) {
        ui[0] = 0.0;
    }

    // ---- HDG volume Jacobian df/duq -------------------------------
    // f_uq[j*(ncu*nd)+i] = df[i]/duq[j];  df[0]/duq[1]=k, df[1]/duq[2]=k.
    KOKKOS_INLINE_FUNCTION static
    void flux_jac_uq(T f_uq[], const T /*x*/[], const T /*uq*/[],
                     const T /*v*/[], const T /*w*/[], const T mu[],
                     const T /*uinf*/[], T /*t*/) {
        for (int i = 0; i < ncu * nd * Nq; ++i) f_uq[i] = 0.0;
        f_uq[1 * (ncu * nd) + 0] = mu[0];
        f_uq[2 * (ncu * nd) + 1] = mu[0];
    }

    // ---- HDG boundary trace residual (FbouHdg), dispatched on ib ---
    //   ib==1: Dirichlet T=0  ->  fb = tau*(0 - uhat) = -tau*uhat
    //   ib==2: interface flux ->  numerical flux fhat = qn, i.e.
    //          fb = f.n + tau*(T - uhat) - qn   (CHT coupling hook)
    KOKKOS_INLINE_FUNCTION static
    void fbou_hdg(T fb[], int ib,
                  const T x[], const T uq[],
                  const T v[], const T w[], const T uh[],
                  const T n[], const T tau[],
                  const T mu[], const T uinf[], T t) {
        if (ib == 2) {
            T f_local[ncu * nd];
            flux(f_local, x, uq, v, w, mu, uinf, t);
            const T qn = mu[3];
            fb[0] = f_local[0] * n[0] + f_local[1] * n[1]
                  + tau[0] * (uq[0] - uh[0]) - qn;
        } else {                                   // ib==1 (default) Dirichlet
            fb[0] = -tau[0] * uh[0];
        }
    }

    // dFbouHdg/duq
    KOKKOS_INLINE_FUNCTION static
    void fbou_hdg_jac_uq(T fb_uq[], int ib,
                         const T /*x*/[], const T /*uq*/[],
                         const T /*v*/[], const T /*w*/[], const T /*uh*/[],
                         const T n[], const T tau[],
                         const T mu[], const T /*uinf*/[], T /*t*/) {
        for (int j = 0; j < ncu * Nq; ++j) fb_uq[j] = 0.0;
        if (ib == 2) {
            fb_uq[0] = tau[0];       // d/duq[0] of tau*uq[0]
            fb_uq[1] = mu[0] * n[0]; // d/duq[1] of f.n
            fb_uq[2] = mu[0] * n[1]; // d/duq[2] of f.n
        }
    }

    // dFbouHdg/duh = -tau  (both boundary types)
    KOKKOS_INLINE_FUNCTION static
    void fbou_hdg_jac_uh(T fb_uh[], int /*ib*/,
                         const T /*x*/[], const T /*uq*/[],
                         const T /*v*/[], const T /*w*/[], const T /*uh*/[],
                         const T /*n*/[], const T tau[],
                         const T /*mu*/[], const T /*uinf*/[], T /*t*/) {
        fb_uh[0] = -tau[0];
    }

    // ---- LDG Fbou / Ubou (present for the model contract) ---------
    KOKKOS_INLINE_FUNCTION static
    void fbou(T fb[], int /*ib*/,
              const T x[], const T uq[],
              const T v[], const T w[], const T uh[],
              const T n[], const T tau[],
              const T mu[], const T uinf[], T t) {
        T f_local[ncu * nd];
        flux(f_local, x, uq, v, w, mu, uinf, t);
        fb[0] = f_local[0] * n[0] + f_local[1] * n[1] + tau[0] * (uq[0] - uh[0]);
    }

    KOKKOS_INLINE_FUNCTION static
    void ubou(T ub[], int /*ib*/,
              const T /*x*/[], const T /*uq*/[],
              const T /*v*/[], const T /*w*/[], const T /*uh*/[],
              const T /*n*/[], const T /*tau*/[],
              const T /*mu*/[], const T /*uinf*/[], T /*t*/) {
        ub[0] = 0.0;
    }

    // ---- Post-processing views -----------------------------------
    // QoI: L2 distance to the manufactured STEADY field phi (used by eval_qoi).
    struct QoI {
        static constexpr int nd=HeatSolid2DT<T>::nd, ncu=HeatSolid2DT<T>::ncu,
                             ncw=HeatSolid2DT<T>::ncw, nco=HeatSolid2DT<T>::nco,
                             nparam=HeatSolid2DT<T>::nparam;
        KOKKOS_INLINE_FUNCTION static
        void qoi_volume(T s[], const T x[], const T uq[],
                        const T /*v*/[], const T /*w*/[], const T mu[],
                        const T /*uinf*/[], T /*t*/) {
            const T Lx = mu[4], Ly = mu[5];
            const T phi = Kokkos::sin(HeatSolid2DT<T>::PI * x[0] / Lx)
                        * Kokkos::sin(HeatSolid2DT<T>::PI * x[1] / Ly);
            s[0] = (uq[0] - phi) * (uq[0] - phi);
            s[1] = uq[0];
        }
        KOKKOS_INLINE_FUNCTION static
        void qoi_boundary(T fb[], int /*ib*/,
                          const T x[], const T uq[],
                          const T v[], const T w[], const T uh[],
                          const T n[], const T tau[],
                          const T mu[], const T uinf[], T t) {
            T f_local[ncu * nd];
            HeatSolid2DT<T>::flux(f_local, x, uq, v, w, mu, uinf, t);
            fb[0] = f_local[0] * n[0] + f_local[1] * n[1] + tau[0] * (uq[0] - uh[0]);
        }
    };

    struct Vis {
        static constexpr int nd=HeatSolid2DT<T>::nd, ncu=HeatSolid2DT<T>::ncu,
                             ncw=HeatSolid2DT<T>::ncw, nco=HeatSolid2DT<T>::nco,
                             nparam=HeatSolid2DT<T>::nparam;
        KOKKOS_INLINE_FUNCTION static
        void vis_scalars(T s[], const T /*x*/[], const T uq[],
                         const T /*v*/[], const T /*w*/[], const T /*mu*/[],
                         const T /*uinf*/[], T /*t*/) {
            s[0] = uq[0];             // temperature T
            s[1] = uq[1] + uq[2];     // grad sum (diagnostic)
        }
        KOKKOS_INLINE_FUNCTION static
        void vis_vectors(T s[], const T /*x*/[], const T uq[],
                         const T /*v*/[], const T /*w*/[], const T /*mu*/[],
                         const T /*uinf*/[], T /*t*/) {
            s[0] = uq[1];             // heat-flux direction (grad T)
            s[1] = uq[2];
        }
        KOKKOS_INLINE_FUNCTION static
        void vis_tensors(T[], const T[], const T[], const T[],
                         const T[], const T[], const T[], T) {}
    };
};

using HeatSolid2D = HeatSolid2DT<double>;
