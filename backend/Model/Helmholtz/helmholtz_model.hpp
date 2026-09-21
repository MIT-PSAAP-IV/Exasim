#pragma once

#include <Kokkos_Core.hpp>
#include "modeldefaults.hpp"

// Internal scalar HDG model used by the optional frozen-AV Helmholtz filter.
// Exasim stores q=-grad(u).  With the mesh-dependent diffusion scale k_h in
// v[1], flux=k_h*q and source=s-u give u-div(k_h grad(u))=s.
template <int SpatialDimension>
struct HelmholtzModel : ModelDefaults<HelmholtzModel<SpatialDimension>> {
    static constexpr int nd = SpatialDimension;
    static constexpr int ncu = 1;
    static constexpr int ncw = 0;
    static constexpr int nco = 2; // v=(sensor source, C_h*sqrt(smoothed nodal jacobian))
    static constexpr int nparam = 0;
    static constexpr int ntau = 1;
    static constexpr int Nq = 1 + nd;

    KOKKOS_INLINE_FUNCTION static
    void flux(dstype f[], const dstype[], const dstype uq[], const dstype v[],
              const dstype[], const dstype[], const dstype[], dstype)
    {
        for (int d = 0; d < nd; ++d) f[d] = v[1] * uq[1 + d];
    }

    KOKKOS_INLINE_FUNCTION static
    void source(dstype f[], const dstype[], const dstype uq[], const dstype v[],
                const dstype[], const dstype[], const dstype[], dstype)
    {
        f[0] = v[0] - uq[0];
    }

    KOKKOS_INLINE_FUNCTION static
    void initu(dstype f[], const dstype[], const dstype[], const dstype[])
    {
        f[0] = 0.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void fbou(dstype f[], int, const dstype[], const dstype uq[], const dstype v[],
              const dstype[], const dstype uh[], const dstype n[], const dstype tau[],
              const dstype[], const dstype[], dstype)
    {
        dstype normal_flux = 0.0;
        for (int d = 0; d < nd; ++d) normal_flux += v[1] * uq[1 + d] * n[d];
        f[0] = normal_flux + tau[0] * (uq[0] - uh[0]);
    }

    KOKKOS_INLINE_FUNCTION static
    void fbou_hdg(dstype f[], int ib, const dstype x[], const dstype uq[], const dstype v[],
                  const dstype w[], const dstype uh[], const dstype n[], const dstype tau[],
                  const dstype mu[], const dstype uinf[], dstype t)
    {
        fbou(f, ib, x, uq, v, w, uh, n, tau, mu, uinf, t);
    }

    KOKKOS_INLINE_FUNCTION static
    void flux_jac_uq(dstype f[], const dstype[], const dstype[], const dstype v[],
                     const dstype[], const dstype[], const dstype[], dstype)
    {
        for (int i = 0; i < nd * Nq; ++i) f[i] = 0.0;
        // Exasim's generated HDG kernels flatten Jacobians with the output
        // index varying fastest: f_i,u_j is stored at i + nout*j.
        for (int d = 0; d < nd; ++d) f[d + nd * (1 + d)] = v[1];
    }

    KOKKOS_INLINE_FUNCTION static
    void source_jac_uq(dstype f[], const dstype[], const dstype[], const dstype[],
                       const dstype[], const dstype[], const dstype[], dstype)
    {
        for (int i = 0; i < Nq; ++i) f[i] = 0.0;
        f[0] = -1.0;
    }

    KOKKOS_INLINE_FUNCTION static
    void fbou_jac_uq(dstype f[], int, const dstype[], const dstype[], const dstype v[],
                     const dstype[], const dstype[], const dstype n[], const dstype tau[],
                     const dstype[], const dstype[], dstype)
    {
        for (int i = 0; i < Nq; ++i) f[i] = 0.0;
        f[0] = tau[0];
        for (int d = 0; d < nd; ++d) f[1 + d] = v[1] * n[d];
    }

    KOKKOS_INLINE_FUNCTION static
    void fbou_jac_uh(dstype f[], int, const dstype[], const dstype[], const dstype[],
                     const dstype[], const dstype[], const dstype[], const dstype tau[],
                     const dstype[], const dstype[], dstype)
    {
        f[0] = -tau[0];
    }

    KOKKOS_INLINE_FUNCTION static
    void fbou_hdg_jac_uq(dstype f[], int ib, const dstype x[], const dstype uq[], const dstype v[],
                         const dstype w[], const dstype uh[], const dstype n[], const dstype tau[],
                         const dstype mu[], const dstype uinf[], dstype t)
    {
        fbou_jac_uq(f, ib, x, uq, v, w, uh, n, tau, mu, uinf, t);
    }

    KOKKOS_INLINE_FUNCTION static
    void fbou_hdg_jac_uh(dstype f[], int ib, const dstype x[], const dstype uq[], const dstype v[],
                         const dstype w[], const dstype uh[], const dstype n[], const dstype tau[],
                         const dstype mu[], const dstype uinf[], dstype t)
    {
        fbou_jac_uh(f, ib, x, uq, v, w, uh, n, tau, mu, uinf, t);
    }
};
