#pragma once

#include <Kokkos_Core.hpp>
#include "modeldefaults.hpp"

// Internal HDG mesh-motion operator. Exasim stores q=-grad(u), so this flux
// is the negative Cauchy stress and div(flux)+force=0 gives -div(sigma)+force=0.
template <int SpatialDimension>
struct LinearElasticityModel : ModelDefaults<LinearElasticityModel<SpatialDimension>> {
    static constexpr int nd = SpatialDimension;
    static constexpr int ncu = nd;
    static constexpr int ncw = 0;
    static constexpr int nco = 2 + nd; // mu, lambda, and force components
    static constexpr int nparam = 3;   // shear, volumetric, and force scales
    static constexpr int ntau = 1;
    static constexpr int Nq = ncu*(1 + nd);
    static constexpr int Nflux = ncu*nd;

    KOKKOS_INLINE_FUNCTION static dstype q(const dstype uq[], int i, int d)
    { return uq[ncu + i + ncu*d]; }

    KOKKOS_INLINE_FUNCTION static dstype stress(const dstype uq[], const dstype v[],
                                                const dstype param[], int i, int d)
    {
        dstype trq = 0.0;
        for (int k = 0; k < nd; ++k) trq += q(uq, k, k);
        return param[0]*v[0]*(q(uq, i, d) + q(uq, d, i))
             + ((i == d) ? param[1]*v[1]*trq : 0.0);
    }

    KOKKOS_INLINE_FUNCTION static void flux(dstype f[], const dstype[], const dstype uq[],
        const dstype v[], const dstype[], const dstype param[], const dstype[], dstype)
    {
        for (int d = 0; d < nd; ++d)
            for (int i = 0; i < ncu; ++i) f[i + ncu*d] = stress(uq, v, param, i, d);
    }

    KOKKOS_INLINE_FUNCTION static void source(dstype f[], const dstype[], const dstype[],
        const dstype v[], const dstype[], const dstype param[], const dstype[], dstype)
    {
        for (int i = 0; i < ncu; ++i) f[i] = param[2]*v[2 + i];
    }

    KOKKOS_INLINE_FUNCTION static void initu(dstype f[], const dstype[], const dstype[], const dstype[])
    { for (int i = 0; i < ncu; ++i) f[i] = 0.0; }

    KOKKOS_INLINE_FUNCTION static void numerical_flux(dstype f[], const dstype uq[],
        const dstype v[], const dstype uh[], const dstype n[], const dstype tau[],
        const dstype param[])
    {
        for (int i = 0; i < ncu; ++i) {
            f[i] = tau[0]*(uq[i] - uh[i]);
            for (int d = 0; d < nd; ++d) f[i] += stress(uq, v, param, i, d)*n[d];
        }
    }

    KOKKOS_INLINE_FUNCTION static void boundary_residual(dstype f[], int ib,
        const dstype uq[], const dstype v[], const dstype uh[], const dstype n[],
        const dstype tau[], const dstype param[])
    {
        dstype fh[3] = {0.0, 0.0, 0.0};
        numerical_flux(fh, uq, v, uh, n, tau, param);
        if (ib == 1) {
            for (int i = 0; i < ncu; ++i) f[i] = fh[i];
        } else if (ib == 2) {
            for (int i = 0; i < ncu; ++i) f[i] = -tau[0]*uh[i];
        } else if (ib == 3) {
            dstype un = 0.0, tn = 0.0;
            for (int i = 0; i < ncu; ++i) { un += uh[i]*n[i]; tn += fh[i]*n[i]; }
            if constexpr (nd == 2) {
                f[0] = -tau[0]*un;
                f[1] = fh[0]*n[1] - fh[1]*n[0];
            } else {
                for (int i = 0; i < ncu; ++i)
                    f[i] = -tau[0]*un*n[i] + fh[i] - tn*n[i];
            }
        } else {
            int fixed = (ib == 4) ? 1 : ((ib == 5) ? 0 : 2);
            if (fixed >= nd) fixed = nd - 1;
            for (int i = 0; i < ncu; ++i) f[i] = (i == fixed) ? -tau[0]*uh[i] : fh[i];
        }
    }

    KOKKOS_INLINE_FUNCTION static void fbou(dstype f[], int, const dstype[],
        const dstype[], const dstype[], const dstype[], const dstype[], const dstype[],
        const dstype[], const dstype[], const dstype[], dstype)
    { for (int i = 0; i < ncu; ++i) f[i] = 0.0; }

    KOKKOS_INLINE_FUNCTION static void fbou_hdg(dstype f[], int ib, const dstype[],
        const dstype uq[], const dstype v[], const dstype[], const dstype uh[],
        const dstype n[], const dstype tau[], const dstype param[], const dstype[], dstype)
    { boundary_residual(f, ib, uq, v, uh, n, tau, param); }

    KOKKOS_INLINE_FUNCTION static dstype dstress_dq(const dstype v[], const dstype param[],
        int i, int d, int k, int l)
    {
        return param[0]*v[0]*(((i == k) && (d == l)) + ((d == k) && (i == l)))
             + param[1]*v[1]*((i == d) && (k == l));
    }

    KOKKOS_INLINE_FUNCTION static void flux_jac_uq(dstype f[], const dstype[],
        const dstype[], const dstype v[], const dstype[], const dstype param[],
        const dstype[], dstype)
    {
        for (int a = 0; a < Nflux*Nq; ++a) f[a] = 0.0;
        for (int k = 0; k < ncu; ++k)
            for (int l = 0; l < nd; ++l)
                for (int i = 0; i < ncu; ++i)
                    for (int d = 0; d < nd; ++d)
                        f[(i + ncu*d) + Nflux*(ncu + k + ncu*l)] =
                            dstress_dq(v, param, i, d, k, l);
    }

    KOKKOS_INLINE_FUNCTION static void source_jac_uq(dstype f[], const dstype[],
        const dstype[], const dstype[], const dstype[], const dstype[], const dstype[], dstype)
    { for (int a = 0; a < ncu*Nq; ++a) f[a] = 0.0; }

    KOKKOS_INLINE_FUNCTION static void numerical_flux_jac_uq(dstype f[], const dstype v[],
        const dstype n[], const dstype tau[], const dstype param[])
    {
        for (int a = 0; a < ncu*Nq; ++a) f[a] = 0.0;
        for (int k = 0; k < ncu; ++k)
            for (int i = 0; i < ncu; ++i) f[i + ncu*k] = tau[0]*(i == k);
        for (int k = 0; k < ncu; ++k)
            for (int l = 0; l < nd; ++l)
                for (int i = 0; i < ncu; ++i)
                    for (int d = 0; d < nd; ++d)
                        f[i + ncu*(ncu + k + ncu*l)] += dstress_dq(v, param, i, d, k, l)*n[d];
    }

    KOKKOS_INLINE_FUNCTION static void boundary_jac_uq(dstype f[], int ib,
        const dstype v[], const dstype n[], const dstype tau[], const dstype param[])
    {
        numerical_flux_jac_uq(f, v, n, tau, param);
        if (ib == 2) {
            for (int a = 0; a < ncu*Nq; ++a) f[a] = 0.0;
        } else if (ib == 3) {
            dstype original[3*12];
            for (int a = 0; a < ncu*Nq; ++a) original[a] = f[a];
            if constexpr (nd == 2) {
                for (int in = 0; in < Nq; ++in) {
                    f[ncu*in] = 0.0;
                    f[1 + ncu*in] = n[1]*original[ncu*in] - n[0]*original[1 + ncu*in];
                }
            } else {
                for (int in = 0; in < Nq; ++in) {
                    dstype normal = 0.0;
                    for (int j = 0; j < ncu; ++j) normal += n[j]*original[j + ncu*in];
                    for (int i = 0; i < ncu; ++i)
                        f[i + ncu*in] = original[i + ncu*in] - n[i]*normal;
                }
            }
        } else if (ib >= 4) {
            int fixed = (ib == 4) ? 1 : ((ib == 5) ? 0 : 2);
            if (fixed >= nd) fixed = nd - 1;
            for (int in = 0; in < Nq; ++in) f[fixed + ncu*in] = 0.0;
        }
    }

    KOKKOS_INLINE_FUNCTION static void boundary_jac_uh(dstype f[], int ib,
        const dstype n[], const dstype tau[])
    {
        for (int k = 0; k < ncu; ++k)
            for (int i = 0; i < ncu; ++i) f[i + ncu*k] = -tau[0]*(i == k);
        if (ib == 3) {
            if constexpr (nd == 2) {
                f[0] = -tau[0]*n[0]; f[2] = -tau[0]*n[1];
                f[1] = -tau[0]*n[1]; f[3] =  tau[0]*n[0];
            } else {
                for (int k = 0; k < ncu; ++k)
                    for (int i = 0; i < ncu; ++i)
                        f[i + ncu*k] = -tau[0]*n[i]*n[k]
                            - tau[0]*((i == k) - n[i]*n[k]);
            }
        } else if (ib >= 4) {
            int fixed = (ib == 4) ? 1 : ((ib == 5) ? 0 : 2);
            if (fixed >= nd) fixed = nd - 1;
            for (int k = 0; k < ncu; ++k) f[fixed + ncu*k] = -tau[0]*(fixed == k);
        }
    }

    KOKKOS_INLINE_FUNCTION static void fbou_jac_uq(dstype f[], int, const dstype[],
        const dstype[], const dstype v[], const dstype[], const dstype[], const dstype n[],
        const dstype tau[], const dstype[], const dstype param[], dstype)
    { numerical_flux_jac_uq(f, v, n, tau, param); }

    KOKKOS_INLINE_FUNCTION static void fbou_jac_uh(dstype f[], int, const dstype[],
        const dstype[], const dstype[], const dstype[], const dstype[], const dstype[],
        const dstype tau[], const dstype[], const dstype[], dstype)
    {
        for (int k = 0; k < ncu; ++k)
            for (int i = 0; i < ncu; ++i) f[i + ncu*k] = -tau[0]*(i == k);
    }

    KOKKOS_INLINE_FUNCTION static void fbou_hdg_jac_uq(dstype f[], int ib,
        const dstype[], const dstype[], const dstype v[], const dstype[], const dstype[],
        const dstype n[], const dstype tau[], const dstype param[], const dstype[], dstype)
    { boundary_jac_uq(f, ib, v, n, tau, param); }

    KOKKOS_INLINE_FUNCTION static void fbou_hdg_jac_uh(dstype f[], int ib,
        const dstype[], const dstype[], const dstype[], const dstype[], const dstype[],
        const dstype n[], const dstype tau[], const dstype[], const dstype[], dstype)
    { boundary_jac_uh(f, ib, n, tau); }
};
