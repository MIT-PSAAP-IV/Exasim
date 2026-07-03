// SPDX-License-Identifier: see LICENSE
//
// <exasim/kernels/boundary.hpp> — boundary kernels: fbou, ubou + their HDG variants.
//
// Boundary kernels operate on faces tagged with `ib` and additionally
// see the trace `uhg`, the outward normal `nlg`, and the stabilization
// `tau`. The HDG variants fill four Jacobian outputs:
//   f_udg = ∂fb/∂uq    (ncu × Nq    × ng)
//   f_wdg = ∂fb/∂w     (ncu × ncw   × ng)  [only if ncw > 0]
//   f_uhg = ∂fb/∂uh    (ncu × ncu   × ng)

#pragma once

#include <Kokkos_Core.hpp>

#include "../common.h"
#include "../model.hpp"

namespace exasim {

template <class M, class T=dstype, class I=Int>
void fbou_kernel(T*       fb,
                 const T* xdg, const T* udg, const T* odg,
                 const T* wdg, const T* uhg, const T* nlg,
                 const T* tau, const T* /*uinf*/, const T* param,
                 T t, int /*modelnumber*/, int ib, int ng,
                 int /*nc*/, int /*ncu*/, int /*nd*/, int /*ncx*/, int /*nco*/, int /*ncw*/)
{
    using dstype=T;
    static_assert(is_boundary_model_v<M>);
    constexpr int nd = M::nd, ncu = M::ncu, ncw = M::ncw, nco = M::nco;
    constexpr int Nq = ncu * (1 + nd);
    constexpr int ncw_buf = (ncw > 0) ? ncw : 1;
    constexpr int nco_buf = (nco > 0) ? nco : 1;

    Kokkos::parallel_for("exasim::fbou_kernel", ng, KOKKOS_LAMBDA(size_t i) {
        (void)odg; (void)wdg;  // HOT.6.2 nvcc force-capture: see /tmp/patch_constexpr_capture.py
        T x[nd], uq[Nq], v[nco_buf], w[ncw_buf], uh[ncu], n[nd], t_[ncu];
        for (int k = 0; k < nd;  ++k) x [k] = xdg[k * ng + i];
        for (int k = 0; k < Nq;  ++k) uq[k] = udg[k * ng + i];
        if (nco > 0) for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
        if (ncw > 0) for (int k = 0; k < ncw; ++k) w[k] = wdg[k * ng + i];
        for (int k = 0; k < ncu; ++k) uh[k] = uhg[k * ng + i];
        for (int k = 0; k < nd;  ++k) n [k] = nlg[k * ng + i];
        for (int k = 0; k < ncu; ++k) t_[k] = tau[k];   // tau is per-component, not per-i

        T fb_local[ncu];
        M::fbou(fb_local, ib, x, uq, v, w, uh, n, t_, param, /*uinf=*/nullptr, t);
        for (int k = 0; k < ncu; ++k) fb[k * ng + i] = fb_local[k];
    });
}

// Value-only HDG fbou kernel — calls M::fbou_hdg (boundary condition),
// matching legacy `HdgFbouonly` (used in residual evaluation, no Jacobians).
template <class M, class T=dstype, class I=Int>
void hdg_fbou_only_kernel(T* fb,
                          const T* xdg, const T* udg, const T* odg,
                          const T* wdg, const T* uhg, const T* nlg,
                          const T* tau, const T* /*uinf*/, const T* param,
                          T t, int /*modelnumber*/, int ib, int ng,
                          int /*nc*/, int /*ncu*/, int /*nd*/, int /*ncx*/, int /*nco*/, int /*ncw*/)
{
    using dstype=T;
    static_assert(is_hdg_boundary_model_v<M>);
    constexpr int nd = M::nd, ncu = M::ncu, ncw = M::ncw, nco = M::nco;
    constexpr int Nq = ncu * (1 + nd);
    constexpr int ncw_buf = (ncw > 0) ? ncw : 1;
    constexpr int nco_buf = (nco > 0) ? nco : 1;

    Kokkos::parallel_for("exasim::hdg_fbou_only_kernel", ng, KOKKOS_LAMBDA(size_t i) {
        (void)odg; (void)wdg;  // HOT.6.2 nvcc force-capture: see /tmp/patch_constexpr_capture.py
        T x[nd], uq[Nq], v[nco_buf], w[ncw_buf], uh[ncu], n[nd], t_[ncu];
        for (int k = 0; k < nd;  ++k) x [k] = xdg[k * ng + i];
        for (int k = 0; k < Nq;  ++k) uq[k] = udg[k * ng + i];
        if (nco > 0) for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
        if (ncw > 0) for (int k = 0; k < ncw; ++k) w[k] = wdg[k * ng + i];
        for (int k = 0; k < ncu; ++k) uh[k] = uhg[k * ng + i];
        for (int k = 0; k < nd;  ++k) n [k] = nlg[k * ng + i];
        for (int k = 0; k < ncu; ++k) t_[k] = tau[k];

        T fb_local[ncu];
        M::fbou_hdg(fb_local, ib, x, uq, v, w, uh, n, t_, param, /*uinf=*/nullptr, t);
        for (int k = 0; k < ncu; ++k) fb[k * ng + i] = fb_local[k];
    });
}

template <class M, class T=dstype, class I=Int>
void hdg_fbou_kernel(T* fb, T* f_udg, T* f_wdg, T* f_uhg,
                     const T* xdg, const T* udg, const T* odg,
                     const T* wdg, const T* uhg, const T* nlg,
                     const T* tau, const T* /*uinf*/, const T* param,
                     T t, int /*modelnumber*/, int ib, int ng,
                     int /*nc*/, int /*ncu*/, int /*nd*/, int /*ncx*/, int /*nco*/, int /*ncw*/)
{
    using dstype=T;
    static_assert(is_hdg_boundary_model_v<M>);
    constexpr int nd = M::nd, ncu = M::ncu, ncw = M::ncw, nco = M::nco;
    constexpr int Nq = ncu * (1 + nd);
    constexpr int ncw_buf = (ncw > 0) ? ncw : 1;
    constexpr int nco_buf = (nco > 0) ? nco : 1;

    Kokkos::parallel_for("exasim::hdg_fbou_kernel", ng, KOKKOS_LAMBDA(size_t i) {
        (void)odg; (void)wdg; (void)f_wdg;  // HOT.6.2 nvcc force-capture: see /tmp/patch_constexpr_capture.py
        T x[nd], uq[Nq], v[nco_buf], w[ncw_buf], uh[ncu], n[nd], t_[ncu];
        for (int k = 0; k < nd;  ++k) x [k] = xdg[k * ng + i];
        for (int k = 0; k < Nq;  ++k) uq[k] = udg[k * ng + i];
        if (nco > 0) for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
        if (ncw > 0) for (int k = 0; k < ncw; ++k) w[k] = wdg[k * ng + i];
        for (int k = 0; k < ncu; ++k) uh[k] = uhg[k * ng + i];
        for (int k = 0; k < nd;  ++k) n [k] = nlg[k * ng + i];
        for (int k = 0; k < ncu; ++k) t_[k] = tau[k];

        // Value — note: HDG path calls fbou_hdg, NOT fbou. Every PDE
        // in apps/ defines `Fbou` (LDG) and `FbouHdg` (HDG) as
        // distinct math; mixing them gives a wrong residual.
        T fb_local[ncu];
        M::fbou_hdg(fb_local, ib, x, uq, v, w, uh, n, t_, param, /*uinf=*/nullptr, t);
        for (int k = 0; k < ncu; ++k) fb[k * ng + i] = fb_local[k];

        // ∂fb/∂uq (Jacobian of fbou_hdg, not fbou)
        T fb_uq[ncu * Nq];
        M::fbou_hdg_jac_uq(fb_uq, ib, x, uq, v, w, uh, n, t_, param, /*uinf=*/nullptr, t);
        for (int k = 0; k < ncu * Nq; ++k) f_udg[k * ng + i] = fb_uq[k];

        // ∂fb/∂w
        if constexpr (ncw > 0) {
            T fb_w[ncu * ncw];
            M::fbou_hdg_jac_w(fb_w, ib, x, uq, v, w, uh, n, t_, param, /*uinf=*/nullptr, t);
            for (int k = 0; k < ncu * ncw; ++k) f_wdg[k * ng + i] = fb_w[k];
        }

        // ∂fb/∂uh
        T fb_uh[ncu * ncu];
        M::fbou_hdg_jac_uh(fb_uh, ib, x, uq, v, w, uh, n, t_, param, /*uinf=*/nullptr, t);
        for (int k = 0; k < ncu * ncu; ++k) f_uhg[k * ng + i] = fb_uh[k];
    });
}

template <class M, class T=dstype, class I=Int>
void ubou_kernel(T* ub,
                 const T* xdg, const T* udg, const T* odg,
                 const T* wdg, const T* uhg, const T* nlg,
                 const T* tau, const T* /*uinf*/, const T* param,
                 T t, int /*modelnumber*/, int ib, int ng,
                 int /*nc*/, int /*ncu*/, int /*nd*/, int /*ncx*/, int /*nco*/, int /*ncw*/)
{
    using dstype=T;
    static_assert(is_boundary_model_v<M>);
    constexpr int nd = M::nd, ncu = M::ncu, ncw = M::ncw, nco = M::nco;
    constexpr int Nq = ncu * (1 + nd);
    constexpr int ncw_buf = (ncw > 0) ? ncw : 1;
    constexpr int nco_buf = (nco > 0) ? nco : 1;

    Kokkos::parallel_for("exasim::ubou_kernel", ng, KOKKOS_LAMBDA(size_t i) {
        (void)odg; (void)wdg;  // HOT.6.2 nvcc force-capture: see /tmp/patch_constexpr_capture.py
        T x[nd], uq[Nq], v[nco_buf], w[ncw_buf], uh[ncu], n[nd], t_[ncu];
        for (int k = 0; k < nd;  ++k) x [k] = xdg[k * ng + i];
        for (int k = 0; k < Nq;  ++k) uq[k] = udg[k * ng + i];
        if (nco > 0) for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
        if (ncw > 0) for (int k = 0; k < ncw; ++k) w[k] = wdg[k * ng + i];
        for (int k = 0; k < ncu; ++k) uh[k] = uhg[k * ng + i];
        for (int k = 0; k < nd;  ++k) n [k] = nlg[k * ng + i];
        for (int k = 0; k < ncu; ++k) t_[k] = tau[k];

        T ub_local[ncu];
        M::ubou(ub_local, ib, x, uq, v, w, uh, n, t_, param, /*uinf=*/nullptr, t);
        for (int k = 0; k < ncu; ++k) ub[k * ng + i] = ub_local[k];
    });
}

} // namespace exasim
