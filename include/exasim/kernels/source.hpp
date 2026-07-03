// SPDX-License-Identifier: see LICENSE
//
// <exasim/kernels/source.hpp> — templated source-term kernels.
//
// Replaces `KokkosSource` and `HdgSource` from libpdemodel.hpp.
//
// The forcing/source pointwise function emits an `ncu`-vector. The
// HDG variant additionally fills `f_udg = ∂s/∂uq` and (when ncw > 0)
// `f_wdg = ∂s/∂w` from the user's hand-written `M::source_jac_uq`
// and `M::source_jac_w` methods.

#pragma once

#include <Kokkos_Core.hpp>

#include "../common.h"
#include "../model.hpp"

namespace exasim {

// Forward path — value only.
template <class M, class T=dstype, class I=Int>
void source_kernel(T*       s,
                   const T* xdg,
                   const T* udg,
                   const T* odg,
                   const T* wdg,
                   const T* /*uinf*/,
                   const T* param,
                   T        t,
                   int           /*modelnumber*/,
                   int           ng,
                   int           /*nc*/,
                   int           ncu_runtime,
                   int           nd_runtime,
                   int           /*ncx*/,
                   int           /*nco*/,
                   int           ncw_runtime)
{
    using dstype=T;
    static_assert(is_source_model_v<M>, "source_kernel<M>: M must satisfy the Model contract.");

    constexpr int nd  = M::nd;
    constexpr int ncu = M::ncu;
    constexpr int ncw = M::ncw;
    constexpr int nco = M::nco;
    constexpr int Nq  = ncu * (1 + nd);
    constexpr int ncw_buf = (ncw > 0) ? ncw : 1;
    constexpr int nco_buf = (nco > 0) ? nco : 1;

    assert(ncu_runtime == ncu && nd_runtime == nd && ncw_runtime == ncw);
    (void)ncu_runtime; (void)nd_runtime; (void)ncw_runtime;

    Kokkos::parallel_for("exasim::source_kernel", ng,
        KOKKOS_LAMBDA(const size_t i) {
            (void)odg; (void)wdg;  // HOT.6.2 nvcc force-capture: see /tmp/patch_constexpr_capture.py
            T x [nd];
            T uq[Nq];
            T v [nco_buf];
            T w [ncw_buf];

            for (int k = 0; k < nd; ++k) x [k] = xdg[k * ng + i];
            for (int k = 0; k < Nq; ++k) uq[k] = udg[k * ng + i];
            if (nco > 0) {
                for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
            }
            if (ncw > 0) {
                for (int k = 0; k < ncw; ++k) w[k] = wdg[k * ng + i];
            }

            T s_local[ncu];
            M::source(s_local, x, uq, v, w, param, /*uinf=*/nullptr, t);

            for (int k = 0; k < ncu; ++k) s[k * ng + i] = s_local[k];
        });
}

// HDG path — value + ∂s/∂uq + ∂s/∂w.
template <class M, class T=dstype, class I=Int>
void hdg_source_kernel(T*       s,
                       T*       s_udg,
                       T*       s_wdg,
                       const T* xdg,
                       const T* udg,
                       const T* odg,
                       const T* wdg,
                       const T* /*uinf*/,
                       const T* param,
                       T        t,
                       int           /*modelnumber*/,
                       int           ng,
                       int           /*nc*/,
                       int           ncu_runtime,
                       int           nd_runtime,
                       int           /*ncx*/,
                       int           /*nco*/,
                       int           ncw_runtime)
{
    using dstype=T;
    static_assert(is_source_model_v<M>, "hdg_source_kernel<M>: M must satisfy the Model contract.");

    constexpr int nd  = M::nd;
    constexpr int ncu = M::ncu;
    constexpr int ncw = M::ncw;
    constexpr int nco = M::nco;
    constexpr int Nq  = ncu * (1 + nd);
    constexpr int ncw_buf = (ncw > 0) ? ncw : 1;
    constexpr int nco_buf = (nco > 0) ? nco : 1;

    assert(ncu_runtime == ncu && nd_runtime == nd && ncw_runtime == ncw);
    (void)ncu_runtime; (void)nd_runtime; (void)ncw_runtime;

    Kokkos::parallel_for("exasim::hdg_source_kernel", ng,
        KOKKOS_LAMBDA(const size_t i) {
            (void)odg; (void)wdg; (void)s_wdg;  // HOT.6.2 nvcc force-capture: see /tmp/patch_constexpr_capture.py
            T x [nd];
            T uq[Nq];
            T v [nco_buf];
            T w [ncw_buf];

            for (int k = 0; k < nd; ++k) x [k] = xdg[k * ng + i];
            for (int k = 0; k < Nq; ++k) uq[k] = udg[k * ng + i];
            if (nco > 0) {
                for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
            }
            if (ncw > 0) {
                for (int k = 0; k < ncw; ++k) w[k] = wdg[k * ng + i];
            }

            // Value
            T s_local[ncu];
            M::source(s_local, x, uq, v, w, param, /*uinf=*/nullptr, t);
            for (int k = 0; k < ncu; ++k) s[k * ng + i] = s_local[k];

            // ∂s/∂uq
            T s_uq[ncu * Nq];
            M::source_jac_uq(s_uq, x, uq, v, w, param, /*uinf=*/nullptr, t);
            for (int k = 0; k < ncu * Nq; ++k) s_udg[k * ng + i] = s_uq[k];

            // ∂s/∂w (only when present)
            if constexpr (ncw > 0) {
                T s_w[ncu * ncw];
                M::source_jac_w(s_w, x, uq, v, w, param, /*uinf=*/nullptr, t);
                for (int k = 0; k < ncu * ncw; ++k) s_wdg[k * ng + i] = s_w[k];
            }
        });
}

} // namespace exasim
