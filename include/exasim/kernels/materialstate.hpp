// SPDX-License-Identifier: see LICENSE
//
// <exasim/kernels/materialstate.hpp> — optional material-state kernel.
//
// materialstate maps the local PDE state (u, q, w, v, x, t, mu, eta) to the
// independent coordinates used by a material database. The HDG variant also
// fills the Jacobians with respect to uq/udg and wdg, mirroring source.hpp.

#pragma once

#include <cassert>
#include <Kokkos_Core.hpp>

#include "../common.h"
#include "../model.hpp"

namespace exasim {

template <class M, class T=dstype, class I=Int>
void materialstate_kernel(T*       state,
                          const T* xdg,
                          const T* udg,
                          const T* odg,
                          const T* wdg,
                          const T* uinf,
                          const T* param,
                          T        t,
                          int      /*modelnumber*/,
                          int      ng,
                          int      /*nc*/,
                          int      ncu_runtime,
                          int      nd_runtime,
                          int      /*ncx*/,
                          int      /*nco*/,
                          int      ncw_runtime,
                          int      nmaterialstate_runtime)
{
    using dstype=T;
    static_assert(is_materialstate_model_v<M>,
                  "materialstate_kernel<M>: M must satisfy the Model contract.");

    constexpr int nd  = M::nd;
    constexpr int ncu = M::ncu;
    constexpr int ncw = M::ncw;
    constexpr int nco = M::nco;
    constexpr int nms = M::nmaterialstate;
    constexpr int Nq  = ncu * (1 + nd);
    constexpr int ncw_buf = (ncw > 0) ? ncw : 1;
    constexpr int nco_buf = (nco > 0) ? nco : 1;
    constexpr int nms_buf = (nms > 0) ? nms : 1;

    assert(ncu_runtime == ncu && nd_runtime == nd && ncw_runtime == ncw);
    assert(nmaterialstate_runtime == nms || nms == 0);
    (void)ncu_runtime; (void)nd_runtime; (void)ncw_runtime; (void)nmaterialstate_runtime;

    Kokkos::parallel_for("exasim::materialstate_kernel", ng,
        KOKKOS_LAMBDA(const size_t i) {
            (void)odg; (void)wdg;  // preserve capture behavior for optional arrays
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

            T state_local[nms_buf];
            M::materialstate(state_local, x, uq, v, w, param, /*uinf=*/nullptr, t);

            for (int k = 0; k < nms; ++k) state[k * ng + i] = state_local[k];
        });
}

template <class M, class T=dstype, class I=Int>
void hdg_materialstate_kernel(T*       state,
                              T*       state_udg,
                              T*       state_wdg,
                              const T* xdg,
                              const T* udg,
                              const T* odg,
                              const T* wdg,
                              const T* /*uinf*/,
                              const T* param,
                              T        t,
                              int      /*modelnumber*/,
                              int      ng,
                              int      /*nc*/,
                              int      ncu_runtime,
                              int      nd_runtime,
                              int      /*ncx*/,
                              int      /*nco*/,
                              int      ncw_runtime,
                              int      nmaterialstate_runtime)
{
    using dstype=T;
    static_assert(is_materialstate_model_v<M>,
                  "hdg_materialstate_kernel<M>: M must satisfy the Model contract.");

    constexpr int nd  = M::nd;
    constexpr int ncu = M::ncu;
    constexpr int ncw = M::ncw;
    constexpr int nco = M::nco;
    constexpr int nms = M::nmaterialstate;
    constexpr int Nq  = ncu * (1 + nd);
    constexpr int ncw_buf = (ncw > 0) ? ncw : 1;
    constexpr int nco_buf = (nco > 0) ? nco : 1;
    constexpr int nms_buf = (nms > 0) ? nms : 1;

    assert(ncu_runtime == ncu && nd_runtime == nd && ncw_runtime == ncw);
    assert(nmaterialstate_runtime == nms || nms == 0);
    (void)ncu_runtime; (void)nd_runtime; (void)ncw_runtime; (void)nmaterialstate_runtime;

    Kokkos::parallel_for("exasim::hdg_materialstate_kernel", ng,
        KOKKOS_LAMBDA(const size_t i) {
            (void)odg; (void)wdg; (void)state_wdg;
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

            T state_local[nms_buf];
            M::materialstate(state_local, x, uq, v, w, param, /*uinf=*/nullptr, t);
            for (int k = 0; k < nms; ++k) state[k * ng + i] = state_local[k];

            T state_uq[(nms * Nq > 0) ? nms * Nq : 1];
            M::materialstate_jac_uq(state_uq, x, uq, v, w, param, /*uinf=*/nullptr, t);
            for (int k = 0; k < nms * Nq; ++k) state_udg[k * ng + i] = state_uq[k];

            if constexpr (ncw > 0) {
                T state_w[nms * ncw];
                M::materialstate_jac_w(state_w, x, uq, v, w, param, /*uinf=*/nullptr, t);
                for (int k = 0; k < nms * ncw; ++k) state_wdg[k * ng + i] = state_w[k];
            }
        });
}

} // namespace exasim
