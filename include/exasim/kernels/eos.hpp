// SPDX-License-Identifier: see LICENSE
//
// <exasim/kernels/eos.hpp> — equation-of-state and AV field kernels.
//
// Replaces KokkosEoS, KokkosEoSdu, KokkosEoSdw, KokkosAvfield, and the
// HDG variant HdgEoS from libpdemodel.hpp. EoS pointwise functions
// produce the same shape as source (ncu output); the *_du / *_dw
// flavors emit Jacobian blocks.
//
// HOT.6.3: dispatch helper inlined into each kernel — see
// `output.hpp` for the same nvcc-extended-lambda restriction.

#pragma once

#include <Kokkos_Core.hpp>

#include "../common.h"
#include "../model.hpp"

namespace exasim {

template <class M, class T=dstype, class I=Int>
void eos_kernel(T* f, const T* xdg, const T* udg, const T* odg,
                const T* wdg, const T* uinf, const T* param, T t,
                int /*modelnumber*/, int ng, int /*nc*/, int /*ncu*/, int /*nd*/,
                int /*ncx*/, int /*nco*/, int /*ncw*/, int /*nce*/, int /*npe*/, int /*ne*/)
{
    using dstype=T;
    static_assert(is_eos_model_v<M>);
    constexpr int nd = M::nd, ncu = M::ncu, ncw = M::ncw, nco = M::nco;
    constexpr int Nq = ncu * (1 + nd);
    constexpr int ncw_buf = (ncw > 0) ? ncw : 1;
    constexpr int nco_buf = (nco > 0) ? nco : 1;

    Kokkos::parallel_for("exasim::eos_kernel", ng, KOKKOS_LAMBDA(size_t i) {
        (void)odg; (void)wdg;
        T x[nd], uq[Nq], v[nco_buf], w[ncw_buf];
        for (int k = 0; k < nd; ++k) x [k] = xdg[k * ng + i];
        for (int k = 0; k < Nq; ++k) uq[k] = udg[k * ng + i];
        if (nco > 0) for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
        if (ncw > 0) for (int k = 0; k < ncw; ++k) w[k] = wdg[k * ng + i];

        T out_local[ncu];
        M::eos(out_local, x, uq, v, w, param, uinf, t);
        for (int k = 0; k < ncu; ++k) f[k * ng + i] = out_local[k];
    });
}

template <class M, class T=dstype, class I=Int>
void eos_du_kernel(T* f, const T* xdg, const T* udg, const T* odg,
                   const T* wdg, const T* uinf, const T* param, T t,
                   int /*modelnumber*/, int ng, int /*nc*/, int /*ncu*/, int /*nd*/,
                   int /*ncx*/, int /*nco*/, int /*ncw*/, int /*nce*/, int /*npe*/, int /*ne*/)
{
    using dstype=T;
    static_assert(is_eos_model_v<M>);
    constexpr int nd = M::nd, ncu = M::ncu, ncw = M::ncw, nco = M::nco;
    constexpr int Nq = ncu * (1 + nd);
    constexpr int ncw_buf = (ncw > 0) ? ncw : 1;
    constexpr int nco_buf = (nco > 0) ? nco : 1;

    Kokkos::parallel_for("exasim::eos_du_kernel", ng, KOKKOS_LAMBDA(size_t i) {
        (void)odg; (void)wdg;
        T x[nd], uq[Nq], v[nco_buf], w[ncw_buf];
        for (int k = 0; k < nd; ++k) x [k] = xdg[k * ng + i];
        for (int k = 0; k < Nq; ++k) uq[k] = udg[k * ng + i];
        if (nco > 0) for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
        if (ncw > 0) for (int k = 0; k < ncw; ++k) w[k] = wdg[k * ng + i];

        T out_local[ncu * Nq];
        M::eos_du(out_local, x, uq, v, w, param, uinf, t);
        for (int k = 0; k < ncu * Nq; ++k) f[k * ng + i] = out_local[k];
    });
}

template <class M, class T=dstype, class I=Int>
void eos_dw_kernel(T* f, const T* xdg, const T* udg, const T* odg,
                   const T* wdg, const T* uinf, const T* param, T t,
                   int /*modelnumber*/, int ng, int /*nc*/, int /*ncu*/, int /*nd*/,
                   int /*ncx*/, int /*nco*/, int /*ncw*/, int /*nce*/, int /*npe*/, int /*ne*/)
{
    using dstype=T;
    static_assert(is_eos_model_v<M>);
    if constexpr (M::ncw > 0) {
        constexpr int nd = M::nd, ncu = M::ncu, ncw = M::ncw, nco = M::nco;
        constexpr int Nq = ncu * (1 + nd);
        constexpr int nco_buf = (nco > 0) ? nco : 1;

        Kokkos::parallel_for("exasim::eos_dw_kernel", ng, KOKKOS_LAMBDA(size_t i) {
            (void)odg; (void)wdg;
            T x[nd], uq[Nq], v[nco_buf], w[ncw];
            for (int k = 0; k < nd; ++k) x [k] = xdg[k * ng + i];
            for (int k = 0; k < Nq; ++k) uq[k] = udg[k * ng + i];
            if (nco > 0) for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
            for (int k = 0; k < ncw; ++k) w[k] = wdg[k * ng + i];

            T out_local[ncu * ncw];
            M::eos_dw(out_local, x, uq, v, w, param, uinf, t);
            for (int k = 0; k < ncu * ncw; ++k) f[k * ng + i] = out_local[k];
        });
    } else {
        (void)f; (void)xdg; (void)udg; (void)odg; (void)wdg; (void)param; (void)t; (void)ng;
    }
}

template <class M, class T=dstype, class I=Int>
void avfield_kernel(T* f, const T* xdg, const T* udg, const T* odg,
                    const T* wdg, const T* uinf, const T* param, T time,
                    int /*modelnumber*/, int ng, int nc_runtime, int /*ncu*/, int /*nd*/,
                    int ncx, int nco_runtime, int ncw_runtime, int nce_runtime, int npe, int /*ne*/)
{
    static_assert(is_avfield_model_v<M>);
    constexpr int nd = M::nd, ncu = M::ncu, nco = M::nco, ncw = M::ncw;
    constexpr int nc = ncu * (1 + nd);

    if (nc_runtime > nc)
        Kokkos::abort("Avfield input count exceeds model capacity.");
    if (nce_runtime > ncu)
        Kokkos::abort("Avfield output count exceeds M::ncu.");

    Kokkos::parallel_for("exasim::avfield_kernel", ng, KOKKOS_LAMBDA(const size_t i) {
        (void)odg; (void)wdg;
        const int p = i % npe;
        const int e = i / npe;
        T x[nd];
        T uq[nc] = {};
        T v[(nco > 0) ? nco : 1];
        T w[(ncw > 0) ? ncw : 1];
        T av_local[ncu] = {};
        for (int k = 0; k < nd; ++k)
            x[k] = xdg[p + npe * k + npe * ncx * e];
        for (int k = 0; k < nc_runtime; ++k)
            uq[k] = udg[p + npe * k + npe * nc_runtime * e];
        for (int k = 0; k < nco; ++k)
            v[k] = odg[p + npe * k + npe * nco_runtime * e];
        for (int k = 0; k < ncw; ++k)
            w[k] = wdg[p + npe * k + npe * ncw_runtime * e];

        M::avfield(av_local, x, uq, v, w, param, uinf, time);

        for (int k = 0; k < nce_runtime; ++k)
            f[p + npe * k + npe * nce_runtime * e] = av_local[k];
    });
}

} // namespace exasim
