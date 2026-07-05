// SPDX-License-Identifier: see LICENSE
//
// <exasim/kernels/init.hpp> — initial-condition kernels.
//
// Replaces KokkosInit{u,q,udg,wdg,odg} from libpdemodel.hpp. The init
// kernels iterate over total nodes (`ng = npe * ne`) rather than
// quadrature points; the SoA layout is element-major:
//   f[node + npe*comp + npe*nc*elem]

#pragma once

#include <Kokkos_Core.hpp>

#include "../common.h"
#include "../model.hpp"

namespace exasim {

namespace detail {

// Common scaffolding for every init kernel. The model method is passed as a
// COMPILE-TIME function pointer `Fn` (not a forwarded lambda): a KOKKOS_LAMBDA
// that captures another lambda makes nvcc's CUDA device-stub generation fail
// (__nv_dl_tag "wrong number of template arguments"). With Fn as a non-type
// template parameter the device lambda captures only trivial pointers, and the
// call resolves to the model's static method at compile time (and inlines).
template <class TT> using InitFn_t = void (*)(TT[], const TT[], const TT[], const TT[]);

// fStride = the DESTINATION buffer's per-element component count. This is NOT always OutputSize:
// initu writes ncu components into udg, whose packed width is nc = ncu+ncq (> ncu when there is a
// flux q). Using OutputSize as the stride scatters the data into the wrong elements -- silently
// masked whenever initu returns 0 (e.g. Poisson), but fatal for a nonzero IC (e.g. NS freestream).
template <class M, class T, int OutputSize, InitFn_t<T> Fn, class I=Int>
void init_dispatch(T* f, const T* xdg, const T* uinf,
                   const T* param, int ng, int npe, int fStride)
{
    using dstype=T;
    constexpr int nd = M::nd;
    Kokkos::parallel_for("exasim::init_kernel", ng, KOKKOS_LAMBDA(size_t i) {
        const int j    = static_cast<int>(i % npe);
        const int elem = static_cast<int>(i / npe);

        T x[nd];
        for (int k = 0; k < nd; ++k) {
            // xdg layout: [npe x ncx x ne], column-major; x at this node:
            x[k] = xdg[j + npe * k + npe * nd * elem];
        }

        T out_local[OutputSize];
        Fn(out_local, x, uinf, param);

        for (int k = 0; k < OutputSize; ++k) {
            f[j + npe * k + npe * fStride * elem] = out_local[k];
        }
    });
}

} // namespace detail

template <class M, class T=dstype, class I=Int>
void initu_kernel(T* f, const T* xdg, const T* uinf,
                  const T* param, int /*modelnumber*/, int ng,
                  int /*ncx*/, int /*nce*/, int npe, int /*ne*/, int nc)
{
    using dstype=T;
    static_assert(is_init_model_v<M>);
    // udg is [npe x nc x ne] (nc = ncu+ncq); initu writes only the ncu u-components -> stride nc.
    detail::init_dispatch<M, T, M::ncu, &M::initu>(f, xdg, uinf, param, ng, npe, nc);
}

template <class M, class T=dstype, class I=Int>
void initq_kernel(T* f, const T* xdg, const T* uinf,
                  const T* param, int /*modelnumber*/, int ng,
                  int /*ncx*/, int /*nce*/, int npe, int /*ne*/)
{
    using dstype=T;
    static_assert(is_init_model_v<M>);
    detail::init_dispatch<M, T, M::ncu * M::nd, &M::initq>(f, xdg, uinf, param, ng, npe, M::ncu * M::nd);
}

template <class M, class T=dstype, class I=Int>
void initudg_kernel(T* f, const T* xdg, const T* uinf,
                    const T* param, int /*modelnumber*/, int ng,
                    int /*ncx*/, int /*nce*/, int npe, int /*ne*/)
{
    using dstype=T;
    static_assert(is_init_model_v<M>);
    constexpr int Nq = M::ncu * (1 + M::nd);   // initudg fills the whole [u,q] block: stride = Nq = nc
    detail::init_dispatch<M, T, Nq, &M::initudg>(f, xdg, uinf, param, ng, npe, Nq);
}

template <class M, class T=dstype, class I=Int>
void initwdg_kernel(T* f, const T* xdg, const T* uinf,
                    const T* param, int /*modelnumber*/, int ng,
                    int /*ncx*/, int /*nce*/, int npe, int /*ne*/)
{
    using dstype=T;
    static_assert(is_init_model_v<M>);
    if constexpr (M::ncw > 0) {   // wdg is its own [npe x ncw x ne] buffer: stride = ncw
        detail::init_dispatch<M, T, M::ncw, &M::initwdg>(f, xdg, uinf, param, ng, npe, M::ncw);
    } else {
        (void)f; (void)xdg; (void)uinf; (void)param; (void)ng; (void)npe;
    }
}

// initodg's output count is determined by the discretization (`nco`),
// not by Self. The runtime nce arg carries the count.
template <class M, class T=dstype, class I=Int>
void initodg_kernel(T* f, const T* xdg, const T* uinf,
                    const T* param, int /*modelnumber*/, int ng,
                    int /*ncx*/, int nce, int npe, int /*ne*/)
{
    using dstype=T;
    static_assert(is_init_model_v<M>);
    Kokkos::parallel_for("exasim::initodg_kernel", ng, KOKKOS_LAMBDA(size_t i) {
        const int j    = static_cast<int>(i % npe);
        const int elem = static_cast<int>(i / npe);
        for (int k = 0; k < nce; ++k) {
            f[j + npe * k + npe * nce * elem] = 0.0;
        }
        // Models that need a non-zero odg field can override M::initodg;
        // we don't pre-pull the user override here because the size isn't
        // known at compile time.
        (void)xdg; (void)uinf; (void)param;
    });
}

} // namespace exasim
