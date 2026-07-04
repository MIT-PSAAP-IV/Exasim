// SPDX-License-Identifier: see LICENSE
//
// <exasim/kernels/sourcew.hpp> — source kernel for the auxiliary `w` field.
//
// Replaces KokkosSourcew / HdgSourcew. Output size is `ncw` (auxiliary
// field count) rather than `ncu`. Only meaningful when `ncw > 0`.

#pragma once

#include <Kokkos_Core.hpp>

#include "../common.h"
#include "../model.hpp"

namespace exasim {

template <class M, class T=dstype, class I=Int>
void sourcew_kernel(T* sw,
                    const T* xdg, const T* udg, const T* odg,
                    const T* wdg, const T* /*uinf*/, const T* param,
                    T t, int /*modelnumber*/, int ng,
                    int /*nc*/, int /*ncu*/, int /*nd*/, int /*ncx*/, int /*nco*/,
                    int /*ncw_runtime*/, int /*nce*/, int /*npe*/, int /*ne*/)
{
    using dstype=T;
    static_assert(is_sourcew_model_v<M>);
    constexpr int nd = M::nd, ncu = M::ncu, ncw = M::ncw, nco = M::nco;
    constexpr int Nq = ncu * (1 + nd);
    constexpr int nco_buf = (nco > 0) ? nco : 1;

    if constexpr (ncw > 0) {
        Kokkos::parallel_for("exasim::sourcew_kernel", ng, KOKKOS_LAMBDA(size_t i) {
            (void)odg; (void)wdg;  // HOT.6.2 nvcc force-capture: see /tmp/patch_constexpr_capture.py
            T x[nd], uq[Nq], v[nco_buf], w[ncw];
            for (int k = 0; k < nd;  ++k) x [k] = xdg[k * ng + i];
            for (int k = 0; k < Nq;  ++k) uq[k] = udg[k * ng + i];
            if (nco > 0) for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
            for (int k = 0; k < ncw; ++k) w [k] = wdg[k * ng + i];

            T sw_local[ncw];
            M::sourcew(sw_local, x, uq, v, w, param, /*uinf=*/nullptr, t);
            for (int k = 0; k < ncw; ++k) sw[k * ng + i] = sw_local[k];
        });
    } else {
        (void)sw; (void)xdg; (void)udg; (void)odg; (void)wdg; (void)param; (void)t; (void)ng;
    }
}

// HDG w-source, full Jacobian: value + ∂sw/∂udg + ∂sw/∂w. Mirrors the ABI
// `HdgSourcew` (backend/Model/.../HdgSourcew.cpp); driven by the implicit
// w-solve in backend/Discretization/wequation.hpp. See the Jacobian layout
// note on `sourcew_jac_uq`/`sourcew_jac_w` in <exasim/model.hpp>:
//   sw_udg[k*ng+i] = buf[k], buf[j*ncw+o] = ∂sw[o]/∂uq[j], size ncw*Nq
//   sw_wdg[k*ng+i] = buf[k], buf[j*ncw+o] = ∂sw[o]/∂w [j], size ncw*ncw
template <class M, class T=dstype, class I=Int>
void hdg_sourcew_kernel(T* sw, T* sw_udg, T* sw_wdg,
                        const T* xdg, const T* udg, const T* odg,
                        const T* wdg, const T* /*uinf*/, const T* param,
                        T t, int /*modelnumber*/, int ng,
                        int /*nc*/, int /*ncu*/, int /*nd*/, int /*ncx*/, int /*nco*/,
                        int /*ncw_runtime*/)
{
    using dstype=T;
    static_assert(is_sourcew_model_v<M>);
    constexpr int nd = M::nd, ncu = M::ncu, ncw = M::ncw, nco = M::nco;
    constexpr int Nq = ncu * (1 + nd);
    constexpr int nco_buf = (nco > 0) ? nco : 1;

    if constexpr (ncw > 0) {
        Kokkos::parallel_for("exasim::hdg_sourcew_kernel", ng, KOKKOS_LAMBDA(size_t i) {
            (void)odg; (void)wdg;  // HOT.6.2 nvcc force-capture: see /tmp/patch_constexpr_capture.py
            T x[nd], uq[Nq], v[nco_buf], w[ncw];
            for (int k = 0; k < nd;  ++k) x [k] = xdg[k * ng + i];
            for (int k = 0; k < Nq;  ++k) uq[k] = udg[k * ng + i];
            if (nco > 0) for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
            for (int k = 0; k < ncw; ++k) w [k] = wdg[k * ng + i];

            // value
            T sw_local[ncw];
            M::sourcew(sw_local, x, uq, v, w, param, /*uinf=*/nullptr, t);
            for (int k = 0; k < ncw; ++k) sw[k * ng + i] = sw_local[k];

            // ∂sw/∂uq  (size ncw*Nq, input-index-outer)
            T sw_uq[ncw * Nq];
            M::sourcew_jac_uq(sw_uq, x, uq, v, w, param, /*uinf=*/nullptr, t);
            for (int k = 0; k < ncw * Nq; ++k) sw_udg[k * ng + i] = sw_uq[k];

            // ∂sw/∂w  (size ncw*ncw, input-index-outer)
            T sw_w[ncw * ncw];
            M::sourcew_jac_w(sw_w, x, uq, v, w, param, /*uinf=*/nullptr, t);
            for (int k = 0; k < ncw * ncw; ++k) sw_wdg[k * ng + i] = sw_w[k];
        });
    } else {
        (void)sw; (void)sw_udg; (void)sw_wdg;
        (void)xdg; (void)udg; (void)odg; (void)wdg; (void)param; (void)t; (void)ng;
    }
}

// HDG w-source, diagonal w-block only: value + ∂sw/∂w (no ∂sw/∂udg). Mirrors
// the ABI `HdgSourcewonly` (backend/Model/.../HdgSourcewonly.cpp) used inside
// the local Newton iteration in wequation.hpp where `u` is frozen.
template <class M, class T=dstype, class I=Int>
void hdg_sourcewonly_kernel(T* sw, T* sw_wdg,
                            const T* xdg, const T* udg, const T* odg,
                            const T* wdg, const T* /*uinf*/, const T* param,
                            T t, int /*modelnumber*/, int ng,
                            int /*nc*/, int /*ncu*/, int /*nd*/, int /*ncx*/, int /*nco*/,
                            int /*ncw_runtime*/)
{
    using dstype=T;
    static_assert(is_sourcew_model_v<M>);
    constexpr int nd = M::nd, ncu = M::ncu, ncw = M::ncw, nco = M::nco;
    constexpr int Nq = ncu * (1 + nd);
    constexpr int nco_buf = (nco > 0) ? nco : 1;

    if constexpr (ncw > 0) {
        Kokkos::parallel_for("exasim::hdg_sourcewonly_kernel", ng, KOKKOS_LAMBDA(size_t i) {
            (void)odg; (void)wdg;  // HOT.6.2 nvcc force-capture: see /tmp/patch_constexpr_capture.py
            T x[nd], uq[Nq], v[nco_buf], w[ncw];
            for (int k = 0; k < nd;  ++k) x [k] = xdg[k * ng + i];
            for (int k = 0; k < Nq;  ++k) uq[k] = udg[k * ng + i];
            if (nco > 0) for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
            for (int k = 0; k < ncw; ++k) w [k] = wdg[k * ng + i];

            // value
            T sw_local[ncw];
            M::sourcew(sw_local, x, uq, v, w, param, /*uinf=*/nullptr, t);
            for (int k = 0; k < ncw; ++k) sw[k * ng + i] = sw_local[k];

            // ∂sw/∂w  (size ncw*ncw, input-index-outer)
            T sw_w[ncw * ncw];
            M::sourcew_jac_w(sw_w, x, uq, v, w, param, /*uinf=*/nullptr, t);
            for (int k = 0; k < ncw * ncw; ++k) sw_wdg[k * ng + i] = sw_w[k];
        });
    } else {
        (void)sw; (void)sw_wdg;
        (void)xdg; (void)udg; (void)odg; (void)wdg; (void)param; (void)t; (void)ng;
    }
}

// ---- ABI-signature forwarders (routed by EXASIM_LEGACY_W_CALL) ----
//
// The templated HDG w-equation chain (backend/Discretization/wequation.hpp)
// dispatches through `EXASIM_LEGACY_W_CALL(HdgSourcew[only], …)`: for the
// AbiAdapter build it calls `common.driver_abi->hdgjac.HdgSourcew[only]`,
// and for a concrete Model `M` it calls `exasim::HdgSourcew[only]<M>` below,
// which forwards to the kernels above. Signatures match the libpdemodel ABI
// `HdgSourcew` / `HdgSourcewonly` exactly, so the dispatch is symmetric.
template <class M, class T=dstype, class I=Int>
inline void HdgSourcew(T* f, T* f_udg, T* f_wdg,
                       const T* xdg, const T* udg, const T* odg, const T* wdg,
                       const T* uinf, const T* param, T time, int modelnumber, int ng,
                       int nc, int ncu, int nd, int ncx, int nco, int ncw)
{
    hdg_sourcew_kernel<M, T>(f, f_udg, f_wdg, xdg, udg, odg, wdg, uinf, param,
                             time, modelnumber, ng, nc, ncu, nd, ncx, nco, ncw);
}

template <class M, class T=dstype, class I=Int>
inline void HdgSourcewonly(T* f, T* f_wdg,
                           const T* xdg, const T* udg, const T* odg, const T* wdg,
                           const T* uinf, const T* param, T time, int modelnumber, int ng,
                           int nc, int ncu, int nd, int ncx, int nco, int ncw)
{
    hdg_sourcewonly_kernel<M, T>(f, f_wdg, xdg, udg, odg, wdg, uinf, param,
                                 time, modelnumber, ng, nc, ncu, nd, ncx, nco, ncw);
}

} // namespace exasim
