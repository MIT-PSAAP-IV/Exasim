#pragma once

#include <driver_abi.hpp>
#include "enzyme_intrinsics.hpp"

#ifdef HAVE_ENZYME
namespace exasim {
namespace enzyme {
namespace detail {

enum class Volume { flux, source, avfield, tdfunc, sourcew, materialstate };

template <class M, class T, Volume V>
struct VolumePoint {
    KOKKOS_INLINE_FUNCTION static void eval(T* f, const T* x, const T* u,
        const T* o, const T* w, const T* param, const T* uinf, T time)
    {
        if constexpr (V == Volume::flux) M::flux(f, x, u, o, w, param, uinf, time);
        if constexpr (V == Volume::source) M::source(f, x, u, o, w, param, uinf, time);
        if constexpr (V == Volume::avfield) M::avfield(f, x, u, o, w, param, uinf, time);
        if constexpr (V == Volume::tdfunc) M::tdfunc(f, x, u, o, w, param, uinf, time);
        if constexpr (V == Volume::sourcew) M::sourcew(f, x, u, o, w, param, uinf, time);
        if constexpr (V == Volume::materialstate) M::materialstate(f, x, u, o, w, param, uinf, time);
    }
};

// AV stores values by node, component, and element.
// Other volume terms store values by quadrature point and component.
// Each point has local arrays and a separate output location.
// The kernel sets output values and derivatives to zero before each call.
template <class M, class T, Volume V, bool ActiveO, bool Nodal = false>
void volume(T* f, T* df, const T* xg, const T* ug, const T* dug,
    const T* og, const T* dog, const T* wg, const T* dwg,
    const T* uinf, const T* param, T time, int ng, int nc, int ncx,
    int nco, int ncw, int nout, int npe = 1)
{
    constexpr int ND = M::nd, NU = M::ncu, NC = NU * (1 + ND);
    constexpr int NO = M::nco, NW = M::ncw;
    constexpr int NF = V == Volume::flux ? NU*ND : (V == Volume::sourcew ? NW : NU);
    Kokkos::parallel_for("exasim::enzyme::volume_jvp", ng, KOKKOS_LAMBDA(int i) {
        (void)og; (void)dog; (void)wg; (void)dwg;
        T x[ND], u[NC] = {}, du[NC] = {}, o[NO > 0 ? NO : 1] = {}, d_o[NO > 0 ? NO : 1] = {};
        T w[NW > 0 ? NW : 1] = {}, dw[NW > 0 ? NW : 1] = {};
        T out[NF > 0 ? NF : 1] = {}, dout[NF > 0 ? NF : 1] = {};
        const int p = Nodal ? i % npe : i;
        const int e = Nodal ? i / npe : 0;
        const int stride = Nodal ? npe : ng;
        for (int k = 0; k < ND; ++k) x[k] = xg[p + stride*(k + ncx*e)];
        for (int k = 0; k < NC && k < nc; ++k) {
            u[k] = ug[p + stride*(k + nc*e)];
            du[k] = dug[p + stride*(k + nc*e)];
        }
        for (int k = 0; k < NO; ++k) {
            o[k] = og[p + stride*(k + nco*e)];
            if constexpr (ActiveO) d_o[k] = dog[p + stride*(k + nco*e)];
        }
        for (int k = 0; k < NW; ++k) {
            w[k] = wg[p + stride*(k + ncw*e)];
            dw[k] = dwg[p + stride*(k + ncw*e)];
        }
        if constexpr (ActiveO) {
            __enzyme_fwddiff((void*)VolumePoint<M,T,V>::eval,
                enzyme_dup, out, dout, enzyme_const, x,
                enzyme_dup, u, du, enzyme_dup, o, d_o, enzyme_dup, w, dw,
                enzyme_const, param, enzyme_const, uinf, enzyme_const, time);
        } else {
            __enzyme_fwddiff((void*)VolumePoint<M,T,V>::eval,
                enzyme_dup, out, dout, enzyme_const, x,
                enzyme_dup, u, du, enzyme_const, o, enzyme_dup, w, dw,
                enzyme_const, param, enzyme_const, uinf, enzyme_const, time);
        }
        for (int k = 0; k < nout; ++k) {
            f[p + stride*(k + nout*e)] = out[k];
            df[p + stride*(k + nout*e)] = dout[k];
        }
    });
}

template <class M, class T, bool Ubou>
struct BoundaryPoint {
    KOKKOS_INLINE_FUNCTION static void eval(T* f, int ib, const T* x,
        const T* u, const T* o, const T* w, const T* uh, const T* n,
        const T* tau, const T* param, const T* uinf, T time)
    {
        if constexpr (Ubou) M::ubou(f, ib, x, u, o, w, uh, n, tau, param, uinf, time);
        else M::fbou(f, ib, x, u, o, w, uh, n, tau, param, uinf, time);
    }
};

template <class M, class T, bool Ubou>
void boundary(T* f, T* df, const T* xg, const T* ug, const T* dug,
    const T* og, const T* dog, const T* wg, const T* dwg,
    const T* uhg, const T* duhg, const T* nlg, const T* tau,
    const T* uinf, const T* param, T time, int ib, int ng, int nc)
{
    constexpr int ND = M::nd, NU = M::ncu, NC = NU*(1 + ND), NO = M::nco, NW = M::ncw;
    Kokkos::parallel_for("exasim::enzyme::boundary_jvp", ng, KOKKOS_LAMBDA(int i) {
        (void)og; (void)dog; (void)wg; (void)dwg; (void)duhg;
        T x[ND], n[ND], u[NC] = {}, du[NC] = {}, uh[NU], duh[NU] = {};
        T o[NO > 0 ? NO : 1] = {}, d_o[NO > 0 ? NO : 1] = {};
        T w[NW > 0 ? NW : 1] = {}, dw[NW > 0 ? NW : 1] = {};
        T out[NU] = {}, dout[NU] = {};
        for (int k = 0; k < ND; ++k) { x[k] = xg[k*ng+i]; n[k] = nlg[k*ng+i]; }
        for (int k = 0; k < NC && k < nc; ++k) { u[k] = ug[k*ng+i]; du[k] = dug[k*ng+i]; }
        for (int k = 0; k < NO; ++k) {
            o[k] = og[k*ng+i];
            if constexpr (!Ubou) d_o[k] = dog[k*ng+i];
        }
        for (int k = 0; k < NW; ++k) { w[k] = wg[k*ng+i]; dw[k] = dwg[k*ng+i]; }
        for (int k = 0; k < NU; ++k) {
            uh[k] = uhg[k*ng+i];
            if (duhg) duh[k] = duhg[k*ng+i];
        }
        if constexpr (Ubou) {
            __enzyme_fwddiff((void*)BoundaryPoint<M,T,Ubou>::eval,
                enzyme_dup, out, dout, enzyme_const, ib, enzyme_const, x,
                enzyme_dup, u, du, enzyme_const, o, enzyme_dup, w, dw,
                enzyme_dup, uh, duh, enzyme_const, n, enzyme_const, tau,
                enzyme_const, param, enzyme_const, uinf, enzyme_const, time);
        } else {
            __enzyme_fwddiff((void*)BoundaryPoint<M,T,Ubou>::eval,
                enzyme_dup, out, dout, enzyme_const, ib, enzyme_const, x,
                enzyme_dup, u, du, enzyme_dup, o, d_o, enzyme_dup, w, dw,
                enzyme_dup, uh, duh, enzyme_const, n, enzyme_const, tau,
                enzyme_const, param, enzyme_const, uinf, enzyme_const, time);
        }
        for (int k = 0; k < NU; ++k) { f[k*ng+i] = out[k]; df[k*ng+i] = dout[k]; }
    });
}

template <class M, class T, bool Stab>
struct FacePoint {
    KOKKOS_INLINE_FUNCTION static void eval(T* f, const T* x,
        const T* u1, const T* u2, const T* o1, const T* o2,
        const T* w1, const T* w2, const T* uh, const T* n,
        const T* tau, const T* param, const T* uinf, T time)
    {
        if constexpr (Stab) M::stab(f,x,u1,u2,o1,o2,w1,w2,uh,n,tau,param,uinf,time);
        else M::fhat(f,x,u1,u2,o1,o2,w1,w2,uh,n,tau,param,uinf,time);
    }
};

template <class M, class T, bool Stab>
void face(T* f, T* df, const T* xg, const T* ug1, const T* dug1,
    const T* ug2, const T* dug2, const T* og1, const T* dog1,
    const T* og2, const T* dog2, const T* wg1, const T* dwg1,
    const T* wg2, const T* dwg2, const T* uhg, const T* duhg,
    const T* nlg, const T* tau, const T* uinf, const T* param, T time, int ng, int nc)
{
    constexpr int ND = M::nd, NU = M::ncu, NC = NU*(1 + ND), NO = M::nco, NW = M::ncw;
    Kokkos::parallel_for("exasim::enzyme::face_jvp", ng, KOKKOS_LAMBDA(int i) {
        (void)og1; (void)dog1; (void)og2; (void)dog2;
        (void)wg1; (void)dwg1; (void)wg2; (void)dwg2;
        T x[ND], n[ND], u1[NC] = {}, du1[NC] = {}, u2[NC] = {}, du2[NC] = {}, uh[NU], duh[NU];
        T o1[NO > 0 ? NO : 1] = {}, do1[NO > 0 ? NO : 1] = {};
        T o2[NO > 0 ? NO : 1] = {}, do2[NO > 0 ? NO : 1] = {};
        T w1[NW > 0 ? NW : 1] = {}, dw1[NW > 0 ? NW : 1] = {};
        T w2[NW > 0 ? NW : 1] = {}, dw2[NW > 0 ? NW : 1] = {};
        T out[NU] = {}, dout[NU] = {};
        for (int k = 0; k < ND; ++k) { x[k] = xg[k*ng+i]; n[k] = nlg[k*ng+i]; }
        for (int k = 0; k < NC && k < nc; ++k) {
            u1[k] = ug1[k*ng+i]; du1[k] = dug1[k*ng+i];
            u2[k] = ug2[k*ng+i]; du2[k] = dug2[k*ng+i];
        }
        for (int k = 0; k < NO; ++k) {
            o1[k] = og1[k*ng+i]; do1[k] = dog1[k*ng+i];
            o2[k] = og2[k*ng+i]; do2[k] = dog2[k*ng+i];
        }
        for (int k = 0; k < NW; ++k) {
            w1[k] = wg1[k*ng+i]; dw1[k] = dwg1[k*ng+i];
            w2[k] = wg2[k*ng+i]; dw2[k] = dwg2[k*ng+i];
        }
        for (int k = 0; k < NU; ++k) { uh[k] = uhg[k*ng+i]; duh[k] = duhg[k*ng+i]; }
        __enzyme_fwddiff((void*)FacePoint<M,T,Stab>::eval,
            enzyme_dup, out, dout, enzyme_const, x,
            enzyme_dup, u1, du1, enzyme_dup, u2, du2,
            enzyme_dup, o1, do1, enzyme_dup, o2, do2,
            enzyme_dup, w1, dw1, enzyme_dup, w2, dw2,
            enzyme_dup, uh, duh, enzyme_const, n, enzyme_const, tau,
            enzyme_const, param, enzyme_const, uinf, enzyme_const, time);
        // These methods replace the output values.
        // This behavior matches the KokkosFhat and KokkosStab value kernels.
        for (int k = 0; k < NU; ++k) { f[k*ng+i] = out[k]; df[k*ng+i] = dout[k]; }
    });
}
} // namespace detail

// Model providers and typed drivers use these functions.
// Their signatures match the driver ABI.
template <class M, class T = dstype>
void flux_jvp(T* f, T* df, const T* x, const T* u, const T* du,
    const T* o, const T* d_o, const T* w, const T* dw,
    const T* uinf, const T* param, T time, int, int ng,
    int nc, int, int, int ncx, int nco, int ncw)
{
    detail::volume<M,T,detail::Volume::flux,true>(f,df,x,u,du,o,d_o,w,dw,
        uinf,param,time,ng,nc,ncx,nco,ncw,M::ncu*M::nd);
}

template <class M, class T = dstype>
void source_jvp(T* f, T* df, const T* x, const T* u, const T* du,
    const T* o, const T* d_o, const T* w, const T* dw,
    const T* uinf, const T* param, T time, int, int ng,
    int nc, int, int, int ncx, int nco, int ncw)
{
    detail::volume<M,T,detail::Volume::source,true>(f,df,x,u,du,o,d_o,w,dw,
        uinf,param,time,ng,nc,ncx,nco,ncw,M::ncu);
}

template <class M, class T = dstype>
void tdfunc_jvp(T* f, T* df, const T* x, const T* u, const T* du,
    const T* o, const T* d_o, const T* w, const T* dw,
    const T* uinf, const T* param, T time, int, int ng,
    int nc, int, int, int ncx, int nco, int ncw)
{
    detail::volume<M,T,detail::Volume::tdfunc,true>(f,df,x,u,du,o,d_o,w,dw,
        uinf,param,time,ng,nc,ncx,nco,ncw,M::ncu);
}

template <class M, class T = dstype>
void avfield_jvp(T* f, T* df, const T* x, const T* u, const T* du,
    const T* o, const T* w, const T* dw, const T* uinf, const T* param,
    T time, int, int ng, int nc, int, int, int ncx, int nco, int ncw,
    int nce, int npe, int)
{
    if (nc > M::ncu * (1 + M::nd))
        Kokkos::abort("Avfield input count exceeds model capacity.");
    if (nce > M::ncu) Kokkos::abort("Avfield JVP output count exceeds Model::ncu.");
    detail::volume<M,T,detail::Volume::avfield,false,true>(f,df,x,u,du,o,nullptr,w,dw,
        uinf,param,time,ng,nc,ncx,nco,ncw,nce,npe);
}

template <class M, class T = dstype>
void fbou_jvp(T* f, T* df, const T* x, const T* u, const T* du,
    const T* o, const T* d_o, const T* w, const T* dw,
    const T* uh, const T* duh, const T* nl, const T* tau,
    const T* uinf, const T* param, T time, int, int ib, int ng,
    int nc, int, int, int, int, int)
{
    detail::boundary<M,T,false>(f,df,x,u,du,o,d_o,w,dw,uh,duh,nl,tau,uinf,param,time,ib,ng,nc);
}

template <class M, class T = dstype>
void ubou_jvp(T* f, T* df, const T* x, const T* u, const T* du,
    const T* o, const T* w, const T* dw, const T* uh, const T* duh, const T* nl,
    const T* tau, const T* uinf, const T* param, T time, int, int ib, int ng,
    int nc, int, int, int, int, int)
{
    detail::boundary<M,T,true>(f,df,x,u,du,o,nullptr,w,dw,uh,duh,nl,tau,uinf,param,time,ib,ng,nc);
}

template <class M, class T = dstype>
void fhat_jvp(T* f, T* df, const T* x, const T* u1, const T* du1,
    const T* u2, const T* du2, const T* o1, const T* do1,
    const T* o2, const T* do2, const T* w1, const T* dw1,
    const T* w2, const T* dw2, const T* uh, const T* duh,
    const T* nl, const T* tau, const T* uinf, const T* param,
    T time, int, int ng, int nc, int, int, int, int, int)
{
    detail::face<M,T,false>(f,df,x,u1,du1,u2,du2,o1,do1,o2,do2,w1,dw1,w2,dw2,
        uh,duh,nl,tau,uinf,param,time,ng,nc);
}

template <class M, class T = dstype>
void stab_jvp(T* f, T* df, const T* x, const T* u1, const T* du1,
    const T* u2, const T* du2, const T* o1, const T* do1,
    const T* o2, const T* do2, const T* w1, const T* dw1,
    const T* w2, const T* dw2, const T* uh, const T* duh,
    const T* nl, const T* tau, const T* uinf, const T* param,
    T time, int, int ng, int nc, int, int, int, int, int)
{
    detail::face<M,T,true>(f,df,x,u1,du1,u2,du2,o1,do1,o2,do2,w1,dw1,w2,dw2,
        uh,duh,nl,tau,uinf,param,time,ng,nc);
}

// Auxiliary equations and material maps need small derivative matrices.
// Enzyme computes one column for each input component.
// The input index is the outer index: jac[(input*NF+output)*ng+i].
template <class M, class T, detail::Volume V, int NF, bool WithU>
void volume_jac_impl(T* f, T* fu, T* fw, const T* xg, const T* ug,
    const T* og, const T* wg, const T* uinf, const T* param, T time, int ng, int nc)
{
    constexpr int ND = M::nd, NC = M::ncu*(1 + ND), NO = M::nco, NW = M::ncw;
    if constexpr (NF > 0) {
        Kokkos::parallel_for("exasim::enzyme::volume_jac", ng, KOKKOS_LAMBDA(int i) {
            (void)og; (void)fu;
            T x[ND], u[NC] = {}, o[NO > 0 ? NO : 1] = {}, w[NW > 0 ? NW : 1] = {};
            for (int k = 0; k < ND; ++k) x[k] = xg[k*ng+i];
            for (int k = 0; k < NC && k < nc; ++k) u[k] = ug[k*ng+i];
            for (int k = 0; k < NO; ++k) o[k] = og[k*ng+i];
            for (int k = 0; k < NW; ++k) w[k] = wg[k*ng+i];
            const int NU = WithU ? nc : 0;
            for (int j = 0; j < NU + NW; ++j) {
                T du[NC] = {}, dw[NW > 0 ? NW : 1] = {}, out[NF] = {}, dout[NF] = {};
                if (j < NU) { if (j < NC) du[j] = 1; }
                else dw[j - NU] = 1;
                __enzyme_fwddiff((void*)detail::VolumePoint<M,T,V>::eval,
                    enzyme_dup, out, dout, enzyme_const, x,
                    enzyme_dup, u, du, enzyme_const, o, enzyme_dup, w, dw,
                    enzyme_const, param, enzyme_const, uinf, enzyme_const, time);
                for (int k = 0; k < NF; ++k) {
                    f[k*ng+i] = out[k];
                    if constexpr (WithU) {
                        if (j < NU) fu[(j*NF+k)*ng+i] = dout[k];
                        else fw[((j-NU)*NF+k)*ng+i] = dout[k];
                    } else fw[(j*NF+k)*ng+i] = dout[k];
                }
            }
        });
    }
}

template <class M, class T = dstype>
void sourcew_jac(T* f, T* fu, T* fw, const T* x, const T* u,
    const T* o, const T* w, const T* uinf, const T* param, T time,
    int, int ng, int nc, int, int, int, int, int)
{
    volume_jac_impl<M,T,detail::Volume::sourcew,M::ncw,true>(f,fu,fw,x,u,o,w,uinf,param,time,ng,nc);
}

template <class M, class T = dstype>
void sourcew_jac_only(T* f, T* fw, const T* x, const T* u,
    const T* o, const T* w, const T* uinf, const T* param, T time,
    int, int ng, int nc, int, int, int, int, int)
{
    volume_jac_impl<M,T,detail::Volume::sourcew,M::ncw,false>(f,nullptr,fw,x,u,o,w,uinf,param,time,ng,nc);
}

template <class M, class T = dstype>
void materialstate_jac(T* f, T* fu, T* fw, const T* x, const T* u,
    const T* o, const T* w, const T* uinf, const T* param, T time,
    int, int ng, int nc, int, int, int, int, int, int)
{
    volume_jac_impl<M,T,detail::Volume::materialstate,M::nmaterialstate,true>(
        f,fu,fw,x,u,o,w,uinf,param,time,ng,nc);
}

} // namespace enzyme
} // namespace exasim
#endif
