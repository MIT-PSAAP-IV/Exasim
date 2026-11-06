// Nonlinear pointwise physics for Enzyme tests. No hand-written derivatives.
#pragma once
#include "modeldefaults.hpp"

struct PdeModel : ModelDefaults<PdeModel> {
    static constexpr int nd = 2, ncu = 2, nco = 2, ncw = 1, nparam = 1, ntau = 2;
    static constexpr int Nq = ncu * (1 + nd);

    KOKKOS_INLINE_FUNCTION static void initu(dstype* f, const dstype*, const dstype*, const dstype*) {
        f[0] = f[1] = 0;
    }
    KOKKOS_INLINE_FUNCTION static void flux(dstype* f, const dstype* x, const dstype* u,
        const dstype* v, const dstype* w, const dstype* mu, const dstype* eta, dstype t) {
        for (int k = 0; k < ncu*nd; ++k)
            f[k] = (k+1)*u[k]*u[k] + u[(k+2)%Nq]*v[0] + v[1]*w[0]
                 + Kokkos::sin(w[0]) + mu[0]*x[0] + eta[0] + t*u[k];
    }
    KOKKOS_INLINE_FUNCTION static void source(dstype* f, const dstype* x, const dstype* u,
        const dstype* v, const dstype* w, const dstype* mu, const dstype* eta, dstype t) {
        for (int k = 0; k < ncu; ++k)
            f[k] = Kokkos::exp(dstype(0.1)*u[k]) + v[0]*u[k+2] + w[0]*w[0]
                 + v[1] + mu[0]*x[1] + eta[0] + t;
    }
    KOKKOS_INLINE_FUNCTION static void avfield(dstype* f, const dstype* x, const dstype* u,
        const dstype* v, const dstype* w, const dstype* mu, const dstype* eta, dstype t) {
        for (int k = 0; k < ncu; ++k)
            f[k] = u[k]*u[k+2] + Kokkos::sin(w[0]) + v[k] + mu[0]*x[0] + eta[0] + t;
    }
    KOKKOS_INLINE_FUNCTION static void tdfunc(dstype* f, const dstype*, const dstype* u,
        const dstype* v, const dstype* w, const dstype*, const dstype*, dstype) {
        for (int k = 0; k < ncu; ++k) f[k] = 2 + u[k]*u[k] + v[k]*w[0];
    }
    KOKKOS_INLINE_FUNCTION static void fbou(dstype* f, int ib, const dstype* x, const dstype* u,
        const dstype* v, const dstype* w, const dstype* h, const dstype* n,
        const dstype* tau, const dstype* mu, const dstype* eta, dstype t) {
        for (int k = 0; k < ncu; ++k)
            f[k] = (ib+1)*u[k]*u[k] + v[0]*u[k+2] + w[0]*h[k]
                 + Kokkos::sin(h[1-k]) + n[k]*tau[k] + mu[0]*x[0] + eta[0] + t;
    }
    KOKKOS_INLINE_FUNCTION static void ubou(dstype* f, int ib, const dstype* x, const dstype* u,
        const dstype* v, const dstype* w, const dstype* h, const dstype* n,
        const dstype* tau, const dstype* mu, const dstype* eta, dstype t) {
        fbou(f, ib, x, u, v, w, h, n, tau, mu, eta, t);
    }
    KOKKOS_INLINE_FUNCTION static void fhat(dstype* f, const dstype* x,
        const dstype* u1, const dstype* u2, const dstype* v1, const dstype* v2,
        const dstype* w1, const dstype* w2, const dstype* h, const dstype* n,
        const dstype* tau, const dstype* mu, const dstype* eta, dstype t) {
        for (int k = 0; k < ncu; ++k)
            f[k] = u1[k]*u2[k+2] + v1[k]*w2[0] + v2[k]*w1[0] + h[k]*h[k]
                 + n[k]*tau[k] + mu[0]*x[1] + eta[0] + t;
    }
    KOKKOS_INLINE_FUNCTION static void stab(dstype* f, const dstype* x,
        const dstype* u1, const dstype* u2, const dstype* v1, const dstype* v2,
        const dstype* w1, const dstype* w2, const dstype* h, const dstype* n,
        const dstype* tau, const dstype* mu, const dstype* eta, dstype t) {
        fhat(f, x, u1, u2, v1, v2, w1, w2, h, n, tau, mu, eta, t);
        for (int k = 0; k < ncu; ++k) f[k] += tau[k]*(u1[k] - u2[k]);
    }
};
