// Numeric-equivalence harness for a generated my_model.hpp.
//
// Compiles the header with dstype=double and a std-backed Kokkos:: shim, then
// evaluates every generated kernel at a fixed, physically-plausible input point
// and prints the results at full precision. Two headers that are semantically
// equivalent (e.g. the C++ text2code golden vs the pyt2c output, which differ
// only in CSE temp ordering) produce byte-identical output here.
//
//   c++ -std=c++17 -O2 -DMODEL_HEADER='"path/to/my_model.hpp"' equiv_harness.cpp -o h
#include <cmath>
#include <cstdio>
#include <vector>

using dstype = double;

namespace Kokkos {
template <class A, class B> inline double pow(A a, B b) { return std::pow((double)a, (double)b); }
inline double sqrt(double x) { return std::sqrt(x); }
inline double exp(double x)  { return std::exp(x); }
inline double log(double x)  { return std::log(x); }
inline double sin(double x)  { return std::sin(x); }
inline double cos(double x)  { return std::cos(x); }
inline double tan(double x)  { return std::tan(x); }
inline double asin(double x) { return std::asin(x); }
inline double acos(double x) { return std::acos(x); }
inline double atan(double x) { return std::atan(x); }
inline double sinh(double x) { return std::sinh(x); }
inline double cosh(double x) { return std::cosh(x); }
inline double tanh(double x) { return std::tanh(x); }
inline double fabs(double x) { return std::fabs(x); }
inline double atan2(double a, double b) { return std::atan2(a, b); }
}  // namespace Kokkos

#define KOKKOS_INLINE_FUNCTION
template <class T> struct ModelDefaults {};

#include MODEL_HEADER

// A deterministic, physically-plausible input point. Positive density/energy so
// the Sutherland viscosity / pressure stay real. Index i just gets a distinct value.
static double val(int i) { return 0.5 + 0.37 * ((i * 2654435761u) % 97) / 97.0; }

int main() {
    constexpr int nd = PdeModel::nd, ncu = PdeModel::ncu;
    constexpr int Nq = PdeModel::Nq, nco = PdeModel::nco, ncw = PdeModel::ncw;
    constexpr int nparam = PdeModel::nparam, ntau = PdeModel::ntau;

    double x[16], uq[64], v[16], w[16], mu[64], uh[16], n[8], tau[8], uinf[16], uext[16];
    for (int i = 0; i < 16; ++i) { x[i]=val(i+1); v[i]=val(i+40); w[i]=val(i+50);
        uh[i]=val(i+60); uinf[i]=val(i+70); uext[i]=val(i+80); }
    for (int i = 0; i < 64; ++i) { uq[i]=val(i+3); mu[i]=val(i+100); }
    for (int i = 0; i < 8;  ++i) { n[i]=val(i+2); tau[i]=1.0 + 0.1*i; }
    // keep density-like states away from 0
    uq[0]=1.2; uh[0]=1.15; mu[0]=1.4; mu[1]=100.0; mu[2]=0.72; mu[3]=0.2;
    double t = 0.3;
    double f[512];

    auto dump = [&](const char* nm, int m) {
        printf("%s:", nm);
        for (int i = 0; i < m; ++i) printf(" %.14g", f[i]);
        printf("\n");
    };
    for (int i = 0; i < 512; ++i) f[i] = 0.0;
#define CALL(expr, nm, m) do { for (int i=0;i<512;++i) f[i]=0.0; expr; dump(nm, m); } while(0)

    CALL(PdeModel::flux(f, x, uq, v, w, mu, uinf, t), "flux", ncu*(1+nd));
    CALL(PdeModel::source(f, x, uq, v, w, mu, uinf, t), "source", ncu);
    CALL(PdeModel::tdfunc(f, x, uq, v, w, mu, uinf, t), "tdfunc", ncu);
    CALL(PdeModel::initu(f, x, uinf, mu), "initu", ncu);
    for (int ib = 1; ib <= 6; ++ib) {
        char b[32]; snprintf(b, 32, "fbou_hdg[%d]", ib);
        CALL(PdeModel::fbou_hdg(f, ib, x, uq, v, w, uh, n, tau, mu, uinf, t), b, ncu);
    }
    CALL(PdeModel::flux_jac_uq(f, x, uq, v, w, mu, uinf, t), "flux_jac_uq", ncu*Nq);
    CALL(PdeModel::source_jac_uq(f, x, uq, v, w, mu, uinf, t), "source_jac_uq", ncu*Nq);
    for (int ib = 1; ib <= 6; ++ib) {
        char b[40]; snprintf(b, 40, "fbou_hdg_jac_uq[%d]", ib);
        CALL(PdeModel::fbou_hdg_jac_uq(f, ib, x, uq, v, w, uh, n, tau, mu, uinf, t), b, ncu*Nq);
        snprintf(b, 40, "fbou_hdg_jac_uh[%d]", ib);
        CALL(PdeModel::fbou_hdg_jac_uh(f, ib, x, uq, v, w, uh, n, tau, mu, uinf, t), b, ncu*ncu);
    }
#ifdef HAS_COUPLING
    CALL(PdeModel::fint(f, 3, x, uq, v, w, uh, n, tau, mu, uinf, t), "fint", PdeModel::nfint);
    CALL(PdeModel::fext(f, 3, x, uq, v, w, uh, n, uext, tau, mu, uinf, t), "fext", PdeModel::nfext);
    CALL(PdeModel::fint_jac_uq(f, 3, x, uq, v, w, uh, n, tau, mu, uinf, t), "fint_jac_uq", PdeModel::nfint*Nq);
    CALL(PdeModel::fext_jac_uq(f, 3, x, uq, v, w, uh, n, uext, tau, mu, uinf, t), "fext_jac_uq", PdeModel::nfext*Nq);
#endif
    return 0;
}
