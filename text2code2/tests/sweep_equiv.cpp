// In-process equivalence comparator: includes the pyt2c-generated and the C++ text2code
// -generated my_model.hpp for the SAME model in two namespaces and compares every guaranteed
// kernel at a fixed input point with a NaN/Inf-aware tolerance. Guaranteed methods (always
// emitted for an Exasim model): flux, source, tdfunc, initu, fbou_hdg + flux_jac_uq,
// source_jac_uq, fbou_hdg_jac_uq, fbou_hdg_jac_uh.
//
//   c++ -std=c++17 -O2 -I<shim> -DPY_HEADER='"a.hpp"' -DCX_HEADER='"b.hpp"' sweep_equiv.cpp -o cmp
#include <cmath>
#include <cstdio>
#include <cstdint>

using dstype = double;
namespace Kokkos {
template <class A, class B> inline double pow(A a, B b) { return std::pow((double)a, (double)b); }
inline double sqrt(double x){return std::sqrt(x);} inline double exp(double x){return std::exp(x);}
inline double log(double x){return std::log(x);}   inline double sin(double x){return std::sin(x);}
inline double cos(double x){return std::cos(x);}   inline double tan(double x){return std::tan(x);}
inline double asin(double x){return std::asin(x);} inline double acos(double x){return std::acos(x);}
inline double atan(double x){return std::atan(x);} inline double sinh(double x){return std::sinh(x);}
inline double cosh(double x){return std::cosh(x);} inline double tanh(double x){return std::tanh(x);}
inline double fabs(double x){return std::fabs(x);} inline double atan2(double a,double b){return std::atan2(a,b);}
}
#define KOKKOS_INLINE_FUNCTION
template <class T> struct ModelDefaults {};

namespace PY { using ::dstype; using ::ModelDefaults; namespace Kokkos = ::Kokkos;
#include PY_HEADER
}
namespace CX { using ::dstype; using ::ModelDefaults; namespace Kokkos = ::Kokkos;
#include CX_HEADER
}

static double val(int i){ return 0.4 + 0.31*((i*2654435761u)%89)/89.0; }
static int mism = 0, checked = 0;

static bool eq(double a, double b){
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) && std::isinf(b)) return (a>0)==(b>0);
    double d = std::fabs(a-b), s = std::fabs(a)+std::fabs(b);
    return d <= 1e-9*(1.0+s);
}
static void cmp(const char* nm, const double* a, const double* b, int m){
    for (int i=0;i<m;++i){ ++checked; if(!eq(a[i],b[i])){ ++mism;
        if (mism<=5) std::printf("    MISMATCH %s[%d]: py=%.15g cx=%.15g\n", nm, i, a[i], b[i]); } }
}

int main(){
    // Separate driver that fills inputs once and calls both models' methods.
    double x[16],uq[64],v[16],w[16],mu[64],uh[16],n[8],tau[8],uinf[16];
    for(int i=0;i<16;++i){x[i]=val(i+1);v[i]=val(i+40);w[i]=val(i+50);uh[i]=val(i+60);uinf[i]=val(i+70);}
    for(int i=0;i<64;++i){uq[i]=val(i+3);mu[i]=0.3+val(i+100);}
    for(int i=0;i<8;++i){n[i]=val(i+2);tau[i]=1.0+0.1*i;}
    uq[0]=1.2; uh[0]=1.15; mu[0]=1.4; double t=0.3;
    static double fp[512], fc[512];
    auto Z=[&](){for(int i=0;i<512;++i){fp[i]=0;fc[i]=0;}};
    const int nd=PY::PdeModel::nd, ncu=PY::PdeModel::ncu, Nq=PY::PdeModel::Nq;
    if (PY::PdeModel::nd!=CX::PdeModel::nd || PY::PdeModel::ncu!=CX::PdeModel::ncu){
        std::printf("    SIZE MISMATCH\n"); return 2; }

    Z(); PY::PdeModel::flux(fp,x,uq,v,w,mu,uinf,t);  CX::PdeModel::flux(fc,x,uq,v,w,mu,uinf,t);  cmp("flux",fp,fc,ncu*(1+nd));
    Z(); PY::PdeModel::source(fp,x,uq,v,w,mu,uinf,t);CX::PdeModel::source(fc,x,uq,v,w,mu,uinf,t);cmp("source",fp,fc,ncu);
    Z(); PY::PdeModel::tdfunc(fp,x,uq,v,w,mu,uinf,t);CX::PdeModel::tdfunc(fc,x,uq,v,w,mu,uinf,t);cmp("tdfunc",fp,fc,ncu);
    Z(); PY::PdeModel::initu(fp,x,uinf,mu);          CX::PdeModel::initu(fc,x,uinf,mu);          cmp("initu",fp,fc,ncu);
    for(int ib=1;ib<=8;++ib){
      Z(); PY::PdeModel::fbou_hdg(fp,ib,x,uq,v,w,uh,n,tau,mu,uinf,t); CX::PdeModel::fbou_hdg(fc,ib,x,uq,v,w,uh,n,tau,mu,uinf,t); cmp("fbou_hdg",fp,fc,ncu);
    }
    Z(); PY::PdeModel::flux_jac_uq(fp,x,uq,v,w,mu,uinf,t);  CX::PdeModel::flux_jac_uq(fc,x,uq,v,w,mu,uinf,t);  cmp("flux_jac_uq",fp,fc,ncu*Nq);
    Z(); PY::PdeModel::source_jac_uq(fp,x,uq,v,w,mu,uinf,t);CX::PdeModel::source_jac_uq(fc,x,uq,v,w,mu,uinf,t);cmp("source_jac_uq",fp,fc,ncu*Nq);
    for(int ib=1;ib<=8;++ib){
      Z(); PY::PdeModel::fbou_hdg_jac_uq(fp,ib,x,uq,v,w,uh,n,tau,mu,uinf,t); CX::PdeModel::fbou_hdg_jac_uq(fc,ib,x,uq,v,w,uh,n,tau,mu,uinf,t); cmp("fbou_hdg_jac_uq",fp,fc,ncu*Nq);
      Z(); PY::PdeModel::fbou_hdg_jac_uh(fp,ib,x,uq,v,w,uh,n,tau,mu,uinf,t); CX::PdeModel::fbou_hdg_jac_uh(fc,ib,x,uq,v,w,uh,n,tau,mu,uinf,t); cmp("fbou_hdg_jac_uh",fp,fc,ncu*ncu);
    }
    std::printf("    checked=%d mismatches=%d\n", checked, mism);
    return mism==0 ? 0 : 1;
}
