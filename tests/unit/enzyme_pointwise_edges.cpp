#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "modeldefaults.hpp"
#include "exasim/enzyme_jvp.hpp"

struct AuxiliaryModel : ModelDefaults<AuxiliaryModel> {
    static constexpr int nd = 2, ncu = 2, nco = 1, ncw = 2;
    static constexpr int nmaterialstate = 2;
    KOKKOS_INLINE_FUNCTION static void sourcew(dstype* s, const dstype*,
        const dstype* u, const dstype* o, const dstype* w,
        const dstype*, const dstype*, dstype)
    {
        s[0] = u[0]*u[0] + u[3]*w[0] + o[0]*w[1];
        s[1] = u[1]*w[0] + Kokkos::sin(w[1]) + u[5];
    }
    KOKKOS_INLINE_FUNCTION static void materialstate(dstype* s, const dstype*,
        const dstype* u, const dstype* o, const dstype* w,
        const dstype*, const dstype*, dstype)
    {
        s[0] = u[0]*w[1] + o[0]*o[0];
        s[1] = u[2]*u[2] + Kokkos::exp(w[0]);
    }
};

// This inviscid model has ncu unknowns.
// It has no gradient arrays or auxiliary arrays.
// The kernels must use runtime nc when they read the input buffers.
// The local arrays have space for ncu*(1+nd) components.
struct NoAuxModel : ModelDefaults<NoAuxModel> {
    static constexpr int nd = 2, ncu = 1, nco = 0, ncw = 0;
    KOKKOS_INLINE_FUNCTION static void flux(dstype* f, const dstype*,
        const dstype* u, const dstype*, const dstype*, const dstype*, const dstype*, dstype)
    { f[0] = u[0]*u[0]; f[1] = u[0]*u[0]*u[0]; }
    KOKKOS_INLINE_FUNCTION static void avfield(dstype* f, const dstype*,
        const dstype* u, const dstype*, const dstype*, const dstype*, const dstype*, dstype)
    { f[0] = u[0]*u[0] + u[1] + 2*u[2]; }
    KOKKOS_INLINE_FUNCTION static void fbou(dstype* f, int ib, const dstype*,
        const dstype* u, const dstype*, const dstype*, const dstype* uh,
        const dstype*, const dstype*, const dstype*, const dstype*, dstype)
    { f[0] = ib*u[0]*uh[0]; }
    KOKKOS_INLINE_FUNCTION static void ubou(dstype* f, int ib, const dstype*,
        const dstype* u, const dstype*, const dstype*, const dstype* uh,
        const dstype*, const dstype*, const dstype*, const dstype*, dstype)
    { f[0] = ib*u[0]*uh[0]; }
    KOKKOS_INLINE_FUNCTION static void fhat(dstype* f, const dstype*,
        const dstype* u1, const dstype* u2, const dstype*, const dstype*,
        const dstype*, const dstype*, const dstype* uh, const dstype*,
        const dstype*, const dstype*, const dstype*, dstype)
    { f[0] = u1[0]*u2[0] + uh[0]*uh[0]; }
};

static void close(dstype got, dstype expected, const char* name)
{
    const dstype tol = sizeof(dstype) == sizeof(float) ? 2e-5 : 2e-12;
    if (!std::isfinite(got) || std::abs(got-expected) > tol*(1+std::abs(expected))) {
        std::fprintf(stderr, "%s: got %.17g, expected %.17g\n", name,
            double(got), double(expected));
        throw std::runtime_error(name);
    }
}

static void auxiliary()
{
    constexpr int ng = 3;
    Kokkos::View<dstype*> x("x",2*ng), u("u",6*ng), o("o",ng), w("w",2*ng);
    Kokkos::View<dstype*> f("f",2*ng), fu("fu",12*ng), fw("fw",4*ng);
    Kokkos::View<dstype*> only("only",2*ng), onlyw("onlyw",4*ng);
    auto hu = Kokkos::create_mirror_view(u);
    auto ho = Kokkos::create_mirror_view(o);
    auto hw = Kokkos::create_mirror_view(w);
    for (int i=0; i<ng; ++i) {
        for (int j=0; j<6; ++j) hu(j*ng+i) = 0.3 + 0.1*j + 0.2*i;
        ho(i) = 0.8 + 0.1*i;
        hw(i) = 0.5 + 0.1*i;
        hw(ng+i) = -0.4 + 0.2*i;
    }
    Kokkos::deep_copy(u,hu); Kokkos::deep_copy(o,ho); Kokkos::deep_copy(w,hw);
    exasim::enzyme::sourcew_jac<AuxiliaryModel,dstype>(f.data(),fu.data(),fw.data(),
        x.data(),u.data(),o.data(),w.data(),nullptr,nullptr,0,0,ng,6,2,2,2,1,2);
    exasim::enzyme::sourcew_jac_only<AuxiliaryModel,dstype>(only.data(),onlyw.data(),
        x.data(),u.data(),o.data(),w.data(),nullptr,nullptr,0,0,ng,6,2,2,2,1,2);
    const auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),f);
    const auto hfu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),fu);
    const auto hfw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),fw);
    const auto hfo = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),only);
    const auto hfwo = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),onlyw);
    for (int i=0; i<ng; ++i) {
        close(hf(i), hu(i)*hu(i) + hu(3*ng+i)*hw(i) + ho(i)*hw(ng+i), "sourcew value0");
        close(hf(ng+i), hu(ng+i)*hw(i) + std::sin(hw(ng+i)) + hu(5*ng+i), "sourcew value1");
        for (int j=0; j<6; ++j) {
            close(hfu((j*2)*ng+i), j==0 ? 2*hu(i) : (j==3 ? hw(i) : 0), "sourcew du0");
            close(hfu((j*2+1)*ng+i), j==1 ? hw(i) : (j==5 ? 1 : 0), "sourcew du1");
        }
        close(hfw(i), hu(3*ng+i), "sourcew dw00");
        close(hfw(ng+i), hu(ng+i), "sourcew dw10");
        close(hfw(2*ng+i), ho(i), "sourcew dw01");
        close(hfw(3*ng+i), std::cos(hw(ng+i)), "sourcew dw11");
        for (int k=0; k<2; ++k) close(hfo(k*ng+i),hf(k*ng+i),"sourcewonly value");
        for (int k=0; k<4; ++k) close(hfwo(k*ng+i),hfw(k*ng+i),"sourcewonly jacobian");
    }

    exasim::enzyme::materialstate_jac<AuxiliaryModel,dstype>(f.data(),fu.data(),fw.data(),
        x.data(),u.data(),o.data(),w.data(),nullptr,nullptr,0,0,ng,6,2,2,2,1,2,2);
    const auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),f);
    const auto hmu = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),fu);
    const auto hmw = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),fw);
    for (int i=0; i<ng; ++i) {
        close(hm(i),hu(i)*hw(ng+i)+ho(i)*ho(i),"material state0");
        close(hm(ng+i),hu(2*ng+i)*hu(2*ng+i)+std::exp(hw(i)),"material state1");
        for (int j=0; j<6; ++j) {
            close(hmu((j*2)*ng+i),j==0 ? hw(ng+i) : 0,"material du0");
            close(hmu((j*2+1)*ng+i),j==2 ? 2*hu(2*ng+i) : 0,"material du1");
        }
        close(hmw(i),0,"material dw00");
        close(hmw(ng+i),std::exp(hw(i)),"material dw10");
        close(hmw(2*ng+i),hu(i),"material dw01");
        close(hmw(3*ng+i),0,"material dw11");
    }
}

static void no_aux()
{
    constexpr int ng = 3;
    Kokkos::View<dstype*> x("x",2*ng), n("n",2*ng), u("u",ng), du("du",ng);
    Kokkos::View<dstype*> v("right",ng), dv("dright",ng), uh("uh",ng), duh("duh",ng);
    Kokkos::View<dstype*> f("f",2*ng), df("df",2*ng), tau("tau",1);
    Kokkos::deep_copy(u,2); Kokkos::deep_copy(du,3);
    Kokkos::deep_copy(v,4); Kokkos::deep_copy(dv,-2);
    Kokkos::deep_copy(uh,5); Kokkos::deep_copy(duh,0.5);
    exasim::enzyme::flux_jvp<NoAuxModel,dstype>(f.data(),df.data(),x.data(),u.data(),du.data(),
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,0,ng,1,1,2,2,0,0);
    auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),f);
    auto hdf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),df);
    for (int i=0; i<ng; ++i) {
        close(hf(i),4,"no-aux flux0"); close(hf(ng+i),8,"no-aux flux1");
        close(hdf(i),12,"no-aux dflux0"); close(hdf(ng+i),36,"no-aux dflux1");
    }
    exasim::enzyme::fbou_jvp<NoAuxModel,dstype>(f.data(),df.data(),x.data(),u.data(),du.data(),
        nullptr,nullptr,nullptr,nullptr,uh.data(),duh.data(),n.data(),tau.data(),
        nullptr,nullptr,0,0,2,ng,1,1,2,2,0,0);
    Kokkos::deep_copy(hf,f); Kokkos::deep_copy(hdf,df);
    for (int i=0; i<ng; ++i) { close(hf(i),20,"no-aux fbou"); close(hdf(i),32,"no-aux dfbou"); }
    exasim::enzyme::ubou_jvp<NoAuxModel,dstype>(f.data(),df.data(),x.data(),u.data(),du.data(),
        nullptr,nullptr,nullptr,uh.data(),duh.data(),n.data(),tau.data(),
        nullptr,nullptr,0,0,2,ng,1,1,2,2,0,0);
    Kokkos::deep_copy(hf,f); Kokkos::deep_copy(hdf,df);
    for (int i=0; i<ng; ++i) { close(hf(i),20,"no-aux ubou"); close(hdf(i),32,"active-trace dubou"); }
    exasim::enzyme::ubou_jvp<NoAuxModel,dstype>(f.data(),df.data(),x.data(),u.data(),du.data(),
        nullptr,nullptr,nullptr,uh.data(),nullptr,n.data(),tau.data(),
        nullptr,nullptr,0,0,2,ng,1,1,2,2,0,0);
    Kokkos::deep_copy(hdf,df);
    for (int i=0; i<ng; ++i) close(hdf(i),30,"fixed-trace dubou");
    exasim::enzyme::fhat_jvp<NoAuxModel,dstype>(f.data(),df.data(),x.data(),u.data(),du.data(),v.data(),dv.data(),
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        uh.data(),duh.data(),n.data(),tau.data(),nullptr,nullptr,0,0,ng,1,1,2,2,0,0);
    Kokkos::deep_copy(hf,f); Kokkos::deep_copy(hdf,df);
    for (int i=0; i<ng; ++i) { close(hf(i),33,"no-aux fhat"); close(hdf(i),13,"no-aux dfhat"); }

    exasim::enzyme::avfield_jvp<NoAuxModel,dstype>(f.data(),df.data(),x.data(),u.data(),du.data(),
        nullptr,nullptr,nullptr,nullptr,nullptr,0,0,ng,1,1,2,2,0,0,1,1,ng);
    Kokkos::deep_copy(hf,f); Kokkos::deep_copy(hdf,df);
    for (int i=0; i<ng; ++i) { close(hf(i),4,"no-gradient avfield"); close(hdf(i),12,"no-gradient avfield JVP"); }
}

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard kokkos(argc,argv);
    try {
        if (argc > 1 && std::strcmp(argv[1], "bad-av-nc") == 0) {
            exasim::enzyme::avfield_jvp<NoAuxModel,dstype>(nullptr,nullptr,nullptr,nullptr,nullptr,
                nullptr,nullptr,nullptr,nullptr,nullptr,0,0,0,4,1,2,2,0,0,1,1,0);
            return 0;
        }
        auxiliary();
        no_aux();
        std::puts("PASS: Enzyme auxiliary/material Jacobians and zero-sized/no-gradient inputs");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"%s\n",e.what());
        return 1;
    }
}
