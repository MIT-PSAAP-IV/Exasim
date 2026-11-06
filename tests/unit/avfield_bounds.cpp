#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "exasim/kernels/eos.hpp"

#ifdef HAVE_ENZYME
#error "This test must check the AV kernel without Enzyme."
#endif

struct AvModel : exasim::ModelDefaults<AvModel> {
    static constexpr int nd = 2, ncu = 1, nco = 0, ncw = 0, nparam = 0;
    KOKKOS_INLINE_FUNCTION static void avfield(double* f, const double* x,
        const double* u, const double*, const double*, const double*, const double*, double)
    { f[0] = u[0]*u[0] + 3*u[1] + 5*u[2] + x[0] + 2*x[1]; }
};

static void check(int nc)
{
    constexpr int npe = 2, ne = 3, ng = npe*ne;
    Kokkos::View<double*> x("x",ng*2), u("u",ng*nc), f("avfield",ng);
    auto hx = Kokkos::create_mirror_view(x);
    auto hu = Kokkos::create_mirror_view(u);
    for (int e=0; e<ne; ++e) {
        for (int p=0; p<npe; ++p) {
            for (int k=0; k<2; ++k)
                hx(p+npe*(k+2*e)) = 0.25 + e + 0.1*p + 0.2*k;
            for (int k=0; k<nc; ++k)
                hu(p+npe*(k+nc*e)) = 1.0 + 0.5*e + 0.1*p + 0.2*k;
        }
    }
    Kokkos::deep_copy(x,hx);
    Kokkos::deep_copy(u,hu);
    exasim::avfield_kernel<AvModel,double>(f.data(),x.data(),u.data(),nullptr,nullptr,
        nullptr,nullptr,0,0,ng,nc,1,2,2,0,0,1,npe,ne);
    const auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),f);
    for (int e=0; e<ne; ++e) {
        for (int p=0; p<npe; ++p) {
            const double u0 = hu(p+npe*nc*e);
            const double u1 = nc>1 ? hu(p+npe*(1+nc*e)) : 0;
            const double u2 = nc>2 ? hu(p+npe*(2+nc*e)) : 0;
            const double expected = u0*u0 + 3*u1 + 5*u2 +
                hx(p+npe*2*e) + 2*hx(p+npe*(1+2*e));
            const double got = hf(p+npe*e);
            if (!std::isfinite(got) || std::abs(got-expected)>1e-12)
                throw std::runtime_error("AV nodal values or absent gradients are incorrect.");
        }
    }
}

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard kokkos(argc,argv);
    try {
        if (argc > 1 && std::strcmp(argv[1], "bad-nc") == 0) {
            exasim::avfield_kernel<AvModel,double>(nullptr,nullptr,nullptr,nullptr,nullptr,
                nullptr,nullptr,0,0,0,4,1,2,2,0,0,1,2,0);
            return 0;
        }
        check(3);
        check(1);
        std::puts("PASS: AV full-state and no-gradient nodal inputs");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"%s\n",e.what());
        return 1;
    }
}
