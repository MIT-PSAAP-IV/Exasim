// Compile against freshly generated Text2Code kernels, not checked-in outputs.
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
using dstype = double;
#include <driver_abi.hpp>
#ifdef EXASIM_TEST_MASS_PROVIDER
#include "testprovider.cpp"
using text2code_shared_lib::KokkosTdfunc;
#else
#include "modeldefaults.hpp"
#include "my_model.hpp"
#include "KokkosTdfunc.cpp"
#include "HdgTdfunc.cpp"
#endif

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int failures = 0;
    auto massJacobian = static_cast<ExasimDriverABI::HdgElementJacFn>(&KokkosTdfunc);
#ifdef EXASIM_TEST_MASS_PROVIDER
    const auto* abi = GetText2CodeExasimDriverABI();
    if (abi->abi_version != kExasimDriverABIVersion ||
        !abi->hdgjac.HdgTdfunc) ++failures;
    if (abi->hdgjac.HdgTdfunc) massJacobian = abi->hdgjac.HdgTdfunc;
#endif
    {
        constexpr int ng = 3, nc = 6, ncu = 2, nw = 1;
        using View = Kokkos::View<double*>;
        View x("x", 2*ng), u("u", nc*ng), w("w", nw*ng);
        View f("f", ncu*ng), values("values", ncu*ng);
        View ju("ju", ncu*nc*ng), jw("jw", ncu*nw*ng);
        auto uh = Kokkos::create_mirror_view(u);
        auto wh = Kokkos::create_mirror_view(w);
        for (int i=0; i<ng; ++i) {
            for (int j=0; j<nc; ++j) uh(j*ng+i) = .1*(j+1) + .02*i;
            wh(i) = .4 + .03*i;
        }
        Kokkos::deep_copy(u, uh); Kokkos::deep_copy(w, wh);
        KokkosTdfunc(values.data(), x.data(), u.data(), nullptr, w.data(),
                     nullptr, nullptr, 0., 0, ng, nc, ncu, 2, 2, 0, nw);
        massJacobian(f.data(), ju.data(), jw.data(), x.data(), u.data(),
                     nullptr, w.data(), nullptr, nullptr, 0., 0, ng, nc,
                     ncu, 2, 2, 0, nw);
        auto fh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, f);
        auto vh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);
        auto juh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ju);
        auto jwh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, jw);
        auto check = [&](double a, double b) {
            if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b)>1e-13) ++failures;
        };
        for (int i=0; i<ng; ++i) {
            double point[nc], wp[1]={wh(i)}, xp[2]={}, mf[ncu], a[ncu*nc], b[ncu];
            for (int j=0; j<nc; ++j) point[j]=uh(j*ng+i);
            PdeModel::tdfunc(mf,xp,point,nullptr,wp,nullptr,nullptr,0.);
            PdeModel::tdfunc_jac_uq(a,xp,point,nullptr,wp,nullptr,nullptr,0.);
            PdeModel::tdfunc_jac_w(b,xp,point,nullptr,wp,nullptr,nullptr,0.);
            for (int k=0; k<ncu; ++k) {
                const double expected = k==0 ? 1+point[0]*point[0]+point[1]*point[2]+wp[0]*wp[0]
                                             : 2+point[3]*wp[0];
                check(fh(k*ng+i),expected); check(vh(k*ng+i),expected); check(mf[k],expected);
                for (int j=0; j<nc; ++j) {
                    const double partial = k==0 ? (j==0 ? 2*point[0] : j==1 ? point[2] : j==2 ? point[1] : 0.)
                                                : (j==3 ? wp[0] : 0.);
                    check(juh((k+ncu*j)*ng+i),partial); check(a[k+ncu*j],partial);
                }
                check(jwh(k*ng+i),k==0 ? 2*wp[0] : point[3]);
                check(b[k],k==0 ? 2*wp[0] : point[3]);
            }
        }
    }
    Kokkos::finalize();
    std::cout << "Text2Code Tdfunc values, uq/w partials and typed/ABI parity: "
              << (failures ? "FAIL" : "PASS") << '\n';
    return failures ? 1 : 0;
}

