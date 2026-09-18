#include <Kokkos_Core.hpp>

#include "../../backend/Common/common.h"
#include "../../backend/Common/kokkosimpl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int failures = 0;
    {
        constexpr int M = 2;
        constexpr int N = 1;
        constexpr int ncu = 8;

        Kokkos::View<dstype*> x_hdg("x_hdg", M*2);
        Kokkos::View<dstype*> x_ldg("x_ldg", M*2);
        Kokkos::View<dstype*> up("up", M*2);
        Kokkos::View<dstype*> vdg("vdg", M*ncu);
        Kokkos::View<dstype*> uhg("uhg", M*ncu);
        Kokkos::View<dstype*> physicsparam("physicsparam", 4);
        Kokkos::View<dstype*> externalparam("externalparam", 1);
        Kokkos::View<dstype*> stgdata("stgdata", N*10);
        Kokkos::View<dstype*> stgparam("stgparam", 3);
        Kokkos::View<dstype*> hdg("hdg", M*ncu);
        Kokkos::View<dstype*> ldg("ldg", M*ncu);

        auto hx = Kokkos::create_mirror_view(x_hdg);
        auto hv = Kokkos::create_mirror_view(vdg);
        auto hu = Kokkos::create_mirror_view(uhg);
        auto hp = Kokkos::create_mirror_view(physicsparam);
        auto he = Kokkos::create_mirror_view(externalparam);
        auto hs = Kokkos::create_mirror_view(stgdata);
        auto hc = Kokkos::create_mirror_view(stgparam);

        hx(0) = 0.1; hx(1) = 0.4;
        hx(2) = 0.2; hx(3) = 0.3;
        for (int m = 0; m < M; ++m) {
            const dstype species[5] = {0.01, 0.03, 0.06, 0.74, 0.16};
            for (int s = 0; s < 5; ++s) hv(m + M*s) = species[s];
            hv(m + M*5) = 1.0;
            hv(m + M*6) = 0.02;
            hv(m + M*7) = 1.0;
            for (int c = 0; c < ncu; ++c) hu(m + M*c) = 0.01*(1 + m + M*c);
        }
        hp(0) = 0.054;
        hp(1) = 4050.0;
        hp(2) = hp(0)*hp(1)*hp(1);
        hp(3) = 1340.0;
        he(0) = 0.0;
        for (int i = 0; i < N*10; ++i) hs(i) = 0.0;
        hs(0*N) = 2.0;
        hs(1*N) = 0.01;
        hs(3*N) = 1.0;
        hs(6*N) = 1.0;
        hc(0) = 1.0; hc(1) = 0.0; hc(2) = 0.0;

        Kokkos::deep_copy(x_hdg, hx);
        Kokkos::deep_copy(x_ldg, hx);
        Kokkos::deep_copy(vdg, hv);
        Kokkos::deep_copy(uhg, hu);
        Kokkos::deep_copy(physicsparam, hp);
        Kokkos::deep_copy(externalparam, he);
        Kokkos::deep_copy(stgdata, hs);
        Kokkos::deep_copy(stgparam, hc);

        StgInFlow2Dchem(hdg.data(), up.data(), x_hdg.data(), vdg.data(), uhg.data(),
                         physicsparam.data(), externalparam.data(), stgdata.data(),
                         stgparam.data(), 0.125, M, N);
        StgInFlowLDGchem(ldg.data(), x_ldg.data(), vdg.data(), physicsparam.data(),
                         externalparam.data(), stgdata.data(), stgparam.data(),
                         0.125, M, N, 2);
        Kokkos::fence();

        auto hhdg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hdg);
        auto hldg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ldg);
        dstype max_error = 0.0;
        for (int i = 0; i < M*ncu; ++i)
            max_error = std::max(max_error, std::abs(hldg(i) - (hhdg(i) + hu(i))));

        const dstype tolerance = 200.0*std::numeric_limits<dstype>::epsilon();
        if (!(max_error <= tolerance)) {
            std::printf("FAIL: LDG chemistry state differs from HDG state by %.17g\n",
                        static_cast<double>(max_error));
            ++failures;
        }
        if (!(hldg(0) > 0.0 && hldg(M*7) > 0.0)) {
            std::printf("FAIL: LDG chemistry state is not physically positive\n");
            ++failures;
        }
    }
    {
        constexpr int M = 2;
        constexpr int N = 1;
        constexpr int ncu = 9;

        Kokkos::View<dstype*> x_hdg("x_hdg_3d", M*3);
        Kokkos::View<dstype*> x_ldg("x_ldg_3d", M*3);
        Kokkos::View<dstype*> up("up_3d", M*3);
        Kokkos::View<dstype*> vdg("vdg_3d", M*ncu);
        Kokkos::View<dstype*> uhg("uhg_3d", M*ncu);
        Kokkos::View<dstype*> physicsparam("physicsparam_3d", 4);
        Kokkos::View<dstype*> externalparam("externalparam_3d", 1);
        Kokkos::View<dstype*> stgdata("stgdata_3d", N*10);
        Kokkos::View<dstype*> stgparam("stgparam_3d", 3);
        Kokkos::View<dstype*> hdg("hdg_3d", M*ncu);
        Kokkos::View<dstype*> ldg("ldg_3d", M*ncu);

        auto hx = Kokkos::create_mirror_view(x_hdg);
        auto hv = Kokkos::create_mirror_view(vdg);
        auto hu = Kokkos::create_mirror_view(uhg);
        auto hp = Kokkos::create_mirror_view(physicsparam);
        auto he = Kokkos::create_mirror_view(externalparam);
        auto hs = Kokkos::create_mirror_view(stgdata);
        auto hc = Kokkos::create_mirror_view(stgparam);

        hx(0) = 0.1; hx(1) = 0.4;
        hx(2) = 0.2; hx(3) = 0.3;
        hx(4) = 0.5; hx(5) = 0.6;
        for (int m = 0; m < M; ++m) {
            const dstype species[5] = {0.01, 0.03, 0.06, 0.74, 0.16};
            for (int s = 0; s < 5; ++s) hv(m + M*s) = species[s];
            hv(m + M*5) = 1.0;
            hv(m + M*6) = 0.02;
            hv(m + M*7) = -0.01;
            hv(m + M*8) = 1.0;
            for (int c = 0; c < ncu; ++c) hu(m + M*c) = 0.01*(1 + m + M*c);
        }
        hp(0) = 0.054;
        hp(1) = 4050.0;
        hp(2) = hp(0)*hp(1)*hp(1);
        hp(3) = 1340.0;
        he(0) = 0.0;
        for (int i = 0; i < N*10; ++i) hs(i) = 0.0;
        hs(0*N) = 2.0;
        hs(1*N) = 0.01;
        hs(3*N) = 1.0;
        hs(5*N) = 0.5;
        hs(6*N) = 1.0;
        hs(8*N) = -0.25;
        hc(0) = 1.0; hc(1) = 0.0; hc(2) = 0.0;

        Kokkos::deep_copy(x_hdg, hx);
        Kokkos::deep_copy(x_ldg, hx);
        Kokkos::deep_copy(vdg, hv);
        Kokkos::deep_copy(uhg, hu);
        Kokkos::deep_copy(physicsparam, hp);
        Kokkos::deep_copy(externalparam, he);
        Kokkos::deep_copy(stgdata, hs);
        Kokkos::deep_copy(stgparam, hc);

        StgInFlow3Dchem(hdg.data(), up.data(), x_hdg.data(), vdg.data(), uhg.data(),
                        physicsparam.data(), externalparam.data(), stgdata.data(),
                        stgparam.data(), 0.125, M, N);
        StgInFlowLDGchem(ldg.data(), x_ldg.data(), vdg.data(), physicsparam.data(),
                         externalparam.data(), stgdata.data(), stgparam.data(),
                         0.125, M, N, 3);
        Kokkos::fence();

        auto hhdg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hdg);
        auto hldg = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ldg);
        dstype max_error = 0.0;
        for (int i = 0; i < M*ncu; ++i)
            max_error = std::max(max_error, std::abs(hldg(i) - (hhdg(i) + hu(i))));

        const dstype tolerance = 200.0*std::numeric_limits<dstype>::epsilon();
        if (!(max_error <= tolerance)) {
            std::printf("FAIL: 3D LDG chemistry state differs from HDG state by %.17g\n",
                        static_cast<double>(max_error));
            ++failures;
        }
        if (!(hldg(0) > 0.0 && hldg(M*8) > 0.0)) {
            std::printf("FAIL: 3D LDG chemistry state is not physically positive\n");
            ++failures;
        }
    }
    {
        // Minimal regression for the full LDG inlet coupling that was missing
        // in auplate-transient2d: GetUhat supplies the chemistry STG state and
        // the boundary flux must use that state in tau*(u-uhat).
        constexpr int M = 1;
        constexpr int N = 1;
        constexpr int ncu = 8;
        constexpr dstype tau = 10.0;

        Kokkos::View<dstype*> x_base("x_base_flux", M*2);
        Kokkos::View<dstype*> x_stg("x_stg_flux", M*2);
        Kokkos::View<dstype*> vdg("vdg_flux", M*ncu);
        Kokkos::View<dstype*> physicsparam("physicsparam_flux", 4);
        Kokkos::View<dstype*> externalparam("externalparam_flux", 1);
        Kokkos::View<dstype*> stgdata_base("stgdata_base_flux", N*10);
        Kokkos::View<dstype*> stgdata_turb("stgdata_turb_flux", N*10);
        Kokkos::View<dstype*> stgparam("stgparam_flux", 3);
        Kokkos::View<dstype*> trace_base("trace_base_flux", M*ncu);
        Kokkos::View<dstype*> trace_stg("trace_stg_flux", M*ncu);

        auto hx = Kokkos::create_mirror_view(x_base);
        auto hv = Kokkos::create_mirror_view(vdg);
        auto hp = Kokkos::create_mirror_view(physicsparam);
        auto he = Kokkos::create_mirror_view(externalparam);
        auto hs0 = Kokkos::create_mirror_view(stgdata_base);
        auto hs1 = Kokkos::create_mirror_view(stgdata_turb);
        auto hc = Kokkos::create_mirror_view(stgparam);

        hx(0) = 0.125;
        hx(1) = 0.25;
        const dstype species[5] = {0.01, 0.03, 0.06, 0.74, 0.16};
        for (int s = 0; s < 5; ++s) hv(s) = species[s];
        hv(5) = 1.0;
        hv(6) = 0.02;
        hv(7) = 1.0;
        hp(0) = 0.054;
        hp(1) = 4050.0;
        hp(2) = hp(0)*hp(1)*hp(1);
        hp(3) = 1340.0;
        he(0) = 0.0;
        for (int i = 0; i < N*10; ++i) {
            hs0(i) = 0.0;
            hs1(i) = 0.0;
        }
        hs0(0*N) = hs1(0*N) = 2.0;  // wavenumber
        hs1(1*N) = 0.01;             // nonzero turbulent amplitude
        hs0(3*N) = hs1(3*N) = 1.0;  // wave-vector x component
        hs0(6*N) = hs1(6*N) = 1.0;  // velocity polarization x
        hs0(7*N) = hs1(7*N) = 0.5;  // velocity polarization y
        hc(0) = 1.0; hc(1) = 0.0; hc(2) = 0.0;

        Kokkos::deep_copy(x_base, hx);
        Kokkos::deep_copy(x_stg, hx);
        Kokkos::deep_copy(vdg, hv);
        Kokkos::deep_copy(physicsparam, hp);
        Kokkos::deep_copy(externalparam, he);
        Kokkos::deep_copy(stgdata_base, hs0);
        Kokkos::deep_copy(stgdata_turb, hs1);
        Kokkos::deep_copy(stgparam, hc);

        StgInFlowLDGchem(trace_base.data(), x_base.data(), vdg.data(),
                         physicsparam.data(), externalparam.data(),
                         stgdata_base.data(), stgparam.data(), 0.125,
                         M, N, 2);
        StgInFlowLDGchem(trace_stg.data(), x_stg.data(), vdg.data(),
                         physicsparam.data(), externalparam.data(),
                         stgdata_turb.data(), stgparam.data(), 0.125,
                         M, N, 2);
        Kokkos::fence();

        auto hbase = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), trace_base);
        auto hstg = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), trace_stg);

        dstype flux_change_norm = 0.0;
        dstype coupling_error = 0.0;
        for (int c = 0; c < ncu; ++c) {
            const dstype interior = 0.2 + 0.01*c;
            const dstype physical_flux = -0.1 + 0.02*c;
            const dstype regular_flux = physical_flux + tau*(interior - hbase(c));
            const dstype turbulent_flux = physical_flux + tau*(interior - hstg(c));
            const dstype flux_change = turbulent_flux - regular_flux;
            const dstype expected_change = tau*(hbase(c) - hstg(c));
            flux_change_norm += flux_change*flux_change;
            coupling_error = std::max(coupling_error,
                                      std::abs(flux_change - expected_change));
        }
        flux_change_norm = std::sqrt(flux_change_norm);

        const dstype tolerance = 500.0*std::numeric_limits<dstype>::epsilon();
        if (!(coupling_error <= tolerance)) {
            std::printf("FAIL: LDG chemistry trace is coupled with the wrong penalty sign; error %.17g\n",
                        static_cast<double>(coupling_error));
            ++failures;
        }
        if (!(flux_change_norm > 1.0e-6)) {
            std::printf("FAIL: nonzero chemistry STG trace does not change the LDG boundary flux\n");
            ++failures;
        }
    }
    Kokkos::finalize();
    return failures == 0 ? 0 : 1;
}
