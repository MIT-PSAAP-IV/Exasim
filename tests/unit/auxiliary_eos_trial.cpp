#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <numeric>
using std::partial_sum;
#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/pblas.h"
#include "../../backend/Common/auxiliary_snapshot.hpp"
#include "../../backend/Discretization/material_properties.hpp"
#include "../../backend/Discretization/wequation.hpp"

// log(w)+u has a positive root, but the full Newton step from w=1,u=2
// crosses the EOS domain boundary. This models a failed CNS temperature trial.
static int sourceCalls = 0;
static void source(double* f, double* fw, const double*, const double* u,
    const double*, const double* w, const double*, const double*, double,
    int model, int ng, int, int, int, int, int, int)
{
    ++sourceCalls;
    Kokkos::parallel_for("trial_eos", ng, KOKKOS_LAMBDA(int i) {
        f[i] = (model==2 ? w[i] : log(w[i])) + u[i];
        fw[i] = model==2 ? 1.0 : 1.0/w[i];
    });
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int failures = 0;
    {
        appstruct app;
        commonstruct common;
        ExasimDriverABI abi{};
        abi.hdgjac.HdgSourcewonly = source;
        common.driver_abi = &abi;
        common.components.ncu=1; common.components.nc=2;
        common.components.ncw=1; common.components.ncx=1; common.components.nco=0;
        common.grid.nd=1; common.modelnumber=1;
        common.timeparams.wave=0; common.timeparams.dae_alpha=0; common.timeparams.dae_beta=0;
        app.materialdb_nprop=0;
        Kokkos::View<double*> w("w",1), u("u",2), zeroes("zeroes",2), tmp("tmp",2);
        ArraySetValue(w.data(),1.0,1); ArraySetValue(u.data(),2.0,2);
        auto solve = [&]() {
            wEquation<exasim::detail::AbiAdapter>(w.data(),zeroes.data(),u.data(),zeroes.data(),zeroes.data(),tmp.data(),app,common,1,0,nullptr);
        };
        {
            AuxiliaryStateSnapshot<double> base(w.data(),1);
            solve();
            auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),w);
            failures += is_finite_bitwise(h(0));
            failures += sourceCalls != 2; // fail promptly, rather than 20 NaN iterations
            base.restore();
            ArraySetValue(u.data(),0.25,2);
            solve();
            h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),w);
            failures += !is_finite_bitwise(h(0)) || fabs(h(0)-exp(-0.25))>1e-6;
            base.commit();
        }
        // The generic auxiliary solver must still permit signed variables.
        common.modelnumber=2;
        ArraySetValue(w.data(),1.0,1); ArraySetValue(u.data(),2.0,2);
        solve();
        auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),w);
        failures += !is_finite_bitwise(h(0)) || fabs(h(0)+2.0)>1e-12;
    }
    Kokkos::finalize();
    std::printf("auxiliary EOS trial recovery: %d failures\n",failures);
    return failures ? 1 : 0;
}
