#include <Kokkos_Core.hpp>
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
using std::partial_sum;
using std::max;
using std::string;
#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/pblas.h"
#include <exasim/drivers.hpp>
#include <exasim/execution_mode.hpp>
#ifdef TEST_LEGACY_SETUP
#include "../../backend/Discretization/setstructs.cpp"
#else
#include "../../backend/Discretization/setstructs.hpp"
#endif

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int failures = 0;
    {
        appstruct app{};
        masterstruct master{};
        meshstruct mesh{};
        Int appdims[64] = {}, problem[32] = {};
        Int masterdims[20] = {}, meshdims[20] = {}, meshsizes[20] = {};
        app.ndims = appdims;
        app.problem = problem;
        master.ndims = masterdims;
        mesh.ndims = meshdims;
        mesh.nsize = meshsizes;
        appdims[AppNdims::nc] = appdims[AppNdims::ncu] = 1;
        appdims[AppNdims::ncx] = 1;
        masterdims[0] = masterdims[5] = masterdims[6] = 1;
        masterdims[7] = masterdims[8] = 1;
        meshdims[1] = meshdims[4] = meshdims[6] = meshdims[8] = 1;
        meshsizes[12] = 1;
        for (Int scheme : {0, 1}) {
            problem[0] = scheme;
            for (Int rb : {0, 1, 10, 64}) {
                problem[16] = rb;
                tempstruct tmp{};
                settempstruct(tmp, app, master, mesh, 0);
                const bool sized = tmp.sztempn >= 2*rb && tmp.sztempg >= 2*rb*rb;
                failures += !sized;
                if (sized && rb > 0) {
                    ArraySetValue(tmp.tempn, 3.0, 2*rb);
                    ArraySetValue(tmp.tempg, 4.0, 2*rb*rb);
                    Kokkos::fence();
                    failures += tmp.tempn[2*rb-1] != 3.0;
                    failures += tmp.tempg[2*rb*rb-1] != 4.0;
                }
                tmp.freememory(0);
            }
        }
    }
    Kokkos::finalize();
    std::printf("RB workspace bounds: %d failures\n", failures);
    return failures ? 1 : 0;
}
