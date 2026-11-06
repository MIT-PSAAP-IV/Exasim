// Check the boundary gather with the actual non-Enzyme workspace allocation.
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
using std::partial_sum;
using std::max;
using std::string;
#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/pblas.h"
#include <exasim/drivers.hpp>
#include <exasim/execution_mode.hpp>
#include <modeldefaults.hpp>
#include "../../backend/Discretization/setstructs.hpp"
#include "../../backend/Discretization/getuhat.hpp"

#ifdef HAVE_ENZYME
#error "This test must exercise the non-Enzyme allocator."
#endif

struct BoundaryModel : ModelDefaults<BoundaryModel> {
    static constexpr int nd = 1, ncu = 1, nco = 14, ncw = 1, nparam = 0;
    KOKKOS_INLINE_FUNCTION static void ubou(dstype* f, int, const dstype* x,
        const dstype* u, const dstype* o, const dstype* w, const dstype* uh,
        const dstype*, const dstype*, const dstype*, const dstype*, dstype)
    {
        f[0] = x[0] + u[0] + u[1] + o[nco-1] + w[0] + uh[0];
    }
};

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    appstruct app{};
    masterstruct master{};
    meshstruct mesh{};
    solstruct sol{};
    resstruct res{};
    commonstruct common{};
    tempstruct tmp{};
    Int appdims[64]{}, problem[32]{}, masterdims[20]{}, meshdims[20]{}, meshsizes[20]{};
    app.ndims = appdims;
    app.problem = problem;
    master.ndims = masterdims;
    mesh.ndims = meshdims;
    mesh.nsize = meshsizes;
    appdims[AppNdims::nc] = 2;
    appdims[AppNdims::ncu] = appdims[AppNdims::ncq] = 1;
    appdims[AppNdims::ncx] = appdims[AppNdims::ncw] = 1;
    appdims[AppNdims::nco] = BoundaryModel::nco;
    masterdims[0] = masterdims[6] = masterdims[8] = 1;
    masterdims[5] = masterdims[7] = 2;
    meshdims[1] = meshdims[6] = meshdims[8] = 1;
    meshdims[4] = 2;
    meshsizes[12] = 3;

    // These LDG dimensions need 21 entries. The old allocator reserved 20.
    settempstruct(tmp, app, master, mesh, 0);
    if (tmp.sztempg < 21) {
        std::fprintf(stderr, "Boundary workspace has %d entries; need at least 21.\n", tmp.sztempg);
        tmp.freememory(0);
        return 1;
    }
    ArraySetValue(tmp.tempn, dstype(-999), tmp.sztempn + tmp.sztempg);

    dstype x[2]{2, 4}, u[4]{3, 5, 7, 11}, w[2]{13, 17}, uh[1]{29};
    dstype o[2*BoundaryModel::nco], shape[1]{1};
    for (int k = 0; k < BoundaryModel::nco; ++k) {
        o[2*k] = 100 + k;
        o[2*k+1] = 200 + k;
    }
    Int facecon[2]{1, 1}, findx[1]{1}, findu[2]{1, 3};
    sol.xdg = x; sol.udg = u; sol.odg = o; sol.wdg = w; sol.uh = uh;
    mesh.facecon = facecon; mesh.findxdg1 = findx; mesh.findudg1 = findu;
    master.shapfnt = shape;
    common.grid.nd = common.grid.npf = 1;
    common.grid.npe = 2;
    common.components.nc = 2;
    common.components.ncu = common.components.ncx = common.components.ncw = 1;
    common.components.nco = BoundaryModel::nco;

    UhatBlock<BoundaryModel>(sol, res, app, master, mesh, tmp, common,
        common.cublasHandle, 1, 2, 1, 2, 1, 1, BoundaryModel::nco, 0, 1, 1, 0);
    Kokkos::fence();
    // The selected face uses node 1 and the incoming trace value 29.
    const dstype expected = 4 + 5 + 11 + 213 + 17 + 29;
    const bool okay = std::isfinite(uh[0]) && std::abs(uh[0] - expected) < 1e-12;
    if (!okay)
        std::fprintf(stderr, "Boundary trace is %.17g; expected %.17g.\n", double(uh[0]), double(expected));
    tmp.freememory(0);
    return okay ? 0 : 1;
}
