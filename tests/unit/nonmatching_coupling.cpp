// Phase 4 checks of equations (31)-(42), (49), (63), and (64).
#ifdef HAVE_MPI
#include <mpi.h>
MPI_Comm EXASIM_COMM_WORLD = MPI_COMM_NULL;
MPI_Comm EXASIM_COMM_LOCAL = MPI_COMM_NULL;
#endif

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <vector>

using namespace std;

#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/pblas.h"
#include "../../backend/Discretization/ioutilities.cpp"
#include "../../backend/Discretization/material_properties.hpp"
#include "../../backend/Discretization/wequation.hpp"
#include "../../backend/Discretization/nonmatchingcoupling.hpp"
#include "../../backend/Discretization/uequation.hpp"

namespace {

int failures = 0;
int worldRank = 0;
int worldSize = 1;
dstype maxScaledError = 0;

void check(bool condition, const char* name)
{
    if (!condition) {
        printf("[rank %d] FAIL %s\n", worldRank, name);
        ++failures;
    }
}

void close_to(dstype actual, dstype expected, const char* name)
{
    const dstype scaled = abs(actual - expected) / max<dstype>(1.0, abs(expected));
    maxScaledError = max(maxScaledError, scaled);
    if (!(scaled <= 2e-13)) {
        printf("[rank %d] FAIL %s: %.17g != %.17g\n",
               worldRank, name, actual, expected);
        ++failures;
    }
}

// Scalar flux from examples/CoupledProblems/Poisson/matching/pdemodel1.m.
struct PoissonCoupled : exasim::ModelDefaults<PoissonCoupled> {
    static constexpr int nd = 2, ncu = 1, ncw = 0, nparam = 1, nfint = 1;
    static constexpr auto disc = exasim::Discretization::HDG;
    static constexpr bool has_external_coupling = true;

    KOKKOS_INLINE_FUNCTION static void flux(double f[], const double[],
            const double[], const double[], const double[], const double[],
            const double[], double)
    {
        for (int k = 0; k < ncu * nd; ++k) f[k] = 0;
    }

    KOKKOS_INLINE_FUNCTION static void fint(double f[], int, const double[],
            const double[], const double[], const double[], const double uh[],
            const double[], const double[], const double[], const double[], double)
    {
        f[0] = -uh[0];
    }

    KOKKOS_INLINE_FUNCTION static void fint_jac_uq(double j[], int, const double[],
            const double[], const double[], const double[], const double[],
            const double[], const double[], const double[], const double[], double)
    {
        for (int k = 0; k < 3; ++k) j[k] = 0;
    }

    KOKKOS_INLINE_FUNCTION static void fint_jac_uh(double j[], int, const double[],
            const double[], const double[], const double[], const double[],
            const double[], const double[], const double[], const double[], double)
    {
        j[0] = -1;
    }
};

// Two-component q-dependent ModelD-style coupling. The component-distinct
// derivatives make u/q and HDG trace-layout mistakes observable.
struct MultiCoupled : exasim::ModelDefaults<MultiCoupled> {
    static constexpr int nd = 2, ncu = 2, ncw = 0, nparam = 1, nfint = 2;
    static constexpr auto disc = exasim::Discretization::HDG;
    static constexpr bool has_external_coupling = true;

    KOKKOS_INLINE_FUNCTION static void flux(double f[], const double[],
            const double[], const double[], const double[], const double[],
            const double[], double)
    {
        for (int k = 0; k < ncu * nd; ++k) f[k] = 0;
    }

    KOKKOS_INLINE_FUNCTION static void fint(double f[], int, const double[],
            const double uq[], const double[], const double[], const double uh[],
            const double[], const double[], const double[], const double[], double)
    {
        f[0] = uq[0] + 0.5 * uq[2] + 2.0 * uh[0];
        f[1] = uq[1] + uq[5] - uh[1];
    }

    KOKKOS_INLINE_FUNCTION static void fint_jac_uq(double j[], int, const double[],
            const double[], const double[], const double[], const double[],
            const double[], const double[], const double[], const double[], double)
    {
        for (int k = 0; k < 12; ++k) j[k] = 0;
        j[0 + 2 * 0] = 1;
        j[0 + 2 * 2] = 0.5;
        j[1 + 2 * 1] = 1;
        j[1 + 2 * 5] = 1;
    }

    KOKKOS_INLINE_FUNCTION static void fint_jac_uh(double j[], int, const double[],
            const double[], const double[], const double[], const double[],
            const double[], const double[], const double[], const double[], double)
    {
        j[0] = 2;
        j[1] = 0;
        j[2] = 0;
        j[3] = -1;
    }
};

struct PairFixture {
    nonmatchinginterface donor;
    masterstruct master;
    resstruct res;
    commonstruct common;
    exasim::nonmatching::CouplingPointData<dstype, Int> point;
    vector<dstype> shape, wj, phi, phibar, psi, offset, dphibar;
    vector<dstype> slotResidual, pairMatrix, C, E, F, Ru;
    vector<Int> pg, pointSlot, pointFace, pointElement;
    vector<Int> pairSlot, pairElement, pairPtr, slotPtr;
    vector<Int> sendRank{0}, sendCount{1};

    PairFixture(bool qfree, Int ncu)
    {
        const Int npe = 2, npf = 2, nfe = 1, nd = 2, npts = 2;
        const Int ncu12 = ncu, ncq = qfree ? 0 : ncu * nd;
        const Int nc = ncu + ncq, n = npe * ncu;
        const Int m = npf * nfe * ncu, m12 = npf * ncu12;

        donor.nd = nd;
        donor.npe = npe;
        donor.npf = npf;
        donor.nfe = nfe;
        donor.ngf = npts;
        donor.ncu = ncu;
        donor.ncq = ncq;
        donor.ncu12 = ncu12;
        donor.npts = npts;
        donor.nslots = 1;
        donor.npairs = 1;

        wj = {0.7, 1.1};
        pg = {0, 1};
        pointSlot = {0, 0};
        pointFace = {0, 0};
        pointElement = {0, 0};
        phi = {0.8, 0.2, 0.3, 0.7};
        phibar = {0.6, 0.4, 0.25, 0.75};
        psi = {0.9, 0.1, 0.2, 0.8};
        offset = {0.12, -0.07, -0.04, 0.09};
        dphibar = {0.4, -0.4, -0.2, 0.2, -0.3, 0.3, 0.5, -0.5};
        pairSlot = {0};
        pairElement = {0};
        pairPtr = {0, 2};
        slotPtr = {0, 2};

        donor.pwJ = wj.data();
        donor.pg = pg.data();
        donor.pslot = pointSlot.data();
        donor.pa = pointFace.data();
        donor.pe = pointElement.data();
        donor.pphi = phi.data();
        donor.pphibar = phibar.data();
        donor.ppsi = psi.data();
        donor.pd = offset.data();
        donor.pdphibar = dphibar.data();
        donor.pairslot = pairSlot.data();
        donor.pairelem = pairElement.data();
        donor.pairptr = pairPtr.data();
        donor.slotptr = slotPtr.data();
        donor.nsendnb = 1;
        donor.sendnb = sendRank.data();
        donor.sendcount = sendCount.data();
        donor.slotrank = sendRank.data();

        shape = {0.75, 0.2, 0.25, 0.8};
        master.shapfgt = shape.data();

        slotResidual.assign(m12, 0);
        pairMatrix.assign(static_cast<size_t>(m12) * m, 0);
        donor.Ri = slotResidual.data();
        donor.Hi = pairMatrix.data();
        donor.szRi = m12;
        donor.szHi = m12 * m;

        common.components.nc = nc;
        common.components.ncu = ncu;
        common.components.ncq = ncq;
        common.meshsizes.ne = 1;
        common.meshsizes.ne1 = 1;
        common.timeparams.wave = 0;

        C.assign(npe * npe * nd, 0);
        E.assign(npe * npf * nfe * nd, 0);
        for (Int d = 0; d < nd; ++d) {
            C[4 * d] = 0.10 + 0.02 * d;
            C[4 * d + 1] = -0.03;
            C[4 * d + 2] = 0.04;
            C[4 * d + 3] = 0.08 + 0.01 * d;
            E[4 * d] = 0.05;
            E[4 * d + 1] = -0.02;
            E[4 * d + 2] = 0.03 + 0.01 * d;
            E[4 * d + 3] = 0.07;
        }
        F.assign(static_cast<size_t>(n) * m, 0);
        Ru.assign(n, 0);
        for (Int j = 0; j < m; ++j)
            for (Int i = 0; i < n; ++i)
                F[i + n * j] = 0.01 * (1 + i + 2 * j);
        for (Int i = 0; i < n; ++i) Ru[i] = 0.02 * (i + 1);
        res.C = C.data();
        res.E = E.data();
        res.F = F.data();
        res.Ru = Ru.data();

        point.flux.resize(npts * ncu12);
        point.fluxUdg.resize(static_cast<size_t>(npts) * ncu12 * nc);
        point.fluxUhg.resize(static_cast<size_t>(npts) * ncu12 * ncu);
        for (Int p = 0; p < npts; ++p)
            for (Int out = 0; out < ncu12; ++out) {
                point.flux[p + npts * out] = 0.3 + 0.1 * p + 0.07 * out;
                for (Int c = 0; c < nc; ++c)
                    point.fluxUdg[p + npts * (out + ncu12 * c)] =
                        0.2 + 0.03 * p + 0.04 * out + 0.05 * c;
                for (Int c = 0; c < ncu; ++c)
                    point.fluxUhg[p + npts * (out + ncu12 * c)] =
                        -0.15 + 0.02 * p + 0.06 * out - 0.03 * c;
            }
    }
};

// Independent direct implementation of equations (39), (41), (42), and (49).
// It intentionally does not use the production layout helpers.
void direct_pair_reference(const PairFixture& fixture,
        vector<dstype>& residual, vector<dstype>& matrix)
{
    const auto& d = fixture.donor;
    const Int npe = d.npe, npf = d.npf, nfe = d.nfe, nd = d.nd;
    const Int ncu = d.ncu, ncu12 = d.ncu12, ncq = d.ncq;
    const Int n = npe * ncu, m = npf * nfe * ncu, m12 = npf * ncu12;
    vector<dstype> K(static_cast<size_t>(m12) * n, 0);
    vector<dstype> G(static_cast<size_t>(nd) * m12 * npe * ncu, 0);
    residual.assign(m12, 0);
    matrix.assign(static_cast<size_t>(m12) * m, 0);

    for (Int p = 0; p < d.npts; ++p)
        for (Int i = 0; i < npf; ++i) {
            const dstype W = d.pwJ[p] *
                fixture.master.shapfgt[d.pg[p] + d.ngf * i];
            for (Int out = 0; out < ncu12; ++out) {
                residual[out + ncu12 * i] -= W * fixture.point.flux[p + d.npts * out];
                for (Int c = 0; c < ncu; ++c) {
                    const dstype fu = fixture.point.fluxUdg[
                        p + d.npts * (out + ncu12 * c)];
                    const dstype fuh = fixture.point.fluxUhg[
                        p + d.npts * (out + ncu12 * c)];
                    for (Int j = 0; j < npe; ++j) {
                        dstype derivative = fu * d.pphi[j + npe * p];
                        if (ncq == 0)
                            for (Int direction = 0; direction < nd; ++direction)
                                derivative += fuh * d.pd[p + d.npts * direction] *
                                    d.pdphibar[j + npe * (direction + nd * p)];
                        K[(out + ncu12 * i) + m12 * (j + npe * c)] += W * derivative;
                    }
                    for (Int j = 0; j < npf; ++j)
                        matrix[(out + ncu12 * i) + m12 * (c + ncu * j)] +=
                            W * fuh * d.ppsi[j + npf * p];

                    if (ncq > 0)
                        for (Int direction = 0; direction < nd; ++direction)
                            for (Int j = 0; j < npe; ++j) {
                                const Int q = ncu + c + ncu * direction;
                                const dstype fq = fixture.point.fluxUdg[
                                    p + d.npts * (out + ncu12 * q)];
                                G[static_cast<size_t>(direction) * m12 * npe * ncu +
                                  (out + ncu12 * i) + m12 * (j + npe * c)] +=
                                    W * (fq * d.pphi[j + npe * p] -
                                         fuh * d.pd[p + d.npts * direction] *
                                         d.pphibar[j + npe * p]);
                            }
                }
            }
        }

    if (ncq > 0)
        for (Int direction = 0; direction < nd; ++direction) {
            const dstype* g = &G[static_cast<size_t>(direction) * m12 * npe * ncu];
            const dstype* C = &fixture.C[direction * npe * npe];
            const dstype* E = &fixture.E[direction * npe * npf * nfe];
            for (Int row = 0; row < m12; ++row)
                for (Int component = 0; component < ncu; ++component) {
                    for (Int j = 0; j < npe; ++j)
                        for (Int a = 0; a < npe; ++a)
                            K[row + m12 * (j + npe * component)] +=
                                g[row + m12 * (a + npe * component)] * C[a + npe * j];
                    for (Int j = 0; j < npf * nfe; ++j)
                        for (Int a = 0; a < npe; ++a)
                            matrix[row + m12 * (component + ncu * j)] -=
                                g[row + m12 * (a + npe * component)] * E[a + npe * j];
                }
        }

    for (Int row = 0; row < m12; ++row) {
        for (Int column = 0; column < m; ++column)
            for (Int a = 0; a < n; ++a)
                matrix[row + m12 * column] +=
                    K[row + m12 * a] * fixture.F[a + n * column];
        for (Int a = 0; a < n; ++a)
            residual[row] -= K[row + m12 * a] * fixture.Ru[a];
    }
}

void pair_case(bool qfree, Int ncu)
{
    PairFixture fixture(qfree, ncu);
    vector<dstype> residual, matrix;
    direct_pair_reference(fixture, residual, matrix);
    exasim::nonmatching::form_and_condense_pairs(fixture.donor, fixture.master,
            fixture.point, fixture.res, fixture.common, 0, 0);
    for (size_t i = 0; i < residual.size(); ++i)
        close_to(fixture.slotResidual[i], residual[i],
                 qfree ? "q-free pair residual" : "q pair residual");
    for (size_t i = 0; i < matrix.size(); ++i)
        close_to(fixture.pairMatrix[i], matrix[i],
                 qfree ? "q-free pair matrix" : "q pair matrix");
}

void point_data_case(Int spatialScheme)
{
    constexpr Int npts = 2, nd = 2, npe = 3, npf = 2, nfe = 2;
    constexpr Int ncu = 2, ncq = 4, nc = 6, nco = 2;
    nonmatchinginterface donor;
    donor.npts = npts;
    donor.nd = nd;
    donor.npe = npe;
    donor.npf = npf;
    donor.nfe = nfe;
    donor.ncu = ncu;
    donor.ncq = ncq;

    vector<Int> pe = {0, 0}, pa = {1, 0};
    vector<dstype> phi = {0.2, 0.3, 0.5, 0.6, 0.1, 0.3};
    vector<dstype> phibar = {0.4, 0.1, 0.5, 0.2, 0.5, 0.3};
    vector<dstype> psi = {0.25, 0.75, 0.8, 0.2};
    vector<dstype> offset = {0.1, -0.2, 0.05, 0.3};
    vector<dstype> x = {1.2, 1.8, -0.4, 0.6};
    vector<dstype> normal = {0.6, -0.8, 0.8, 0.6};
    donor.pe = pe.data();
    donor.pa = pa.data();
    donor.pphi = phi.data();
    donor.pphibar = phibar.data();
    donor.ppsi = psi.data();
    donor.pd = offset.data();
    donor.px = x.data();
    donor.pnl = normal.data();

    vector<Int> elemcon = {0, 1, 2, 3};
    meshstruct mesh;
    mesh.elemcon = elemcon.data();
    vector<dstype> udg(npe * nc), odg(npe * nco), uh(npf * ncu * nfe);
    for (Int c = 0; c < nc; ++c)
        for (Int j = 0; j < npe; ++j) udg[j + npe * c] = 100 * c + 10 * j + 1;
    for (Int c = 0; c < nco; ++c)
        for (Int j = 0; j < npe; ++j) odg[j + npe * c] = 1000 + 100 * c + 10 * j + 3;
    if (spatialScheme == 0) {
        for (Int face = 0; face < nfe; ++face)
            for (Int c = 0; c < ncu; ++c)
                for (Int node = 0; node < npf; ++node)
                    uh[node + npf * (c + ncu * face)] = 1000 * face + 100 * c + 10 * node + 7;
    }
    else {
        for (Int encoded = 0; encoded < npf * nfe; ++encoded)
            for (Int c = 0; c < ncu; ++c)
                uh[c + ncu * encoded] = 1000 * encoded + 100 * c + 7;
    }
    solstruct sol;
    sol.udg = udg.data();
    sol.odg = odg.data();
    sol.uh = uh.data();
    commonstruct common;
    common.components.nc = nc;
    common.components.ncu = ncu;
    common.components.ncq = ncq;
    common.components.nco = nco;
    common.components.ncw = 0;
    common.components.ncx = nd;
    common.spatialScheme = spatialScheme;

    exasim::nonmatching::CouplingPointData<dstype, Int> data;
    exasim::nonmatching::build_coupling_point_data(data, donor, sol, mesh, common);
    for (Int p = 0; p < npts; ++p) {
        for (Int direction = 0; direction < nd; ++direction) {
            close_to(data.x[p + npts * direction], x[p + npts * direction],
                     "point coordinate");
            close_to(data.nl[p + npts * direction], -normal[p + npts * direction],
                     "point normal");
        }
        for (Int c = 0; c < nc; ++c) {
            dstype expected = 0;
            for (Int j = 0; j < npe; ++j) expected += phi[j + npe * p] * udg[j + npe * c];
            close_to(data.udg[p + npts * c], expected, "point udg interpolation");
        }
        for (Int c = 0; c < nco; ++c) {
            dstype expected = 0;
            for (Int j = 0; j < npe; ++j) expected += phi[j + npe * p] * odg[j + npe * c];
            close_to(data.odg[p + npts * c], expected, "point odg interpolation");
        }
        for (Int c = 0; c < ncu; ++c) {
            dstype expected = 0;
            for (Int j = 0; j < npf; ++j) {
                const Int encoded = elemcon[j + npf * pa[p]];
                const dstype value = spatialScheme == 0
                    ? uh[(encoded % npf) + npf * (c + ncu * (encoded / npf))]
                    : uh[c + ncu * encoded];
                expected += psi[j + npf * p] * value;
            }
            for (Int direction = 0; direction < nd; ++direction) {
                dstype qbar = 0;
                const Int q = ncu + c + ncu * direction;
                for (Int j = 0; j < npe; ++j)
                    qbar += phibar[j + npe * p] * udg[j + npe * q];
                expected -= offset[p + npts * direction] * qbar;
            }
            close_to(data.uhg[p + npts * c], expected,
                     spatialScheme == 0 ? "HDG corrected trace" : "EDG corrected trace");
        }
    }
}

template <class Model>
void fill_expected_flux(PairFixture& fixture)
{
    const Int npts = fixture.donor.npts;
    const Int ncu = fixture.donor.ncu;
    fill(fixture.point.flux.begin(), fixture.point.flux.end(), 0);
    fill(fixture.point.fluxUdg.begin(), fixture.point.fluxUdg.end(), 0);
    fill(fixture.point.fluxUhg.begin(), fixture.point.fluxUhg.end(), 0);
    for (Int p = 0; p < npts; ++p) {
        if constexpr (is_same_v<Model, PoissonCoupled>) {
            fixture.point.flux[p] = -fixture.point.uhg[p];
            fixture.point.fluxUhg[p] = -1;
        }
        else {
            fixture.point.flux[p] = fixture.point.udg[p] +
                0.5 * fixture.point.udg[p + 2 * npts] + 2 * fixture.point.uhg[p];
            fixture.point.flux[p + npts] = fixture.point.udg[p + npts] +
                fixture.point.udg[p + 5 * npts] - fixture.point.uhg[p + npts];
            fixture.point.fluxUdg[p + npts * (0 + ncu * 0)] = 1;
            fixture.point.fluxUdg[p + npts * (0 + ncu * 2)] = 0.5;
            fixture.point.fluxUdg[p + npts * (1 + ncu * 1)] = 1;
            fixture.point.fluxUdg[p + npts * (1 + ncu * 5)] = 1;
            fixture.point.fluxUhg[p + npts * (0 + ncu * 0)] = 2;
            fixture.point.fluxUhg[p + npts * (1 + ncu * 1)] = -1;
        }
    }
}

template <class Model>
void compute_case(Int ncu, bool matching)
{
    PairFixture fixture(false, ncu);
    auto& d = fixture.donor;
    d.isdonor = 1;
    if (matching) {
        fill(fixture.offset.begin(), fixture.offset.end(), 0);
        fixture.phi = fixture.psi;
        d.pphi = fixture.phi.data();
    }

    const Int npe = d.npe, npf = d.npf, nc = fixture.common.components.nc;
    vector<dstype> udg(npe * nc), uh(npf * ncu), px(d.npts * d.nd), pnl(d.npts * d.nd);
    vector<Int> elemcon = {0, 1};
    for (Int c = 0; c < nc; ++c)
        for (Int j = 0; j < npe; ++j) udg[j + npe * c] = 10 * c + 2 * j + 1;
    for (Int c = 0; c < ncu; ++c)
        for (Int j = 0; j < npf; ++j) uh[j + npf * c] = 100 * c + 3 * j + 5;
    for (Int direction = 0; direction < d.nd; ++direction)
        for (Int p = 0; p < d.npts; ++p) {
            px[p + d.npts * direction] = 0.2 + p + 0.1 * direction;
            pnl[p + d.npts * direction] = direction == 0 ? 1.0 : 0.0;
        }
    d.px = px.data();
    d.pnl = pnl.data();

    solstruct sol;
    sol.udg = udg.data();
    sol.uh = uh.data();
    meshstruct mesh;
    mesh.elemcon = elemcon.data();
    fixture.common.components.ncx = 2;
    fixture.common.components.nco = 0;
    fixture.common.components.ncw = 0;
    fixture.common.grid.nd = 2;
    fixture.common.spatialScheme = 0;
    fixture.common.couplingparams.coupledcondition = 1;

    // Build the expected point fields independently from the documented HDG
    // layouts, including the q offset ncu+c+ncu*d.
    fixture.point.x.assign(d.npts * d.nd, 0);
    fixture.point.nl.assign(d.npts * d.nd, 0);
    fixture.point.udg.assign(d.npts * nc, 0);
    fixture.point.odg.clear();
    fixture.point.wdg.clear();
    fixture.point.wsrc.clear();
    fixture.point.uhg.assign(d.npts * ncu, 0);
    for (Int p = 0; p < d.npts; ++p) {
        for (Int direction = 0; direction < d.nd; ++direction) {
            fixture.point.x[p + d.npts * direction] = px[p + d.npts * direction];
            fixture.point.nl[p + d.npts * direction] = -pnl[p + d.npts * direction];
        }
        for (Int c = 0; c < nc; ++c)
            for (Int j = 0; j < npe; ++j)
                fixture.point.udg[p + d.npts * c] +=
                    d.pphi[j + npe * p] * udg[j + npe * c];
        for (Int c = 0; c < ncu; ++c) {
            for (Int j = 0; j < npf; ++j) {
                const Int encoded = elemcon[j];
                fixture.point.uhg[p + d.npts * c] +=
                    d.ppsi[j + npf * p] * uh[(encoded % npf) + npf * c];
            }
            for (Int direction = 0; direction < d.nd; ++direction) {
                dstype qbar = 0;
                const Int q = ncu + c + ncu * direction;
                for (Int j = 0; j < npe; ++j)
                    qbar += d.pphibar[j + npe * p] * udg[j + npe * q];
                fixture.point.uhg[p + d.npts * c] -=
                    d.pd[p + d.npts * direction] * qbar;
            }
        }
    }
    fill_expected_flux<Model>(fixture);
    vector<dstype> expectedResidual, expectedMatrix;
    direct_pair_reference(fixture, expectedResidual, expectedMatrix);

    nonmatchingdata nm;
    nm.interfacetype = 1;
    nm.dir[0] = d;
    appstruct app;
    vector<dstype> tau(ncu, 1), param(1, 1), uinf(1, 0);
    app.tau = tau.data();
    app.physicsparam = param.data();
    app.uinf = uinf.data();
    app.materialdb_nprop = 0;
    tempstruct tmp;
    exasim::nonmatching::NonmatchingRuntime<dstype, Int> runtime;
    exasim::nonmatching::refresh_nonmatching_runtime(runtime, nm, app, mesh, fixture.common, Int(0));
    const long long initialAllocations = runtime.allocations;
    exasim::nonmatching::compute_nonmatching_coupling<Model>(runtime, sol, fixture.res,
            app, fixture.master, mesh, tmp, fixture.common, 0, true, 0);
    for (int iteration = 0; iteration < 10; ++iteration) {
        exasim::nonmatching::compute_nonmatching_coupling<Model>(runtime, sol, fixture.res,
                app, fixture.master, mesh, tmp, fixture.common, 0, false, 0);
        exasim::nonmatching::compute_nonmatching_coupling<Model>(runtime, sol, fixture.res,
                app, fixture.master, mesh, tmp, fixture.common, 0, true, 0);
    }
    check(runtime.allocations == initialAllocations,
          "repeated residual/Jacobian calls allocate no runtime storage");

    // C7: the host-executed Kokkos path must preserve the accepted CPU bits.
    exasim::nonmatching::compute_nonmatching_coupling<Model>(nm, sol, fixture.res,
            app, fixture.master, mesh, tmp, fixture.common, 0, true, 0);

    for (size_t i = 0; i < expectedResidual.size(); ++i)
        close_to(runtime.dir[0].Ri.data[i], expectedResidual[i],
                 matching ? "matching compute residual" : "offset compute residual");
    for (size_t i = 0; i < expectedMatrix.size(); ++i)
        close_to(runtime.dir[0].Hi.data[i], expectedMatrix[i],
                 matching ? "matching compute matrix" : "offset compute matrix");
    for (size_t i = 0; i < expectedResidual.size(); ++i)
        check(runtime.dir[0].Ri.data[i] == fixture.slotResidual[i],
              matching ? "matching runtime residual is bitwise CPU" : "offset runtime residual is bitwise CPU");
    for (size_t i = 0; i < expectedMatrix.size(); ++i)
        check(runtime.dir[0].Hi.data[i] == fixture.pairMatrix[i],
              matching ? "matching runtime matrix is bitwise CPU" : "offset runtime matrix is bitwise CPU");
}

// On a matching face, the Phase 3 point operators reduce to the conformal
// face quadrature.  Exercise both production paths from the same nodal state
// and compare their condensed interface residual and matrix.
template <class Model>
void conformal_equivalence_case(Int ncu)
{
    constexpr Int nd = 2, npe = 2, npf = 2, nfe = 1, ngf = 2, ne = 1;
    const Int ncu12 = ncu, ncq = nd * ncu, nc = ncu + ncq;
    const Int n = npe * ncu, m = npf * nfe * ncu, m12 = npf * ncu12;

    vector<dstype> faceShape = {1, 0, 0, 1};
    vector<dstype> faceWeight = faceShape;
    vector<dstype> faceMass(static_cast<size_t>(ngf) * npf * npf, 0);
    for (Int g = 0; g < ngf; ++g)
        for (Int j = 0; j < npf; ++j)
            for (Int i = 0; i < npf; ++i)
                faceMass[g + ngf * (i + npf * j)] =
                    faceShape[g + ngf * i] * faceShape[g + ngf * j];
    masterstruct master;
    master.shapfgt = faceShape.data();
    master.shapfgw = faceWeight.data();
    master.shapfgwdotshapfg = faceMass.data();

    vector<Int> perm = {0, 1}, elemcon = {0, 1}, boufaces = {0};
    meshstruct mesh;
    mesh.perm = perm.data();
    mesh.elemcon = elemcon.data();
    mesh.boufaces = boufaces.data();

    vector<dstype> udg(static_cast<size_t>(npe) * nc);
    vector<dstype> uh(static_cast<size_t>(npf) * ncu);
    for (Int c = 0; c < nc; ++c)
        for (Int j = 0; j < npe; ++j)
            udg[j + npe * c] = 1.0 + 2.0 * j + 10.0 * c;
    for (Int c = 0; c < ncu; ++c)
        for (Int j = 0; j < npf; ++j)
            uh[j + npf * c] = 5.0 + 3.0 * j + 100.0 * c;

    vector<dstype> elemfaceg(ngf * (nd + nd + 1), 0);
    vector<dstype> x = {0.2, 0.8, -0.3, 0.4};
    vector<dstype> donorNormal = {0.6, 0.8, -0.8, -0.6};
    vector<dstype> jac = {0.7, 1.1};
    for (Int d = 0; d < nd; ++d)
        for (Int g = 0; g < ngf; ++g) {
            elemfaceg[g + ngf * d] = x[g + ngf * d];
            elemfaceg[ngf * nd + g + ngf * d] = donorNormal[g + ngf * d];
        }
    for (Int g = 0; g < ngf; ++g)
        elemfaceg[ngf * (2 * nd) + g] = jac[g];

    solstruct sol;
    sol.udg = udg.data();
    sol.uh = uh.data();
    sol.elemfaceg = elemfaceg.data();

    vector<Int> eblks = {1, 1, -1}, nboufaces = {0, 1};
    commonstruct common;
    common.components.nc = nc;
    common.components.ncu = ncu;
    common.components.ncq = ncq;
    common.components.nco = 0;
    common.components.ncw = 0;
    common.components.ncx = nd;
    common.grid.nd = nd;
    common.grid.npe = npe;
    common.grid.npf = npf;
    common.grid.ngf = ngf;
    common.meshsizes.ne = ne;
    common.meshsizes.ne1 = ne;
    common.meshsizes.nfe = nfe;
    common.meshsizes.maxnbc = 1;
    common.eblks = eblks.data();
    common.nboufaces = nboufaces.data();
    common.spatialScheme = 0;
    common.timeparams.wave = 0;
    common.couplingparams.coupledcondition = 1;
    common.couplingparams.coupledboundarycondition = 1;
    common.couplingparams.coupledinterface = 1;
    common.szinterfacefluxmap = ncu12;

    appstruct app;
    vector<dstype> tau(ncu, 1), param(1, 1), uinf(1, 0);
    app.tau = tau.data();
    app.physicsparam = param.data();
    app.uinf = uinf.data();
    app.materialdb_nprop = 0;

    vector<dstype> C(static_cast<size_t>(nd) * npe * npe, 0);
    vector<dstype> E(static_cast<size_t>(nd) * npe * npf * nfe, 0);
    vector<dstype> D(static_cast<size_t>(n) * n, 0);
    vector<dstype> B(static_cast<size_t>(nd) * n * n, 0);
    vector<dstype> desiredF(static_cast<size_t>(n) * m, 0);
    vector<dstype> desiredRu(n, 0);
    for (Int d = 0; d < nd; ++d) {
        C[4 * d] = 0.10 + 0.02 * d;
        C[4 * d + 1] = -0.03;
        C[4 * d + 2] = 0.04;
        C[4 * d + 3] = 0.08 + 0.01 * d;
        E[4 * d] = 0.05;
        E[4 * d + 1] = -0.02;
        E[4 * d + 2] = 0.03 + 0.01 * d;
        E[4 * d + 3] = 0.07;
    }
    for (Int j = 0; j < m; ++j)
        for (Int i = 0; i < n; ++i)
            desiredF[i + n * j] = 0.01 * (1 + i + 2 * j);
    for (Int i = 0; i < n; ++i) desiredRu[i] = 0.02 * (i + 1);

    vector<dstype> confRu(n, 0), confRh(m, 0), confF(static_cast<size_t>(n) * m, 0);
    vector<dstype> confG(static_cast<size_t>(nd) * m * n, 0);
    vector<dstype> confK(256, 0), confH(static_cast<size_t>(m) * m, 0);
    vector<dstype> confRi(m12, 0), confKi(static_cast<size_t>(m12) * n, 0);
    vector<dstype> confGi(static_cast<size_t>(nd) * m12 * n, 0);
    vector<dstype> confHi(static_cast<size_t>(m12) * m, 0);
    vector<Int> ipiv(n, 0), tempi(64, 0);
    vector<dstype> tempn(4096, 0), tempg(4096, 0);
    resstruct conformal;
    conformal.Ru = confRu.data();
    conformal.Rh = confRh.data();
    conformal.C = C.data();
    conformal.E = E.data();
    conformal.D = D.data();
    conformal.B = B.data();
    conformal.F = confF.data();
    conformal.G = confG.data();
    conformal.K = confK.data();
    conformal.H = confH.data();
    conformal.Ri = confRi.data();
    conformal.Ki = confKi.data();
    conformal.Gi = confGi.data();
    conformal.Hi = confHi.data();
    conformal.ipiv = ipiv.data();
    tempstruct tmp;
    tmp.tempn = tempn.data();
    tmp.tempg = tempg.data();
    tmp.tempi = tempi.data();

    uEquationElemFaceBlock<Model>(sol, conformal, app, master, mesh, tmp,
            common, 0, 0, 0, 0);
    fill(D.begin(), D.end(), 0);
    // Native pre-Schur layout: npe x npe x ne x ncu x ncu.
    for (Int c = 0; c < ncu; ++c)
        for (Int i = 0; i < npe; ++i)
            D[i + npe * i + npe * npe * c + npe * npe * ncu * c] = 1;
    fill(B.begin(), B.end(), 0);
    fill(confF.begin(), confF.end(), 0);
    for (Int traceNode = 0; traceNode < npf; ++traceNode)
        for (Int traceComponent = 0; traceComponent < ncu; ++traceComponent)
            for (Int elementComponent = 0; elementComponent < ncu; ++elementComponent)
                for (Int elementNode = 0; elementNode < npe; ++elementNode) {
                    const Int transformed = elementNode + npe * elementComponent +
                        n * (traceComponent + ncu * traceNode);
                    const Int native = elementNode + npe * traceNode +
                        npe * npf * (elementComponent + ncu * traceComponent);
                    confF[native] = -desiredF[transformed];
                }
    confRu = desiredRu;
    uEquationSchurBlock<Model>(sol, conformal, app, master, mesh, tmp,
            common, 0, 0, 0, 0);

    nonmatchinginterface donor;
    donor.isdonor = 1;
    donor.nd = nd;
    donor.npe = npe;
    donor.npf = npf;
    donor.nfe = nfe;
    donor.ngf = ngf;
    donor.ncu = ncu;
    donor.ncq = ncq;
    donor.ncu12 = ncu12;
    donor.npts = ngf;
    donor.nslots = 1;
    donor.npairs = 1;
    vector<Int> pg = {0, 1}, zeros = {0, 0}, pair = {0}, pairptr = {0, 2};
    vector<Int> sendRank = {0}, sendCount = {1};
    vector<dstype> ownerNormal(donorNormal.size()), offset(ngf * nd, 0);
    for (size_t i = 0; i < ownerNormal.size(); ++i)
        ownerNormal[i] = -donorNormal[i];
    donor.pg = pg.data();
    donor.pe = zeros.data();
    donor.pa = zeros.data();
    donor.pslot = zeros.data();
    donor.px = x.data();
    donor.pnl = ownerNormal.data();
    donor.pwJ = jac.data();
    donor.pphi = faceShape.data();
    donor.pphibar = faceShape.data();
    donor.ppsi = faceShape.data();
    donor.pd = offset.data();
    donor.pairslot = pair.data();
    donor.pairelem = pair.data();
    donor.pairptr = pairptr.data();
    donor.slotptr = pairptr.data();
    donor.nsendnb = 1;
    donor.sendnb = sendRank.data();
    donor.sendcount = sendCount.data();
    donor.slotrank = sendRank.data();
    vector<dstype> nmRi(m12, 0), nmHi(static_cast<size_t>(m12) * m, 0);
    donor.Ri = nmRi.data();
    donor.Hi = nmHi.data();
    donor.szRi = m12;
    donor.szHi = m12 * m;
    nonmatchingdata nm;
    nm.interfacetype = 1;
    nm.dir[0] = donor;
    resstruct nonmatching;
    nonmatching.C = C.data();
    nonmatching.E = E.data();
    nonmatching.F = desiredF.data();
    nonmatching.Ru = desiredRu.data();
    tempstruct nmTmp;
    exasim::nonmatching::NonmatchingRuntime<dstype, Int> runtime;
    exasim::nonmatching::refresh_nonmatching_runtime(runtime, nm, app, mesh, common, Int(0));
    exasim::nonmatching::compute_nonmatching_coupling<Model>(runtime, sol,
            nonmatching, app, master, mesh, nmTmp, common, 0, true, 0);

    for (Int i = 0; i < m12; ++i)
        close_to(runtime.dir[0].Ri.data[i], confRi[i], "matching conformal residual");
    for (Int i = 0; i < m12 * m; ++i)
        close_to(runtime.dir[0].Hi.data[i], confHi[i], "matching conformal matrix");
}

#ifdef HAVE_MPI
void exchange_case()
{
    check(worldSize >= 2 && worldSize % 2 == 0, "equal model rank counts");
    if (worldSize < 2 || worldSize % 2) return;
    const Int half = worldSize / 2;
    const Int model = worldRank >= half;
    const Int local = worldRank % half;
    const Int peer = model ? local : half + local;
    const Int ownerDirection = model;
    const Int donorDirection = 1 - model;
    nonmatchingdata nm;
    nm.interfacetype = 1;
    vector<dstype> ri(1), pairMatrix(2), send(1), recv(1), rh(2);
    vector<Int> one = {1}, peerList = {peer}, zero = {0};
    vector<Int> range = {0, 1};
    vector<Int> ownerFaces = {0, 1}, second = {1};
    vector<Int> pointElement = {0}, pointFace = {0}, pointSlot = {0}, gauss = {0};
    vector<dstype> unit = {1}, pointX = {0.25, 0.5}, pointNormal = {1, 0};
    for (Int direction = 0; direction < 2; ++direction) {
        nm.dir[direction].direction = direction;
        nm.dir[direction].npf = 1;
        nm.dir[direction].ncu12 = 1;
        nm.dir[direction].nfe = 2;
    }
    auto& donor = nm.dir[donorDirection];
    donor.isdonor = 1;
    donor.nd = 2;
    donor.npe = 1;
    donor.ngf = 1;
    donor.ncu = 1;
    donor.ncq = 2;
    donor.npts = 1;
    donor.nslots = 1;
    donor.npairs = 1;
    donor.pe = pointElement.data();
    donor.pa = pointFace.data();
    donor.pslot = pointSlot.data();
    donor.pg = gauss.data();
    donor.slotptr = range.data();
    donor.pairslot = zero.data();
    donor.pairelem = zero.data();
    donor.pairptr = range.data();
    donor.px = pointX.data();
    donor.pnl = pointNormal.data();
    donor.pwJ = unit.data();
    donor.pphi = unit.data();
    donor.pphibar = unit.data();
    donor.ppsi = unit.data();
    vector<dstype> zeroOffset = {0, 0};
    donor.pd = zeroOffset.data();
    donor.nsendnb = 1;
    donor.sendnb = peerList.data();
    donor.sendcount = one.data();
    donor.slotrank = peerList.data();
    donor.Ri = ri.data();
    donor.Hi = pairMatrix.data();
    donor.szRi = 1;
    donor.szHi = 2;
    donor.sendbuf = send.data();
    donor.szsendbuf = 1;
    auto& owner = nm.dir[ownerDirection];
    owner.isowner = 1;
    owner.nownfaces = 2;
    owner.nrecvnb = 1;
    owner.recvnb = peerList.data();
    owner.recvcount = one.data();
    owner.recvface = second.data();
    owner.ownface = ownerFaces.data();
    owner.recvbuf = recv.data();
    owner.szrecvbuf = 1;
    meshstruct mesh;
    vector<Int> intfaces = {0, 1};
    vector<Int> elemcon = {0, 1};
    mesh.intfaces = intfaces.data();
    mesh.szintfaces = static_cast<Int>(intfaces.size());
    mesh.elemcon = elemcon.data();
    appstruct app;
    vector<Int> map = {0};
    vector<dstype> tau = {1}, param = {1}, uinf = {0};
    app.interfacefluxmap = map.data();
    app.tau = tau.data();
    app.physicsparam = param.data();
    app.uinf = uinf.data();
    app.materialdb_nprop = 0;
    commonstruct common;
    common.components.nc = 3;
    common.components.ncu = 1;
    common.components.ncq = 2;
    common.components.ncx = 2;
    common.components.nco = 0;
    common.components.ncw = 0;
    common.grid.nd = 2;
    common.meshsizes.ne = 1;
    common.meshsizes.ne1 = 1;
    common.spatialScheme = 0;
    common.timeparams.wave = 0;
    common.couplingparams.coupledcondition = 1;
    vector<dstype> udg = {1, 2, 3};
    vector<dstype> uh = {10.0 + worldRank};
    solstruct sol;
    sol.udg = udg.data();
    sol.uh = uh.data();
    resstruct res;
    masterstruct master;
    master.shapfgt = unit.data();
    tempstruct tmp;
    exasim::nonmatching::NonmatchingRuntime<dstype, Int> runtime;
    exasim::nonmatching::refresh_nonmatching_runtime(runtime, nm, app, mesh, common, Int(0));
    check(runtime.dir[ownerDirection].ownerTarget.data[0] == 0 &&
          runtime.dir[ownerDirection].ownerTarget.data[1] == 1,
          "owner targets preserve interface-face ordering");
    exasim::nonmatching::compute_nonmatching_coupling<PoissonCoupled>(runtime, sol, res,
            app, master, mesh, tmp, common, 0, false, 0);
    vector<MPI_Request> requests(2);
    vector<MPI_Status> statuses(2);
    common.requests = requests.data();
    common.statuses = statuses.data();
    Int count = 0;
    exasim::nonmatching::begin_coupling_slot_exchange(runtime, common, count);
    MPI_Waitall(count, requests.data(), statuses.data());
    exasim::nonmatching::assemble_coupling_slot_owner(
        rh.data(), runtime, mesh, app, common);
    close_to(rh[0], 0, "unaddressed owner face remains unchanged");
    close_to(rh[1], 10.0 + peer,
             "computed residual exchange and nonzero owner-face assembly");
}
#endif

} // namespace

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    EXASIM_COMM_LOCAL = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
#endif
    Kokkos::initialize(argc, argv);
    pair_case(false, 1);
    pair_case(false, 2);
    pair_case(true, 2);
    point_data_case(0);
    point_data_case(1);
    compute_case<PoissonCoupled>(1, true);
    compute_case<MultiCoupled>(2, true);
    compute_case<MultiCoupled>(2, false);
    conformal_equivalence_case<PoissonCoupled>(1);
    conformal_equivalence_case<MultiCoupled>(2);
#ifdef HAVE_MPI
    exchange_case();
#endif
    Kokkos::finalize();

    int globalFailures = failures;
    dstype globalMax = maxScaledError;
#ifdef HAVE_MPI
    MPI_Allreduce(&failures, &globalFailures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&maxScaledError, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (worldRank == 0)
        printf("nonmatching coupling: %s (%d ranks), max scaled error %.3e\n",
               globalFailures ? "FAIL" : "PASS", worldSize, globalMax);
    MPI_Finalize();
#endif
    return globalFailures ? EXIT_FAILURE : EXIT_SUCCESS;
}
