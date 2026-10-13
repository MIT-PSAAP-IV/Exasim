// Phase 8 host-executed Kokkos/runtime ownership test. The same plain-pointer
// views and kernels are compiled for CUDA/HIP and will be run on GPU hardware.

#ifdef HAVE_MPI
#include <mpi.h>
MPI_Comm EXASIM_COMM_WORLD = MPI_COMM_NULL;
MPI_Comm EXASIM_COMM_LOCAL = MPI_COMM_NULL;
#endif

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
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

namespace {
int failures = 0;
void check(bool condition, const char* message)
{
    if (!condition) { ++failures; std::fprintf(stderr, "FAIL: %s\n", message); }
}

struct Fixture {
    nonmatchingdata host;
    appstruct app;
    meshstruct mesh;
    commonstruct common;
    std::vector<Int> pe{0, 0}, pa{0, 0}, pslot{0, 0}, pg{0, 1};
    std::vector<Int> slotptr{0, 2}, pairslot{0}, pairelem{0}, pairptr{0, 2};
    std::vector<dstype> px{0.2, 0.8, 0.0, 0.0}, pnl{1, 1, 0, 0};
    std::vector<dstype> pwJ{0.5, 0.5}, pphi{1, 1}, pphibar{1, 1};
    std::vector<dstype> ppsi{1, 1}, pd{0, 0, 0, 0};
    std::vector<dstype> Ri{0}, Hi{2};
    std::vector<Int> sendnb, sendcount, slotrank;

    Fixture(int rank, int size)
    {
        host.interfacetype = 1;
        auto& d = host.dir[0];
        d.direction = 0; d.isdonor = 1;
        d.nd = 2; d.npe = 1; d.npf = 1; d.nfe = 1; d.ngf = 2;
        d.ncu = 1; d.ncq = 2; d.ncu12 = 1;
        d.npts = 2; d.nslots = 1; d.npairs = 1;
        d.pe = pe.data(); d.pa = pa.data(); d.pslot = pslot.data(); d.pg = pg.data();
        d.slotptr = slotptr.data(); d.pairslot = pairslot.data();
        d.pairelem = pairelem.data(); d.pairptr = pairptr.data();
        d.px = px.data(); d.pnl = pnl.data(); d.pwJ = pwJ.data();
        d.pphi = pphi.data(); d.pphibar = pphibar.data(); d.ppsi = ppsi.data(); d.pd = pd.data();
        d.Ri = Ri.data(); d.Hi = Hi.data(); d.szRi = 1; d.szHi = 1;
        if (size > 1) {
            sendnb = {(rank + 1) % size}; sendcount = {1}; slotrank = sendnb;
            d.nsendnb = 1; d.sendnb = sendnb.data(); d.sendcount = sendcount.data();
            d.slotrank = slotrank.data();
        }
        common.components.nc = 3; common.components.ncu = 1;
        common.components.ncq = 2; common.components.ncx = 2;
        common.components.nco = 0; common.components.ncw = 0;
        common.meshsizes.ne = 1; common.meshsizes.ne1 = 1;
        app.materialdb_nprop = 0; app.materialdb_nstate = 0; app.materialdb_porder = 0;
    }
};

// [nonmatching] Phase 9 (Phase 8 carried R1-R3): a q-absent fixture with several
// slots, two pairs per slot, an owner side and an interface-face list, sized by
// `scale` so that a rebuild can change every count.  All values are deterministic
// and identical on every rank, so the ring exchange delivers the sender's Ri.
struct RichFixture {
    nonmatchingdata host;
    appstruct app;
    meshstruct mesh;
    commonstruct common;
    Int nslots = 0, npairs = 0, npts = 0, ne = 0;
    static constexpr Int nd = 2, npe = 3, npf = 2, nfe = 3, ngf = 2, ncu = 1, ncu12 = 1;
    std::vector<Int> pe, pa, pslot, pg, slotptr, pairslot, pairelem, pairptr;
    std::vector<Int> sendnb, sendcount, slotrank, recvnb, recvcount, recvface, ownface, bf;
    std::vector<dstype> px, pnl, pwJ, pphi, pphibar, ppsi, pd, pdphibar, Ri, Hi;

    RichFixture(int rank, int size, Int scale)
    {
        nslots = scale + 1; npairs = 2 * nslots; npts = 2 * npairs; ne = scale + 2;
        const Int m12 = npf * ncu12, m = npf * nfe * ncu;
        for (Int q = 0; q < npairs; ++q) {
            pairslot.push_back(q / 2);
            pairelem.push_back(q % ne);
            pairptr.push_back(2 * q);
        }
        pairptr.push_back(npts);
        for (Int s = 0; s <= nslots; ++s) slotptr.push_back(4 * s);
        for (Int p = 0; p < npts; ++p) {
            const Int q = p / 2;
            pe.push_back(pairelem[q]); pa.push_back(p % nfe);
            pslot.push_back(pairslot[q]); pg.push_back(p % ngf);
            pwJ.push_back(0.25 + 0.01 * p);
        }
        auto fill = [](std::vector<dstype>& v, std::size_t n, double a, double b) {
            v.resize(n);
            for (std::size_t i = 0; i < n; ++i) v[i] = a + b * std::sin(0.37 * double(i) + a);
        };
        fill(px, npts * nd, 0.5, 0.25); fill(pnl, npts * nd, 0.1, 1.0);
        fill(pphi, npe * npts, 0.3, 0.5); fill(pphibar, npe * npts, 0.2, 0.5);
        fill(ppsi, npf * npts, 0.4, 0.5); fill(pd, npts * nd, 0.0, 0.01);
        fill(pdphibar, npe * nd * npts, 0.6, 2.0);
        Ri.assign(m12 * nslots, 0.0);
        fill(Hi, std::size_t(m12) * m * npairs, 0.7, 1.5);

        host.interfacetype = 1;
        auto& d = host.dir[0];
        d.direction = 0; d.isdonor = 1;
        d.nd = nd; d.npe = npe; d.npf = npf; d.nfe = nfe; d.ngf = ngf;
        d.ncu = ncu; d.ncq = 0; d.ncu12 = ncu12;
        d.npts = npts; d.nslots = nslots; d.npairs = npairs;
        d.pe = pe.data(); d.pa = pa.data(); d.pslot = pslot.data(); d.pg = pg.data();
        d.slotptr = slotptr.data(); d.pairslot = pairslot.data();
        d.pairelem = pairelem.data(); d.pairptr = pairptr.data();
        d.px = px.data(); d.pnl = pnl.data(); d.pwJ = pwJ.data();
        d.pphi = pphi.data(); d.pphibar = pphibar.data(); d.ppsi = ppsi.data();
        d.pd = pd.data(); d.pdphibar = pdphibar.data();
        d.Ri = Ri.data(); d.Hi = Hi.data();
        d.szRi = static_cast<Int>(Ri.size()); d.szHi = static_cast<Int>(Hi.size());

        // Owner side of the same direction (the exchange tag is per direction): one
        // received block per own face.
        auto& o = host.dir[0];
        o.isowner = 1; o.nownfaces = nslots;
        for (Int f = 0; f < nslots; ++f) { ownface.push_back(f); recvface.push_back(f); }
        o.ownface = ownface.data();
        o.szrecvbuf = m12 * nslots;
        if (size > 1) {
            sendnb = {(rank + 1) % size}; sendcount = {nslots};
            slotrank.assign(nslots, sendnb[0]);
            d.nsendnb = 1; d.sendnb = sendnb.data(); d.sendcount = sendcount.data();
            d.slotrank = slotrank.data();
            recvnb = {(rank + size - 1) % size}; recvcount = {nslots};
            o.nrecvnb = 1; o.recvnb = recvnb.data(); o.recvcount = recvcount.data();
            o.recvface = recvface.data();
        }
        else {
            o.szrecvbuf = 0;
        }
        // Interface faces: local face 1 of every element carries boundary condition 2.
        bf.assign(nfe * ne, 1);
        for (Int e = 0; e < ne; ++e) bf[1 + nfe * e] = 2;
        mesh.bf = bf.data();
        common.couplingparams.coupledboundarycondition = 2;
        common.components.nc = 1; common.components.ncu = ncu;
        common.components.ncq = 0; common.components.ncx = 2;
        common.components.nco = 0; common.components.ncw = 0;
        common.meshsizes.ne = ne; common.meshsizes.ne1 = ne;
        app.materialdb_nprop = 0; app.materialdb_nstate = 0; app.materialdb_porder = 0;
    }
};

template <class V, class I>
bool mirror_equals(const std::vector<V>& host, const exasim::nonmatching::RuntimeBuffer<V, I>& buffer)
{
    // backend 0: the runtime buffer is host memory. The GPU stage must copy back first.
    return buffer.size == static_cast<I>(host.size()) &&
           (host.empty() || std::equal(host.begin(), host.end(), buffer.data));
}

void check_rich_runtime(const RichFixture& f, const exasim::nonmatching::NonmatchingRuntime<dstype, Int>& rt,
                        const char* tag)
{
    const auto& d = rt.dir[0];
    const auto& o = rt.dir[0];
    auto msg = [tag](const char* what) { static std::string s; s = std::string(tag) + ": " + what; return s.c_str(); };
    check(mirror_equals(f.pe, d.pe) && mirror_equals(f.pa, d.pa) && mirror_equals(f.pslot, d.pslot) &&
          mirror_equals(f.pg, d.pg), msg("point index mirrors match the host cache"));
    check(mirror_equals(f.slotptr, d.slotptr) && mirror_equals(f.pairslot, d.pairslot) &&
          mirror_equals(f.pairelem, d.pairelem) && mirror_equals(f.pairptr, d.pairptr),
          msg("slot/pair mirrors match the host cache"));
    check(mirror_equals(f.px, d.px) && mirror_equals(f.pnl, d.pnl) && mirror_equals(f.pwJ, d.pwJ),
          msg("px/pnl/pwJ mirrors match the host cache"));
    check(mirror_equals(f.pphi, d.pphi) && mirror_equals(f.pphibar, d.pphibar) &&
          mirror_equals(f.ppsi, d.ppsi) && mirror_equals(f.pd, d.pd),
          msg("basis and displacement mirrors match the host cache"));
    check(mirror_equals(f.pdphibar, d.pdphibar), msg("q-absent pdphibar mirror matches the host cache"));
    check(mirror_equals(f.Hi, d.Hi), msg("Hi mirror matches the host cache"));
    std::vector<Int> slotPairs(f.nslots + 1, 0);
    for (Int s = 0; s <= f.nslots; ++s) slotPairs[s] = 2 * s;
    check(mirror_equals(slotPairs, d.pairSlotPtr), msg("pairSlotPtr is the contiguous pair range of each slot"));
    check(d.view.npts == f.npts && d.view.nslots == f.nslots && d.view.npairs == f.npairs &&
          d.Ri.size == f.nslots * RichFixture::npf * RichFixture::ncu12,
          msg("view counts and Ri size follow the host cache"));
    check(d.view.pe == d.pe.data && d.view.pdphibar == d.pdphibar.data && d.view.Hi == d.Hi.data &&
          d.view.Ri == d.Ri.data && o.view.ownerTarget == o.ownerTarget.data,
          msg("views point at the current buffers (no stale pointers)"));

    // Owner targets against the finalized mesh.intfaces ordering, built with the same
    // routine finalizeConstruction uses (getinterfacefaces over ne1 elements).
    std::vector<Int> bf = f.bf;
    const int nint = getinterfacefaces(bf.data(), RichFixture::nfe, f.ne, 2);
    std::vector<int> intfaces(nint, 0);
    getinterfacefaces(intfaces.data(), bf.data(), RichFixture::nfe, f.ne, 2, nint);
    std::vector<Int> expected;
    for (Int own : f.ownface)
        expected.push_back(RichFixture::ncu * RichFixture::npf * intfaces[own]);
    check(mirror_equals(expected, o.ownerTarget), msg("owner targets equal the finalized mesh.intfaces decoding"));
}
}

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    EXASIM_COMM_LOCAL = MPI_COMM_WORLD;
#endif
    Kokkos::initialize(argc, argv);
    int rank = 0, size = 1;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
    {
        Fixture fixture(rank, size);
        exasim::nonmatching::NonmatchingRuntime<dstype, Int> runtime;
        exasim::nonmatching::refresh_nonmatching_runtime(
            runtime, fixture.host, fixture.app, fixture.mesh, fixture.common, Int(0));
        const auto& mirror = runtime.dir[0];
        check(std::equal(fixture.pe.begin(), fixture.pe.end(), mirror.pe.data),
              "runtime pe mirror matches host setup");
        check(std::equal(fixture.pa.begin(), fixture.pa.end(), mirror.pa.data),
              "runtime pa mirror matches host setup");
        check(std::equal(fixture.pg.begin(), fixture.pg.end(), mirror.pg.data),
              "runtime pg mirror matches host setup");
        check(std::equal(fixture.pphi.begin(), fixture.pphi.end(), mirror.pphi.data),
              "runtime pphi mirror matches host setup");
        check(std::equal(fixture.pphibar.begin(), fixture.pphibar.end(), mirror.pphibar.data),
              "runtime pphibar mirror matches host setup");
        check(std::equal(fixture.ppsi.begin(), fixture.ppsi.end(), mirror.ppsi.data),
              "runtime ppsi mirror matches host setup");
        check(std::equal(fixture.pd.begin(), fixture.pd.end(), mirror.pd.data),
              "runtime displacement mirror matches host setup");
        const long long initialAllocations = runtime.allocations;
        dstype* const initialRi = runtime.dir[0].Ri.data;
        dstype* const initialVpair = runtime.dir[0].vpair.data;

        std::vector<dstype> ve{3.0};
        for (int iteration = 0; iteration < 100; ++iteration)
            exasim::nonmatching::nonmatching_matvec_products(
                runtime, ve.data(), Int(1), Int(1), 0, Int(0));
        check(runtime.allocations == initialAllocations,
              "repeated matvec calls allocate no runtime storage");
        check(runtime.dir[0].Ri.data == initialRi && runtime.dir[0].vpair.data == initialVpair,
              "repeated matvec calls preserve workspace addresses");
        check(runtime.dir[0].Ri.data[0] == 6.0, "pair product and slot reduction");

        exasim::nonmatching::refresh_nonmatching_runtime(
            runtime, fixture.host, fixture.app, fixture.mesh, fixture.common, Int(0));
        check(runtime.allocations == initialAllocations,
              "same-size refresh reuses established capacity");
        check(runtime.dir[0].Ri.data == initialRi,
              "same-size refresh preserves runtime allocation");

        nonmatchingdata inactive;
        exasim::nonmatching::refresh_nonmatching_runtime(
            runtime, inactive, fixture.app, fixture.mesh, fixture.common, Int(0));
        check(runtime.interfacetype == 0 && runtime.dir[0].Ri.data == nullptr,
              "active-to-inactive refresh releases runtime arrays");
        exasim::nonmatching::refresh_nonmatching_runtime(
            runtime, fixture.host, fixture.app, fixture.mesh, fixture.common, Int(0));
        check(runtime.interfacetype == 1 && runtime.dir[0].Ri.data != nullptr,
              "inactive-to-active refresh reconstructs runtime arrays");

        // Preserve the legacy block-then-component owner accumulation order,
        // including a non-injective interfacefluxmap.  A component-parallel
        // GPU kernel would race on this fixture.
        exasim::nonmatching::NonmatchingRuntime<dstype, Int> ownerRuntime;
        ownerRuntime.interfacetype = 1;
        auto& owner = ownerRuntime.dir[0];
        owner.view.isowner = 1;
        owner.view.nownfaces = 1;
        owner.view.npf = 1;
        owner.view.ncu12 = 2;
        std::vector<Int> ownerTarget{0}, faceBlockPtr{0, 2}, faceBlocks{0, 1};
        std::vector<dstype> received{1.0e16, 1.0, -1.0e16, 2.0};
        owner.view.ownerTarget = ownerTarget.data();
        owner.view.faceBlockPtr = faceBlockPtr.data();
        owner.view.faceBlocks = faceBlocks.data();
        owner.recvbuf.copy_from_host(received.data(), static_cast<Int>(received.size()), Int(0));
        std::vector<Int> duplicateMap{0, 0};
        appstruct ownerApp;
        ownerApp.interfacefluxmap = duplicateMap.data();
        commonstruct ownerCommon;
        ownerCommon.components.ncu = 1;
        meshstruct ownerMesh;
        dstype assembled = 1.0;
        dstype reference = assembled;
        for (Int block = 0; block < 2; ++block)
            for (Int component = 0; component < 2; ++component)
                reference += received[component + 2 * block];
        exasim::nonmatching::assemble_coupling_slot_owner(
            &assembled, ownerRuntime, ownerMesh, ownerApp, ownerCommon);
        Kokkos::fence();
        check(assembled == reference,
              "owner assembly preserves block/component order bitwise");


        // [nonmatching] Phase 9: changed-count rebuild of one runtime object, against
        // the host cache and a fresh construction, followed by matvec, ring exchange
        // and owner assembly.
        {
            RichFixture small(rank, size, 1), large(rank, size, 3);
            exasim::nonmatching::NonmatchingRuntime<dstype, Int> rebuilt, fresh;
            exasim::nonmatching::refresh_nonmatching_runtime(rebuilt, small.host, small.app, small.mesh,
                                                             small.common, Int(0));
            check_rich_runtime(small, rebuilt, "rich runtime before rebuild");
            const long long before = rebuilt.allocations;
            exasim::nonmatching::refresh_nonmatching_runtime(rebuilt, large.host, large.app, large.mesh,
                                                             large.common, Int(0));
            check(rebuilt.allocations > before, "changed-count rebuild grows runtime storage");
            check_rich_runtime(large, rebuilt, "rich runtime after changed-count rebuild");
            exasim::nonmatching::refresh_nonmatching_runtime(fresh, large.host, large.app, large.mesh,
                                                             large.common, Int(0));
            check_rich_runtime(large, fresh, "rich runtime fresh construction");
            const long long settled = rebuilt.allocations;
            exasim::nonmatching::refresh_nonmatching_runtime(rebuilt, large.host, large.app, large.mesh,
                                                             large.common, Int(0));
            check(rebuilt.allocations == settled, "same-size rebuild after growth allocates nothing");

            const Int m = RichFixture::npf * RichFixture::nfe * RichFixture::ncu;
            const Int m12 = RichFixture::npf * RichFixture::ncu12;
            std::vector<dstype> ve(static_cast<std::size_t>(m) * large.ne);
            for (std::size_t i = 0; i < ve.size(); ++i) ve[i] = std::cos(0.11 * double(i));
            exasim::nonmatching::nonmatching_matvec_products(rebuilt, ve.data(), m, large.ne, 0, Int(0));
            exasim::nonmatching::nonmatching_matvec_products(fresh, ve.data(), m, large.ne, 0, Int(0));
            Kokkos::fence();
            const auto& rRi = rebuilt.dir[0].Ri;
            check(std::equal(rRi.data, rRi.data + rRi.size, fresh.dir[0].Ri.data),
                  "matvec after rebuild equals a fresh construction bitwise");
            double maxError = 0, maxValue = 0;
            for (Int slot = 0; slot < large.nslots; ++slot)
                for (Int row = 0; row < m12; ++row) {
                    double value = 0;
                    for (Int q = 2 * slot; q < 2 * slot + 2; ++q)
                        for (Int col = 0; col < m; ++col)
                            value += large.Hi[row + m12 * (col + static_cast<std::size_t>(m) * q)] *
                                     ve[col + static_cast<std::size_t>(m) * large.pairelem[q]];
                    maxError = std::max(maxError, std::abs(value - rRi.data[row + m12 * slot]));
                    maxValue = std::max(maxValue, std::abs(value));
                }
            check(maxError <= 1e-14 * std::max(1.0, maxValue), "matvec after rebuild matches an independent product");

#ifdef HAVE_MPI
            if (size > 1) {
                auto exchange_and_assemble = [&](exasim::nonmatching::NonmatchingRuntime<dstype, Int>& rt) {
                    commonstruct ring;
                    ring.components.ncu = RichFixture::ncu;
                    ring.requests = static_cast<MPI_Request*>(std::malloc(4 * sizeof(MPI_Request)));
                    ring.statuses = static_cast<MPI_Status*>(std::malloc(4 * sizeof(MPI_Status)));
                    Int counter = 0;
                    exasim::nonmatching::begin_coupling_slot_exchange(rt, ring, counter);
                    MPI_Waitall(counter, ring.requests, ring.statuses);
                    std::vector<dstype> residual(static_cast<std::size_t>(RichFixture::ncu) * RichFixture::npf *
                                                 RichFixture::nfe * large.ne, 1.0);
                    exasim::nonmatching::assemble_coupling_slot_owner(residual.data(), rt, large.mesh,
                                                                      large.app, ring);
                    Kokkos::fence();
                    ring.freememory();
                    return residual;
                };
                Int fluxmap = 0;
                large.app.interfacefluxmap = &fluxmap;
                const std::vector<dstype> afterRebuild = exchange_and_assemble(rebuilt);
                const std::vector<dstype> afterFresh = exchange_and_assemble(fresh);
                check(afterRebuild == afterFresh, "exchange and owner assembly after rebuild equal a fresh construction");
                // Every rank's donor Ri is identical, so own face f receives slot f of Ri.
                bool delivered = true;
                for (Int f = 0; f < large.nslots; ++f)
                    for (Int i = 0; i < RichFixture::npf; ++i) {
                        const Int face = 1 + RichFixture::nfe * f;   // ownface f -> interface face f
                        const dstype expected = 1.0 + rRi.data[i * RichFixture::ncu12 + m12 * f];
                        delivered = delivered &&
                            afterRebuild[RichFixture::ncu * (i + RichFixture::npf * face)] == expected;
                    }
                check(delivered, "owner assembly after rebuild adds the received slot into the target face");
                large.app.interfacefluxmap = nullptr;
            }
#endif
        }

#ifdef HAVE_MPI
        commonstruct teardown;
        teardown.requests = static_cast<MPI_Request*>(std::malloc(2 * sizeof(MPI_Request)));
        teardown.statuses = static_cast<MPI_Status*>(std::malloc(2 * sizeof(MPI_Status)));
        teardown.freememory();
        check(teardown.requests == nullptr && teardown.statuses == nullptr,
              "common teardown frees and nulls request storage");
        teardown.freememory();
        check(teardown.requests == nullptr && teardown.statuses == nullptr,
              "common teardown is repeat-safe for request storage");
#endif
    }
    Kokkos::finalize();
#ifdef HAVE_MPI
    int globalFailures = 0;
    MPI_Allreduce(&failures, &globalFailures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) std::printf("nonmatching runtime: %s (%d ranks)\n",
                               globalFailures ? "FAIL" : "PASS", size);
    MPI_Finalize();
    return globalFailures ? 1 : 0;
#else
    return failures ? 1 : 0;
#endif
}
