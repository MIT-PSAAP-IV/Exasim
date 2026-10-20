// [nonmatching] Phase 14 T1 executable runtime contracts.
// Production code is observed, never replaced: the wrapper validates every
// published runtime cache immediately after transactional rebuild.

#ifdef HAVE_MPI
#include <mpi.h>
extern MPI_Comm EXASIM_COMM_WORLD;
extern MPI_Comm EXASIM_COMM_LOCAL;
#endif

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__APPLE__)
#include <malloc/malloc.h>
#endif

#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/pblas.h"
#include "../../backend/Discretization/residual.hpp"
void FintDriver(...);
#include "../../backend/Discretization/nonmatchingpreprocess.hpp"
#include "../../backend/Discretization/nonmatchingcoupling.hpp"

namespace phase14_allocations {
static std::atomic<long long> count{0};
static std::atomic<bool> enabled{false};

enum CallKind {
    ComputeSameResidual,
    ComputeSameJacobian,
    ComputeMixedResidual,
    ComputeMixedJacobian,
    MatvecSame,
    MatvecMixed,
    FintResidual,
    FintJacobian,
    WEquationResidual,
    WEquationJacobian,
    KokkosShortLabel,
    KokkosLongLabel,
    CallKindCount
};

static const char* const callNames[CallKindCount] = {
    "compute-same-residual", "compute-same-jacobian",
    "compute-mixed-residual", "compute-mixed-jacobian",
    "matvec-same", "matvec-mixed", "fint-residual", "fint-jacobian",
    "wequation-residual", "wequation-jacobian",
    "kokkos-short-label", "kokkos-long-label"
};

static std::atomic<long long> calls[CallKindCount];
static std::atomic<long long> newCalls[CallKindCount];
static std::atomic<long long> blockDelta[CallKindCount];
static std::atomic<long long> byteDelta[CallKindCount];
static std::atomic<long long> maxNewCalls[CallKindCount];
}

void* operator new(std::size_t bytes)
{
    if (phase14_allocations::enabled.load(std::memory_order_relaxed))
        phase14_allocations::count.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(bytes)) return p;
    throw std::bad_alloc();
}

void* operator new[](std::size_t bytes)
{
    return ::operator new(bytes);
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace phase14_contracts {

struct AllocationScope {
    long long blocksBefore = 0;
    long long bytesBefore = 0;
    AllocationScope()
    {
#if defined(__APPLE__)
        malloc_statistics_t statistics{};
        malloc_zone_statistics(malloc_default_zone(), &statistics);
        blocksBefore = static_cast<long long>(statistics.blocks_in_use);
        bytesBefore = static_cast<long long>(statistics.size_in_use);
#endif
        phase14_allocations::count.store(0, std::memory_order_relaxed);
        phase14_allocations::enabled.store(true, std::memory_order_relaxed);
    }
    ~AllocationScope()
    {
        phase14_allocations::enabled.store(false, std::memory_order_relaxed);
    }
    long long allocations() const
    {
        return phase14_allocations::count.load(std::memory_order_relaxed);
    }
    std::pair<long long, long long> heap_delta() const
    {
#if defined(__APPLE__)
        malloc_statistics_t statistics{};
        malloc_zone_statistics(malloc_default_zone(), &statistics);
        return {static_cast<long long>(statistics.blocks_in_use) - blocksBefore,
                static_cast<long long>(statistics.size_in_use) - bytesBefore};
#else
        return {0, 0};
#endif
    }
};

struct AllocationObservation {
    long long allocations = 0;
    long long blockDelta = 0;
    long long byteDelta = 0;
};

template <class Function>
AllocationObservation observe_allocations(Function&& function)
{
    AllocationObservation observation;
    {
        AllocationScope scope;
        function();
        observation.allocations = scope.allocations();
        const auto heapDelta = scope.heap_delta();
        observation.blockDelta = heapDelta.first;
        observation.byteDelta = heapDelta.second;
    }
    return observation;
}

void record_allocations(phase14_allocations::CallKind kind,
        const AllocationObservation& observation)
{
    phase14_allocations::calls[kind].fetch_add(1, std::memory_order_relaxed);
    phase14_allocations::newCalls[kind].fetch_add(
        observation.allocations, std::memory_order_relaxed);
    phase14_allocations::blockDelta[kind].fetch_add(
        observation.blockDelta, std::memory_order_relaxed);
    phase14_allocations::byteDelta[kind].fetch_add(
        observation.byteDelta, std::memory_order_relaxed);
    long long previous = phase14_allocations::maxNewCalls[kind].load(
        std::memory_order_relaxed);
    while (previous < observation.allocations &&
            !phase14_allocations::maxNewCalls[kind].compare_exchange_weak(
                previous, observation.allocations, std::memory_order_relaxed)) {}
}

template <class T, class I>
bool has_rectangular_direction(
        const exasim::nonmatching::NonmatchingRuntime<T, I>& runtime)
{
    return runtime.dir[0].view.rectangular || runtime.dir[1].view.rectangular;
}

template <class T, class I>
void poison_compute_outputs(
        exasim::nonmatching::NonmatchingRuntime<T, I>& runtime,
        bool withJacobian)
{
    const T poison = std::numeric_limits<T>::quiet_NaN();
    for (int direction = 0; direction < 2; ++direction) {
        auto& d = runtime.dir[direction];
        if (!d.view.isdonor) continue;
        std::fill(d.Ri.data, d.Ri.data + d.Ri.size, poison);
        if (!withJacobian) continue;
        std::fill(d.K.data, d.K.data + d.K.size, poison);
        std::fill(d.Hi.data, d.Hi.data + d.Hi.size, poison);
        std::fill(d.pairResidual.data,
                  d.pairResidual.data + d.pairResidual.size, poison);
        std::fill(d.Fpair.data, d.Fpair.data + d.Fpair.size, poison);
        std::fill(d.Rupair.data, d.Rupair.data + d.Rupair.size, poison);
    }
}

template <class T, class I>
void poison_matvec_outputs(
        exasim::nonmatching::NonmatchingRuntime<T, I>& runtime)
{
    const T poison = std::numeric_limits<T>::quiet_NaN();
    for (int direction = 0; direction < 2; ++direction) {
        auto& d = runtime.dir[direction];
        if (!d.view.isdonor) continue;
        std::fill(d.Ri.data, d.Ri.data + d.Ri.size, poison);
        std::fill(d.vpair.data, d.vpair.data + d.vpair.size, poison);
        std::fill(d.rpair.data, d.rpair.data + d.rpair.size, poison);
    }
}

template <class T, class I>
std::vector<T> snapshot(const exasim::nonmatching::RuntimeBuffer<T, I>& buffer)
{
    if (buffer.size == 0) return {};
    return std::vector<T>(buffer.data, buffer.data + buffer.size);
}

template <class T>
void require_bytes(const std::vector<T>& reference, const T* candidate,
        std::size_t count, const char* name)
{
    if (reference.size() != count || (count > 0 &&
            std::memcmp(reference.data(), candidate, count * sizeof(T)) != 0))
        throw std::runtime_error(std::string(
            "Phase 14 same-family/rectangular equivalence failed: ") + name);
}

template <class T, class I>
struct DirectionSnapshot {
    bool donor = false;
    std::vector<T> Ri, K, Hi, pairResidual, rpair;
};

template <class T, class I>
DirectionSnapshot<T, I> snapshot_direction(
        const exasim::nonmatching::NonmatchingRuntimeDirection<T, I>& direction)
{
    DirectionSnapshot<T, I> result;
    result.donor = direction.view.isdonor != 0;
    if (!result.donor) return result;
    result.Ri = snapshot(direction.Ri);
    result.K = snapshot(direction.K);
    result.Hi = snapshot(direction.Hi);
    result.pairResidual = snapshot(direction.pairResidual);
    result.rpair = snapshot(direction.rpair);
    return result;
}

template <class T, class I>
void require_direction(const DirectionSnapshot<T, I>& reference,
        const exasim::nonmatching::NonmatchingRuntimeDirection<T, I>& direction,
        bool withPairs)
{
    if (!reference.donor) return;
    require_bytes(reference.Ri, direction.Ri.data,
                  static_cast<std::size_t>(direction.Ri.size), "Ri");
    if (withPairs) {
        require_bytes(reference.K, direction.K.data,
                      static_cast<std::size_t>(direction.K.size), "K");
        require_bytes(reference.Hi, direction.Hi.data,
                      static_cast<std::size_t>(direction.Hi.size), "Hi");
        require_bytes(reference.pairResidual, direction.pairResidual.data,
                      static_cast<std::size_t>(direction.pairResidual.size),
                      "pairResidual");
    }
}

template <class T, class I>
bool enable_equivalent_rectangular(
        exasim::nonmatching::NonmatchingRuntime<T, I>& runtime,
        const masterstructT<T, I>* master)
{
    bool changed = false;
    for (int k = 0; k < 2; ++k) {
        auto& direction = runtime.dir[k];
        auto& view = direction.view;
        if (!view.isdonor || view.rectangular) continue;
        if (master) {
            const I count = exasim::nonmatching::runtime_count<I>(
                {view.npf, view.npts}, "Phase 14 expanded owner test");
            direction.ptest.resize(count, runtime.backend);
            std::vector<T> values(static_cast<std::size_t>(count));
            for (I p = 0; p < view.npts; ++p)
                for (I i = 0; i < view.npf; ++i)
                    values[i + static_cast<std::size_t>(view.npf) * p] =
                        master->shapfgt[view.pg[p] +
                            static_cast<std::size_t>(view.ngf) * i];
            TemplateCopytoDevice(direction.ptest.data, values.data(), count,
                                 runtime.backend);
            view.ptest = direction.ptest.data;
        }
        view.ownerNpf = view.npf;
        view.ownerNgf = view.ngf;
        view.donorNpe = view.npe;
        view.donorNpf = view.npf;
        view.donorNfe = view.nfe;
        view.rectangular = 1;
        changed = true;
    }
    return changed;
}

template <class T, class I>
void restore_same_family(exasim::nonmatching::NonmatchingRuntime<T, I>& runtime)
{
    for (int k = 0; k < 2; ++k) {
        auto& direction = runtime.dir[k];
        if (!direction.view.isdonor) continue;
        direction.view.rectangular = 0;
        direction.view.ptest = nullptr;
    }
}

void audit_host_only_metadata()
{
#ifndef EXASIM_PHASE14_SOURCE_DIR
#error "EXASIM_PHASE14_SOURCE_DIR must identify the source checkout"
#endif
    const char* files[] = {
        "backend/Discretization/nonmatchingruntime.hpp",
        "backend/Discretization/nonmatchingruntimekernels.hpp",
        "backend/Discretization/nonmatchingrectangular.hpp"
    };
    const char* forbidden[] = {"sendnb", "sendcount", "recvnb", "recvcount", "OwnerTestData"};
    for (const char* relative : files) {
        const std::string path = std::string(EXASIM_PHASE14_SOURCE_DIR) + "/" + relative;
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Phase 14 source audit cannot open " + path);
        const std::string source((std::istreambuf_iterator<char>(input)),
                                 std::istreambuf_iterator<char>());
        if (source.find("KOKKOS_LAMBDA") == std::string::npos) continue;
        for (const char* name : forbidden)
            if (source.find(name) != std::string::npos)
                throw std::runtime_error(std::string("Phase 14 host-only metadata appears in a device-kernel header: ") + name);
    }

    const char* hotPathFiles[] = {
        "backend/Discretization/nonmatchingruntimekernels.hpp",
        "backend/Discretization/nonmatchingrectangular.hpp"
    };
    const char* allocationTokens[] = {
        "std::vector", "std::malloc", "TemplateMalloc", ".resize(", "new "
    };
    for (const char* relative : hotPathFiles) {
        const std::string path = std::string(EXASIM_PHASE14_SOURCE_DIR) + "/" + relative;
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Phase 14 source audit cannot open " + path);
        const std::string source((std::istreambuf_iterator<char>(input)),
                                 std::istreambuf_iterator<char>());
        for (const char* token : allocationTokens)
            if (source.find(token) != std::string::npos)
                throw std::runtime_error(std::string(
                    "Phase 14 hot-path source contains an allocation token: ") + token);
    }
}

void measure_kokkos_label_baseline()
{
    Kokkos::parallel_for("p14short", 1, KOKKOS_LAMBDA(const int) {});
    Kokkos::fence();
    record_allocations(phase14_allocations::KokkosShortLabel,
        observe_allocations([] {
            Kokkos::parallel_for("p14short", 1, KOKKOS_LAMBDA(const int) {});
            Kokkos::fence();
        }));

    Kokkos::parallel_for("Phase14LongKernelLabelProbe", 1,
                         KOKKOS_LAMBDA(const int) {});
    Kokkos::fence();
    record_allocations(phase14_allocations::KokkosLongLabel,
        observe_allocations([] {
            Kokkos::parallel_for("Phase14LongKernelLabelProbe", 1,
                                 KOKKOS_LAMBDA(const int) {});
            Kokkos::fence();
        }));
}

void verify_and_print_allocation_report(int rank)
{
    long long local[5 * phase14_allocations::CallKindCount] = {};
    long long global[5 * phase14_allocations::CallKindCount] = {};
    for (int kind = 0; kind < phase14_allocations::CallKindCount; ++kind) {
        local[kind] = phase14_allocations::calls[kind].load(
            std::memory_order_relaxed);
        local[phase14_allocations::CallKindCount + kind] =
            phase14_allocations::newCalls[kind].load(std::memory_order_relaxed);
        local[2 * phase14_allocations::CallKindCount + kind] =
            phase14_allocations::blockDelta[kind].load(std::memory_order_relaxed);
        local[3 * phase14_allocations::CallKindCount + kind] =
            phase14_allocations::byteDelta[kind].load(std::memory_order_relaxed);
        local[4 * phase14_allocations::CallKindCount + kind] =
            phase14_allocations::maxNewCalls[kind].load(std::memory_order_relaxed);
    }
#ifdef HAVE_MPI
    MPI_Reduce(local, global, 4 * phase14_allocations::CallKindCount,
               MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local + 4 * phase14_allocations::CallKindCount,
               global + 4 * phase14_allocations::CallKindCount,
               phase14_allocations::CallKindCount, MPI_LONG_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);
#else
    std::copy(local, local + 5 * phase14_allocations::CallKindCount, global);
#endif
    if (rank != 0) return;

    for (int kind = 0; kind < phase14_allocations::CallKindCount; ++kind) {
        const long long calls = global[kind];
        if (calls == 0) continue;
        const long long newCalls =
            global[phase14_allocations::CallKindCount + kind];
        std::printf("Phase 14 allocation %-25s calls %lld new-total %lld "
                    "new-average %.3f new-max %lld net-blocks %lld "
                    "net-bytes %lld\n",
            phase14_allocations::callNames[kind], calls, newCalls,
            static_cast<double>(newCalls) / static_cast<double>(calls),
            global[4 * phase14_allocations::CallKindCount + kind],
            global[2 * phase14_allocations::CallKindCount + kind],
            global[3 * phase14_allocations::CallKindCount + kind]);
    }

    const phase14_allocations::CallKind required[] = {
        phase14_allocations::ComputeSameJacobian,
        phase14_allocations::ComputeMixedResidual,
        phase14_allocations::ComputeMixedJacobian,
        phase14_allocations::MatvecSame,
        phase14_allocations::MatvecMixed,
        phase14_allocations::FintResidual,
        phase14_allocations::FintJacobian,
        phase14_allocations::WEquationResidual,
        phase14_allocations::WEquationJacobian,
        phase14_allocations::KokkosShortLabel,
        phase14_allocations::KokkosLongLabel
    };
    for (const auto kind : required)
        if (global[kind] == 0)
            throw std::runtime_error(std::string(
                "Phase 14 allocation attribution has no observations for ") +
                phase14_allocations::callNames[kind]);
}

template <class V, class I>
void extent(const exasim::nonmatching::RuntimeBuffer<V, I>& buffer,
        std::size_t expected, I backend, const char* name)
{
    if (expected > static_cast<std::size_t>(std::numeric_limits<I>::max()) ||
            buffer.size != static_cast<I>(expected) || buffer.capacity < buffer.size ||
            buffer.backend != backend) {
        throw std::runtime_error(std::string("Phase 14 runtime extent contract failed: ") + name);
    }
}

template <class T, class I>
void verify_direction(const nonmatchinginterfaceT<T, I>& host,
        const exasim::nonmatching::NonmatchingRuntimeDirection<T, I>& runtime,
        const appstructT<T, I>& app, const meshstructT<T, I>& mesh,
        const commonstructT<T, I>& common, I backend,
        const exasim::nonmatching::OwnerTestDirection<T, I>* ownerTests)
{
    const auto& v = runtime.view;
    const std::size_t npts = static_cast<std::size_t>(host.npts);
    const std::size_t nslots = static_cast<std::size_t>(host.nslots);
    const std::size_t npairs = static_cast<std::size_t>(host.npairs);
    const I rowNpf = v.rectangular ? v.ownerNpf : host.npf;
    const I donorNpe = v.rectangular ? v.donorNpe : host.npe;
    const I donorNpf = v.rectangular ? v.donorNpf : host.npf;
    const I donorNfe = v.rectangular ? v.donorNfe : host.nfe;
    const std::size_t ro = static_cast<std::size_t>(host.ncu12) * rowNpf;
    const std::size_t ne = static_cast<std::size_t>(donorNpe) * host.ncu;
    const std::size_t cd = static_cast<std::size_t>(host.ncu) * donorNpf * donorNfe;
    I recvBlocks = 0;
    for (I i = 0; i < host.nrecvnb; ++i) recvBlocks += host.recvcount[i];

#define CHECK_BUFFER(field, count) extent(runtime.field, static_cast<std::size_t>(count), backend, #field)
    CHECK_BUFFER(pe, npts); CHECK_BUFFER(pa, npts); CHECK_BUFFER(pslot, npts);
    CHECK_BUFFER(pg, npts); CHECK_BUFFER(slotptr, nslots + (nslots > 0));
    CHECK_BUFFER(pairslot, npairs); CHECK_BUFFER(pairelem, npairs);
    CHECK_BUFFER(pairptr, npairs + (npairs > 0)); CHECK_BUFFER(pairSlotPtr, nslots + 1);
    CHECK_BUFFER(recvface, recvBlocks); CHECK_BUFFER(ownface, host.nownfaces);
    CHECK_BUFFER(ownerTarget, host.nownfaces); CHECK_BUFFER(faceBlockPtr, host.nownfaces + 1);
    CHECK_BUFFER(faceBlocks, recvBlocks);
    CHECK_BUFFER(px, npts * host.nd); CHECK_BUFFER(pnl, npts * host.nd);
    CHECK_BUFFER(pwJ, npts); CHECK_BUFFER(pphi, npts * host.npe);
    CHECK_BUFFER(pphibar, npts * host.npe); CHECK_BUFFER(ppsi, npts * host.npf);
    CHECK_BUFFER(pd, npts * host.nd);
    CHECK_BUFFER(pdphibar, host.ncq == 0 ? npts * host.nd * host.npe : 0);
    CHECK_BUFFER(ptest, ownerTests && !ownerTests->values.empty() ?
        static_cast<std::size_t>(rowNpf) * host.npts : 0);
    CHECK_BUFFER(Ri, ro * nslots); CHECK_BUFFER(Hi, ro * cd * npairs);
    CHECK_BUFFER(recvbuf, host.szrecvbuf);
    CHECK_BUFFER(x, npts * common.components.ncx);
    CHECK_BUFFER(udg, npts * common.components.nc);
    CHECK_BUFFER(odg, npts * common.components.nco);
    CHECK_BUFFER(wdg, npts * common.components.ncw);
    CHECK_BUFFER(uhg, npts * host.ncu); CHECK_BUFFER(nl, npts * host.nd);
    CHECK_BUFFER(wsrc, npts * common.components.ncw);
    CHECK_BUFFER(flux, npts * host.ncu12);
    CHECK_BUFFER(fluxUdg, npts * host.ncu12 * common.components.nc);
    CHECK_BUFFER(fluxWdg, npts * host.ncu12 * common.components.ncw);
    CHECK_BUFFER(fluxUhg, npts * host.ncu12 * host.ncu);
    CHECK_BUFFER(wdgUdg, npts * common.components.ncw * common.components.nc);
    CHECK_BUFFER(equationUdg, npts * common.components.nc);
    CHECK_BUFFER(K, ro * ne * npairs);
    CHECK_BUFFER(G, host.ncq > 0 ? static_cast<std::size_t>(rowNpf) * donorNpe *
        npairs * host.ncu12 * host.ncu : 0);
    CHECK_BUFFER(Cpair, host.ncq > 0 ? static_cast<std::size_t>(donorNpe) *
        donorNpe * npairs : 0);
    CHECK_BUFFER(Epair, host.ncq > 0 ? static_cast<std::size_t>(donorNpe) *
        donorNpf * donorNfe * npairs : 0);
    CHECK_BUFFER(pairResidual, ro * npairs); CHECK_BUFFER(vpair, cd * npairs);
    CHECK_BUFFER(rpair, ro * npairs); CHECK_BUFFER(Fpair, ne * cd * npairs);
    CHECK_BUFFER(Rupair, ne * npairs);
#undef CHECK_BUFFER

    if (runtime.sendnb.size() != static_cast<std::size_t>(host.nsendnb) ||
            runtime.sendcount.size() != static_cast<std::size_t>(host.nsendnb) ||
            runtime.recvnb.size() != static_cast<std::size_t>(host.nrecvnb) ||
            runtime.recvcount.size() != static_cast<std::size_t>(host.nrecvnb))
        throw std::runtime_error("Phase 14 runtime neighbor-vector contract failed");

    std::vector<I> interfaceFaces;
    if (host.nownfaces > 0) {
        const I boundary = common.couplingparams.coupledboundarycondition;
        if (mesh.bf && boundary > 0) {
            for (I element = 0; element < common.meshsizes.ne1; ++element)
                for (I localFace = 0; localFace < host.nfe; ++localFace)
                    if (mesh.bf[localFace + host.nfe * element] == boundary)
                        interfaceFaces.push_back(localFace + host.nfe * element);
        }
        else if (mesh.intfaces && mesh.szintfaces > 0)
            interfaceFaces.assign(mesh.intfaces, mesh.intfaces + mesh.szintfaces);
    }
    for (I own = 0; own < host.nownfaces; ++own) {
        const I encoded = interfaceFaces.at(static_cast<std::size_t>(host.ownface[own]));
        const I face = encoded;
        const I expected = common.components.ncu * host.npf * face;
        if (runtime.ownerTarget.data[own] != expected)
            throw std::runtime_error("Phase 14 owner-target contract failed");
    }
    (void) app;
}

template <class T, class I>
void verify_runtime(const nonmatchingdataT<T, I>& host,
        const exasim::nonmatching::NonmatchingRuntime<T, I>& runtime,
        const appstructT<T, I>& app, const meshstructT<T, I>& mesh,
        const commonstructT<T, I>& common, I backend,
        const exasim::nonmatching::OwnerTestData<T, I>* ownerTests)
{
    if (runtime.interfacetype != host.interfacetype || runtime.backend != backend)
        throw std::runtime_error("Phase 14 runtime publication contract failed");
    for (int direction = 0; direction < 2; ++direction)
        verify_direction(host.dir[direction], runtime.dir[direction], app, mesh,
            common, backend, ownerTests ? &ownerTests->dir[direction] : nullptr);
}

} // namespace phase14_contracts

namespace exasim {
namespace nonmatching {

template <class T, class I>
inline bool phase14_observed_rebuild_nonmatching_interface(appstructT<T, I>& app,
        masterstructT<T, I>& master, meshstructT<T, I>& mesh,
        solstructT<T, I>& sol, commonstructT<T, I>& common,
        I mpiprocs, I mpirank, I fileoffset, nonmatchingdataT<T, I>& host,
        std::unique_ptr<NonmatchingRuntime<T, I>>& runtime, I backend,
        DonorSearchResult<T, I>* result = nullptr,
        const DonorSearchParams& parameters = DonorSearchParams(),
        bool performDistribution = true,
        std::unique_ptr<OwnerTestData<T, I>>* ownerTests = nullptr)
{
    const bool ran = rebuild_nonmatching_interface(app, master, mesh, sol, common,
        mpiprocs, mpirank, fileoffset, host, runtime, backend, result, parameters,
        performDistribution, ownerTests);
    if (ran && runtime)
        phase14_contracts::verify_runtime(host, *runtime, app, mesh, common, backend,
            ownerTests && *ownerTests ? ownerTests->get() : nullptr);
    if (ran) {
        long long sends = 0, receives = 0;
        for (int direction = 0; direction < 2; ++direction) {
            sends += host.dir[direction].nsendnb;
            receives += host.dir[direction].nrecvnb;
        }
        const long long legacy = std::max(
            2LL * common.nnbsd + 2LL * common.nnbintf,
            2LL * common.nnbsd + sends + receives);
        const I consolidated = nonmatching_request_capacity(common, host,
            "Phase 14 negative request-neighbor contract",
            "Phase 14 request-capacity contract exceeds integer range");
        if (legacy != consolidated)
            throw std::runtime_error(
                "Phase 14 consolidated request-capacity contract failed");
    }
    return ran;
}

template <class M, class T, class I>
inline void phase14_contract_compute_nonmatching_coupling(
        NonmatchingRuntime<T, I>& runtime, solstructT<T, I>& sol,
        resstructT<T, I>& res, appstructT<T, I>& app,
        masterstructT<T, I>& master, meshstructT<T, I>& mesh,
        tempstructT<T, I>& tmp, commonstructT<T, I>& common,
        cublasHandle_t handle, bool withJacobian, I backend)
{
    // First call warms the exact production path.  The second call below is
    // measured without changing its roles, views, or dispatch.
    compute_nonmatching_coupling<M>(runtime, sol, res, app, master, mesh, tmp,
                                    common, handle, withJacobian, backend);
    phase14_contracts::DirectionSnapshot<T, I> reference[2] = {
        phase14_contracts::snapshot_direction(runtime.dir[0]),
        phase14_contracts::snapshot_direction(runtime.dir[1])};
    const bool mixed = phase14_contracts::has_rectangular_direction(runtime);
    const long long ownedBefore = runtime.allocations;
    phase14_contracts::poison_compute_outputs(runtime, withJacobian);
    const auto productionObservation = phase14_contracts::observe_allocations([&] {
        compute_nonmatching_coupling<M>(runtime, sol, res, app, master, mesh,
                                        tmp, common, handle, withJacobian,
                                        backend);
    });
    const auto productionKind = mixed
        ? (withJacobian ? phase14_allocations::ComputeMixedJacobian
                        : phase14_allocations::ComputeMixedResidual)
        : (withJacobian ? phase14_allocations::ComputeSameJacobian
                        : phase14_allocations::ComputeSameResidual);
    phase14_contracts::record_allocations(productionKind,
                                          productionObservation);
    for (int k = 0; k < 2; ++k)
        phase14_contracts::require_direction(reference[k], runtime.dir[k],
                                             withJacobian);
    if (runtime.allocations != ownedBefore)
        throw std::runtime_error(
            "Phase 14 nonmatching-owned allocation contract failed in compute");

    // Attribute the two model-level calls separately.  These observations
    // include their Kokkos launches; the short/long-label probes printed by
    // this executable quantify that launch-layer component.
    for (int direction = 0; direction < 2; ++direction) {
        auto& d = runtime.dir[direction];
        if (!d.view.isdonor || d.view.npts == 0) continue;
        if (common.components.ncw > 0 && common.timeparams.wave == 0) {
            const auto observation = phase14_contracts::observe_allocations([&] {
                if (withJacobian)
                    wEquation<M>(d.wdg.data, d.wdgUdg.data, d.x.data,
                        d.equationUdg.data, d.odg.data, d.wsrc.data,
                        d.wScratch.data, app, common, d.view.npts, backend,
                        d.iScratch.data);
                else
                    wEquation<M>(d.wdg.data, d.x.data, d.equationUdg.data,
                        d.odg.data, d.wsrc.data, d.wScratch.data, app, common,
                        d.view.npts, backend, d.iScratch.data);
            });
            phase14_contracts::record_allocations(
                withJacobian ? phase14_allocations::WEquationJacobian
                             : phase14_allocations::WEquationResidual,
                observation);
        }
        const auto observation = phase14_contracts::observe_allocations([&] {
            if (withJacobian)
                FintDriver<M>(d.flux.data, d.fluxUdg.data, d.fluxWdg.data,
                    d.fluxUhg.data, d.x.data, d.udg.data, d.odg.data,
                    d.wdg.data, d.uhg.data, d.nl.data, mesh, master, app, sol,
                    tmp, common, d.view.npts,
                    common.couplingparams.coupledcondition, backend);
            else
                FintDriver<M>(d.flux.data, d.x.data, d.udg.data, d.odg.data,
                    d.wdg.data, d.uhg.data, d.nl.data, mesh, master, app, sol,
                    tmp, common, d.view.npts,
                    common.couplingparams.coupledcondition, backend);
        });
        phase14_contracts::record_allocations(
            withJacobian ? phase14_allocations::FintJacobian
                         : phase14_allocations::FintResidual,
            observation);
    }

    // Standalone driver attribution writes the driver outputs.  Recreate the
    // complete production result (including the w-chain rule) before returning
    // to the embedded numerical fixture.
    phase14_contracts::poison_compute_outputs(runtime, withJacobian);
    compute_nonmatching_coupling<M>(runtime, sol, res, app, master, mesh, tmp,
                                    common, handle, withJacobian, backend);
    for (int k = 0; k < 2; ++k)
        phase14_contracts::require_direction(reference[k], runtime.dir[k],
                                             withJacobian);

    // The separate same-family/rectangular equivalence check is deliberately
    // outside the production allocation measurement.
    if (mixed || !phase14_contracts::enable_equivalent_rectangular(runtime, &master))
        return;
    phase14_contracts::poison_compute_outputs(runtime, withJacobian);
    compute_nonmatching_coupling<M>(runtime, sol, res, app, master, mesh, tmp,
                                    common, handle, withJacobian, backend);
    for (int k = 0; k < 2; ++k)
        phase14_contracts::require_direction(reference[k], runtime.dir[k],
                                             withJacobian);
    phase14_contracts::restore_same_family(runtime);
}

template <class T, class I>
inline void phase14_contract_nonmatching_matvec_products(
        NonmatchingRuntime<T, I>& runtime, const T* ve, I m, I ne1,
        cublasHandle_t handle, I backend)
{
    // Warm and snapshot the unmodified production dispatch.
    nonmatching_matvec_products(runtime, ve, m, ne1, handle, backend);
    phase14_contracts::DirectionSnapshot<T, I> reference[2] = {
        phase14_contracts::snapshot_direction(runtime.dir[0]),
        phase14_contracts::snapshot_direction(runtime.dir[1])};
    const bool mixed = phase14_contracts::has_rectangular_direction(runtime);
    const long long ownedBefore = runtime.allocations;
    phase14_contracts::poison_matvec_outputs(runtime);
    const auto productionObservation = phase14_contracts::observe_allocations([&] {
        nonmatching_matvec_products(runtime, ve, m, ne1, handle, backend);
    });
    phase14_contracts::record_allocations(
        mixed ? phase14_allocations::MatvecMixed
              : phase14_allocations::MatvecSame,
        productionObservation);
    for (int k = 0; k < 2; ++k) {
        if (!reference[k].donor) continue;
        phase14_contracts::require_bytes(reference[k].Ri,
            runtime.dir[k].Ri.data,
            static_cast<std::size_t>(runtime.dir[k].Ri.size), "matvec Ri");
        phase14_contracts::require_bytes(reference[k].rpair,
            runtime.dir[k].rpair.data,
            static_cast<std::size_t>(runtime.dir[k].rpair.size), "matvec rpair");
    }
    if (runtime.allocations != ownedBefore)
        throw std::runtime_error(
            "Phase 14 nonmatching-owned allocation contract failed in matvec");

    if (mixed || !phase14_contracts::enable_equivalent_rectangular(
            runtime, static_cast<const masterstructT<T, I>*>(nullptr)))
        return;
    phase14_contracts::poison_matvec_outputs(runtime);
    nonmatching_matvec_products(runtime, ve, m, ne1, handle, backend);
    for (int k = 0; k < 2; ++k) {
        if (!reference[k].donor) continue;
        phase14_contracts::require_bytes(reference[k].Ri,
            runtime.dir[k].Ri.data,
            static_cast<std::size_t>(runtime.dir[k].Ri.size), "matvec Ri");
        phase14_contracts::require_bytes(reference[k].rpair,
            runtime.dir[k].rpair.data,
            static_cast<std::size_t>(runtime.dir[k].rpair.size), "matvec rpair");
    }
    phase14_contracts::restore_same_family(runtime);
}

} // namespace nonmatching
} // namespace exasim

#define rebuild_nonmatching_interface phase14_observed_rebuild_nonmatching_interface
#define compute_nonmatching_coupling phase14_contract_compute_nonmatching_coupling
#define nonmatching_matvec_products phase14_contract_nonmatching_matvec_products
#define main phase14_embedded_nonmatching_3d_main
#include "nonmatching_3d.cpp"
#undef main
#undef nonmatching_matvec_products
#undef compute_nonmatching_coupling
#undef rebuild_nonmatching_interface

namespace {

int contract_split(int argc, char** argv, int size)
{
    int split = -1;
    bool mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--split") == 0) {
            if (++i >= argc) throw std::runtime_error("--split requires a rank count");
            char* end = nullptr;
            const long value = std::strtol(argv[i], &end, 10);
            if (!end || *end || value <= 0 || value >= size)
                throw std::runtime_error("--split must satisfy 0 < split < world size");
            split = static_cast<int>(value);
        }
        else if (std::strcmp(argv[i], "--contracts") == 0) mode = true;
        else throw std::runtime_error(std::string("unexpected argument: ") + argv[i]);
    }
    if (split < 0 || !mode) throw std::runtime_error("--split k and --contracts are required");
    return split;
}

} // namespace

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
    const int rank = 0, size = 1;
#endif
    int failed = 0;
    try {
        const int split = contract_split(argc, argv, size);
        phase14_contracts::audit_host_only_metadata();
        Kokkos::initialize(argc, argv);
        phase14_contracts::measure_kokkos_label_baseline();
        for (int p : {1, 4}) {
            model_operator_case<NLCoupled3D>("phase14-tet-contract", 0, p, rank, size,
                                             0.0, split, false, false);
            model_operator_case<NLCoupled3D>("phase14-hex-contract", 1, p, rank, size,
                                             0.0, split, false, false);
            model_operator_case<WCoupled3D>("phase14-mixed-A-contract", 0, p, rank, size,
                                             0.0, split, true, false);
            model_operator_case<WCoupled3D>("phase14-mixed-B-contract", 0, p, rank, size,
                                             0.0, split, true, true);
        }
        if (failures) throw std::runtime_error("Phase 14 inherited contract checks failed");
        phase14_contracts::verify_and_print_allocation_report(rank);
        Kokkos::finalize();
        if (rank == 0)
            std::printf("PASS Phase 14 runtime contracts split %d; "
                        "production RuntimeBuffer growth 0\n", split);
    }
    catch (const std::exception& exception) {
        std::fprintf(stderr, "[Rank %d] ERROR: %s\n", rank, exception.what());
        failed = 1;
        if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
    }
#ifdef HAVE_MPI
    int globalFailed = 0;
    MPI_Allreduce(&failed, &globalFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    failed = globalFailed;
    MPI_Finalize();
#endif
    return failed ? 1 : 0;
}
