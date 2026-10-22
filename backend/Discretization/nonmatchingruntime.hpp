/*
    Backend-resident run-time cache for non-matching coupling.

    The frozen nonmatchinginterfaceT remains the host setup product.  This
    object owns the arrays used after setup and the persistent workspaces used
    by residual, Jacobian and matrix-vector evaluations.  Kernels consume the
    plain-pointer RuntimeDirectionView, which is valid in Kokkos host, CUDA and
    HIP execution spaces.
*/

#ifndef __NONMATCHINGRUNTIME_HPP
#define __NONMATCHINGRUNTIME_HPP

#include "nonmatchinginterface.hpp"
#include "nonmatchinginteraction.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace exasim {
namespace nonmatching {

// Contract between topology-specific setup and the non-matching algebra.
//
// SharedFaceSpace is the optimized line/line, tet/tet and hex/hex path: the
// owner test rows and donor trace columns use the same face master.  The
// IndependentFaceSpaces path is used when those spaces differ (currently the
// 3D tet/hex pair).  In either case the algebra consumes only the dimensions
// below; element-family detection and basis construction remain setup duties.
enum NonmatchingSpaceRelation {
    SharedFaceSpace = 0,
    IndependentFaceSpaces = 1
};

template <class I>
struct NonmatchingDirectionContract {
    I relation = SharedFaceSpace;
    I ownerElemtype = 0;
    I ownerFaceNodes = 0;
    I ownerFaceQuadraturePoints = 0;
    I donorElemtype = 0;
    I donorVolumeNodes = 0;
    I donorFaceNodes = 0;
    I donorFacesPerElement = 0;
};

template <class I>
inline bool uses_independent_face_spaces(
        const NonmatchingDirectionContract<I>& contract)
{
    return contract.relation == IndependentFaceSpaces;
}

// Owner face test values transported during setup when the two models use
// different volume-element families.  The values are node-first [npf x npts]
// and are independent of the removed composite-quadrature feature.
template <class T, class I>
struct OwnerTestDirection {
    I enabled = 0;
    NonmatchingDirectionContract<I> contract;
    std::vector<T> values;
    void clear()
    {
        enabled = 0;
        contract = NonmatchingDirectionContract<I>();
        values.clear();
    }
};

template <class T, class I>
struct OwnerTestData {
    OwnerTestDirection<T,I> dir[2];
    void clear() { dir[0].clear(); dir[1].clear(); }
    long long host_bytes() const
    {
        return static_cast<long long>(sizeof(T)) *
               static_cast<long long>(dir[0].values.size()+dir[1].values.size());
    }
};

template <class T, class I>
struct OwnerTestView {
    const T* values=nullptr;
    I npf=0;
};

template <class T, class I>
inline OwnerTestView<T,I> owner_test_view(const OwnerTestData<T,I>* tests,
        int direction,I npf)
{
    OwnerTestView<T,I> view;
    if(tests&&!tests->dir[direction].values.empty()){
        view.values=tests->dir[direction].values.data();view.npf=npf;
    }
    return view;
}

template <class V, class I>
struct RuntimeBuffer {
    V* data = nullptr;
    I size = 0;
    I capacity = 0;
    I backend = 0;

    RuntimeBuffer() = default;
    RuntimeBuffer(const RuntimeBuffer&) = delete;
    RuntimeBuffer& operator=(const RuntimeBuffer&) = delete;

    RuntimeBuffer(RuntimeBuffer&& other) noexcept { swap(other); }
    RuntimeBuffer& operator=(RuntimeBuffer&& other) noexcept
    {
        if (this != &other) {
            release();
            swap(other);
        }
        return *this;
    }

    ~RuntimeBuffer() { release(); }

    void swap(RuntimeBuffer& other) noexcept
    {
        std::swap(data, other.data);
        std::swap(size, other.size);
        std::swap(capacity, other.capacity);
        std::swap(backend, other.backend);
    }

    void release()
    {
        if (data) TemplateFree(data, backend);
        size = 0;
        capacity = 0;
    }

    bool resize(I requested, I allocationBackend)
    {
        if (requested < 0) error("non-matching runtime: negative buffer size");
        bool allocated = false;
        if (requested > capacity || (data && backend != allocationBackend)) {
            release();
            backend = allocationBackend;
            if (requested > 0) {
                TemplateMalloc(&data, requested, backend);
                if (!data) error("non-matching runtime: allocation failed");
                capacity = requested;
                allocated = true;
            }
        }
        else {
            backend = allocationBackend;
        }
        size = requested;
        return allocated;
    }

    bool copy_from_host(const V* source, I count, I allocationBackend)
    {
        const bool allocated = resize(count, allocationBackend);
        if (count > 0) {
            if (!source) error("non-matching runtime: missing setup array");
            TemplateCopytoDevice(data, source, count, backend);
        }
        return allocated;
    }
};

template <class T, class I>
struct RuntimeDirectionView {
    I direction = -1, isowner = 0, isdonor = 0;
    I nd = 0, npe = 0, npf = 0, nfe = 0, ngf = 0;
    I ncu = 0, ncq = 0, ncu12 = 0;
    NonmatchingDirectionContract<I> contract;
    I nownfaces = 0, npts = 0, nslots = 0, npairs = 0, nrecvblocks = 0;

    const I* pe = nullptr;
    const I* pa = nullptr;
    const I* pslot = nullptr;
    const I* pg = nullptr;
    const I* slotptr = nullptr;
    const I* pairslot = nullptr;
    const I* pairelem = nullptr;
    const I* pairptr = nullptr;
    const I* pairSlotPtr = nullptr;
    const I* recvface = nullptr;
    const I* ownface = nullptr;
    const I* ownerTarget = nullptr;
    const I* faceBlockPtr = nullptr;
    const I* faceBlocks = nullptr;

    const T* px = nullptr;
    const T* pnl = nullptr;
    const T* pwJ = nullptr;
    const T* pphi = nullptr;
    const T* pphibar = nullptr;
    const T* ppsi = nullptr;
    const T* pd = nullptr;
    const T* pdphibar = nullptr;
    // Mixed-family owner test values psi_i(xi_p), node-first [npf x npts].
    // nullptr selects the common master face rule.
    const T* ptest = nullptr;

    T* Ri = nullptr;
    T* Hi = nullptr;
    T* recvbuf = nullptr;
};

template <class T, class I>
struct NonmatchingRuntimeDirection {
    RuntimeDirectionView<T, I> view;

    RuntimeBuffer<I, I> pe, pa, pslot, pg, slotptr;
    RuntimeBuffer<I, I> pairslot, pairelem, pairptr, recvface, ownface;
    RuntimeBuffer<I, I> pairSlotPtr, ownerTarget, faceBlockPtr, faceBlocks;
    RuntimeBuffer<T, I> px, pnl, pwJ, pphi, pphibar, ppsi, pd, pdphibar;
    RuntimeBuffer<T, I> ptest;
    RuntimeBuffer<T, I> Ri, Hi, recvbuf;

    // Point data and model-driver workspaces.
    RuntimeBuffer<T, I> x, udg, odg, wdg, uhg, nl, wsrc;
    RuntimeBuffer<T, I> flux, fluxUdg, fluxWdg, fluxUhg, wdgUdg;
    RuntimeBuffer<T, I> equationUdg, wScratch;
    RuntimeBuffer<I, I> iScratch;

    // Pair condensation and matrix-vector workspaces.
    RuntimeBuffer<T, I> K, G, Cpair, Epair;
    RuntimeBuffer<T, I> pairResidual, vpair, rpair, Fpair, Rupair;

    std::vector<I> sendnb, sendcount, recvnb, recvcount;

    void bind_view()
    {
        view.pe = pe.data; view.pa = pa.data; view.pslot = pslot.data; view.pg = pg.data;
        view.slotptr = slotptr.data; view.pairslot = pairslot.data;
        view.pairelem = pairelem.data; view.pairptr = pairptr.data;
        view.pairSlotPtr = pairSlotPtr.data;
        view.recvface = recvface.data; view.ownface = ownface.data;
        view.ownerTarget = ownerTarget.data;
        view.faceBlockPtr = faceBlockPtr.data; view.faceBlocks = faceBlocks.data;
        view.px = px.data; view.pnl = pnl.data; view.pwJ = pwJ.data;
        view.pphi = pphi.data; view.pphibar = pphibar.data; view.ppsi = ppsi.data;
        view.pd = pd.data; view.pdphibar = pdphibar.data;
        view.ptest = ptest.size > 0 ? ptest.data : nullptr;
        view.Ri = Ri.data; view.Hi = Hi.data; view.recvbuf = recvbuf.data;
    }
};

template <class T, class I>
struct NonmatchingRuntime {
    I interfacetype = 0;
    I backend = 0;
    I ordinal = 0;
    I condition = 0;
    I tag[2] = {27100, 27101};
    long long allocations = 0;
    RuntimeBuffer<I, I> fluxmap;
    NonmatchingRuntimeDirection<T, I> dir[2];

    NonmatchingRuntime() = default;
    NonmatchingRuntime(const NonmatchingRuntime&) = delete;
    NonmatchingRuntime& operator=(const NonmatchingRuntime&) = delete;
    NonmatchingRuntime(NonmatchingRuntime&&) noexcept = default;
    NonmatchingRuntime& operator=(NonmatchingRuntime&&) noexcept = default;
};

template <class I>
inline I checked_launch_extent(std::size_t count)
{
    static_assert(std::is_integral<I>::value && std::is_signed<I>::value,
                  "non-matching runtime indices must be signed integers");
    const std::size_t limit = std::min(
        static_cast<std::size_t>(std::numeric_limits<I>::max()),
        static_cast<std::size_t>(std::numeric_limits<int>::max()));
    if (count > limit)
        error("non-matching runtime: buffer extent exceeds index range");
    return static_cast<I>(count);
}

template <class I>
inline I runtime_count(std::initializer_list<I> factors, const char* what)
{
    std::size_t count = 1;
    for (I factor : factors) {
        if (factor < 0 || (factor != 0 && count >
                static_cast<std::size_t>(std::numeric_limits<int>::max()) /
                static_cast<std::size_t>(factor))) {
            (void) what;
            error("non-matching runtime: buffer extent exceeds index range");
        }
        count *= static_cast<std::size_t>(factor);
    }
    return checked_launch_extent<I>(count);
}

template <class V, class I>
inline void refresh_buffer(RuntimeBuffer<V, I>& destination, const V* source,
        I count, I backend, long long& allocations, const char* field)
{
    if (count > 0 && !source) {
        std::string message = "non-matching runtime: missing setup array ";
        message += field;
        error(message);
    }
    if (destination.copy_from_host(source, count, backend)) ++allocations;
}

template <class V, class I>
inline void resize_buffer(RuntimeBuffer<V, I>& destination, I count,
        I backend, long long& allocations)
{
    if (destination.resize(count, backend)) ++allocations;
}

template <class T, class I>
inline void refresh_runtime_direction(NonmatchingRuntimeDirection<T, I>& runtime,
        const nonmatchinginterfaceT<T, I>& host, const appstructT<T, I>& app,
        const meshstructT<T, I>& mesh, const commonstructT<T, I>& common,
        I backend, long long& allocations, I boundary = 0,
        bool allowLegacyFaceList = true,
        const OwnerTestDirection<T, I>* ownerTests = nullptr)
{
    auto& v = runtime.view;
    v.direction = host.direction; v.isowner = host.isowner; v.isdonor = host.isdonor;
    v.nd = host.nd; v.npe = host.npe; v.npf = host.npf; v.nfe = host.nfe; v.ngf = host.ngf;
    v.ncu = host.ncu; v.ncq = host.ncq; v.ncu12 = host.ncu12;
    v.contract = ownerTests ? ownerTests->contract : NonmatchingDirectionContract<I>();
    if (!ownerTests || (v.contract.relation == SharedFaceSpace &&
            v.contract.ownerFaceNodes == 0 &&
            v.contract.ownerFaceQuadraturePoints == 0 &&
            v.contract.donorVolumeNodes == 0 &&
            v.contract.donorFaceNodes == 0 &&
            v.contract.donorFacesPerElement == 0)) {
        v.contract.relation = SharedFaceSpace;
        v.contract.ownerFaceNodes = host.npf;
        v.contract.ownerFaceQuadraturePoints = host.ngf;
        v.contract.donorVolumeNodes = host.npe;
        v.contract.donorFaceNodes = host.npf;
        v.contract.donorFacesPerElement = host.nfe;
    }
    v.nownfaces = host.nownfaces; v.npts = host.npts; v.nslots = host.nslots; v.npairs = host.npairs;
    v.nrecvblocks = 0;
    if (host.nrecvnb > 0 && (!host.recvnb || !host.recvcount))
        error("non-matching runtime: owner receive metadata are incomplete");
    for (I n = 0; n < host.nrecvnb; ++n) {
        if (host.recvcount[n] < 0 ||
            v.nrecvblocks > std::numeric_limits<I>::max() - host.recvcount[n])
            error("non-matching runtime: owner receive count is invalid");
        v.nrecvblocks += host.recvcount[n];
    }

    if (host.nslots < 0 || host.npairs < 0 || host.npts < 0 || host.nownfaces < 0 ||
        host.nrecvnb < 0 || host.nsendnb < 0 || v.nrecvblocks < 0)
        error("non-matching runtime: negative setup extent");
    if (host.nslots > 0 && (!host.slotptr || host.slotptr[0] != 0 ||
                            host.slotptr[host.nslots] != host.npts))
        error("non-matching runtime: slot point ranges are incomplete");
    if (host.npairs > 0 && (!host.pairslot || !host.pairelem || !host.pairptr ||
                            host.pairptr[0] != 0 || host.pairptr[host.npairs] != host.npts))
        error("non-matching runtime: pair data are incomplete");
    for (I slot = 0; slot < host.nslots; ++slot)
        if (host.slotptr[slot + 1] < host.slotptr[slot])
            error("non-matching runtime: slot point ranges are not monotone");
    for (I pair = 0; pair < host.npairs; ++pair) {
        if (host.pairslot[pair] < 0 || host.pairslot[pair] >= host.nslots ||
            (pair > 0 && host.pairslot[pair] < host.pairslot[pair - 1]))
            error("non-matching runtime: pair slots are not ordered");
        if (host.pairelem[pair] < 0 || host.pairelem[pair] >= common.meshsizes.ne1)
            error("non-matching runtime: pair element is outside the extended element range");
        if (host.pairptr[pair + 1] < host.pairptr[pair])
            error("non-matching runtime: pair point ranges are not monotone");
    }

    I sendSlots = 0;
    for (I n = 0; n < host.nsendnb; ++n) {
        if (!host.sendnb || !host.sendcount || host.sendcount[n] < 0 ||
            sendSlots > std::numeric_limits<I>::max() - host.sendcount[n])
            error("non-matching runtime: donor send metadata are invalid");
        if (host.sendcount[n] > 0 && !host.slotrank)
            error("non-matching runtime: donor slot-rank list is missing");
        for (I slot = 0; slot < host.sendcount[n]; ++slot)
            if (host.slotrank[sendSlots + slot] != host.sendnb[n])
                error("non-matching runtime: donor slots do not follow send-neighbor order");
        sendSlots += host.sendcount[n];
    }
    if (sendSlots != host.nslots)
        error("non-matching runtime: donor send ranges do not cover the slots");
    if (v.nrecvblocks > 0 && !host.recvface)
        error("non-matching runtime: owner receive-face list is missing");
    for (I block = 0; block < v.nrecvblocks; ++block)
        if (host.recvface[block] < 0 || host.recvface[block] >= host.nownfaces)
            error("non-matching runtime: owner receive-face index is out of range");

    std::vector<I> hostPairSlotPtr(static_cast<std::size_t>(host.nslots) + 1, 0);
    I pair = 0;
    for (I slot = 0; slot < host.nslots; ++slot) {
        hostPairSlotPtr[slot] = pair;
        while (pair < host.npairs && host.pairslot[pair] == slot) ++pair;
    }
    hostPairSlotPtr[host.nslots] = pair;
    if (pair != host.npairs)
        error("non-matching runtime: pair slots do not form contiguous ranges");

    std::vector<I> hostFaceBlockPtr(static_cast<std::size_t>(host.nownfaces) + 1, 0);
    std::vector<I> hostFaceBlocks(static_cast<std::size_t>(v.nrecvblocks), 0);
    for (I block = 0; block < v.nrecvblocks; ++block)
        ++hostFaceBlockPtr[host.recvface[block] + 1];
    for (I face = 0; face < host.nownfaces; ++face)
        hostFaceBlockPtr[face + 1] += hostFaceBlockPtr[face];
    std::vector<I> cursor = hostFaceBlockPtr;
    for (I block = 0; block < v.nrecvblocks; ++block)
        hostFaceBlocks[cursor[host.recvface[block]]++] = block;

    // [nonmatching] Build the exact owner target from the same element-major
    // boundary-face ordering used later to construct mesh.intfaces.  The
    // runtime kernel therefore does not depend on that array being available
    // when the atomic setup transaction runs.
    std::vector<I> interfaceFaces;
    if (host.nownfaces > 0) {
        if (boundary <= 0)
            boundary = common.couplingparams.coupledboundarycondition;
        if (mesh.bf && boundary > 0) {
            for (I element = 0; element < common.meshsizes.ne1; ++element)
                for (I localFace = 0; localFace < host.nfe; ++localFace)
                    if (mesh.bf[localFace + host.nfe * element] == boundary)
                        interfaceFaces.push_back(localFace + host.nfe * element);
        }
        else if (allowLegacyFaceList && mesh.intfaces && mesh.szintfaces > 0) {
            interfaceFaces.assign(mesh.intfaces, mesh.intfaces + mesh.szintfaces);
        }
        else {
            error("non-matching runtime: coupled boundary-face data are incomplete");
        }
    }
    std::vector<I> hostOwnerTarget(static_cast<std::size_t>(host.nownfaces), 0);
    for (I own = 0; own < host.nownfaces; ++own) {
        if (!host.ownface || host.ownface[own] < 0 ||
            host.ownface[own] >= static_cast<I>(interfaceFaces.size()))
            error("non-matching runtime: owner interface-face index is out of range");
        const I encoded = interfaceFaces[host.ownface[own]];
        const I element = encoded / host.nfe;
        const I localFace = encoded - host.nfe * element;
        if (element < 0 || element >= common.meshsizes.ne ||
            localFace < 0 || localFace >= host.nfe)
            error("non-matching runtime: owner element face is out of range");
        const I face = localFace + host.nfe * element;
        hostOwnerTarget[own] = runtime_count<I>(
            {common.components.ncu, host.npf, face}, "owner target");
    }

    refresh_buffer(runtime.pe, host.pe, host.npts, backend, allocations, "pe");
    refresh_buffer(runtime.pa, host.pa, host.npts, backend, allocations, "pa");
    refresh_buffer(runtime.pslot, host.pslot, host.npts, backend, allocations, "pslot");
    refresh_buffer(runtime.pg, host.pg, host.npts, backend, allocations, "pg");
    refresh_buffer(runtime.slotptr, host.slotptr, host.nslots + (host.nslots > 0), backend, allocations, "slotptr");
    refresh_buffer(runtime.pairslot, host.pairslot, host.npairs, backend, allocations, "pairslot");
    refresh_buffer(runtime.pairelem, host.pairelem, host.npairs, backend, allocations, "pairelem");
    refresh_buffer(runtime.pairptr, host.pairptr, host.npairs + (host.npairs > 0), backend, allocations, "pairptr");
    refresh_buffer(runtime.pairSlotPtr, hostPairSlotPtr.data(), host.nslots + 1,
                   backend, allocations, "pairSlotPtr");
    refresh_buffer(runtime.recvface, host.recvface, v.nrecvblocks, backend, allocations, "recvface");
    refresh_buffer(runtime.ownface, host.ownface, host.nownfaces, backend, allocations, "ownface");
    refresh_buffer(runtime.ownerTarget, hostOwnerTarget.data(), host.nownfaces,
                   backend, allocations, "ownerTarget");
    refresh_buffer(runtime.faceBlockPtr, hostFaceBlockPtr.data(), host.nownfaces + 1,
                   backend, allocations, "faceBlockPtr");
    refresh_buffer(runtime.faceBlocks, hostFaceBlocks.data(), v.nrecvblocks,
                   backend, allocations, "faceBlocks");

    refresh_buffer(runtime.px, host.px, runtime_count<I>({host.npts, host.nd}, "px"), backend, allocations, "px");
    refresh_buffer(runtime.pnl, host.pnl, runtime_count<I>({host.npts, host.nd}, "pnl"), backend, allocations, "pnl");
    refresh_buffer(runtime.pwJ, host.pwJ, host.npts, backend, allocations, "pwJ");
    refresh_buffer(runtime.pphi, host.pphi, runtime_count<I>({host.npe, host.npts}, "pphi"), backend, allocations, "pphi");
    refresh_buffer(runtime.pphibar, host.pphibar, runtime_count<I>({host.npe, host.npts}, "pphibar"), backend, allocations, "pphibar");
    refresh_buffer(runtime.ppsi, host.ppsi, runtime_count<I>({host.npf, host.npts}, "ppsi"), backend, allocations, "ppsi");
    refresh_buffer(runtime.pd, host.pd, runtime_count<I>({host.npts, host.nd}, "pd"), backend, allocations, "pd");
    refresh_buffer(runtime.pdphibar, host.pdphibar,
            host.ncq == 0 ? runtime_count<I>({host.npe, host.nd, host.npts}, "pdphibar") : 0,
            backend, allocations, "pdphibar");

    const auto& contract = v.contract;
    if (contract.relation != SharedFaceSpace &&
            contract.relation != IndependentFaceSpaces)
        error("non-matching runtime: unknown owner/donor space relation");
    const bool independentSpaces = uses_independent_face_spaces(contract);
    const bool ownerFamilyMatches = ownerTests &&
            ((contract.ownerElemtype == 0 && host.nfe == 4) ||
             (contract.ownerElemtype == 1 && host.nfe == 6));
    if (independentSpaces && (!ownerTests || contract.ownerFaceNodes <= 0 ||
            contract.ownerFaceQuadraturePoints <= 0 ||
            contract.donorVolumeNodes <= 0 || contract.donorFaceNodes <= 0 ||
            contract.donorFacesPerElement <= 0 ||
            (host.isowner && (contract.ownerFaceNodes != host.npf ||
                              !ownerFamilyMatches)) ||
            (host.isdonor && (contract.donorVolumeNodes != host.npe ||
                              contract.donorFaceNodes != host.npf ||
                              contract.donorFacesPerElement != host.nfe))))
        error("non-matching runtime: rectangular role dimensions are incomplete");
    if (!independentSpaces &&
            ((host.isowner &&
                (contract.ownerFaceNodes != host.npf ||
                 contract.ownerFaceQuadraturePoints != host.ngf)) ||
             (host.isdonor &&
                (contract.donorVolumeNodes != host.npe ||
                 contract.donorFaceNodes != host.npf ||
                 contract.donorFacesPerElement != host.nfe))))
        error("non-matching runtime: shared-space role dimensions are inconsistent");
    if (independentSpaces && (host.szRi != 0 || host.szHi != 0 || host.szsendbuf != 0 ||
            host.Ri || host.Hi || host.sendbuf))
        error("non-matching runtime: rectangular work arrays must not use frozen storage");
    const bool hasOwnerTests = ownerTests && !ownerTests->values.empty();
    const I testNpf = contract.ownerFaceNodes;
    if (independentSpaces && host.npts > 0 && !hasOwnerTests)
        error("non-matching runtime: rectangular owner test values are missing");
    if (hasOwnerTests && static_cast<long long>(ownerTests->values.size()) !=
                         static_cast<long long>(testNpf) * host.npts)
        error("non-matching runtime: owner test values do not match the point count");
    refresh_buffer(runtime.ptest, hasOwnerTests ? ownerTests->values.data() : static_cast<const T*>(nullptr),
                   hasOwnerTests ? runtime_count<I>({testNpf, host.npts}, "ptest") : 0,
                   backend, allocations, "ptest");

    const I rowNpf = contract.ownerFaceNodes;
    const I donorNpe = contract.donorVolumeNodes;
    const I donorNpf = contract.donorFaceNodes;
    const I donorNfe = contract.donorFacesPerElement;
    const I ro = runtime_count<I>({host.ncu12, rowNpf}, "coupling rows");
    const I ne = runtime_count<I>({donorNpe, host.ncu}, "donor volume columns");
    const I cd = runtime_count<I>({host.ncu, donorNpf, donorNfe}, "donor trace columns");
    if (independentSpaces) {
        resize_buffer(runtime.Ri, runtime_count<I>({ro, host.nslots}, "Ri"),
                      backend, allocations);
        resize_buffer(runtime.Hi, runtime_count<I>({ro, cd, host.npairs}, "Hi"),
                      backend, allocations);
        const std::vector<T> zeroRi(static_cast<std::size_t>(runtime.Ri.size),
                                    static_cast<T>(0));
        const std::vector<T> zeroHi(static_cast<std::size_t>(runtime.Hi.size),
                                    static_cast<T>(0));
        if (runtime.Ri.size > 0)
            TemplateCopytoDevice(runtime.Ri.data, zeroRi.data(), runtime.Ri.size, backend);
        if (runtime.Hi.size > 0)
            TemplateCopytoDevice(runtime.Hi.data, zeroHi.data(), runtime.Hi.size, backend);
    }
    else {
        refresh_buffer(runtime.Ri, host.Ri, host.szRi, backend, allocations, "Ri");
        refresh_buffer(runtime.Hi, host.Hi, host.szHi, backend, allocations, "Hi");
    }
    resize_buffer(runtime.recvbuf, host.szrecvbuf, backend, allocations);

    const I npts = host.npts;
    const I nc = common.components.nc, ncx = common.components.ncx;
    const I nco = common.components.nco, ncw = common.components.ncw;
    resize_buffer(runtime.x, runtime_count<I>({npts, ncx}, "x"), backend, allocations);
    resize_buffer(runtime.udg, runtime_count<I>({npts, nc}, "udg"), backend, allocations);
    resize_buffer(runtime.odg, runtime_count<I>({npts, nco}, "odg"), backend, allocations);
    resize_buffer(runtime.wdg, runtime_count<I>({npts, ncw}, "wdg"), backend, allocations);
    resize_buffer(runtime.uhg, runtime_count<I>({npts, host.ncu}, "uhg"), backend, allocations);
    resize_buffer(runtime.nl, runtime_count<I>({npts, host.nd}, "nl"), backend, allocations);
    resize_buffer(runtime.wsrc, runtime_count<I>({npts, ncw}, "wsrc"), backend, allocations);
    resize_buffer(runtime.flux, runtime_count<I>({npts, host.ncu12}, "flux"), backend, allocations);
    resize_buffer(runtime.fluxUdg, runtime_count<I>({npts, host.ncu12, nc}, "fluxUdg"), backend, allocations);
    resize_buffer(runtime.fluxWdg, runtime_count<I>({npts, host.ncu12, ncw}, "fluxWdg"), backend, allocations);
    resize_buffer(runtime.fluxUhg, runtime_count<I>({npts, host.ncu12, host.ncu}, "fluxUhg"), backend, allocations);
    resize_buffer(runtime.wdgUdg, runtime_count<I>({npts, ncw, nc}, "wdgUdg"), backend, allocations);
    resize_buffer(runtime.equationUdg, runtime_count<I>({npts, nc}, "equationUdg"), backend, allocations);

    const I nprop = app.materialdb_nprop;
    const I nstate = app.materialdb_nstate;
    const I np = app.materialdb_porder + 1;
    const I ncwa = ncw - nprop;
    std::size_t differentiated = static_cast<std::size_t>(ncw) + static_cast<std::size_t>(ncw) * ncw + 1;
    if (nprop > 0) {
        differentiated += static_cast<std::size_t>(ncwa) * ncwa +
                          static_cast<std::size_t>(ncw) * nc +
                          static_cast<std::size_t>(ncwa) * nc +
                          static_cast<std::size_t>(nstate) * (1 + nc + ncw + nc) +
                          static_cast<std::size_t>(nprop) * nstate +
                          static_cast<std::size_t>(nstate) * (2 + static_cast<std::size_t>(3) * np);
    }
    const std::size_t ordinary = static_cast<std::size_t>(ncw) +
                                 static_cast<std::size_t>(ncw) * ncw +
                                 static_cast<std::size_t>(std::max<I>(0, ncwa)) * std::max<I>(0, ncwa);
    const std::size_t material = nprop > 0 ? static_cast<std::size_t>(nstate) *
                                             (3 + static_cast<std::size_t>(2) * np) : 0;
    const std::size_t scratchPerPoint = std::max(differentiated, std::max(ordinary, material));
    if (scratchPerPoint > static_cast<std::size_t>(std::numeric_limits<I>::max()) /
                              static_cast<std::size_t>(std::max<I>(1, npts)))
        error("non-matching runtime: w-equation workspace exceeds index range");
    resize_buffer(runtime.wScratch, static_cast<I>(scratchPerPoint * npts), backend, allocations);
    resize_buffer(runtime.iScratch, runtime_count<I>({npts, nprop > 0 ? nstate + 1 : 1}, "iScratch"),
                  backend, allocations);

    resize_buffer(runtime.K, runtime_count<I>({ro, ne, host.npairs}, "K"), backend, allocations);
    resize_buffer(runtime.G, host.ncq > 0 ?
        runtime_count<I>({rowNpf, donorNpe, host.npairs, host.ncu12, host.ncu}, "G") : 0,
        backend, allocations);
    resize_buffer(runtime.Cpair, host.ncq > 0 ?
        runtime_count<I>({donorNpe, donorNpe, host.npairs}, "Cpair") : 0,
        backend, allocations);
    resize_buffer(runtime.Epair, host.ncq > 0 ?
        runtime_count<I>({donorNpe, donorNpf, donorNfe, host.npairs}, "Epair") : 0,
        backend, allocations);
    resize_buffer(runtime.pairResidual, runtime_count<I>({ro, host.npairs}, "pairResidual"), backend, allocations);
    resize_buffer(runtime.vpair, runtime_count<I>({cd, host.npairs}, "vpair"), backend, allocations);
    resize_buffer(runtime.rpair, runtime_count<I>({ro, host.npairs}, "rpair"), backend, allocations);
    resize_buffer(runtime.Fpair, runtime_count<I>({ne, cd, host.npairs}, "Fpair"), backend, allocations);
    resize_buffer(runtime.Rupair, runtime_count<I>({ne, host.npairs}, "Rupair"), backend, allocations);

    runtime.sendnb.clear(); runtime.sendcount.clear();
    runtime.recvnb.clear(); runtime.recvcount.clear();
    if (host.nsendnb > 0) {
        runtime.sendnb.assign(host.sendnb, host.sendnb + host.nsendnb);
        runtime.sendcount.assign(host.sendcount, host.sendcount + host.nsendnb);
    }
    if (host.nrecvnb > 0) {
        runtime.recvnb.assign(host.recvnb, host.recvnb + host.nrecvnb);
        runtime.recvcount.assign(host.recvcount, host.recvcount + host.nrecvnb);
    }
    runtime.bind_view();
}

template <class T, class I>
inline void refresh_nonmatching_runtime(NonmatchingRuntime<T, I>& runtime,
        const nonmatchingdataT<T, I>& host, const appstructT<T, I>& app,
        const meshstructT<T, I>& mesh, const commonstructT<T, I>& common, I backend,
        const OwnerTestData<T, I>* ownerTests = nullptr, I boundary = 0,
        const I* fluxmap = nullptr, I fluxmapSize = -1,
        I condition = -1, I ordinal = 0)
{
    if (backend < 0 || backend > 3) error("non-matching runtime: unsupported backend");
    runtime.backend = backend;
    runtime.interfacetype = host.interfacetype;
    if (host.interfacetype != 1) {
        runtime = NonmatchingRuntime<T, I>();
        runtime.backend = backend;
        return;
    }
    const bool graph = fluxmapSize >= 0;
    if (!graph) {
        fluxmap = app.interfacefluxmap;
        condition = common.couplingparams.coupledcondition;
    }
    runtime.ordinal = ordinal;
    runtime.condition = condition;
    try {
        runtime.tag[0] = checked_interaction_tag<I>(ordinal, 0,
                std::numeric_limits<I>::max());
        runtime.tag[1] = checked_interaction_tag<I>(ordinal, 1,
                std::numeric_limits<I>::max());
    }
    catch (...) { error("non-matching runtime: interaction tag exceeds integer range"); }
    if (graph)
        refresh_buffer(runtime.fluxmap, fluxmap, fluxmapSize, backend,
                       runtime.allocations, "fluxmap");
    else
        runtime.fluxmap.release();
    for (int direction = 0; direction < 2; ++direction)
        refresh_runtime_direction(runtime.dir[direction], host.dir[direction], app, mesh, common,
                                  backend, runtime.allocations, boundary, !graph,
                                  ownerTests ? &ownerTests->dir[direction] : nullptr);
}

} // namespace nonmatching
} // namespace exasim

#endif
