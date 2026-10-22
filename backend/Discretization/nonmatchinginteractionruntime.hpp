#ifndef __NONMATCHING_INTERACTION_RUNTIME_HPP__
#define __NONMATCHING_INTERACTION_RUNTIME_HPP__

#include "nonmatchinginteraction.hpp"
#include "nonmatchingpreprocess.hpp"
#include "nonmatchingcoupling.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace exasim {
namespace nonmatching {

#ifdef HAVE_MPI
template <class T, class I>
inline std::vector<InteractionRecord<I>> validate_interaction_graph(
        const appstructT<T, I>& app, const meshstructT<T, I>& mesh,
        const commonstructT<T, I>& common, I localDomain,
        const std::vector<I>& domainRankOffsets)
{
    int localFailure = 0;
    std::vector<InteractionRecord<I>> records;
    if (!app.interactions || app.szinteractions < 3 ||
            app.szinteractions > static_cast<I>(std::numeric_limits<int>::max()))
        localFailure = 1;
    if (!localFailure) {
        try {
            records = parse_interaction_table<I>(app.interactions,
                    static_cast<std::size_t>(app.szinteractions));
        }
        catch (...) { localFailure = 1; }
    }
    int anyFailure = 0;
    MPI_Allreduce(&localFailure, &anyFailure, 1, MPI_INT, MPI_MAX,
                  EXASIM_COMM_WORLD);
    if (anyFailure)
        error("generalized interactions: invalid serialized interaction table");

    int localLength = static_cast<int>(app.szinteractions);
    int minLength = 0, maxLength = 0;
    MPI_Allreduce(&localLength, &minLength, 1, MPI_INT, MPI_MIN,
                  EXASIM_COMM_WORLD);
    MPI_Allreduce(&localLength, &maxLength, 1, MPI_INT, MPI_MAX,
                  EXASIM_COMM_WORLD);
    if (minLength != maxLength)
        error("generalized interactions: table length differs between ranks");
    std::vector<I> reference(static_cast<std::size_t>(localLength));
    if (common.mpiRank == 0)
        std::copy(app.interactions, app.interactions + localLength, reference.begin());
    MPI_Bcast(reference.data(), localLength, mpi_type<I>(), 0, EXASIM_COMM_WORLD);
    int differs = !std::equal(reference.begin(), reference.end(), app.interactions);
    MPI_Allreduce(MPI_IN_PLACE, &differs, 1, MPI_INT, MPI_MAX,
                  EXASIM_COMM_WORLD);
    if (differs)
        error("generalized interactions: table values differ between ranks");

    const I domainCount = app.interactions[1];
    if (domainCount < 2 || domainRankOffsets.size() !=
            static_cast<std::size_t>(domainCount) + 1 || localDomain < 0 ||
            localDomain >= domainCount)
        localFailure = 1;
    MPI_Allreduce(&localFailure, &anyFailure, 1, MPI_INT, MPI_MAX,
                  EXASIM_COMM_WORLD);
    if (anyFailure)
        error("generalized interactions: domain-rank metadata are invalid");

    I boundaryCount = mesh.nsize ? mesh.nsize[27] : static_cast<I>(-1);
    I minBoundaryCount = 0, maxBoundaryCount = 0;
    MPI_Allreduce(&boundaryCount, &minBoundaryCount, 1, mpi_type<I>(), MPI_MIN,
                  EXASIM_COMM_LOCAL);
    MPI_Allreduce(&boundaryCount, &maxBoundaryCount, 1, mpi_type<I>(), MPI_MAX,
                  EXASIM_COMM_LOCAL);
    localFailure = boundaryCount < 0 || minBoundaryCount != maxBoundaryCount ||
            boundaryCount > static_cast<I>(std::numeric_limits<int>::max()) ||
            (boundaryCount > 0 && !mesh.boundaryConditions);
    int domainFailure = 0;
    MPI_Allreduce(&localFailure, &domainFailure, 1, MPI_INT, MPI_MAX,
                  EXASIM_COMM_LOCAL);
    localFailure = domainFailure;
    std::vector<I> boundaryReference(
            static_cast<std::size_t>(std::max<I>(0, boundaryCount)));
    const I localDomainRank = common.mpiRank - domainRankOffsets[localDomain];
    if (!localFailure && localDomainRank == 0)
        std::copy(mesh.boundaryConditions, mesh.boundaryConditions + boundaryCount,
                  boundaryReference.begin());
    if (!domainFailure && boundaryCount > 0)
        MPI_Bcast(boundaryReference.data(), static_cast<int>(boundaryCount),
                  mpi_type<I>(), 0, EXASIM_COMM_LOCAL);
    if (!domainFailure && !std::equal(boundaryReference.begin(),
            boundaryReference.end(), mesh.boundaryConditions))
        localFailure = 1;
    MPI_Allreduce(&localFailure, &anyFailure, 1, MPI_INT, MPI_MAX,
                  EXASIM_COMM_WORLD);
    if (anyFailure)
        error("generalized interactions: boundary selector table differs within a domain");

    std::vector<I> selectors, boundaries;
    I donatedWidth = -1;
    if (!localFailure) {
        for (const auto& record : records) {
            int side = -1;
            if (record.endpoint[0].domain == localDomain) side = 0;
            else if (record.endpoint[1].domain == localDomain) side = 1;
            if (side < 0) continue;
            const auto& endpoint = record.endpoint[side];
            const I selector = endpoint.interfaceSelector;
            if (!mesh.boundaryConditions || !mesh.nsize || selector < 1 ||
                    selector > mesh.nsize[27]) { localFailure = 1; break; }
            const I boundary = mesh.boundaryConditions[selector - 1];
            I occurrences = 0;
            for (I i = 0; i < mesh.nsize[27]; ++i)
                occurrences += mesh.boundaryConditions[i] == boundary ? 1 : 0;
            if (boundary <= 0 || occurrences != 1 || endpoint.condition <= 0 ||
                    endpoint.fluxmap.empty() ||
                    std::find(selectors.begin(), selectors.end(), selector) != selectors.end() ||
                    std::find(boundaries.begin(), boundaries.end(), boundary) != boundaries.end()) {
                localFailure = 1; break;
            }
            for (I component : endpoint.fluxmap)
                if (component < 0 || component >= common.components.ncu) {
                    localFailure = 1; break;
                }
            if (localFailure) break;
            const I localDonated = static_cast<I>(record.endpoint[1 - side].fluxmap.size());
            if (donatedWidth < 0) donatedWidth = localDonated;
            else if (donatedWidth != localDonated) { localFailure = 1; break; }
            selectors.push_back(selector);
            boundaries.push_back(boundary);
        }
    }
    MPI_Allreduce(&localFailure, &anyFailure, 1, MPI_INT, MPI_MAX,
                  EXASIM_COMM_WORLD);
    if (anyFailure)
        error("generalized interactions: local endpoint metadata are invalid");

    // These inputs control world-wide Newton/GMRES/time-step collectives and
    // therefore must be bitwise-identical across the graph run.
    const I controls[] = {
        app.problem[2], app.problem[3], app.problem[4], app.problem[12],
        app.problem[13], app.problem[14], app.problem[15], app.flag[5],
        app.flag[6], app.flag[7]
    };
    I controlReference[10];
    if (common.mpiRank == 0) std::copy(controls, controls + 10, controlReference);
    MPI_Bcast(controlReference, 10, mpi_type<I>(), 0, EXASIM_COMM_WORLD);
    localFailure = !std::equal(controls, controls + 10, controlReference);
    I dtLength = app.nsize[4], referenceDtLength = dtLength;
    MPI_Bcast(&referenceDtLength, 1, mpi_type<I>(), 0, EXASIM_COMM_WORLD);
    if (dtLength != referenceDtLength || dtLength < 0 || (dtLength > 0 && !app.dt) ||
            dtLength > static_cast<I>(std::numeric_limits<int>::max()))
        localFailure = 1;
    std::vector<T> referenceDt(static_cast<std::size_t>(std::max<I>(0, referenceDtLength)));
    if (!localFailure && common.mpiRank == 0 && dtLength > 0)
        std::copy(app.dt, app.dt + dtLength, referenceDt.begin());
    if (referenceDtLength > 0)
        MPI_Bcast(referenceDt.data(), static_cast<int>(referenceDtLength),
                  mpi_type<T>(), 0, EXASIM_COMM_WORLD);
    if (!localFailure && dtLength > 0 &&
            (!app.dt || !std::equal(referenceDt.begin(), referenceDt.end(), app.dt)))
        localFailure = 1;
    I solverLength = app.nsize[7], referenceSolverLength = solverLength;
    MPI_Bcast(&referenceSolverLength, 1, mpi_type<I>(), 0, EXASIM_COMM_WORLD);
    if (solverLength != referenceSolverLength || solverLength < 2 || !app.solversparam ||
            solverLength > static_cast<I>(std::numeric_limits<int>::max()))
        localFailure = 1;
    std::vector<T> referenceSolver(
            static_cast<std::size_t>(std::max<I>(0, referenceSolverLength)));
    if (!localFailure && common.mpiRank == 0)
        std::copy(app.solversparam, app.solversparam + solverLength,
                  referenceSolver.begin());
    if (referenceSolverLength > 0)
        MPI_Bcast(referenceSolver.data(), static_cast<int>(referenceSolverLength),
                  mpi_type<T>(), 0, EXASIM_COMM_WORLD);
    if (!localFailure && (!app.solversparam || !std::equal(referenceSolver.begin(),
            referenceSolver.end(), app.solversparam)))
        localFailure = 1;
    MPI_Allreduce(&localFailure, &anyFailure, 1, MPI_INT, MPI_MAX,
                  EXASIM_COMM_WORLD);
    if (anyFailure)
        error("generalized interactions: solver controls differ between domains");

    int* tagUpperBound = nullptr;
    int hasTagUpperBound = 0;
    MPI_Comm_get_attr(EXASIM_COMM_WORLD, MPI_TAG_UB, &tagUpperBound,
                      &hasTagUpperBound);
    localFailure = !hasTagUpperBound || !tagUpperBound;
    if (!localFailure) {
        try {
            I checkedUpper = std::numeric_limits<I>::max();
            if (static_cast<unsigned long long>(*tagUpperBound) <
                    static_cast<unsigned long long>(checkedUpper))
                checkedUpper = static_cast<I>(*tagUpperBound);
            (void) checked_interaction_tag<I>(static_cast<I>(records.size() - 1),
                    1, checkedUpper);
        }
        catch (...) { localFailure = 1; }
    }
    MPI_Allreduce(&localFailure, &anyFailure, 1, MPI_INT, MPI_MAX,
                  EXASIM_COMM_WORLD);
    if (anyFailure)
        error("generalized interactions: interaction message tag exceeds MPI_TAG_UB");
    return records;
}
#endif

template <class T, class I>
struct NonmatchingDataDeleter {
    void operator()(nonmatchingdataT<T, I>* data) const noexcept
    {
        if (!data) return;
        free_nonmatching_data(*data);
        delete data;
    }
};

// Owns one invocation of the existing pairwise type-1 implementation.  The
// class intentionally contains no numerical or communication algorithm: every
// operation below forwards to the accepted pairwise function.
template <class T, class I>
class NonmatchingInteraction {
public:
    NonmatchingInteraction() = default;
    NonmatchingInteraction(const NonmatchingInteraction&) = delete;
    NonmatchingInteraction& operator=(const NonmatchingInteraction&) = delete;
    NonmatchingInteraction(NonmatchingInteraction&&) noexcept = default;
    NonmatchingInteraction& operator=(NonmatchingInteraction&&) = delete;

    ~NonmatchingInteraction()
    {
        runtime_.reset();
        ownerTests_.reset();
        setup_.reset();
    }

    static bool setupLegacy(appstructT<T, I>& app, masterstructT<T, I>& master,
            meshstructT<T, I>& mesh, solstructT<T, I>& sol,
            commonstructT<T, I>& common, I mpiprocs, I mpirank, I fileoffset,
            I backend, NonmatchingInteraction& out)
    {
        NonmatchingInteraction candidate;
        candidate.setup_.reset(new nonmatchingdataT<T, I>());
        DonorSearchParams parameters;
        const bool built = rebuild_nonmatching_interface(
                app, master, mesh, sol, common, mpiprocs, mpirank, fileoffset,
                *candidate.setup_, candidate.runtime_, backend,
                static_cast<DonorSearchResult<T, I>*>(nullptr), parameters,
                true, &candidate.ownerTests_);
        if (!built) return false;

        candidate.ordinal_ = 0;
        candidate.side_ = fileoffset > 0 ? 1 : 0;
        candidate.peerDomain_ = 1 - candidate.side_;
        candidate.interface_ = app.problem[28];
        candidate.condition_ = app.problem[29];
        candidate.boundary_ = app.problem[30];
        candidate.receivedWidth_ = app.nsize[14];
        candidate.donatedWidth_ = app.nsize[14];
        if (candidate.receivedWidth_ > 0 && app.interfacefluxmap)
            candidate.fluxmap_.assign(app.interfacefluxmap,
                    app.interfacefluxmap + candidate.receivedWidth_);

        // Publish only after the complete rebuild and metadata copy succeed.
        // This leaves out untouched on every exceptional path without needing
        // a move-assignment operator on the ownership type.
        out.ordinal_ = candidate.ordinal_;
        out.side_ = candidate.side_;
        out.peerDomain_ = candidate.peerDomain_;
        out.interface_ = candidate.interface_;
        out.boundary_ = candidate.boundary_;
        out.condition_ = candidate.condition_;
        out.receivedWidth_ = candidate.receivedWidth_;
        out.donatedWidth_ = candidate.donatedWidth_;
        out.fluxmap_ = std::move(candidate.fluxmap_);
        out.setup_ = std::move(candidate.setup_);
        out.ownerTests_ = std::move(candidate.ownerTests_);
        out.runtime_ = std::move(candidate.runtime_);
        return true;
    }

#ifdef HAVE_MPI
    static bool setupCollective(const InteractionRecord<I>& record, I localDomain,
            const NonmatchingEndpoint<I>* endpoint,
            appstructT<T, I>& app, masterstructT<T, I>& master,
            meshstructT<T, I>& mesh, solstructT<T, I>& sol,
            commonstructT<T, I>& common, I mpiprocs, I mpirank,
            I fileoffset, I backend, NonmatchingInteraction& out)
    {
        I side = -1;
        if (localDomain == record.endpoint[0].domain) side = 0;
        else if (localDomain == record.endpoint[1].domain) side = 1;
        const int local[3] = {
            side >= 0 && (!endpoint || endpoint->side != side || !app.problem ||
                    !app.nsize || !master.ndims || !mesh.ndims || !mesh.bf ||
                    !mesh.perm || !sol.xdg),
            side == 0 ? 1 : 0,
            side == 1 ? 1 : 0
        };
        int global[3] = {0, 0, 0};
        MPI_Allreduce(local, global, 3, MPI_INT, MPI_SUM, EXASIM_COMM_WORLD);
        if (global[0] != 0 || global[1] == 0 || global[2] == 0)
            error("non-matching interaction: endpoint setup is not ready on every endpoint domain");

        if (side < 0) {
            auto communicator = preprocess::make_interface_communicator<I>(false, mpirank, 0);
            (void) communicator;
            return false;
        }

        NonmatchingInteraction candidate;
        candidate.setup_.reset(new nonmatchingdataT<T, I>());
        DonorSearchParams parameters;
        const bool built = rebuild_nonmatching_interface(
                app, master, mesh, sol, common, mpiprocs, mpirank, fileoffset,
                *candidate.setup_, candidate.runtime_, backend,
                static_cast<DonorSearchResult<T, I>*>(nullptr), parameters,
                true, &candidate.ownerTests_, endpoint, false);
        if (!built || !candidate.runtime_)
            error("non-matching interaction: endpoint pairwise setup did not build type 1");

        candidate.ordinal_ = record.ordinal;
        candidate.side_ = side;
        candidate.peerDomain_ = record.endpoint[1 - side].domain;
        candidate.interface_ = record.endpoint[side].interfaceSelector;
        candidate.boundary_ = endpoint->boundary;
        candidate.condition_ = record.endpoint[side].condition;
        candidate.receivedWidth_ = endpoint->receivedWidth;
        candidate.donatedWidth_ = endpoint->donatedWidth;
        candidate.fluxmap_.assign(endpoint->fluxmap,
                endpoint->fluxmap + endpoint->receivedWidth);
        candidate.runtime_->ordinal = record.ordinal;
        candidate.runtime_->condition = candidate.condition_;
        candidate.runtime_->tag[0] = checked_interaction_tag<I>(record.ordinal, 0,
                std::numeric_limits<I>::max());
        candidate.runtime_->tag[1] = checked_interaction_tag<I>(record.ordinal, 1,
                std::numeric_limits<I>::max());

        out.ordinal_ = candidate.ordinal_;
        out.side_ = candidate.side_;
        out.peerDomain_ = candidate.peerDomain_;
        out.interface_ = candidate.interface_;
        out.boundary_ = candidate.boundary_;
        out.condition_ = candidate.condition_;
        out.receivedWidth_ = candidate.receivedWidth_;
        out.donatedWidth_ = candidate.donatedWidth_;
        out.fluxmap_ = std::move(candidate.fluxmap_);
        out.setup_ = std::move(candidate.setup_);
        out.ownerTests_ = std::move(candidate.ownerTests_);
        out.runtime_ = std::move(candidate.runtime_);
        return true;
    }
#endif

    template <class M>
    void compute(solstructT<T, I>& sol, resstructT<T, I>& res,
            appstructT<T, I>& app, masterstructT<T, I>& master,
            meshstructT<T, I>& mesh, tempstructT<T, I>& tmp,
            commonstructT<T, I>& common, cublasHandle_t handle,
            bool withJacobian, I backend)
    {
        compute_nonmatching_coupling<M>(*runtime_, sol, res, app, master, mesh,
                                        tmp, common, handle, withJacobian, backend);
    }

    void matvecProducts(const T* ve, I m, I ne1, cublasHandle_t handle, I backend)
    {
        nonmatching_matvec_products(*runtime_, ve, m, ne1, handle, backend);
    }

    void beginExchange(commonstructT<T, I>& common, I& requestCounter)
    {
        begin_coupling_slot_exchange(*runtime_, common, requestCounter);
    }

    void assembleOwner(T* elementResidual, const meshstructT<T, I>& mesh,
            const appstructT<T, I>& app, const commonstructT<T, I>& common) const
    {
        assemble_coupling_slot_owner(elementResidual, *runtime_, mesh, app, common);
    }

    long long requestCount() const
    {
        if (!setup_) return 0;
        long long count = 0;
        const long long limit = std::numeric_limits<int>::max();
        for (int direction = 0; direction < 2; ++direction) {
            const I nsend = setup_->dir[direction].nsendnb;
            const I nrecv = setup_->dir[direction].nrecvnb;
            if (nsend < 0 || nrecv < 0 ||
                    static_cast<unsigned long long>(nsend) >
                            static_cast<unsigned long long>(limit) ||
                    static_cast<unsigned long long>(nrecv) >
                            static_cast<unsigned long long>(limit))
                error("non-matching runtime: MPI request capacity exceeds integer range");
            const long long send = static_cast<long long>(nsend);
            const long long receive = static_cast<long long>(nrecv);
            if (count > limit - send || count + send > limit - receive)
                error("non-matching runtime: MPI request capacity exceeds integer range");
            count += send + receive;
        }
        return count;
    }

    void ensureRequestCapacity(commonstructT<T, I>& common) const
    {
        ensure_nonmatching_request_capacity(common, *setup_);
    }

    I ordinal() const noexcept { return ordinal_; }
    I side() const noexcept { return side_; }
    I peerDomain() const noexcept { return peerDomain_; }
    I interfaceSelector() const noexcept { return interface_; }
    I boundary() const noexcept { return boundary_; }
    I condition() const noexcept { return condition_; }
    I receivedWidth() const noexcept { return receivedWidth_; }
    I donatedWidth() const noexcept { return donatedWidth_; }
    const std::vector<I>& fluxmap() const noexcept { return fluxmap_; }
    bool hasSetup() const noexcept { return static_cast<bool>(setup_); }
    bool hasOwnerTests() const noexcept { return static_cast<bool>(ownerTests_); }
    bool hasRuntime() const noexcept { return static_cast<bool>(runtime_); }
    I setupInterfacetype() const noexcept { return setup_ ? setup_->interfacetype : 0; }
    I sendNeighborCount(int direction) const noexcept
    {
        return setup_ && direction >= 0 && direction < 2 ? setup_->dir[direction].nsendnb : 0;
    }
    I receiveNeighborCount(int direction) const noexcept
    {
        return setup_ && direction >= 0 && direction < 2 ? setup_->dir[direction].nrecvnb : 0;
    }
    const nonmatchingdataT<T, I>* setupData() const noexcept { return setup_.get(); }
    const NonmatchingRuntime<T, I>* runtimeData() const noexcept { return runtime_.get(); }

private:
    I ordinal_ = 0;
    I side_ = 0;
    I peerDomain_ = -1;
    I interface_ = 0;
    I boundary_ = 0;
    I condition_ = 0;
    I receivedWidth_ = 0;
    I donatedWidth_ = 0;
    std::vector<I> fluxmap_;
    std::unique_ptr<nonmatchingdataT<T, I>, NonmatchingDataDeleter<T, I>> setup_;
    std::unique_ptr<OwnerTestData<T, I>> ownerTests_;
    std::unique_ptr<NonmatchingRuntime<T, I>> runtime_;
};

#ifdef HAVE_MPI
template <class T, class I>
inline void ensure_nonmatching_request_capacity(commonstructT<T, I>& common,
        const std::vector<NonmatchingInteraction<T, I>>& interactions)
{
    const long long halo = 2LL * static_cast<long long>(common.nnbsd);
    const long long conformal = halo + 2LL * static_cast<long long>(common.nnbintf);
    long long nonmatching = halo;
    for (const auto& interaction : interactions) {
        const long long count = interaction.requestCount();
        if (count < 0 || nonmatching > std::numeric_limits<int>::max() - count)
            error("non-matching setup: MPI request count exceeds integer limits");
        nonmatching += count;
    }
    const long long capacity = std::max(conformal, nonmatching);
    if (halo < 0 || conformal < 0 || capacity > std::numeric_limits<int>::max())
        error("non-matching setup: MPI request count exceeds integer limits");
    if (common.requests) CPUFREE(common.requests);
    if (common.statuses) CPUFREE(common.statuses);
    common.requests = nullptr;
    common.statuses = nullptr;
    if (capacity > 0) {
        common.requests = static_cast<MPI_Request*>(std::malloc(
                static_cast<std::size_t>(capacity) * sizeof(MPI_Request)));
        common.statuses = static_cast<MPI_Status*>(std::malloc(
                static_cast<std::size_t>(capacity) * sizeof(MPI_Status)));
        if (!common.requests || !common.statuses)
            error("non-matching setup: MPI request allocation failed");
    }
}

// Shared graph-construction loop used by CDiscretization and the focused MPI
// contract test.  Keeping this orchestration in one place guarantees that
// both execute the same global record order, endpoint descriptor construction,
// collective setup, publication checks, and one-time request sizing.
template <class T, class I>
inline void setup_nonmatching_interactions(
        const std::vector<InteractionRecord<I>>& records, I localDomain,
        appstructT<T, I>& app, masterstructT<T, I>& master,
        meshstructT<T, I>& mesh, solstructT<T, I>& sol,
        commonstructT<T, I>& common, I mpiprocs, I mpirank,
        I fileoffset, I backend,
        std::vector<NonmatchingInteraction<T, I>>& interactions)
{
    if (!interactions.empty())
        error("generalized interactions: construction target is not empty");
    std::size_t incidentCount = 0;
    for (const auto& record : records)
        incidentCount += record.endpoint[0].domain == localDomain ||
                         record.endpoint[1].domain == localDomain ? 1 : 0;
    interactions.reserve(incidentCount);
    for (const auto& record : records) {
        I side = -1;
        if (record.endpoint[0].domain == localDomain) side = 0;
        else if (record.endpoint[1].domain == localDomain) side = 1;
        NonmatchingEndpoint<I> endpoint;
        const NonmatchingEndpoint<I>* endpointPointer = nullptr;
        if (side >= 0) {
            const auto& local = record.endpoint[side];
            endpoint.side = side;
            endpoint.boundary = mesh.boundaryConditions[local.interfaceSelector - 1];
            endpoint.receivedWidth = static_cast<I>(local.fluxmap.size());
            endpoint.donatedWidth = static_cast<I>(record.endpoint[1 - side].fluxmap.size());
            endpoint.fluxmap = local.fluxmap.data();
            endpointPointer = &endpoint;
        }
        NonmatchingInteraction<T, I> candidate;
        if (NonmatchingInteraction<T, I>::setupCollective(
                record, localDomain, endpointPointer, app, master, mesh, sol,
                common, mpiprocs, mpirank, fileoffset, backend, candidate))
            interactions.emplace_back(std::move(candidate));
    }
    if (interactions.size() != incidentCount)
        error("generalized interactions: local interaction count is inconsistent");
    for (std::size_t i = 1; i < interactions.size(); ++i)
        if (interactions[i - 1].ordinal() >= interactions[i].ordinal())
            error("generalized interactions: local interaction order is invalid");
    ensure_nonmatching_request_capacity(common, interactions);
}
#endif

} // namespace nonmatching
} // namespace exasim

#endif
