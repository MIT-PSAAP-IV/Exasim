// Focused Phase 3 ownership/lifecycle test.  The accepted 3D fixture is
// included so setup goes through the production donor search and rebuild.

#ifdef HAVE_MPI
#include <mpi.h>
extern MPI_Comm EXASIM_COMM_WORLD;
extern MPI_Comm EXASIM_COMM_LOCAL;
#endif
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <stdexcept>
#include <string>

#if __has_include("../../backend/Discretization/nonmatchinginteractionruntime.hpp")
#define EXASIM_PHASE3_INTERACTION_CLASS 1
#endif

#include "../../backend/Common/common.h"

// This translation unit needs to inspect transactional publication after a
// controlled setup failure.  Convert the production fatal-error macro to an
// exception only here; production and every other test retain MPI_Abort/exit.
[[noreturn]] inline void phase3_test_error(const std::string& message)
{
    throw std::runtime_error(message);
}
#undef error
#define error(message) phase3_test_error(message)

#define main phase3_embedded_nonmatching_3d_main
#include "nonmatching_3d.cpp"
#undef main

namespace {

#ifdef EXASIM_PHASE3_INTERACTION_CLASS
using Interaction = exasim::nonmatching::NonmatchingInteraction<dstype, Int>;
#endif

void require(bool value, const char* message, int& failures)
{
    if (!value) {
        std::fprintf(stderr, "FAIL Phase 3 lifecycle: %s\n", message);
        ++failures;
    }
}

void free_requests(commonstruct& common)
{
    CPUFREE(common.requests);
    CPUFREE(common.statuses);
    common.requests = nullptr;
    common.statuses = nullptr;
}

} // namespace

int main(int argc, char** argv)
{
#ifndef HAVE_MPI
    (void)argc;
    (void)argv;
    std::printf("SKIP Phase 3 lifecycle requires MPI\n");
    return 0;
#else
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    EXASIM_COMM_LOCAL = MPI_COMM_WORLD;
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Kokkos::initialize(argc, argv);

    int failures = 0;
    if (size < 2 || size % 2 != 0) {
        if (rank == 0) std::fprintf(stderr, "Phase 3 lifecycle requires an even np >= 2\n");
        failures = 1;
    }
    else {
        const Int split = size / 2;
        const Int model = rank < split ? 0 : 1;
        const Int part = rank - model * split;
        const bool noCoupledFace = split > 1 && part == split - 1;
        const Int ne = 2;
        SetupMaster3D masterData = setup_master(1, 1);
        SetupRank3D rankData;
        setup_multi_rank_3d(rankData, masterData, model, part, split,
                            0.0, 1, 0, 0, 1, ne);
        if (noCoupledFace) {
            std::fill(rankData.bf.begin(), rankData.bf.end(), Int(1));
            rankData.intfaces.clear();
            rankData.mesh.intfaces = nullptr;
            rankData.mesh.szintfaces = 0;
        }

        commonstruct common{};
        common.components.nc = 1;
        common.components.ncu = 1;
        common.components.ncx = 3;
        common.grid.nd = 3;
        common.grid.npe = masterData.npe;
        common.grid.npf = masterData.npf;
        common.grid.ngf = masterData.ngf;
        common.meshsizes.ne = ne;
        common.meshsizes.ne0 = ne;
        common.meshsizes.ne1 = ne;
        common.meshsizes.nfe = masterData.nfe;
        common.spatialScheme = 1;
        common.szinterfacefluxmap = 1;
        common.couplingparams.coupledcondition = 1;
        common.couplingparams.coupledboundarycondition = 2;
        common.couplingparams.coupledinterface = 1;
        common.couplingparams.interfacetype = 1;

        // Reference-available capacity emitter.  This block intentionally
        // uses no Phase 3 API, so this source can be copied to the immutable
        // Phase 0 tree and produce the same manifest fields.
        nonmatchingdata capacitySetup;
        std::unique_ptr<exasim::nonmatching::NonmatchingRuntime<dstype, Int>>
            capacityRuntime;
        std::unique_ptr<exasim::nonmatching::OwnerTestData<dstype, Int>>
            capacityOwnerTests;
        const bool capacitySelected = exasim::nonmatching::rebuild_nonmatching_interface(
            rankData.app, rankData.master, rankData.mesh, rankData.sol, common,
            static_cast<Int>(size), static_cast<Int>(rank), model * split,
            capacitySetup, capacityRuntime, Int(0),
            static_cast<exasim::nonmatching::DonorSearchResult<dstype, Int>*>(nullptr),
            exasim::nonmatching::DonorSearchParams(), true, &capacityOwnerTests);
        require(capacitySelected, "reference-available setup was not selected", failures);
        const Int d0Send = capacitySetup.dir[0].nsendnb;
        const Int d0Recv = capacitySetup.dir[0].nrecvnb;
        const Int d1Send = capacitySetup.dir[1].nsendnb;
        const Int d1Recv = capacitySetup.dir[1].nrecvnb;
        const long long pairRequests = static_cast<long long>(d0Send) + d0Recv +
                                       d1Send + d1Recv;
        const Int productionCapacity = exasim::nonmatching::nonmatching_request_capacity(
            common, capacitySetup,
            "non-matching runtime: MPI request capacity exceeds integer range",
            "non-matching runtime: MPI request capacity exceeds integer range");
        std::printf("PHASE3_CAPACITY rank=%d local_faces=%d d0_send=%d d0_recv=%d "
                    "d1_send=%d d1_recv=%d pair=%lld production=%d\n",
            rank, noCoupledFace ? 0 : int(ne), int(d0Send), int(d0Recv),
            int(d1Send), int(d1Recv),
            pairRequests, int(productionCapacity));
        capacityRuntime.reset();
        capacityOwnerTests.reset();
        exasim::nonmatching::free_nonmatching_data(capacitySetup);
        free_requests(common);

#ifdef EXASIM_PHASE3_INTERACTION_CLASS
        Interaction original;
        const bool selected = Interaction::setupLegacy(rankData.app, rankData.master,
            rankData.mesh, rankData.sol, common, static_cast<Int>(size),
            static_cast<Int>(rank), model * split, Int(0), original);
        require(selected, "type 1 was not selected", failures);
        require(original.hasSetup() && original.hasRuntime() && original.hasOwnerTests(),
                "type 1 did not publish all three owned resources", failures);
        require(original.ordinal() == 0 && original.side() == model &&
                original.peerDomain() == 1 - model, "legacy endpoint identity", failures);
        require(original.interfaceSelector() == rankData.problem[28] &&
                original.condition() == rankData.problem[29] &&
                original.boundary() == rankData.problem[30], "legacy endpoint metadata", failures);
        require(original.receivedWidth() == 1 && original.donatedWidth() == 1 &&
                original.fluxmap().size() == 1 && original.fluxmap()[0] == 0,
                "legacy width and flux map", failures);

        const long long classRequests = original.requestCount();
        const Int classCapacity = exasim::nonmatching::nonmatching_request_capacity(
            common, *original.setupData(),
            "non-matching runtime: MPI request capacity exceeds integer range",
            "non-matching runtime: MPI request capacity exceeds integer range");
        require(classRequests == pairRequests && classCapacity == productionCapacity,
                "class request count differs from common manifest", failures);
        if (classCapacity > 0)
            require(common.requests != nullptr && common.statuses != nullptr,
                    "request storage was not allocated", failures);

        Interaction moved(std::move(original));
        require(!original.hasSetup() && !original.hasRuntime() && !original.hasOwnerTests(),
                "move source retained ownership", failures);
        require(moved.hasSetup() && moved.hasRuntime() && moved.hasOwnerTests(),
                "move destination lost ownership", failures);

        std::vector<Interaction> one;
        one.reserve(1);
        one.emplace_back(std::move(moved));
        require(exasim::nonmatching::validate_interactions(common, one, Int(0),
                "phase3 lifecycle"), "one-entry validation was not active", failures);
        bool gpuRejected = false;
        try {
            (void)exasim::nonmatching::validate_interactions(common, one, Int(2),
                                                              "phase3 lifecycle");
        }
        catch (const std::runtime_error& exception) {
            gpuRejected = std::string(exception.what()).find("awaits device validation") !=
                          std::string::npos;
        }
        require(gpuRejected, "one-entry GPU diagnostic changed", failures);

        std::vector<Interaction> two;
        two.reserve(2);
        two.emplace_back();
        two.emplace_back();
        bool multipleRejected = false;
        try {
            (void)exasim::nonmatching::validate_interactions(common, two, Int(0),
                                                              "phase3 lifecycle");
        }
        catch (const std::runtime_error& exception) {
            multipleRejected = std::string(exception.what()).find(
                "Phase 3 supports at most one interaction") != std::string::npos;
        }
        require(multipleRejected, "size>1 invariant was not rejected first", failures);

        one.clear();
        free_requests(common);

        Interaction empty;
        rankData.problem[34] = 0;
        require(!Interaction::setupLegacy(rankData.app, rankData.master, rankData.mesh,
                    rankData.sol, common, static_cast<Int>(size), static_cast<Int>(rank),
                    model * split, Int(0), empty) && !empty.hasSetup(),
                "type 0 created an interaction", failures);
        rankData.problem[34] = -1;
        require(!Interaction::setupLegacy(rankData.app, rankData.master, rankData.mesh,
                    rankData.sol, common, static_cast<Int>(size), static_cast<Int>(rank),
                    model * split, Int(0), empty) && !empty.hasSetup(),
                "uncoupled input created an interaction", failures);

        rankData.problem[34] = rank == 0 ? 1 : 0;
        bool inconsistentRejected = false;
        try {
            (void)Interaction::setupLegacy(rankData.app, rankData.master, rankData.mesh,
                    rankData.sol, common, static_cast<Int>(size), static_cast<Int>(rank),
                    model * split, Int(0), empty);
        }
        catch (const std::runtime_error& exception) {
            inconsistentRejected = std::string(exception.what()).find(
                "coupling selection is inconsistent across coupled ranks") !=
                std::string::npos;
        }
        require(inconsistentRejected && !empty.hasSetup(),
                "inconsistent selection was not rejected uniformly", failures);

        rankData.problem[34] = 1;
        common.nnbsd = -1;
        bool failedTransaction = false;
        try {
            (void)Interaction::setupLegacy(rankData.app, rankData.master, rankData.mesh,
                    rankData.sol, common, static_cast<Int>(size), static_cast<Int>(rank),
                    model * split, Int(0), empty);
        }
        catch (const std::runtime_error& exception) {
            failedTransaction = std::string(exception.what()).find(
                "MPI request capacity exceeds integer range") != std::string::npos;
        }
        require(failedTransaction && !empty.hasSetup() && !empty.hasRuntime() &&
                !empty.hasOwnerTests(), "failed setup published partial ownership", failures);
        common.nnbsd = 0;
        free_requests(common);
#endif
    }

    int globalFailures = 0;
    MPI_Allreduce(&failures, &globalFailures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    Kokkos::finalize();
    MPI_Finalize();
    if (globalFailures == 0 && rank == 0) std::printf("PASS Phase 3 interaction lifecycle\n");
    return globalFailures == 0 ? 0 : 1;
#endif
}
