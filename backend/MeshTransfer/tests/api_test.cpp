#include <mesh_transfer.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cmath>
#include <type_traits>

using Adapter = exasim::meshtransfer::MeshAdapter<double, std::int32_t>;
using Transfer = exasim::meshtransfer::DistributedMeshTransfer<double, std::int32_t>;

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    exasim::meshtransfer::Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
#else
    (void)argc;
    (void)argv;
    exasim::meshtransfer::Comm comm = 0;
    int rank = 0;
#endif
    Kokkos::initialize(argc, argv);
    {
        static_assert(!std::is_copy_constructible_v<Adapter>);
        static_assert(std::is_move_constructible_v<Adapter>);
        static_assert(!std::is_copy_constructible_v<Transfer>);
        static_assert(std::is_move_constructible_v<Transfer>);

        Kokkos::View<double*> xe("xe", 2);
        Kokkos::View<double*> xpe("xpe", 2);
        Kokkos::View<std::int32_t*> nr("nr", 2);
        Kokkos::View<std::int32_t*> ne("ne", 2);
        auto xpeHost = Kokkos::create_mirror_view(xpe);
        xpeHost(0) = 0.0;
        xpeHost(1) = 1.0;
        Kokkos::deep_copy(xpe, xpeHost);
        Kokkos::deep_copy(xe, xpe);
        Kokkos::deep_copy(nr, std::int32_t(-1));
        Kokkos::deep_copy(ne, std::int32_t(-1));
        Adapter mesh(xe.data(), xpe.data(), nr.data(), ne.data(),
                     1, 1, 2, 2, 1, 1, rank);
        Transfer transfer(comm, std::move(mesh));

        Kokkos::View<double*> point("component_point", 1);
        Kokkos::View<double*> source("component_source", 4);
        Kokkos::View<double*> targetWithNc("component_target_with_nc", 2);
        Kokkos::View<double*> targetExplicit("component_target_explicit", 2);
        Kokkos::View<std::int32_t*> pointRank("component_rank", 1);
        Kokkos::View<std::int32_t*> pointElem("component_elem", 1);
        Kokkos::View<double*> pointXi("component_xi", 1);
        auto sourceHost = Kokkos::create_mirror_view(source);
        sourceHost(0) = 1.0;
        sourceHost(1) = 3.0;
        sourceHost(2) = 2.0;
        sourceHost(3) = 6.0;
        Kokkos::deep_copy(source, sourceHost);
        Kokkos::deep_copy(point, 0.25);
        Kokkos::deep_copy(pointRank, static_cast<std::int32_t>(rank));
        Kokkos::deep_copy(pointElem, std::int32_t(0));
        Kokkos::deep_copy(pointXi, 0.5);

        transfer.transfer(
            point.data(), source.data(), targetWithNc.data(),
            pointRank.data(), pointElem.data(), pointXi.data(), 1, 2);
        assert(transfer.num_components() == 2);
        auto targetWithNcHost = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), targetWithNc);
        assert(std::abs(targetWithNcHost(0) - 1.5) < 1e-12);
        assert(std::abs(targetWithNcHost(1) - 3.0) < 1e-12);

        Adapter evaluateMesh(xe.data(), xpe.data(), nr.data(), ne.data(),
                             1, 1, 2, 2, 1, 1, rank);
        Transfer evaluator(comm, std::move(evaluateMesh));
        Kokkos::View<double*> evaluatedWithNc(
            "component_evaluated_with_nc", 2);
        evaluator.evaluate(
            source.data(), evaluatedWithNc.data(), pointRank.data(),
            pointElem.data(), pointXi.data(), 1, 2);
        assert(evaluator.num_components() == 2);
        auto evaluatedWithNcHost = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), evaluatedWithNc);
        assert(std::abs(evaluatedWithNcHost(0) - 1.5) < 1e-12);
        assert(std::abs(evaluatedWithNcHost(1) - 3.0) < 1e-12);

        Kokkos::View<double*> evaluatedExplicit(
            "component_evaluated_explicit", 2);
        evaluator.set_num_components(2);
        evaluator.evaluate(
            source.data(), evaluatedExplicit.data(), pointRank.data(),
            pointElem.data(), pointXi.data(), 1);
        auto evaluatedExplicitHost = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), evaluatedExplicit);
        assert(std::abs(
            evaluatedExplicitHost(0) - evaluatedWithNcHost(0)) < 1e-12);
        assert(std::abs(
            evaluatedExplicitHost(1) - evaluatedWithNcHost(1)) < 1e-12);

        Kokkos::View<double*> evaluatedOneComponent(
            "component_evaluated_one_component", 1);
        evaluator.evaluate(
            source.data(), evaluatedOneComponent.data(), pointRank.data(),
            pointElem.data(), pointXi.data(), 1, 1);
        assert(evaluator.num_components() == 1);
        auto evaluatedOneComponentHost = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), evaluatedOneComponent);
        assert(std::abs(evaluatedOneComponentHost(0) - 1.5) < 1e-12);

        bool evaluateRejectedInvalidNc = false;
        try {
            evaluator.evaluate(
                source.data(), evaluatedWithNc.data(), pointRank.data(),
                pointElem.data(), pointXi.data(), 1, 0);
        }
        catch (const exasim::meshtransfer::MeshTransferError&) {
            evaluateRejectedInvalidNc = true;
        }
        assert(evaluateRejectedInvalidNc);

        Kokkos::deep_copy(pointRank, static_cast<std::int32_t>(rank));
        Kokkos::deep_copy(pointElem, std::int32_t(0));
        Kokkos::deep_copy(pointXi, 0.5);
        transfer.set_num_components(2);
        transfer.transfer(
            point.data(), source.data(), targetExplicit.data(),
            pointRank.data(), pointElem.data(), pointXi.data(), 1);
        auto targetExplicitHost = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), targetExplicit);
        assert(std::abs(targetExplicitHost(0) - targetWithNcHost(0)) < 1e-12);
        assert(std::abs(targetExplicitHost(1) - targetWithNcHost(1)) < 1e-12);

        Kokkos::View<double*> targetOneComponent(
            "component_target_one_component", 1);
        Kokkos::deep_copy(pointRank, static_cast<std::int32_t>(rank));
        Kokkos::deep_copy(pointElem, std::int32_t(0));
        Kokkos::deep_copy(pointXi, 0.5);
        transfer.transfer(
            point.data(), source.data(), targetOneComponent.data(),
            pointRank.data(), pointElem.data(), pointXi.data(), 1, 1);
        assert(transfer.num_components() == 1);
        auto targetOneComponentHost = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), targetOneComponent);
        assert(std::abs(targetOneComponentHost(0) - 1.5) < 1e-12);

        bool rejectedInvalidNc = false;
        try {
            transfer.transfer(
                point.data(), source.data(), targetWithNc.data(),
                pointRank.data(), pointElem.data(), pointXi.data(), 1, 0);
        }
        catch (const exasim::meshtransfer::MeshTransferError&) {
            rejectedInvalidNc = true;
        }
        assert(rejectedInvalidNc);

        Adapter registeredMesh(xe.data(), xpe.data(), nr.data(), ne.data(),
                               1, 1, 2, 2, 1, 1, rank);
        Transfer registeredTransfer(comm, std::move(registeredMesh));
        Kokkos::View<double*> registeredTargetWithNc(
            "registered_target_with_nc", 2);
        Kokkos::View<double*> registeredTargetExplicit(
            "registered_target_explicit", 2);
        Kokkos::deep_copy(pointRank, static_cast<std::int32_t>(rank));
        Kokkos::deep_copy(pointElem, std::int32_t(0));
        Kokkos::deep_copy(pointXi, 0.5);
        registeredTransfer.set_target_points(
            point.data(), 1, pointRank.data(), pointElem.data(), pointXi.data());
        registeredTransfer.transfer(
            source.data(), registeredTargetWithNc.data(), 2);
        assert(registeredTransfer.num_components() == 2);
        auto registeredTargetWithNcHost = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), registeredTargetWithNc);
        assert(std::abs(registeredTargetWithNcHost(0) - 1.5) < 1e-12);
        assert(std::abs(registeredTargetWithNcHost(1) - 3.0) < 1e-12);

        Kokkos::deep_copy(pointRank, static_cast<std::int32_t>(rank));
        Kokkos::deep_copy(pointElem, std::int32_t(0));
        Kokkos::deep_copy(pointXi, 0.5);
        registeredTransfer.set_num_components(2);
        registeredTransfer.transfer(
            source.data(), registeredTargetExplicit.data());
        auto registeredTargetExplicitHost =
            Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), registeredTargetExplicit);
        assert(std::abs(
            registeredTargetExplicitHost(0) -
            registeredTargetWithNcHost(0)) < 1e-12);
        assert(std::abs(
            registeredTargetExplicitHost(1) -
            registeredTargetWithNcHost(1)) < 1e-12);

        Kokkos::View<double*> registeredTargetOneComponent(
            "registered_target_one_component", 1);
        Kokkos::deep_copy(pointRank, static_cast<std::int32_t>(rank));
        Kokkos::deep_copy(pointElem, std::int32_t(0));
        Kokkos::deep_copy(pointXi, 0.5);
        registeredTransfer.transfer(
            source.data(), registeredTargetOneComponent.data(), 1);
        assert(registeredTransfer.num_components() == 1);
        auto registeredTargetOneComponentHost =
            Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), registeredTargetOneComponent);
        assert(std::abs(registeredTargetOneComponentHost(0) - 1.5) < 1e-12);

        bool registeredRejectedInvalidNc = false;
        try {
            registeredTransfer.transfer(
                source.data(), registeredTargetWithNc.data(), 0);
        }
        catch (const exasim::meshtransfer::MeshTransferError&) {
            registeredRejectedInvalidNc = true;
        }
        assert(registeredRejectedInvalidNc);

        transfer.set_num_components(3);
        transfer.reserve(8);
        const auto capacity = transfer.point_capacity();
        transfer.reserve(4);
        assert(transfer.point_capacity() == capacity);

        Kokkos::View<double*> x("x", 4);
        Kokkos::View<std::int32_t*> erank("erank", 4);
        Kokkos::View<std::int32_t*> elem("elem", 4);
        Kokkos::View<double*> xi("xi", 4);
        transfer.set_target_points(
            x.data(), 4, erank.data(), elem.data(), xi.data());
        assert(transfer.num_target_points() == 4);
        assert(transfer.spatial_dimension() == 1);
        assert(transfer.num_components() == 3);
        assert(transfer.registered_points() == x.data());
        assert(transfer.registered_ranks() == erank.data());
        assert(transfer.registered_elements() == elem.data());
        assert(transfer.registered_reference_coordinates() == xi.data());

        Transfer moved(std::move(transfer));
        assert(moved.num_target_points() == 4);
        assert(moved.point_capacity() == capacity);
    }
    Kokkos::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
