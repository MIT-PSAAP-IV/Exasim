#include <mesh_transfer.hpp>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cassert>
#include <cmath>
#include <cstdint>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Kokkos::initialize(argc, argv);
    {
        using Index = std::int32_t;
        using Adapter = exasim::meshtransfer::MeshAdapter<double, Index>;
        using Transfer = exasim::meshtransfer::DistributedMeshTransfer<double, Index>;
        constexpr Index npe = 4;
        constexpr Index nd = 2;
        constexpr Index nfe = 4;
        const double xpeValues[8] = {0,1,0,1, 0,0,1,1};
        const double xeValues[8] = {
            double(rank),double(rank+1),double(rank),double(rank+1),
            0,0,1,1};
        const Index rankValues[4] = {
            -1, rank + 1 < size ? rank + 1 : -1, -1, rank > 0 ? rank - 1 : -1};
        const Index elemValues[4] = {
            -1, rank + 1 < size ? 0 : -1, -1, rank > 0 ? 0 : -1};
        Kokkos::View<double*> xpe("xpe", 8);
        Kokkos::View<double*> xe("xe", 8);
        Kokkos::View<Index*> nr("nr", 4);
        Kokkos::View<Index*> ne("ne", 4);
        auto hxpe = Kokkos::create_mirror_view(xpe);
        auto hxe = Kokkos::create_mirror_view(xe);
        auto hnr = Kokkos::create_mirror_view(nr);
        auto hne = Kokkos::create_mirror_view(ne);
        for (int i = 0; i < 8; ++i) {
            hxpe(i) = xpeValues[i];
            hxe(i) = xeValues[i];
        }
        for (int i = 0; i < 4; ++i) {
            hnr(i) = rankValues[i];
            hne(i) = elemValues[i];
        }
        Kokkos::deep_copy(xpe, hxpe);
        Kokkos::deep_copy(xe, hxe);
        Kokkos::deep_copy(nr, hnr);
        Kokkos::deep_copy(ne, hne);
        Adapter adapter(xe.data(), xpe.data(), nr.data(), ne.data(),
                        1, 1, npe, nfe, nd, 1, rank);
        Transfer transfer(MPI_COMM_WORLD, std::move(adapter));

        Kokkos::View<double*> x("x", 2);
        Kokkos::View<Index*> erank("erank", 1);
        Kokkos::View<Index*> elem("elem", 1);
        Kokkos::View<double*> xi("xi", 2);
        auto hx = Kokkos::create_mirror_view(x);
        auto hr = Kokkos::create_mirror_view(erank);
        auto he = Kokkos::create_mirror_view(elem);
        auto hxi = Kokkos::create_mirror_view(xi);
        const int expectedRank = (rank + size - 1) % size;
        hx(0) = expectedRank + 0.25;
        hx(1) = 0.4;
        hr(0) = rank;
        he(0) = 0;
        hxi(0) = 0.5;
        hxi(1) = 0.5;
        Kokkos::deep_copy(x, hx);
        Kokkos::deep_copy(erank, hr);
        Kokkos::deep_copy(elem, he);
        Kokkos::deep_copy(xi, hxi);
        transfer.locate(x.data(), erank.data(), elem.data(), xi.data(), 1);
        hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), erank);
        he = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), elem);
        hxi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xi);
        assert(hr(0) == expectedRank);
        assert(he(0) == 0);
        assert(std::abs(hxi(0) - 0.25) < 1.0e-11);
        assert(std::abs(hxi(1) - 0.4) < 1.0e-11);

        Kokkos::View<double*> source("source", 2 * npe);
        Kokkos::View<double*> target("target", 2);
        auto hsource = Kokkos::create_mirror_view(source);
        for (Index a = 0; a < npe; ++a) {
            const double gx = xeValues[a];
            const double gy = xeValues[a + npe];
            hsource(a) = 1.0 + 2.0 * gx - 3.0 * gy;
            hsource(a + npe) = gx * gy;
        }
        Kokkos::deep_copy(source, hsource);
        transfer.set_num_components(2);
        transfer.evaluate(source.data(), target.data(), erank.data(), elem.data(), xi.data(), 1);
        auto htarget =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), target);
        const double px = expectedRank + 0.25;
        assert(std::abs(htarget(0) - (1.0 + 2.0 * px - 3.0 * 0.4)) < 1.0e-11);
        assert(std::abs(htarget(1) - px * 0.4) < 1.0e-11);
        const Index sendQueryCapacity = transfer.send_query_capacity();
        const Index recvQueryCapacity = transfer.receive_query_capacity();
        const Index sendResultCapacity = transfer.send_result_capacity();
        const Index recvResultCapacity = transfer.receive_result_capacity();

        for (Index i = 0; i < 2 * npe; ++i) hsource(i) *= 2.0;
        Kokkos::deep_copy(source, hsource);
        transfer.evaluate(source.data(), target.data(), erank.data(), elem.data(), xi.data(), 1);
        htarget = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), target);
        assert(std::abs(htarget(0) - 2.0 * (1.0 + 2.0 * px - 3.0 * 0.4)) < 1.0e-11);
        assert(std::abs(htarget(1) - 2.0 * px * 0.4) < 1.0e-11);
        assert(transfer.send_query_capacity() == sendQueryCapacity);
        assert(transfer.receive_query_capacity() == recvQueryCapacity);
        assert(transfer.send_result_capacity() == sendResultCapacity);
        assert(transfer.receive_result_capacity() == recvResultCapacity);

        transfer.set_target_points(
            x.data(), 1, erank.data(), elem.data(), xi.data());
        transfer.transfer(source.data(), target.data());
        htarget = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), target);
        assert(std::abs(htarget(0) - 2.0 * (1.0 + 2.0 * px - 3.0 * 0.4)) < 1.0e-11);
        assert(std::abs(htarget(1) - 2.0 * px * 0.4) < 1.0e-11);
    }
    Kokkos::finalize();
    MPI_Finalize();
}
