#include <mesh_transfer.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
        MPI_Finalize();
        return 0;
    }
    exasim::meshtransfer::Comm comm = MPI_COMM_WORLD;
#else
    int rank = 0;
    exasim::meshtransfer::Comm comm = 0;
#endif
    Kokkos::initialize(argc, argv);
    {
        using Index = std::int32_t;
        using Adapter = exasim::meshtransfer::MeshAdapter<double, Index>;
        using Transfer = exasim::meshtransfer::DistributedMeshTransfer<double, Index>;
        constexpr Index npe = 4;
        constexpr Index nd = 2;
        constexpr Index nfe = 4;
        constexpr Index nelem = 2;
        const double xpeValues[8] = {0,1,0,1, 0,0,1,1};
        const double xeValues[16] = {
            0,1,0,1, 0,0,1,1,
            1,2,1,2, 0,0,1,1};
        const Index rankValues[8] = {
            -1,rank,-1,-1, -1,-1,-1,rank};
        const Index elemValues[8] = {
            -1,1,-1,-1, -1,-1,-1,0};
        Kokkos::View<double*> xpe("xpe", 8);
        Kokkos::View<double*> xe("xe", 16);
        Kokkos::View<Index*> nr("nr", 8);
        Kokkos::View<Index*> ne("ne", 8);
        auto hxpe = Kokkos::create_mirror_view(xpe);
        auto hxe = Kokkos::create_mirror_view(xe);
        auto hnr = Kokkos::create_mirror_view(nr);
        auto hne = Kokkos::create_mirror_view(ne);
        for (int i = 0; i < 8; ++i) {
            hxpe(i) = xpeValues[i];
            hnr(i) = rankValues[i];
            hne(i) = elemValues[i];
        }
        for (int i = 0; i < 16; ++i) hxe(i) = xeValues[i];
        Kokkos::deep_copy(xpe, hxpe);
        Kokkos::deep_copy(xe, hxe);
        Kokkos::deep_copy(nr, hnr);
        Kokkos::deep_copy(ne, hne);
        Adapter adapter(xe.data(), xpe.data(), nr.data(), ne.data(),
                        1, 1, npe, nfe, nd, nelem, rank);
        Transfer transfer(comm, std::move(adapter));

        Kokkos::View<double*> x("x", 6);
        Kokkos::View<Index*> erank("erank", 3);
        Kokkos::View<Index*> elem("elem", 3);
        Kokkos::View<double*> xi("xi", 6);
        auto hx = Kokkos::create_mirror_view(x);
        auto hr = Kokkos::create_mirror_view(erank);
        auto he = Kokkos::create_mirror_view(elem);
        auto hxi = Kokkos::create_mirror_view(xi);
        const double points[6] = {0.25,0.5, 1.5,0.5, -0.25,0.5};
        for (int i = 0; i < 6; ++i) hx(i) = points[i];
        for (int i = 0; i < 3; ++i) {
            hr(i) = rank;
            he(i) = 0;
            hxi(2*i) = 0.5;
            hxi(2*i+1) = 0.5;
        }
        Kokkos::deep_copy(x, hx);
        Kokkos::deep_copy(erank, hr);
        Kokkos::deep_copy(elem, he);
        Kokkos::deep_copy(xi, hxi);
        transfer.set_target_points(x.data(), 3, erank.data(), elem.data(), xi.data());
        transfer.locate(x.data(), erank.data(), elem.data(), xi.data(), 3);
        he = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), elem);
        hxi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xi);
        assert(he(0) == 0);
        assert(he(1) == 1);
        assert(he(2) == 0);
        assert(std::abs(hxi(0) - 0.25) < 1.0e-12);
        assert(std::abs(hxi(2) - 0.5) < 1.0e-12);
        assert(std::abs(hxi(4) + 0.25) < 1.0e-12);
    }
    Kokkos::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
