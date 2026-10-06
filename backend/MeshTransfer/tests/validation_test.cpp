#include <mesh_transfer.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace {

template <class Scalar, class Index>
void check_alternate_types(int rank, exasim::meshtransfer::Comm comm)
{
    using Adapter = exasim::meshtransfer::MeshAdapter<Scalar, Index>;
    using Transfer = exasim::meshtransfer::DistributedMeshTransfer<Scalar, Index>;
    constexpr Index npe = 3;
    const Scalar nodes[3] = {Scalar(0), Scalar(0.5), Scalar(1)};
    Kokkos::View<Scalar*> xpe("typed_xpe", npe);
    Kokkos::View<Scalar*> xe("typed_xe", npe);
    Kokkos::View<Index*> nr("typed_nr", 2);
    Kokkos::View<Index*> ne("typed_ne", 2);
    auto hxpe = Kokkos::create_mirror_view(xpe);
    for (Index i = 0; i < npe; ++i) hxpe(i) = nodes[i];
    Kokkos::deep_copy(xpe, hxpe);
    Kokkos::deep_copy(xe, xpe);
    Kokkos::deep_copy(nr, Index(-1));
    Kokkos::deep_copy(ne, Index(-1));
    Adapter adapter(xe.data(), xpe.data(), nr.data(), ne.data(),
                    1, 2, npe, 2, 1, 1, static_cast<Index>(rank));
    Transfer transfer(comm, std::move(adapter));
    Kokkos::View<Scalar*> points("typed_points", npe);
    Kokkos::deep_copy(points, xpe);
    Kokkos::View<Scalar*> shape("typed_shape", npe * npe);
    transfer.evaluate_shape(points.data(), shape.data(), npe);
    auto hshape = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shape);
    const Scalar tolerance = std::is_same_v<Scalar, float> ? Scalar(2e-5) : Scalar(1e-12);
    for (Index i = 0; i < npe; ++i)
        for (Index a = 0; a < npe; ++a)
            assert(std::abs(hshape(a + npe * i) - (a == i ? Scalar(1) : Scalar(0))) < tolerance);
}

void check_curved_inverse(int rank)
{
    using Index = std::int32_t;
    using Adapter = exasim::meshtransfer::MeshAdapter<double, Index>;
    constexpr Index p = 2;
    constexpr Index npe = 9;
    constexpr Index nd = 2;
    Kokkos::View<double*> xpe("curved_xpe", npe * nd);
    Kokkos::View<double*> xe("curved_xe", npe * nd);
    Kokkos::View<Index*> nr("curved_nr", 4);
    Kokkos::View<Index*> ne("curved_ne", 4);
    auto hxpe = Kokkos::create_mirror_view(xpe);
    auto hxe = Kokkos::create_mirror_view(xe);
    Index a = 0;
    for (Index j = 0; j <= p; ++j)
        for (Index i = 0; i <= p; ++i, ++a) {
            const double r = 0.5 * i;
            const double s = 0.5 * j;
            hxpe(a) = r;
            hxpe(a + npe) = s;
            hxe(a) = r + 0.2 * r * s;
            hxe(a + npe) = s + 0.1 * r * r;
        }
    Kokkos::deep_copy(xpe, hxpe);
    Kokkos::deep_copy(xe, hxe);
    Kokkos::deep_copy(nr, Index(-1));
    Kokkos::deep_copy(ne, Index(-1));
    Adapter adapter(xe.data(), xpe.data(), nr.data(), ne.data(),
                    1, p, npe, 4, nd, 1, rank);
    adapter.set_newton_options(20, 1e-13, 1e-11);
    const auto mesh = adapter.device_view();
    Kokkos::View<double*> recovered("curved_recovered", nd);
    Kokkos::View<int*> status("curved_status", 1);
    Kokkos::parallel_for("curved_inverse", 1, KOKKOS_LAMBDA(int) {
        const double exact[2] = {0.31, 0.43};
        const double physical[2] = {
            exact[0] + 0.2 * exact[0] * exact[1],
            exact[1] + 0.1 * exact[0] * exact[0]};
        double xi[3] = {0.5, 0.5, 0};
        double shape[exasim::meshtransfer::detail::max_element_nodes];
        double dshape[3 * exasim::meshtransfer::detail::max_element_nodes];
        double phi[exasim::meshtransfer::detail::max_element_nodes];
        double dphi[3 * exasim::meshtransfer::detail::max_element_nodes];
        status(0) = static_cast<int>(mesh.inverse_map(
            physical, 0, xi, shape, dshape, phi, dphi));
        recovered(0) = xi[0];
        recovered(1) = xi[1];
    });
    auto hs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), status);
    auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recovered);
    assert(hs(0) == static_cast<int>(exasim::meshtransfer::InverseMapStatus::Converged));
    assert(std::abs(hr(0) - 0.31) < 2e-12);
    assert(std::abs(hr(1) - 0.43) < 2e-12);
    if (rank == 0)
        std::cout << "curved_inverse_error "
                  << std::max(std::abs(hr(0) - 0.31),
                              std::abs(hr(1) - 0.43)) << '\n';
}

void check_outside_extrapolation(int rank, exasim::meshtransfer::Comm comm)
{
    using Index = std::int32_t;
    using Adapter = exasim::meshtransfer::MeshAdapter<double, Index>;
    using Transfer = exasim::meshtransfer::DistributedMeshTransfer<double, Index>;
    constexpr Index npe = 4;
    const double nodes[8] = {0,1,0,1, 0,0,1,1};
    Kokkos::View<double*> xpe("outside_xpe", 8);
    Kokkos::View<double*> xe("outside_xe", 8);
    Kokkos::View<Index*> nr("outside_nr", 4);
    Kokkos::View<Index*> ne("outside_ne", 4);
    auto hn = Kokkos::create_mirror_view(xpe);
    for (int i = 0; i < 8; ++i) hn(i) = nodes[i];
    Kokkos::deep_copy(xpe, hn);
    Kokkos::deep_copy(xe, xpe);
    Kokkos::deep_copy(nr, Index(-1));
    Kokkos::deep_copy(ne, Index(-1));
    Adapter adapter(xe.data(), xpe.data(), nr.data(), ne.data(),
                    1, 1, npe, 4, 2, 1, rank);
    Transfer transfer(comm, std::move(adapter));
    transfer.set_num_components(1);
    Kokkos::View<double*> x("outside_x", 2);
    Kokkos::View<double*> source("outside_source", npe);
    Kokkos::View<double*> target("outside_target", 1);
    Kokkos::View<Index*> erank("outside_rank", 1);
    Kokkos::View<Index*> elem("outside_elem", 1);
    Kokkos::View<double*> xi("outside_xi", 2);
    auto hx = Kokkos::create_mirror_view(x);
    auto hs = Kokkos::create_mirror_view(source);
    auto hr = Kokkos::create_mirror_view(erank);
    auto he = Kokkos::create_mirror_view(elem);
    auto hxi = Kokkos::create_mirror_view(xi);
    hx(0) = -0.2; hx(1) = 0.4;
    for (Index i = 0; i < npe; ++i)
        hs(i) = 1.0 + 2.0 * nodes[i] - 3.0 * nodes[i + npe];
    hr(0) = rank; he(0) = 0; hxi(0) = 0.5; hxi(1) = 0.5;
    Kokkos::deep_copy(x, hx);
    Kokkos::deep_copy(source, hs);
    Kokkos::deep_copy(erank, hr);
    Kokkos::deep_copy(elem, he);
    Kokkos::deep_copy(xi, hxi);
    transfer.transfer(
        x.data(), source.data(), target.data(),
        erank.data(), elem.data(), xi.data(), 1);
    auto ht = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), target);
    auto hxr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), xi);
    assert(std::abs(hxr(0) + 0.2) < 1e-12);
    assert(std::abs(ht(0) - (1.0 + 2.0 * -0.2 - 3.0 * 0.4)) < 1e-12);
    if (rank == 0)
        std::cout << "outside_extrapolation_error "
                  << std::abs(ht(0) - (1.0 + 2.0 * -0.2 - 3.0 * 0.4))
                  << '\n';
}

} // namespace

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    exasim::meshtransfer::Comm comm = MPI_COMM_WORLD;
#else
    int rank = 0;
    exasim::meshtransfer::Comm comm = 0;
#endif
    Kokkos::initialize(argc, argv);
    {
        check_alternate_types<float, std::int64_t>(rank, comm);
        check_curved_inverse(rank);
        check_outside_extrapolation(rank, comm);
    }
    Kokkos::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
