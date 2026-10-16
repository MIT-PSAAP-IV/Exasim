#include <mesh_transfer.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

std::vector<double> linear_nodes(int elemtype, int nd)
{
    if (nd == 1) return {0.0, 1.0};
    if (elemtype == 0 && nd == 2) return {0, 1, 0, 0, 0, 1};
    if (elemtype == 1 && nd == 2) return {0, 1, 0, 1, 0, 0, 1, 1};
    if (elemtype == 0)
        return {0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    return {
        0,1,0,1,0,1,0,1,
        0,0,1,1,0,0,1,1,
        0,0,0,0,1,1,1,1};
}

void check(int elemtype, int nd, int rank)
{
    using Index = std::int32_t;
    using Adapter = exasim::meshtransfer::MeshAdapter<double, Index>;
    const Index npe = exasim::meshtransfer::detail::expected_nodes<double>(elemtype, 1, nd);
    const Index nfe = nd + (nd - 1) * elemtype + 1;
    const auto hnodes = linear_nodes(elemtype, nd);
    Kokkos::View<double*> xpe("xpe", npe * nd);
    Kokkos::View<double*> xe("xe", npe * nd);
    auto hxpe = Kokkos::create_mirror_view(xpe);
    for (Index i = 0; i < npe * nd; ++i) hxpe(i) = hnodes[i];
    Kokkos::deep_copy(xpe, hxpe);
    Kokkos::deep_copy(xe, xpe);
    Kokkos::View<Index*> nr("nr", nfe);
    Kokkos::View<Index*> ne("ne", nfe);
    Kokkos::deep_copy(nr, Index(-1));
    Kokkos::deep_copy(ne, Index(-1));
    Adapter adapter(xe.data(), xpe.data(), nr.data(), ne.data(),
                    elemtype, 1, npe, nfe, nd, 1, rank);
    adapter.set_newton_options(12, 1.0e-13, 1.0e-12);
    const auto mesh = adapter.device_view();
    Kokkos::View<int*> result("result", 4);
    Kokkos::View<double*> recovered("recovered", nd);
    Kokkos::parallel_for("geometry_test", 1, KOKKOS_LAMBDA(int) {
        double target[3] = {0.2, 0.25, 0.15};
        double xi[3] = {0.45, 0.45, 0.45};
        double shape[exasim::meshtransfer::detail::max_element_nodes];
        double dshape[3 * exasim::meshtransfer::detail::max_element_nodes];
        double phi[exasim::meshtransfer::detail::max_element_nodes];
        double dphi[3 * exasim::meshtransfer::detail::max_element_nodes];
        const auto status = mesh.inverse_map(
            target, 0, xi, shape, dshape, phi, dphi);
        result(0) = static_cast<int>(status);
        result(1) = mesh.inside(xi) ? 1 : 0;
        double outside[3] = {-0.1, 0.2, 0.2};
        result(2) = mesh.exit_face(outside);
        Index nextRank = -2;
        Index nextElem = -2;
        result(3) = mesh.neighbor(0, 0, nextRank, nextElem) ? 1 : 0;
        for (Index d = 0; d < mesh.nd; ++d) recovered(d) = xi[d];
    });
    auto hresult = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
    auto hrecovered = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recovered);
    assert(hresult(0) == static_cast<int>(exasim::meshtransfer::InverseMapStatus::Converged));
    assert(hresult(1) == 1);
    assert(hresult(2) == (elemtype == 0 ? 1 : (nd == 1 ? 0 : (nd == 2 ? 3 : 5))));
    assert(hresult(3) == 0);
    for (Index d = 0; d < nd; ++d)
        assert(std::abs(hrecovered(d) - (d == 0 ? 0.2 : (d == 1 ? 0.25 : 0.15))) < 1.0e-12);
}

} // namespace

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    int rank = 0;
#endif
    Kokkos::initialize(argc, argv);
    {
        check(1, 1, rank);
        check(0, 2, rank);
        check(1, 2, rank);
        check(0, 3, rank);
        check(1, 3, rank);
    }
    Kokkos::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
