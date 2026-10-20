#include <mesh_transfer.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

std::vector<double> nodes(int elemtype, int p, int nd)
{
    const int npe = exasim::meshtransfer::detail::expected_nodes<double>(elemtype, p, nd);
    std::vector<double> x(static_cast<std::size_t>(npe * nd), 0.0);
    int a = 0;
    if (elemtype != 0) {
        if (nd == 1) {
            for (int i = 0; i <= p; ++i, ++a) x[a] = double(i) / p;
        }
        else if (nd == 2) {
            for (int j = 0; j <= p; ++j)
                for (int i = 0; i <= p; ++i, ++a) {
                    x[a] = double(i) / p;
                    x[a + npe] = double(j) / p;
                }
        }
        else {
            for (int k = 0; k <= p; ++k)
                for (int j = 0; j <= p; ++j)
                    for (int i = 0; i <= p; ++i, ++a) {
                        x[a] = double(i) / p;
                        x[a + npe] = double(j) / p;
                        x[a + 2 * npe] = double(k) / p;
                    }
        }
    }
    else if (nd == 1) {
        for (int i = 0; i <= p; ++i, ++a) x[a] = double(i) / p;
    }
    else if (nd == 2) {
        for (int j = 0; j <= p; ++j)
            for (int i = 0; i <= p - j; ++i, ++a) {
                x[a] = double(i) / p;
                x[a + npe] = double(j) / p;
            }
    }
    else {
        for (int k = 0; k <= p; ++k)
            for (int j = 0; j <= p - k; ++j)
                for (int i = 0; i <= p - j - k; ++i, ++a) {
                    x[a] = double(i) / p;
                    x[a + npe] = double(j) / p;
                    x[a + 2 * npe] = double(k) / p;
                }
    }
    return x;
}

void check_element(int elemtype, int p, int nd, int rank)
{
    using Index = std::int32_t;
    using Adapter = exasim::meshtransfer::MeshAdapter<double, Index>;
    using Transfer = exasim::meshtransfer::DistributedMeshTransfer<double, Index>;
    const Index npe = exasim::meshtransfer::detail::expected_nodes<double>(elemtype, p, nd);
    const Index nfe = nd + (nd - 1) * elemtype + 1;
    const auto hostNodes = nodes(elemtype, p, nd);
    Kokkos::View<double*> xpe("xpe", npe * nd);
    Kokkos::View<double*> xe("xe", npe * nd);
    auto xpeHost = Kokkos::create_mirror_view(xpe);
    for (Index i = 0; i < npe * nd; ++i) xpeHost(i) = hostNodes[i];
    Kokkos::deep_copy(xpe, xpeHost);
    Kokkos::deep_copy(xe, xpe);
    Kokkos::View<Index*> nr("nr", nfe);
    Kokkos::View<Index*> ne("ne", nfe);
    Kokkos::deep_copy(nr, Index(-1));
    Kokkos::deep_copy(ne, Index(-1));
    Adapter adapter(xe.data(), xpe.data(), nr.data(), ne.data(),
                    elemtype, p, npe, nfe, nd, 1, rank);
    assert(std::isfinite(adapter.vandermonde_condition_estimate()));
    assert(adapter.vandermonde_condition_estimate() > 0.0);
    if (rank == 0)
        std::cout << "condition elemtype=" << elemtype
                  << " nd=" << nd << " p=" << p
                  << " value=" << adapter.vandermonde_condition_estimate()
                  << '\n';

#ifdef HAVE_MPI
    exasim::meshtransfer::Comm comm = MPI_COMM_WORLD;
#else
    exasim::meshtransfer::Comm comm = 0;
#endif
    Transfer transfer(comm, std::move(adapter));
    Kokkos::View<double*> points("points", npe * nd);
    auto pointsHost = Kokkos::create_mirror_view(points);
    for (Index i = 0; i < npe; ++i)
        for (Index d = 0; d < nd; ++d)
            pointsHost(d + nd * i) = hostNodes[i + npe * d];
    Kokkos::deep_copy(points, pointsHost);
    Kokkos::View<double*> shape("shape", npe * npe);
    transfer.evaluate_shape(points.data(), shape.data(), npe);
    auto shapeHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shape);
    double maxError = 0.0;
    for (Index i = 0; i < npe; ++i)
        for (Index a = 0; a < npe; ++a)
            maxError = std::max(maxError,
                std::abs(shapeHost(a + npe * i) - (a == i ? 1.0 : 0.0)));
    if (rank == 0)
        std::cout << "cardinality elemtype=" << elemtype
                  << " nd=" << nd << " p=" << p
                  << " error=" << maxError << '\n';
    if (maxError > 2.0e-10) {
        std::cerr << "cardinality failure elemtype=" << elemtype
                  << " p=" << p << " nd=" << nd
                  << " error=" << maxError << '\n';
        std::abort();
    }
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
        for (int p = 1; p <= 4; ++p) {
            check_element(1, p, 1, rank);
            check_element(0, p, 2, rank);
            check_element(1, p, 2, rank);
            check_element(0, p, 3, rank);
            check_element(1, p, 3, rank);
        }
    }
    Kokkos::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
