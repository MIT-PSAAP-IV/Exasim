#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

extern "C" {
void dgetrf_(const int*, const int*, double*, const int*, int*, int*);
void dgetri_(const int*, double*, const int*, const int*, double*, const int*, int*);
void dgemm_(const char*, const char*, const int*, const int*, const int*,
            const double*, const double*, const int*, const double*, const int*,
            const double*, double*, const int*);
}

template <class T>
void xiny2(
    int* output,
    const T* first,
    const T* second,
    int m,
    int n,
    int dimension,
    double tolerance)
{
    for (int i = 0; i < m; ++i) {
        output[i] = -1;
        for (int j = 0; j < n; ++j) {
            bool match = true;
            for (int d = 0; d < dimension; ++d)
                match = match &&
                    std::abs(first[i + m * d] - second[j + n * d]) <= tolerance;
            if (match) {
                output[i] = j;
                break;
            }
        }
    }
}

#define DGETRF dgetrf_
#define DGETRI dgetri_
#define DGEMM dgemm_
#define CPUFREE(pointer) std::free(pointer)
#define error(message) throw std::runtime_error(message)
#include "../../Preprocessing/makemaster.hpp"
#undef error
#undef CPUFREE
#undef DGEMM
#undef DGETRI
#undef DGETRF

#include <mesh_transfer.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cstdint>
#include <iostream>

namespace {

std::vector<double> nodes(int elemtype, int p, int nd)
{
    const int npe = exasim::meshtransfer::detail::expected_nodes<double>(
        elemtype, p, nd);
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

void compare(int elemtype, int p, int nd, int rank, exasim::meshtransfer::Comm comm)
{
    using Index = std::int32_t;
    using Adapter = exasim::meshtransfer::MeshAdapter<double, Index>;
    using Transfer = exasim::meshtransfer::DistributedMeshTransfer<double, Index>;
    const Index npe = exasim::meshtransfer::detail::expected_nodes<double>(
        elemtype, p, nd);
    const Index nfe = nd + (nd - 1) * elemtype + 1;
    auto plocal = nodes(elemtype, p, nd);
    const int npoints = 5;
    std::vector<double> points(static_cast<std::size_t>(npoints * nd), 0.0);
    for (int q = 0; q < npoints; ++q) {
        const double t = (q + 1.0) / (npoints + 2.0);
        if (elemtype == 0) {
            for (int d = 0; d < nd; ++d)
                points[q + npoints * d] = t / (nd + 1.0);
        }
        else {
            for (int d = 0; d < nd; ++d)
                points[q + npoints * d] = std::fmod(t + 0.17 * d, 0.9);
        }
    }
    std::vector<double> reference(
        static_cast<std::size_t>(npe * npoints * (nd + 1)), 0.0);
    auto referencePoints = points;
    auto referenceNodes = plocal;
    mkshape(reference, referenceNodes, referencePoints, npoints,
            elemtype, p, nd, npe);

    Kokkos::View<double*> xpe("mkshape_xpe", npe * nd);
    Kokkos::View<double*> xe("mkshape_xe", npe * nd);
    Kokkos::View<Index*> nr("mkshape_nr", nfe);
    Kokkos::View<Index*> ne("mkshape_ne", nfe);
    auto hxpe = Kokkos::create_mirror_view(xpe);
    for (Index i = 0; i < npe * nd; ++i) hxpe(i) = plocal[i];
    Kokkos::deep_copy(xpe, hxpe);
    Kokkos::deep_copy(xe, xpe);
    Kokkos::deep_copy(nr, Index(-1));
    Kokkos::deep_copy(ne, Index(-1));
    Adapter adapter(xe.data(), xpe.data(), nr.data(), ne.data(),
                    elemtype, p, npe, nfe, nd, 1, rank);
    Transfer transfer(comm, std::move(adapter));
    Kokkos::View<double*> query("mkshape_query", npoints * nd);
    auto hquery = Kokkos::create_mirror_view(query);
    for (int q = 0; q < npoints; ++q)
        for (int d = 0; d < nd; ++d)
            hquery(d + nd * q) = points[q + npoints * d];
    Kokkos::deep_copy(query, hquery);
    Kokkos::View<double*> shape("mkshape_shape", npe * npoints);
    transfer.evaluate_shape(query.data(), shape.data(), npoints);
    auto hshape = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shape);
    double maxError = 0.0;
    for (int q = 0; q < npoints; ++q)
        for (Index a = 0; a < npe; ++a)
            maxError = std::max(
                maxError, std::abs(hshape(a + npe * q) - reference[a + npe * q]));
    if (rank == 0)
        std::cout << "mkshape elemtype=" << elemtype << " nd=" << nd
                  << " p=" << p << " error=" << maxError << '\n';
    assert(maxError < 2.0e-10);
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
        for (int p = 1; p <= 4; ++p) {
            compare(1, p, 1, rank, comm);
            compare(0, p, 2, rank, comm);
            compare(1, p, 2, rank, comm);
            compare(0, p, 3, rank, comm);
            compare(1, p, 3, rank, comm);
        }
    }
    Kokkos::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
