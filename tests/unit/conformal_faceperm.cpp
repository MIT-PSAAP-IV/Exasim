#ifdef HAVE_MPI
#include <mpi.h>
MPI_Comm EXASIM_COMM_WORLD = MPI_COMM_NULL;
MPI_Comm EXASIM_COMM_LOCAL = MPI_COMM_NULL;
#endif

#include <Kokkos_Core.hpp>
#include "../../backend/Common/common.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Discretization/interfacepartition.hpp"

#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message)
{
    if (!condition) {
        std::printf("FAIL conformal_faceperm: %s\n", message);
        ++failures;
    }
}

void check_layouts()
{
    const int npf = 3;
    const std::vector<int> common = {2, 0, 1, 2, 0, 1, 2, 0, 1};
    const auto compact = exasim::interfacepartition::select_faceperm_layout(common, npf, 3);
    expect(compact == std::vector<int>({2, 0, 1}),
            "uniform face permutations did not retain compact storage");

    // All six orientations of a p=1 triangular face.
    const std::vector<int> six = {
        0, 1, 2,
        0, 2, 1,
        1, 0, 2,
        1, 2, 0,
        2, 0, 1,
        2, 1, 0
    };
    const auto expanded = exasim::interfacepartition::select_faceperm_layout(six, npf, 6);
    expect(expanded == six, "mixed triangular orientations were not retained per face");

    // A legacy compact input remains valid even when geometry requires the
    // expanded runtime representation, provided its first face agrees.
    exasim::interfacepartition::verify_existing_faceperm(
            0, six.data(), npf, six, npf, 6);
    exasim::interfacepartition::verify_existing_faceperm(
            0, six.data(), static_cast<int>(six.size()), six, npf, 6);
}

void check_scatter(const std::vector<int>& faceperm, int stride, int nf)
{
    const int npf = 3;
    const int nfe = nf;
    const int ncu = 2;
    const int nc = 2;
    std::vector<int> faces(static_cast<std::size_t>(nf));
    std::vector<int> components = {0, 1};
    std::vector<dstype> received(static_cast<std::size_t>(ncu*npf*nf));
    std::vector<dstype> assembled(static_cast<std::size_t>(nc*npf*nfe), 0.0);

    for (int k = 0; k < nf; ++k) {
        faces[static_cast<std::size_t>(k)] = k;
        for (int m = 0; m < npf; ++m)
            for (int c = 0; c < ncu; ++c)
                received[static_cast<std::size_t>(c + ncu*m + ncu*npf*k)] =
                    static_cast<dstype>(1000*k + 10*m + c);
    }

    PutBoudaryNodes(assembled.data(), received.data(), faces.data(), faceperm.data(),
            components.data(), nfe, npf, ncu, nc, stride, nf);
    Kokkos::fence();

    for (int k = 0; k < nf; ++k)
        for (int n = 0; n < npf; ++n)
            for (int c = 0; c < ncu; ++c) {
                const int donor = faceperm[static_cast<std::size_t>(n + stride*k)];
                const dstype expected = static_cast<dstype>(1000*k + 10*donor + c);
                const dstype got = assembled[static_cast<std::size_t>(c + nc*n + nc*npf*k)];
                if (got != expected) {
                    std::printf("FAIL conformal_faceperm scatter face %d node %d component %d: %.17g != %.17g\n",
                            k, n, c, static_cast<double>(got), static_cast<double>(expected));
                    ++failures;
                }
            }
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        check_layouts();
        check_scatter({2, 0, 1}, 0, 3);
        check_scatter({0, 1, 2, 0, 2, 1, 1, 0, 2,
                       1, 2, 0, 2, 0, 1, 2, 1, 0}, 3, 6);
    }
    Kokkos::finalize();

    if (failures == 0)
        std::printf("PASS conformal_faceperm compact and six triangular orientations\n");
    return failures == 0 ? 0 : 1;
}
