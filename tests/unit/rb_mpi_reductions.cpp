#include <Kokkos_Core.hpp>
#include <mpi.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
using std::partial_sum;
#ifndef HAVE_MPI
#define HAVE_MPI
#endif
MPI_Comm EXASIM_COMM_WORLD = MPI_COMM_NULL;
MPI_Comm EXASIM_COMM_LOCAL = MPI_COMM_NULL;
#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/pblas.h"

template <class T> int checkReductions(int rank, int ranks)
{
    T alpha = 1, beta = 0;
    T A[] = {T(rank+1), 0, 0, T(rank+2)}, x[] = {1, 2};
    std::array<T, 8> y, ylocal, gram, gramlocal;
    // Real storage after the intended destinations detects a wrong MPI element
    // type without depending on an out-of-bounds access or a sanitizer.
    y.fill(T(101)); ylocal.fill(T(103));
    gram.fill(T(107)); gramlocal.fill(T(109));
    PGEMTV(0, 2, 2, &alpha, A, 2, x, 1, &beta, y.data(), 1, ylocal.data(), 0);
    PGEMTM(0, 2, 2, 2, &alpha, A, 2, A, 2, &beta, gram.data(), 2, gramlocal.data(), 0);
    T expectedY0 = 0, expectedY1 = 0, expectedG0 = 0, expectedG3 = 0;
    for (int r=0; r<ranks; ++r) {
        expectedY0 += T(r+1);
        expectedY1 += T(2*(r+2));
        expectedG0 += T((r+1)*(r+1));
        expectedG3 += T((r+2)*(r+2));
    }
    int failures = y[0] != expectedY0 || y[1] != expectedY1;
    failures += gram[0] != expectedG0 || gram[1] != 0 || gram[2] != 0 || gram[3] != expectedG3;
    for (int i=2; i<8; ++i) failures += y[i] != T(101);
    for (int i=4; i<8; ++i) failures += gram[i] != T(107);
    std::printf("rank %d %s RB reductions/guards: %d failures\n", rank,
                sizeof(T)==sizeof(float) ? "float" : "double", failures);
    return failures;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = EXASIM_COMM_LOCAL = MPI_COMM_WORLD;
    Kokkos::initialize(argc, argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const int localFailures = checkReductions<double>(rank, ranks) + checkReductions<float>(rank, ranks);
    int failures = 0;
    MPI_Allreduce(&localFailures, &failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    Kokkos::finalize();
    MPI_Finalize();
    return failures ? 1 : 0;
}
