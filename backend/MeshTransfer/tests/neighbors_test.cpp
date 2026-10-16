#include <mesh_neighbors.hpp>

#include <cassert>
#include <cstdint>

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1) {
        MPI_Finalize();
        return 0;
    }
#else
    (void)argc;
    (void)argv;
#endif
    using Index = std::int32_t;
    constexpr Index nfe = 4;
    constexpr Index ne = 2;
    constexpr Index nf = 7;
    const Index e2f[nfe * ne] = {0,1,2,3, 4,5,6,1};
    const Index f2e[4 * nf] = {
        0,0,-1,-1,
        0,1,1,3,
        0,2,-1,-1,
        0,3,-1,-1,
        1,0,-1,-1,
        1,1,-1,-1,
        1,2,-1,-1};
    const Index elempartpts[2] = {2,0};
    std::vector<Index> neighborRank;
    std::vector<Index> neighborElem;
    exasim::meshtransfer::buildNeighborTables<Index>(
        neighborRank, neighborElem,
        e2f, f2e, nullptr, nullptr, nullptr,
        nullptr, nullptr, elempartpts,
        nfe, nf, ne, 0, 0, 0, 0,
#ifdef HAVE_MPI
        MPI_COMM_WORLD
#else
        0
#endif
    );
    for (Index q = 0; q < nfe * ne; ++q) {
        if (q == 1) {
            assert(neighborRank[q] == 0);
            assert(neighborElem[q] == 1);
        }
        else if (q == 7) {
            assert(neighborRank[q] == 0);
            assert(neighborElem[q] == 0);
        }
        else {
            assert(neighborRank[q] == -1);
            assert(neighborElem[q] == -1);
        }
    }
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
}
