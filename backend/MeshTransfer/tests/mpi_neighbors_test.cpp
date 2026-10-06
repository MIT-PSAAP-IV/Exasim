#include <mesh_neighbors.hpp>

#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <vector>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    assert(size == 2);

    using Index = std::int32_t;
    constexpr Index nfe = 4;
    constexpr Index nf = 4;
    constexpr Index neTotal = 2;
    const Index otherRank = 1 - rank;
    const Index sharedFace = rank == 0 ? 1 : 3;
    const Index remoteFace = rank == 0 ? 3 : 1;
    const Index e2f[nfe] = {0,1,2,3};
    Index f2e[4 * nf];
    for (Index face = 0; face < nf; ++face) {
        f2e[4 * face] = 0;
        f2e[4 * face + 1] = face;
        f2e[4 * face + 2] = -1;
        f2e[4 * face + 3] = -1;
    }
    f2e[4 * sharedFace] = 0;
    f2e[4 * sharedFace + 1] = sharedFace;
    f2e[4 * sharedFace + 2] = 1;
    f2e[4 * sharedFace + 3] = remoteFace;
    const Index nbsd[1] = {otherRank};
    const Index elemsend[1] = {0};
    const Index elemrecv[1] = {1};
    const Index elemsendpts[1] = {1};
    const Index elemrecvpts[1] = {1};
    const Index elempartpts[2] = {1,0};
    std::vector<Index> neighborRank;
    std::vector<Index> neighborElem;
    exasim::meshtransfer::buildNeighborTables(
        neighborRank, neighborElem, e2f, f2e, nbsd,
        elemsend, elemrecv, elemsendpts, elemrecvpts, elempartpts,
        nfe, nf, neTotal, Index(1), Index(1), Index(1),
        Index(rank), MPI_COMM_WORLD);
    for (Index face = 0; face < nfe; ++face) {
        if (face == sharedFace) {
            assert(neighborRank[face] == otherRank);
            assert(neighborElem[face] == 0);
        }
        else {
            assert(neighborRank[face] == -1);
            assert(neighborElem[face] == -1);
        }
    }
    MPI_Finalize();
}
