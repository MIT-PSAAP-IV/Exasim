#ifndef EXASIM_MESH_TRANSFER_MESH_NEIGHBORS_HPP
#define EXASIM_MESH_TRANSFER_MESH_NEIGHBORS_HPP

#include <limits>
#include <stdexcept>
#include <vector>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace exasim {
namespace meshtransfer {

#ifdef HAVE_MPI
using MeshNeighborComm = MPI_Comm;
#else
using MeshNeighborComm = int;
#endif

/**
 * Build direct neighbor tables for every face of every element owned by the
 * current MPI rank.
 *
 * The outputs use the flattened index face + nfe*elem and contain:
 *   physical boundary: neighborRank = -1,     neighborElem = -1
 *   local face:        neighborRank = myrank, neighborElem = local element
 *   MPI face:          neighborRank = r',     neighborElem = element on r'
 *
 * neTotal includes owned and ghost elements. The number of owned elements is
 * elempartpts[0] + elempartpts[1]. The function performs one integer exchange
 * so that the local ghost indices in elemrecv are paired with the owning
 * rank's local indices in elemsend.
 */
template <class I>
inline void buildNeighborTables(
    std::vector<I>& neighborRank,
    std::vector<I>& neighborElem,
    const I* e2f,
    const I* f2e,
    const I* nbsd,
    const I* elemsend,
    const I* elemrecv,
    const I* elemsendpts,
    const I* elemrecvpts,
    const I* elempartpts,
    I nfe,
    I nf,
    I neTotal,
    I nnbsd,
    I nelemsend,
    I nelemrecv,
    I myrank,
    MeshNeighborComm comm,
    int mpiTag = 9301)
{
    const I invalid = I(-1);

    if (nfe <= 0 || nf < 0 || neTotal < 0 || nnbsd < 0 ||
        nelemsend < 0 || nelemrecv < 0)
        throw std::invalid_argument("buildNeighborTables: invalid dimensions");

    if (e2f == nullptr || f2e == nullptr || elempartpts == nullptr)
        throw std::invalid_argument("buildNeighborTables: null mesh array");

    if (nnbsd > 0 &&
        (nbsd == nullptr || elemsendpts == nullptr || elemrecvpts == nullptr))
        throw std::invalid_argument("buildNeighborTables: null neighbor schedule");

    if (nelemsend > 0 && elemsend == nullptr)
        throw std::invalid_argument("buildNeighborTables: null elemsend array");

    if (nelemrecv > 0 && elemrecv == nullptr)
        throw std::invalid_argument("buildNeighborTables: null elemrecv array");

    const I neOwned = elempartpts[0] + elempartpts[1];
    if (neOwned < 0 || neOwned > neTotal)
        throw std::invalid_argument("buildNeighborTables: invalid owned-element count");

    I sendTotal = 0;
    I recvTotal = 0;
    for (I j = 0; j < nnbsd; ++j) {
        if (elemsendpts[j] < 0 || elemrecvpts[j] < 0)
            throw std::invalid_argument("buildNeighborTables: negative neighbor count");
        sendTotal += elemsendpts[j];
        recvTotal += elemrecvpts[j];
    }

    if (sendTotal != nelemsend || recvTotal != nelemrecv)
        throw std::invalid_argument("buildNeighborTables: inconsistent neighbor counts");

    std::vector<I> ghostOwner(static_cast<std::size_t>(neTotal), invalid);
    I recvOffset = 0;
    for (I j = 0; j < nnbsd; ++j) {
        for (I k = 0; k < elemrecvpts[j]; ++k) {
            const I ghost = elemrecv[recvOffset + k];
            if (ghost < neOwned || ghost >= neTotal)
                throw std::runtime_error("buildNeighborTables: invalid ghost index");
            ghostOwner[static_cast<std::size_t>(ghost)] = nbsd[j];
        }
        recvOffset += elemrecvpts[j];
    }

    std::vector<I> receivedRemoteElem(
        static_cast<std::size_t>(nelemrecv), invalid);

#ifdef HAVE_MPI
    if (comm != MPI_COMM_NULL) {
        if (sendTotal > I(std::numeric_limits<int>::max()) ||
            recvTotal > I(std::numeric_limits<int>::max()))
            throw std::overflow_error("buildNeighborTables: MPI count exceeds int range");

        MPI_Datatype indexType = MPI_DATATYPE_NULL;
        if (MPI_Type_match_size(MPI_TYPECLASS_INTEGER, sizeof(I), &indexType) !=
            MPI_SUCCESS)
            throw std::runtime_error(
                "buildNeighborTables: no MPI datatype matches the index type");

        std::vector<MPI_Request> requests;
        requests.reserve(static_cast<std::size_t>(2 * nnbsd));

        recvOffset = 0;
        for (I j = 0; j < nnbsd; ++j) {
            const I count = elemrecvpts[j];
            if (count > 0) {
                MPI_Request request;
                MPI_Irecv(receivedRemoteElem.data() + recvOffset,
                          static_cast<int>(count), indexType,
                          static_cast<int>(nbsd[j]), mpiTag, comm, &request);
                requests.push_back(request);
            }
            recvOffset += count;
        }

        I sendOffset = 0;
        for (I j = 0; j < nnbsd; ++j) {
            const I count = elemsendpts[j];
            if (count > 0) {
                MPI_Request request;
                MPI_Isend(elemsend + sendOffset, static_cast<int>(count),
                          indexType, static_cast<int>(nbsd[j]), mpiTag,
                          comm, &request);
                requests.push_back(request);
            }
            sendOffset += count;
        }

        if (!requests.empty())
            MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                        MPI_STATUSES_IGNORE);
    } else if (nnbsd > 0) {
        throw std::invalid_argument(
            "buildNeighborTables: null MPI communicator with neighbor ranks");
    }
#else
    (void)comm;
    (void)mpiTag;
    if (nnbsd > 0 || nelemsend > 0 || nelemrecv > 0)
        throw std::runtime_error(
            "buildNeighborTables: MPI neighbor data requires HAVE_MPI");
#endif

    std::vector<I> remoteLocalIndex(
        static_cast<std::size_t>(neTotal), invalid);
    for (I q = 0; q < nelemrecv; ++q) {
        const I ghost = elemrecv[q];
        const I remoteElem = receivedRemoteElem[static_cast<std::size_t>(q)];
        if (remoteElem < 0)
            throw std::runtime_error("buildNeighborTables: invalid remote element index");
        remoteLocalIndex[static_cast<std::size_t>(ghost)] = remoteElem;
    }

    const std::size_t tableSize = static_cast<std::size_t>(nfe) *
                                  static_cast<std::size_t>(neOwned);
    neighborRank.assign(tableSize, invalid);
    neighborElem.assign(tableSize, invalid);

    for (I elem = 0; elem < neOwned; ++elem) {
        for (I localFace = 0; localFace < nfe; ++localFace) {
            const I index = localFace + nfe * elem;
            const I face = e2f[index];
            if (face < 0 || face >= nf)
                throw std::runtime_error("buildNeighborTables: invalid e2f entry");

            const I elem1 = f2e[4 * face + 0];
            const I face1 = f2e[4 * face + 1];
            const I elem2 = f2e[4 * face + 2];
            const I face2 = f2e[4 * face + 3];

            I other = invalid;
            if (elem1 == elem && face1 == localFace)
                other = elem2;
            else if (elem2 == elem && face2 == localFace)
                other = elem1;
            else
                throw std::runtime_error(
                    "buildNeighborTables: inconsistent e2f/f2e connectivity");

            if (other < 0)
                continue;
            if (other >= neTotal)
                throw std::runtime_error(
                    "buildNeighborTables: neighbor index exceeds local partition");

            const std::size_t outputIndex = static_cast<std::size_t>(index);
            if (other < neOwned) {
                neighborRank[outputIndex] = myrank;
                neighborElem[outputIndex] = other;
            } else {
                const I remoteRank = ghostOwner[static_cast<std::size_t>(other)];
                const I remoteElem = remoteLocalIndex[static_cast<std::size_t>(other)];
                if (remoteRank < 0 || remoteElem < 0)
                    throw std::runtime_error(
                        "buildNeighborTables: incomplete ghost metadata");
                neighborRank[outputIndex] = remoteRank;
                neighborElem[outputIndex] = remoteElem;
            }
        }
    }
}

} // namespace meshtransfer
} // namespace exasim

#endif
