#ifndef __NONMATCHINGPREPROCESSMPI_HPP
#define __NONMATCHINGPREPROCESSMPI_HPP

// Distributed broad-phase routing for prepared interface faces. The curved,
// high-order closest-point projection and deterministic global-face tie rule
// remain in the local search layer, while Exasim-specific D6-D7 payload setup
// remains in nonmatchingpreprocess.hpp. External spatial-index libraries can
// replace this broad phase only if those numerical and ownership contracts are
// preserved. This setup path uses host MPI buffers; runtime kernels use Kokkos.

#include "nonmatchingpreprocesssearch.hpp"

#include <climits>
#include <sstream>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace exasim {
namespace nonmatching {
namespace preprocess {

template <class T, class I>
struct DistributedDonor {
    I rank = -1, elem = -1, face = -1, localFace = -1;
    T xi[2] = {0, 0};
    T distance = std::numeric_limits<T>::max(), hface = 0, cmax = 0;
    bool converged = false;
};

struct DistributedStatistics {
    long long ownerPoints=0,d4Queries=0,maxD4FanOut=0;
    long long facesEvaluated=0,facesPruned=0,maxFacesPerPoint=0;
    long long projectionStarts=0,projectionIterations=0;
    long long queryBytes=0,responseBytes=0,countBytes=0,boxBytes=0;
};

#ifdef HAVE_MPI

template <class I> inline MPI_Datatype preprocess_mpi_type();
template <> inline MPI_Datatype preprocess_mpi_type<int>() { return MPI_INT; }
template <> inline MPI_Datatype preprocess_mpi_type<long long>() { return MPI_LONG_LONG; }
template <> inline MPI_Datatype preprocess_mpi_type<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype preprocess_mpi_type<float>() { return MPI_FLOAT; }

template <class I>
struct InterfaceCommunicator {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = -1, size = 0;
    std::vector<I> world, model;
    std::vector<int> worldToComm;
    long long* collectiveCounter = nullptr;
    InterfaceCommunicator() = default;
    InterfaceCommunicator(const InterfaceCommunicator&) = delete;
    InterfaceCommunicator& operator=(const InterfaceCommunicator&) = delete;
    InterfaceCommunicator(InterfaceCommunicator&& other) noexcept
        : comm(other.comm), rank(other.rank), size(other.size),
          world(std::move(other.world)), model(std::move(other.model)),
          worldToComm(std::move(other.worldToComm)), collectiveCounter(other.collectiveCounter)
    { other.comm = MPI_COMM_NULL; }
    void collective() const { if (collectiveCounter) ++(*collectiveCounter); }
    void release() { if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm); }
    ~InterfaceCommunicator() { release(); }
};

struct InterfaceCommunicatorView {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = -1, size = 0;
};

template <class I>
inline InterfaceCommunicator<I> make_interface_communicator(bool member, I worldRank, I model)
{
    InterfaceCommunicator<I> result;
    MPI_Comm_split(EXASIM_COMM_WORLD, member ? 1 : MPI_UNDEFINED,
        static_cast<int>(worldRank), &result.comm);
    if (result.comm == MPI_COMM_NULL) return result;
    MPI_Comm_rank(result.comm, &result.rank);
    MPI_Comm_size(result.comm, &result.size);
    const I mine[2] = {worldRank, model};
    std::vector<I> records(static_cast<std::size_t>(2) * result.size);
    MPI_Allgather(mine, 2, preprocess_mpi_type<I>(), records.data(), 2,
        preprocess_mpi_type<I>(), result.comm);
    result.world.resize(result.size); result.model.resize(result.size);
    int worldSize = 0;
    MPI_Comm_size(EXASIM_COMM_WORLD, &worldSize);
    result.worldToComm.assign(static_cast<std::size_t>(worldSize), -1);
    bool present[2] = {false, false};
    for (int q = 0; q < result.size; ++q) {
        result.world[q] = records[2*q]; result.model[q] = records[2*q+1];
        if (result.world[q] < 0 || result.world[q] >= worldSize ||
            result.worldToComm[result.world[q]] >= 0)
            throw std::runtime_error("non-matching preprocessing: invalid interface rank map");
        result.worldToComm[result.world[q]] = q;
        if (result.model[q] == 0 || result.model[q] == 1) present[result.model[q]] = true;
    }
    if (!present[0] || !present[1])
        throw std::runtime_error("non-matching preprocessing: interface communicator lacks one model");
    return result;
}

template <class T>
struct ProcessorBox {
    int world = -1, model = -1, nfaces = 0;
    T minimum[3] = {0,0,0}, maximum[3] = {0,0,0};
    T cmax = 0, beta = 0;
};

template <class T, class I, class Communicator>
inline std::vector<ProcessorBox<T>> exchange_processor_boxes(const Communicator &communicator,
                                                             const PreparedFaces<T, I> &local,
                                                             I worldRank, I model) {
    const int nd = local.master.nd;
    std::vector<I> sendInteger(3), allInteger(static_cast<std::size_t>(3) * communicator.size);
    sendInteger[0] = worldRank;
    sendInteger[1] = model;
    sendInteger[2] = local.count;
    MPI_Allgather(sendInteger.data(), 3, preprocess_mpi_type<I>(), allInteger.data(), 3,
                  preprocess_mpi_type<I>(), communicator.comm);
    std::vector<T> sendReal(static_cast<std::size_t>(2 * nd + 2), 0),
        allReal(static_cast<std::size_t>(2 * nd + 2) * communicator.size);
    if (local.count > 0) {
        for (int d = 0; d < nd; ++d) {
            sendReal[d] = local.boxMin[d];
            sendReal[nd + d] = local.boxMax[d];
        }
        for (I f = 0; f < local.count; ++f)
            for (int d = 0; d < nd; ++d) {
                sendReal[d] = std::min(sendReal[d], local.boxMin[d + nd * f]);
                sendReal[nd + d] = std::max(sendReal[nd + d], local.boxMax[d + nd * f]);
            }
        for (I f = 0; f < local.count; ++f) {
            sendReal[2 * nd] = std::max(sendReal[2 * nd], local.cmax[f]);
            sendReal[2 * nd + 1] = std::max(sendReal[2 * nd + 1], local.beta[f]);
        }
    }
    MPI_Allgather(sendReal.data(), 2 * nd + 2, preprocess_mpi_type<T>(), allReal.data(), 2 * nd + 2,
                  preprocess_mpi_type<T>(), communicator.comm);
    std::vector<ProcessorBox<T>> boxes(communicator.size);
    for (int q = 0; q < communicator.size; ++q) {
        auto &box = boxes[q];
        box.world = allInteger[3 * q];
        box.model = allInteger[3 * q + 1];
        box.nfaces = allInteger[3 * q + 2];
        for (int d = 0; d < nd; ++d) {
            box.minimum[d] = allReal[(2 * nd + 2) * q + d];
            box.maximum[d] = allReal[(2 * nd + 2) * q + nd + d];
        }
        box.cmax = allReal[(2 * nd + 2) * q + 2 * nd];
        box.beta = allReal[(2 * nd + 2) * q + 2 * nd + 1];
    }
    return boxes;
}

template <class T>
inline T processor_box_distance(const ProcessorBox<T> &box, const T *point, int nd) {
    T square = 0;
    for (int d = 0; d < nd; ++d) {
        T gap = 0;
        if (point[d] < box.minimum[d])
            gap = box.minimum[d] - point[d];
        else if (point[d] > box.maximum[d])
            gap = point[d] - box.maximum[d];
        square += gap * gap;
    }
    return down(std::sqrt(square));
}

template <class T>
inline T processor_margin(const ProcessorBox<T> &box, const T *point, int nd, int porder) {
    T magnitude = 0;
    for (int d = 0; d < nd; ++d)
        magnitude = std::max(magnitude, std::abs(point[d]));
    return screen_margin(box.cmax, box.beta, magnitude, nd, porder);
}

template <class T, class I> struct RecordWorkspace {
    std::vector<int> sendCount, recvCount, sendDisplacement, recvDisplacement;
    std::vector<I> sendInteger, recvInteger;
    std::vector<T> sendReal, recvReal;
};

template <class T, class I, class Communicator>
inline void exchange_records(const Communicator &communicator, int ni, int nr,
                             const std::vector<std::vector<I>> &integers,
                             const std::vector<std::vector<T>> &reals,
                             RecordWorkspace<T, I> &workspace) {
    const int P = communicator.size;
    int localLayoutError = integers.size() != static_cast<std::size_t>(P) ||
                           reals.size() != static_cast<std::size_t>(P) || ni <= 0 || nr <= 0;
    if (!localLayoutError)
        for (int q = 0; q < P; ++q) {
            const std::size_t integerSize = integers[q].size();
            const std::size_t records = integerSize / static_cast<std::size_t>(ni);
            if (integerSize % static_cast<std::size_t>(ni) != 0 ||
                records > static_cast<std::size_t>(INT_MAX) ||
                records > static_cast<std::size_t>(INT_MAX) / static_cast<std::size_t>(ni) ||
                records > static_cast<std::size_t>(INT_MAX) / static_cast<std::size_t>(nr) ||
                reals[q].size() != records * static_cast<std::size_t>(nr))
                localLayoutError = 1;
        }
    int anyLayoutError = 0;
    MPI_Allreduce(&localLayoutError, &anyLayoutError, 1, MPI_INT, MPI_MAX, communicator.comm);
    if (anyLayoutError)
        throw std::runtime_error("non-matching preprocessing: inconsistent compact-record layout");
    workspace.sendCount.resize(P);
    workspace.recvCount.resize(P);
    for (int q = 0; q < P; ++q)
        workspace.sendCount[q] =
            static_cast<int>(integers[q].size() / static_cast<std::size_t>(ni));
    MPI_Alltoall(workspace.sendCount.data(), 1, MPI_INT, workspace.recvCount.data(), 1, MPI_INT,
                 communicator.comm);
    workspace.sendDisplacement.resize(P);
    workspace.recvDisplacement.resize(P);
    std::size_t sendRecords = 0, recvRecords = 0;
    int localCountOverflow = 0;
    for (int q = 0; q < P; ++q) {
        if (sendRecords > static_cast<std::size_t>(INT_MAX) ||
            recvRecords > static_cast<std::size_t>(INT_MAX))
            localCountOverflow = 1;
        workspace.sendDisplacement[q] = localCountOverflow ? 0 : static_cast<int>(sendRecords);
        workspace.recvDisplacement[q] = localCountOverflow ? 0 : static_cast<int>(recvRecords);
        sendRecords += static_cast<std::size_t>(workspace.sendCount[q]);
        recvRecords += static_cast<std::size_t>(workspace.recvCount[q]);
    }
    if (sendRecords > static_cast<std::size_t>(INT_MAX) ||
        recvRecords > static_cast<std::size_t>(INT_MAX) ||
        sendRecords > static_cast<std::size_t>(INT_MAX) / static_cast<std::size_t>(ni) ||
        recvRecords > static_cast<std::size_t>(INT_MAX) / static_cast<std::size_t>(ni) ||
        sendRecords > static_cast<std::size_t>(INT_MAX) / static_cast<std::size_t>(nr) ||
        recvRecords > static_cast<std::size_t>(INT_MAX) / static_cast<std::size_t>(nr))
        localCountOverflow = 1;
    int anyCountOverflow = 0;
    MPI_Allreduce(&localCountOverflow, &anyCountOverflow, 1, MPI_INT, MPI_MAX, communicator.comm);
    if (anyCountOverflow)
        throw std::runtime_error(
            "non-matching preprocessing: compact-record count exceeds the MPI count range");
    workspace.sendInteger.resize(sendRecords * ni);
    workspace.sendReal.resize(sendRecords * nr);
    workspace.recvInteger.resize(recvRecords * ni);
    workspace.recvReal.resize(recvRecords * nr);
    for (int q = 0; q < P; ++q) {
        std::copy(integers[q].begin(), integers[q].end(),
                  workspace.sendInteger.begin() +
                      static_cast<std::size_t>(workspace.sendDisplacement[q]) * ni);
        std::copy(reals[q].begin(), reals[q].end(),
                  workspace.sendReal.begin() +
                      static_cast<std::size_t>(workspace.sendDisplacement[q]) * nr);
    }
    std::vector<int> sic(P), ric(P), sid(P), rid(P), src(P), rrc(P), srd(P), rrd(P);
    for (int q = 0; q < P; ++q) {
        sic[q] = workspace.sendCount[q] * ni;
        ric[q] = workspace.recvCount[q] * ni;
        sid[q] = workspace.sendDisplacement[q] * ni;
        rid[q] = workspace.recvDisplacement[q] * ni;
        src[q] = workspace.sendCount[q] * nr;
        rrc[q] = workspace.recvCount[q] * nr;
        srd[q] = workspace.sendDisplacement[q] * nr;
        rrd[q] = workspace.recvDisplacement[q] * nr;
    }
    MPI_Alltoallv(workspace.sendInteger.data(), sic.data(), sid.data(), preprocess_mpi_type<I>(),
                  workspace.recvInteger.data(), ric.data(), rid.data(), preprocess_mpi_type<I>(),
                  communicator.comm);
    MPI_Alltoallv(workspace.sendReal.data(), src.data(), srd.data(), preprocess_mpi_type<T>(),
                  workspace.recvReal.data(), rrc.data(), rrd.data(), preprocess_mpi_type<T>(),
                  communicator.comm);
}

template <class T, class I, class Communicator>
inline std::vector<DistributedDonor<T, I>>
distributed_search(const Communicator &communicator, const PreparedFaces<T, I> &local,
                   const std::vector<T> &ownerPoints, I worldRank, I model, T tolerance,
                   I maxIterations, DistributedStatistics *statistics = nullptr) {
    const int nd = local.master.nd, ndf = local.master.ndf, P = communicator.size;
    if (ownerPoints.size() % nd != 0)
        throw std::runtime_error("non-matching preprocessing: invalid owner-point array");
    const I npoints = static_cast<I>(ownerPoints.size() / nd);
    const auto boxes = exchange_processor_boxes(communicator, local, worldRank, model);
    if (statistics) {
        statistics->ownerPoints += npoints;
        statistics->boxBytes +=
            static_cast<long long>(boxes.size()) *
            (3 * sizeof(int) + static_cast<std::size_t>(2 * nd + 2) * sizeof(T));
    }
    std::vector<int> opposite;
    for (int q = 0; q < P; ++q)
        if (boxes[q].model != model && boxes[q].nfaces > 0)
            opposite.push_back(q);
    if (opposite.empty())
        throw std::runtime_error("non-matching preprocessing: opposite model has no donor faces");
    std::vector<DistributedDonor<T, I>> result(npoints);
    std::vector<T> firstUpper(npoints, std::numeric_limits<T>::max());
    std::vector<int> first(npoints, -1);
    RecordWorkspace<T, I> query, response;
    I failedPoint = std::numeric_limits<I>::max();
    I failedDonor = std::numeric_limits<I>::max();
    I failedFace = std::numeric_limits<I>::max();
    int protocolError = 0;
    auto perform = [&](bool additional) {
        std::vector<std::vector<I>> qi(P), ri(P);
        std::vector<std::vector<T>> qr(P), rr(P);
        for (I point = 0; point < npoints; ++point) {
            const T *x = &ownerPoints[static_cast<std::size_t>(point) * nd];
            if (!additional) {
                T best = 0;
                for (int q : opposite) {
                    const T bound = processor_box_distance(boxes[q], x, nd);
                    if (first[point] < 0 || bound < best ||
                        (bound == best && boxes[q].world < boxes[first[point]].world)) {
                        first[point] = q;
                        best = bound;
                    }
                }
            }
            long long pointQueries = 0;
            for (int q : opposite) {
                if ((!additional && q != first[point]) || (additional && q == first[point]))
                    continue;
                if (additional && result[point].converged) {
                    const T window =
                        tie_window(firstUpper[point], boxes[q].cmax, result[point].cmax);
                    if (processor_box_distance(boxes[q], x, nd) -
                            processor_margin(boxes[q], x, nd, local.master.porder) >
                        up(firstUpper[point] + window))
                        continue;
                }
                qi[q].push_back(point);
                for (int d = 0; d < nd; ++d)
                    qr[q].push_back(x[d]);
                ++pointQueries;
            }
            if (additional && statistics) {
                statistics->d4Queries += pointQueries;
                statistics->maxD4FanOut = std::max(statistics->maxD4FanOut, pointQueries);
            }
        }
        if (statistics) {
            long long records = 0;
            for (int q = 0; q < P; ++q)
                records += qi[q].size();
            statistics->queryBytes +=
                records * (sizeof(I) + static_cast<long long>(nd) * sizeof(T));
            statistics->countBytes += 2LL * P * sizeof(int);
        }
        exchange_records(communicator, 1, nd, qi, qr, query);
        ProjectionWorkspace<T, I> projection;
        projection.faceOrder.reserve(static_cast<std::size_t>(local.count));
        projection.faceLower.reserve(static_cast<std::size_t>(local.count));
        for (int source = 0; source < P; ++source)
            for (int record = 0; record < query.recvCount[source]; ++record) {
                const std::size_t k =
                    static_cast<std::size_t>(query.recvDisplacement[source] + record);
                const I point = query.recvInteger[k];
                const T *x = &query.recvReal[k * nd];
                const auto localResult =
                    search_local_faces(local, x, tolerance, maxIterations, projection);
                if (statistics) {
                    statistics->facesEvaluated += localResult.projectedFaces;
                    statistics->facesPruned += localResult.prunedFaces;
                    statistics->projectionStarts += localResult.projectionStarts;
                    statistics->projectionIterations += localResult.projectionIterations;
                    statistics->maxFacesPerPoint = std::max<long long>(statistics->maxFacesPerPoint,
                                                                       localResult.projectedFaces);
                }
                const I failedGlobal =
                    localResult.failedFace >= 0 ? local.globalFace[localResult.failedFace] : I(-1);
                ri[source].insert(
                    ri[source].end(),
                    {point, localResult.failed ? I(1) : I(0),
                     localResult.face >= 0 ? local.globalFace[localResult.face] : I(-1),
                     localResult.face >= 0 ? local.elem[localResult.face] : I(-1),
                     localResult.face >= 0 ? local.localFace[localResult.face] : I(-1), worldRank,
                     failedGlobal});
                rr[source].push_back(localResult.projection.distance);
                rr[source].push_back(localResult.projection.xi[0]);
                if (ndf == 2)
                    rr[source].push_back(localResult.projection.xi[1]);
                rr[source].push_back(localResult.face >= 0 ? local.hface[localResult.face] : T(0));
                rr[source].push_back(localResult.face >= 0 ? local.cmax[localResult.face] : T(0));
            }
        if (statistics) {
            long long records = 0;
            for (int q = 0; q < P; ++q)
                records += ri[q].size() / 7;
            statistics->responseBytes +=
                records * (7 * sizeof(I) + static_cast<long long>(ndf + 3) * sizeof(T));
            statistics->countBytes += 2LL * P * sizeof(int);
        }
        exchange_records(communicator, 7, ndf + 3, ri, rr, response);
        for (int source = 0; source < P; ++source)
            for (int record = 0; record < response.recvCount[source]; ++record) {
                const std::size_t k =
                    static_cast<std::size_t>(response.recvDisplacement[source] + record);
                const I *meta = &response.recvInteger[k * 7];
                const T *data = &response.recvReal[k * (ndf + 3)];
                const I point = meta[0];
                if (point < 0 || point >= npoints) {
                    protocolError = 1;
                    continue;
                }
                if (meta[1]) {
                    const I donor = meta[5], face = meta[6];
                    if (failedPoint == std::numeric_limits<I>::max() ||
                        std::tie(point, donor, face) <
                            std::tie(failedPoint, failedDonor, failedFace)) {
                        failedPoint = point;
                        failedDonor = donor;
                        failedFace = face;
                    }
                    continue;
                }
                if (!additional)
                    firstUpper[point] = data[0];
                auto &current = result[point];
                const bool tie = current.converged &&
                                 tied(data[0], current.distance, data[ndf + 2], current.cmax);
                const bool take =
                    !current.converged || (!tie && data[0] < current.distance) ||
                    (tie && std::tie(meta[2], meta[5]) < std::tie(current.face, current.rank));
                if (take) {
                    current.converged = true;
                    current.distance = data[0];
                    current.face = meta[2];
                    current.elem = meta[3];
                    current.localFace = meta[4];
                    current.rank = meta[5];
                    current.xi[0] = data[1];
                    if (ndf == 2)
                        current.xi[1] = data[2];
                    current.hface = data[ndf + 1];
                    current.cmax = data[ndf + 2];
                }
            }
    };
    perform(false);
    perform(true);
    int anyProtocol = 0;
    MPI_Allreduce(&protocolError, &anyProtocol, 1, MPI_INT, MPI_MAX, communicator.comm);
    if (anyProtocol)
        throw std::runtime_error("non-matching preprocessing: invalid donor-search response");
    const I none = std::numeric_limits<I>::max();
    const I localOwner = failedPoint == none ? none : worldRank;
    I failedOwner = none;
    MPI_Allreduce(&localOwner, &failedOwner, 1, preprocess_mpi_type<I>(), MPI_MIN,
                  communicator.comm);
    if (failedOwner != none) {
        I localTuple[3] = {none, none, none}, tuple[3] = {none, none, none};
        if (worldRank == failedOwner) {
            localTuple[0] = failedPoint;
            localTuple[1] = failedDonor;
            localTuple[2] = failedFace;
        }
        MPI_Allreduce(localTuple, tuple, 3, preprocess_mpi_type<I>(), MPI_MIN, communicator.comm);
        std::ostringstream message;
        message << "non-matching preprocessing: donor face projection failed for owner rank "
                << failedOwner << ", point " << tuple[0] << ", donor rank " << tuple[1]
                << ", global face " << tuple[2];
        throw std::runtime_error(message.str());
    }
    return result;
}

#endif

} // namespace preprocess
} // namespace nonmatching
} // namespace exasim

#endif
