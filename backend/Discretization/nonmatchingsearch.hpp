/*
    nonmatchingsearch.hpp

    Phase 7 of the non-matching interface solver: scalable donor search on a
    setup-only interface communicator (docs/nonmatching/nonmatching_multiphysics.tex,
    Section 3.1 "Interface communicator", Section 3.2 Steps D1-D5).

    Include only from nonmatchingsetup.hpp, after donor_search: this file uses
    InterfaceFaceSet, DonorSearchResult, PreparedFaces, evaluate_candidate_face,
    select_certified_face and the interval helpers defined there.

    comm_Gamma. Every active world rank calls MPI_Comm_split once (color: owns a
    coupled interface face; key: world rank). Members form comm_Gamma; it is
    reused for the master check, Steps D2-D7 and the diagnostics, and freed when
    the setup returns. Nonmembers get MPI_COMM_NULL and enter no interface
    collective. Deviation from Section 3.1: P^1/P^2 are not in the partition data,
    so the group is formed by one split on EXASIM_COMM_WORLD instead of
    MPI_Comm_create_group (P^1/P^2 in the partition data: Phase 9).

    Search (Steps D1-D5), the bound-based form of eq. (3):
    - D1-D2: every member boxes its own interface faces with the outward-rounded
      Bernstein coefficient intervals (convex-hull property: each face lies in
      the hull of its exact coefficients, which the intervals enclose), and the
      boxes are all-gathered with cmax_q and beta_q (maxima over the faces).
    - D3: each owner point goes to the opposite-model processor with the
      smallest box lower bound (ties: smallest world rank). It returns
      U_q = min upper_m over its faces and every face with lower_m <= U_q.
    - D4: the point goes to every other opposite-model processor with
      lo(boxLB_q - G_q(x)) <= U_first, where G_q(x) =
      certified_gap_target_upper(cmax_q, beta_q, |x|_inf), the processor value of
      the shared acceptance predicate. It returns U_q and every face with
      lower_m <= U_first.
    - D5: U = min_q U_q over the queried processors; the donor is the smallest
      global face index among the returned faces with lower_m <= U.
    Design argument (reports/phase7_report.md, Section 4.4): under standard
    round-to-nearest arithmetic with the outward rounding of the interval
    helpers, every accepted face has upper - lower <= gapUpper <= gapTarget <= G_q,
    so a face of a pruned processor has lower >= dist - G_q >= boxLB_q - G_q >
    U_first >= U and is not in the tie set; every face of a queried processor
    with lower <= U is returned. The production build uses -ffast-math
    (maintainer decision 2026-10-09), which does not preserve those arithmetic
    assumptions: the boxes, the margin and the gap rule are conservative
    best-effort bounds there, not a formal guarantee. Input data are assumed
    finite and valid, with no intermediate arithmetic overflow (an overflowed
    interval end rounds to max(), which is not an enclosure); there is no
    NaN/infinity or overflow validation (Exasim's residual-norm check is the
    production detector).
    Failure policy (maintainer decision): a point fails if any face on a queried
    processor fails; pruned faces are never evaluated. If the D3 processor fails
    or returns no usable U_q, UfirstKnown stays false and every other
    opposite-model processor is queried in D4. Failures travel as response data;
    the caller reduces them on comm_Gamma after all rounds and reports them
    uniformly.
*/

#ifndef __NONMATCHINGSEARCH_HPP
#define __NONMATCHINGSEARCH_HPP

#ifndef __NONMATCHINGSETUP_HPP
#error "nonmatchingsearch.hpp is included only from nonmatchingsetup.hpp"
#endif

namespace exasim {
namespace nonmatching {

#ifdef HAVE_MPI

// Creates minus frees of comm_Gamma (tests check the balance).
inline long long& nonmatching_comm_balance()
{
    static long long balance = 0;
    return balance;
}

// The interface communicator and its rank maps; all ranks stored elsewhere are WORLD ranks.
template <class I>
struct InterfaceComm {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = -1, size = 0;
    std::vector<I> world;          // comm rank -> world rank
    std::vector<I> model;          // comm rank -> model (0, 1)
    std::vector<int> worldToComm;  // world rank -> comm rank, -1 for nonmembers
    SearchStatistics* stats = nullptr;
    void collective() const { if (stats) stats->interfaceCollectives++; }
    InterfaceComm() = default;
    InterfaceComm(const InterfaceComm&) = delete;
    InterfaceComm& operator=(const InterfaceComm&) = delete;
    void release()
    {
        if (comm != MPI_COMM_NULL) {
            MPI_Comm_free(&comm);
            comm = MPI_COMM_NULL;
            nonmatching_comm_balance()--;
        }
    }
    ~InterfaceComm() { release(); }
};

// One split per rebuild (collective on EXASIM_COMM_WORLD), then the rank maps on
// comm_Gamma. Both models must have an interface rank.
template <class I>
inline void create_interface_comm(InterfaceComm<I>& ic, bool member, I mpirank, I mpiprocs, I model,
        SearchStatistics* stats)
{
    ic.stats = stats;
    MPI_Comm_split(EXASIM_COMM_WORLD, member ? 1 : MPI_UNDEFINED, static_cast<int>(mpirank), &ic.comm);
    if (stats) stats->member = member ? 1 : 0;
    if (ic.comm == MPI_COMM_NULL) return;
    nonmatching_comm_balance()++;
    MPI_Comm_rank(ic.comm, &ic.rank);
    MPI_Comm_size(ic.comm, &ic.size);
    if (stats) stats->interfaceSize = ic.size;
    const I mine[2] = {mpirank, model};
    std::vector<I> all(static_cast<std::size_t>(2) * ic.size);
    MPI_Allgather(mine, 2, mpi_type<I>(), all.data(), 2, mpi_type<I>(), ic.comm);
    ic.collective();
    ic.world.resize(ic.size);
    ic.model.resize(ic.size);
    ic.worldToComm.assign(static_cast<std::size_t>(mpiprocs), -1);
    bool seen[2] = {false, false};
    for (int q = 0; q < ic.size; q++) {
        ic.world[q] = all[2 * q];
        ic.model[q] = all[2 * q + 1];
        if (ic.world[q] < 0 || ic.world[q] >= mpiprocs || ic.worldToComm[ic.world[q]] >= 0 ||
            (ic.model[q] != 0 && ic.model[q] != 1))
            error("non-matching search: invalid or duplicate world rank in the interface communicator");
        ic.worldToComm[ic.world[q]] = q;
        seen[ic.model[q]] = true;
    }
    if (!seen[0] || !seen[1])
        error("non-matching donor search: the other model has no interface faces");
}

// MPI int count from a wide value.
inline int checked_search_count(long long value, const char* what)
{
    if (value < 0 || value > std::numeric_limits<int>::max()) {
        std::ostringstream oss;
        oss << "non-matching search: " << what << " exceeds MPI integer limits";
        error(oss.str());
    }
    return static_cast<int>(value);
}

// Setup-local storage of exchange_records: counts, displacements and the flat
// send buffers, shared by the query and response exchanges of D3 and D4 (one
// instance per distributed_donor_search call; no global state).
template <class T, class I>
struct RecordExchangeWorkspace {
    std::vector<int> sendCount, sci, sdi, rci, rdi, scr, sdr, rcr, rdr;
    std::vector<I> sbi;
    std::vector<T> sbr;
};

// Irregular exchange of fixed-size records (nInt integers and nReal reals per
// record) on comm_Gamma: MPI_Alltoall of the record counts, then two
// MPI_Alltoallv. Every member calls it, also with zero records. Received records
// are grouped by source comm rank, in the sender's order. Buffers live in ws and
// keep their capacity between calls.
template <class T, class I>
inline void exchange_records(const InterfaceComm<I>& ic, int nInt, int nReal,
        const std::vector<std::vector<I>>& sendInt, const std::vector<std::vector<T>>& sendReal,
        std::vector<I>& recvInt, std::vector<T>& recvReal, std::vector<int>& recvCount,
        long long* payloadBytes, RecordExchangeWorkspace<T, I>& ws)
{
    const int P = ic.size;
    std::vector<int>& sendCount = ws.sendCount;
    sendCount.assign(P, 0);
    for (int q = 0; q < P; q++) {
        const std::size_t n = sendInt[q].size() / static_cast<std::size_t>(nInt);
        if (sendInt[q].size() != n * nInt || sendReal[q].size() != n * nReal)
            error("non-matching search: inconsistent record buffer");
        sendCount[q] = checked_search_count(static_cast<long long>(n), "record count");
    }
    recvCount.assign(P, 0);
    MPI_Alltoall(sendCount.data(), 1, MPI_INT, recvCount.data(), 1, MPI_INT, ic.comm);
    ic.collective();
    if (ic.stats) ic.stats->countBytes += 2LL * P * static_cast<long long>(sizeof(int));

    std::vector<int>& sci = ws.sci; std::vector<int>& sdi = ws.sdi;
    std::vector<int>& rci = ws.rci; std::vector<int>& rdi = ws.rdi;
    std::vector<int>& scr = ws.scr; std::vector<int>& sdr = ws.sdr;
    std::vector<int>& rcr = ws.rcr; std::vector<int>& rdr = ws.rdr;
    sci.resize(P); sdi.resize(P); rci.resize(P); rdi.resize(P);
    scr.resize(P); sdr.resize(P); rcr.resize(P); rdr.resize(P);
    long long si = 0, sr = 0, ri = 0, rr = 0;
    for (int q = 0; q < P; q++) {
        sdi[q] = checked_search_count(si, "send displacement");
        sdr[q] = checked_search_count(sr, "send displacement");
        rdi[q] = checked_search_count(ri, "receive displacement");
        rdr[q] = checked_search_count(rr, "receive displacement");
        sci[q] = checked_search_count(static_cast<long long>(sendCount[q]) * nInt, "send count");
        scr[q] = checked_search_count(static_cast<long long>(sendCount[q]) * nReal, "send count");
        if (recvCount[q] < 0) error("non-matching search: negative receive count");
        rci[q] = checked_search_count(static_cast<long long>(recvCount[q]) * nInt, "receive count");
        rcr[q] = checked_search_count(static_cast<long long>(recvCount[q]) * nReal, "receive count");
        si += sci[q]; sr += scr[q]; ri += rci[q]; rr += rcr[q];
    }
    checked_search_count(si, "send extent"); checked_search_count(sr, "send extent");
    checked_search_count(ri, "receive extent"); checked_search_count(rr, "receive extent");
    std::vector<I>& sbi = ws.sbi;
    std::vector<T>& sbr = ws.sbr;
    sbi.resize(static_cast<std::size_t>(si));
    sbr.resize(static_cast<std::size_t>(sr));
    for (int q = 0; q < P; q++) {
        std::copy(sendInt[q].begin(), sendInt[q].end(), sbi.begin() + sdi[q]);
        std::copy(sendReal[q].begin(), sendReal[q].end(), sbr.begin() + sdr[q]);
    }
    recvInt.resize(static_cast<std::size_t>(ri));
    recvReal.resize(static_cast<std::size_t>(rr));
    MPI_Alltoallv(sbi.data(), sci.data(), sdi.data(), mpi_type<I>(),
            recvInt.data(), rci.data(), rdi.data(), mpi_type<I>(), ic.comm);
    ic.collective();
    MPI_Alltoallv(sbr.data(), scr.data(), sdr.data(), mpi_type<T>(),
            recvReal.data(), rcr.data(), rdr.data(), mpi_type<T>(), ic.comm);
    ic.collective();
    if (payloadBytes)
        *payloadBytes += si * static_cast<long long>(sizeof(I)) + sr * static_cast<long long>(sizeof(T));
}

// Processor record of Step D2, sent as two records: integers [world rank,
// model, valid, number of faces, local nfe] and reals
// [xmin[2], xmax[2], cmax_q, beta_q]. The local nfe lets an owner validate a
// returned donor face without interpreting it through the owner's family.
constexpr int ProcessorBoxInts = 5;
constexpr int ProcessorBoxReals = 6;

template <class T>
struct ProcessorBox {
    int world = -1, model = -1, nfaces = 0, nfe = 0;
    bool valid = false;
    T xmin[2] = {0, 0}, xmax[2] = {0, 0};
    T cmax = 0, beta = 0;
};

// Step D1: the outward-rounded box of the local interface faces from their
// Bernstein coefficient intervals only (maintainer decision 2026-10-09; no
// element or volume nodes). No MPI.
template <class T, class I>
inline ProcessorBox<T> local_processor_box(const PreparedFaces<T, I>& local, I mpirank, I model, I nfe)
{
    ProcessorBox<T> b;
    const int n = static_cast<int>(local.bm.p) + 1;
    // Finite sentinels (max()/lowest()) instead of infinities: the production
    // library is built with -ffast-math, so no code path relies on infinity.
    T xmin[2] = {std::numeric_limits<T>::max(), std::numeric_limits<T>::max()};
    T xmax[2] = {std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest()};
    T cmax = 0, beta = 0;
    bool valid = local.nc > 0 && local.nd == 2 &&
                 static_cast<long long>(local.nc) <= std::numeric_limits<int>::max();
    for (I m = 0; m < local.nc && valid; m++) {
        const BernsteinFace<T>& bf = local.bfaces[m];
        if (!bf.ok || bf.beta < 0 || bf.cmax < 0) {
            valid = false;
            break;
        }
        cmax = std::max(cmax, bf.cmax);
        beta = std::max(beta, bf.beta);
        for (int d = 0; d < 2; d++)
            for (int k = 0; k < n; k++) {
                const Interval<T>& c = bf.ci[k + n * d];
                if (c.lo > c.hi) valid = false;
                xmin[d] = std::min(xmin[d], c.lo);
                xmax[d] = std::max(xmax[d], c.hi);
            }
    }
    for (int d = 0; d < 2; d++) {
        xmin[d] = ia_down(xmin[d]); // widen outward
        xmax[d] = ia_up(xmax[d]);
        if (xmin[d] > xmax[d]) valid = false;
        b.xmin[d] = xmin[d];
        b.xmax[d] = xmax[d];
    }
    b.world = static_cast<int>(mpirank);
    b.model = static_cast<int>(model);
    b.valid = valid;
    b.cmax = cmax;
    b.beta = beta;
    b.nfaces = valid ? static_cast<int>(local.nc) : 0;
    b.nfe = static_cast<int>(nfe);
    return b;
}

// Validity of a received processor record (ordered box, nonnegative cmax/beta,
// positive face count, metadata equal to the rank map). Structural checks only:
// the data are assumed finite (no NaN/infinity validation in production code).
template <class T>
inline bool processor_box_valid(const ProcessorBox<T>& b, int world, int model)
{
    if (!b.valid || b.world != world || b.model != model || b.nfaces <= 0 || b.nfe <= 0) return false;
    for (int d = 0; d < 2; d++)
        if (b.xmin[d] > b.xmax[d]) return false;
    return b.cmax >= 0 && b.beta >= 0;
}

// Step D2: all-gather of the processor records on comm_Gamma (integer and real
// parts separately; no integer travels as a floating-point value).
template <class T, class I>
inline std::vector<ProcessorBox<T>> exchange_processor_boxes(const InterfaceComm<I>& ic,
        const PreparedFaces<T, I>& local, I mpirank, I model, I nfe)
{
    const ProcessorBox<T> mine = local_processor_box(local, mpirank, model, nfe);
    const I recI[ProcessorBoxInts] = {static_cast<I>(mine.world), static_cast<I>(mine.model),
                                      static_cast<I>(mine.valid ? 1 : 0), static_cast<I>(mine.nfaces),
                                      static_cast<I>(mine.nfe)};
    const T recR[ProcessorBoxReals] = {mine.xmin[0], mine.xmin[1], mine.xmax[0], mine.xmax[1], mine.cmax, mine.beta};
    std::vector<I> allI(static_cast<std::size_t>(ProcessorBoxInts) * ic.size);
    std::vector<T> allR(static_cast<std::size_t>(ProcessorBoxReals) * ic.size);
    MPI_Allgather(recI, ProcessorBoxInts, mpi_type<I>(), allI.data(), ProcessorBoxInts, mpi_type<I>(), ic.comm);
    ic.collective();
    MPI_Allgather(recR, ProcessorBoxReals, mpi_type<T>(), allR.data(), ProcessorBoxReals, mpi_type<T>(), ic.comm);
    ic.collective();
    if (ic.stats)
        ic.stats->boxBytes += static_cast<long long>(allI.size() * sizeof(I) + allR.size() * sizeof(T));
    std::vector<ProcessorBox<T>> boxes(ic.size);
    for (int q = 0; q < ic.size; q++) {
        const I* ri = &allI[static_cast<std::size_t>(ProcessorBoxInts) * q];
        const T* rr = &allR[static_cast<std::size_t>(ProcessorBoxReals) * q];
        ProcessorBox<T>& b = boxes[q];
        if (ri[0] < 0 || ri[0] > std::numeric_limits<int>::max() || (ri[1] != 0 && ri[1] != 1) ||
            (ri[2] != 0 && ri[2] != 1) || ri[3] < 0 || ri[3] > std::numeric_limits<int>::max() ||
            ri[4] <= 0 || ri[4] > std::numeric_limits<int>::max())
            error("non-matching search: invalid processor box metadata");
        b.world = static_cast<int>(ri[0]); b.model = static_cast<int>(ri[1]);
        b.valid = ri[2] == 1; b.nfaces = static_cast<int>(ri[3]); b.nfe = static_cast<int>(ri[4]);
        b.xmin[0] = rr[0]; b.xmin[1] = rr[1]; b.xmax[0] = rr[2]; b.xmax[1] = rr[3];
        b.cmax = rr[4]; b.beta = rr[5];
        // Every member sees the same records, so this error is uniform.
        if (!processor_box_valid(b, static_cast<int>(ic.world[q]), static_cast<int>(ic.model[q])))
            error("non-matching search: invalid (empty, unordered or inconsistent) processor box of interface faces");
    }
    return boxes;
}

// Outward-rounded lower bound of dist(x, B) for an axis-aligned box B.
template <class T>
inline T box_distance_lower(const T* x, const ProcessorBox<T>& b)
{
    Interval<T> s(static_cast<T>(0));
    for (int d = 0; d < 2; d++) {
        T g = 0;
        if (x[d] < b.xmin[d]) g = ia_sub(Interval<T>(b.xmin[d]), Interval<T>(x[d])).lo;
        else if (x[d] > b.xmax[d]) g = ia_sub(Interval<T>(x[d]), Interval<T>(b.xmax[d])).lo;
        g = std::max(static_cast<T>(0), g);
        s = ia_add(s, ia_sqr(Interval<T>(g)));
    }
    return std::max(static_cast<T>(0), ia_sqrt(s).lo);
}

// Pruning margin G_q(x): the gap-target bound of the shared acceptance predicate
// (certified_gap_target_upper) for the processor maxima cmax_q, beta_q. Under the
// arithmetic assumptions of the design argument every accepted face m of q has
// upper - lower <= gapUpper_m <= gapTarget_m <= G_q(x); best-effort under fast math.
template <class T>
inline T pruning_margin_upper(const T* x, const ProcessorBox<T>& b, int porder)
{
    const T ax = std::max(std::abs(x[0]), std::abs(x[1]));
    return certified_gap_target_upper(b.cmax, b.beta, ax, porder);
}

// D4 routing test: lo(boxLB - G) <= U_first.
template <class T>
inline bool d4_must_query(const T* x, const ProcessorBox<T>& b, int porder, T Ufirst)
{
    const T lb = box_distance_lower(x, b);
    const T G = pruning_margin_upper(x, b, porder);
    return ia_sub(Interval<T>(lb), Interval<T>(G)).lo <= Ufirst;
}

// Query records: integers [point, mode (0: D3, 1: D4)]; reals [x, y, threshold].
// Response records: integers [point, kind, status, global face, element, local
// face, world rank]; kind 0 (summary, one per query): status 0 ok, 1 a local face
// failed (its smallest failing global face in the face field), 2 invalid query;
// reals [U_q, 0...]. Kind 1 (face): status 0 accepted, 1 failed; reals [lower,
// upper, dist, xi, hface, achieved gap bound gapUpper].
constexpr int SearchQueryInts = 2;
constexpr int SearchQueryReals = 3;
constexpr int SearchResponseInts = 7;
constexpr int SearchResponseReals = 6;

// Evaluate received queries against the local faces (D3 mode: return faces with
// lower <= U_q; D4 mode: lower <= threshold sent with the query). Invalid queries
// are answered with status 2 (no exit between collectives). The inner response
// vectors are cleared, not reallocated, so their capacity is reused across rounds.
template <class T, class I>
inline void answer_queries(const PreparedFaces<T, I>& local, const std::vector<I>& qInt,
        const std::vector<T>& qReal, const std::vector<int>& qCount, I mpirank,
        const DonorSearchParams& prm, int P, std::vector<std::vector<I>>& rInt,
        std::vector<std::vector<T>>& rReal, SearchStatistics* stats)
{
    rInt.resize(P);
    rReal.resize(P);
    for (int q = 0; q < P; q++) { rInt[q].clear(); rReal[q].clear(); }
    const I nc = local.nc;
    for (int q = 0; q < P; q++) {
        const std::size_t records = static_cast<std::size_t>(qCount[q]) * (1 + static_cast<std::size_t>(nc));
        rInt[q].reserve(records * SearchResponseInts);
        rReal[q].reserve(records * SearchResponseReals);
    }
    std::vector<FaceEvaluation<T>> fe(static_cast<std::size_t>(nc));
    std::size_t k = 0;
    for (int q = 0; q < P; q++)
        for (int j = 0; j < qCount[q]; j++, k++) {
            const I point = qInt[SearchQueryInts * k], mode = qInt[SearchQueryInts * k + 1];
            const T xp[3] = {qReal[SearchQueryReals * k], qReal[SearchQueryReals * k + 1], 0};
            const T threshold = qReal[SearchQueryReals * k + 2];
            const bool validQuery = point >= 0 && (mode == 0 || mode == 1);
            T Uq = std::numeric_limits<T>::max();
            I status = 0, firstFail = -1;
            if (validQuery) {
                for (I m = 0; m < nc; m++) {
                    fe[m] = evaluate_candidate_face(local, m, xp, prm);
                    Uq = std::min(Uq, fe[m].upper);
                    if (!fe[m].ok) {
                        status = 1;
                        if (firstFail < 0 || local.faces->global[m] < firstFail) firstFail = local.faces->global[m];
                    }
                    else if (stats && fe[m].gapTarget > 0)
                        stats->maxGapRatio = std::max(stats->maxGapRatio,
                                static_cast<double>(fe[m].gapUpper) / static_cast<double>(fe[m].gapTarget));
                }
                if (stats) stats->facesEvaluated += nc;
                if (!(Uq < std::numeric_limits<T>::max()) && status == 0) status = 1;
            }
            else status = 2;
            const I sumInt[SearchResponseInts] = {point, 0, status, firstFail, -1, -1, mpirank};
            const T sumReal[SearchResponseReals] = {Uq, 0, 0, 0, 0, 0};
            rInt[q].insert(rInt[q].end(), sumInt, sumInt + SearchResponseInts);
            rReal[q].insert(rReal[q].end(), sumReal, sumReal + SearchResponseReals);
            if (!validQuery) continue;
            const T bound = (mode == 0) ? Uq : threshold;
            for (I m = 0; m < nc; m++) {
                if (!(fe[m].lower <= bound)) continue;
                const I fInt[SearchResponseInts] = {point, 1, fe[m].ok ? 0 : 1, local.faces->global[m],
                    local.faces->elem[m], local.faces->lface[m], mpirank};
                const T fReal[SearchResponseReals] = {fe[m].lower, fe[m].upper, fe[m].dist, fe[m].xi[0],
                    local.hf[m], fe[m].gapUpper};
                rInt[q].insert(rInt[q].end(), fInt, fInt + SearchResponseInts);
                rReal[q].insert(rReal[q].end(), fReal, fReal + SearchResponseReals);
            }
        }
}

// Candidate record of one owner point.
template <class T, class I>
struct SearchCandidate {
    I global = -1, elem = -1, lface = -1, rank = -1;
    T lower = 0, upper = 0, dist = 0, xi = 0, hface = 0;
};

// Protocol error codes of the owner-side response validation (0: none).
enum SearchProtocolError : int {
    SearchProtocolOk = 0,
    SearchProtocolBadPoint = 1,       // point index out of range
    SearchProtocolUnexpected = 2,     // response from a processor not queried for that point
    SearchProtocolBadKind = 3,        // kind not 0 or 1, or a bad status
    SearchProtocolBadRank = 4,        // donor rank field differs from the sender
    SearchProtocolSummaryCount = 5,   // not exactly one summary per query
    SearchProtocolBadPayload = 6,     // inconsistent ordering (lower > upper, xi outside [0, 1]) or metadata
    SearchProtocolNoCandidate = 7,    // successful point without a candidate
    SearchProtocolBoundMismatch = 8,  // min returned upper differs from min_q U_q
    SearchProtocolInvalidQuery = 9    // a processor reported an invalid query
};

inline const char* search_protocol_error_text(int code)
{
    switch (code) {
    case SearchProtocolBadPoint: return "response point index out of range";
    case SearchProtocolUnexpected: return "response from a processor that was not queried";
    case SearchProtocolBadKind: return "response kind or status invalid";
    case SearchProtocolBadRank: return "response donor rank differs from its sender";
    case SearchProtocolSummaryCount: return "missing or duplicate response summary";
    case SearchProtocolBadPayload: return "inconsistent response ordering or metadata";
    case SearchProtocolNoCandidate: return "successful point without a returned candidate";
    case SearchProtocolBoundMismatch: return "returned candidates do not attain min U_q";
    case SearchProtocolInvalidQuery: return "a donor processor received an invalid query";
    default: return "no error";
    }
}

// Owner-side state of the search, per own point.
template <class T, class I>
struct SearchOwnerState {
    I npts = 0, nfe = 0;
    std::vector<std::vector<int>> queried;     // per point: comm ranks queried in this round (sorted)
    std::vector<std::vector<int>> summaries;   // per point: summaries received per queried rank
    std::vector<std::vector<SearchCandidate<T, I>>> cand;
    std::vector<T> U, Ufirst;                  // finite sentinels (max()), see UfirstKnown
    std::vector<char> UfirstKnown;             // D3 returned a usable bound (else query all in D4)
    std::vector<char> failed;
    std::vector<long long> facesForPoint;
    SearchFailures<I> failures;
    int error = SearchProtocolOk;
    void set_error(int code) { if (error == SearchProtocolOk) error = code; }
    void init(I n, I nfaces)
    {
        npts = n; nfe = nfaces;
        queried.assign(n, std::vector<int>());
        summaries.assign(n, std::vector<int>());
        cand.assign(n, std::vector<SearchCandidate<T, I>>());
        U.assign(n, std::numeric_limits<T>::max());
        Ufirst.assign(n, std::numeric_limits<T>::max());
        UfirstKnown.assign(n, 0);
        failed.assign(n, 0);
        facesForPoint.assign(n, 0);
        failures.reset(n);
    }
    void begin_round()
    {
        for (I p = 0; p < npts; p++) { queried[p].clear(); summaries[p].clear(); }
    }
    void expect(I p, int q) { queried[p].push_back(q); summaries[p].push_back(0); }
};

// Validates and absorbs the responses of one round (owner side). No MPI; all
// problems are recorded in st.error and reported after all collectives.
template <class T, class I>
inline void absorb_responses(SearchOwnerState<T, I>& st, bool d3, const std::vector<I>& raI,
        const std::vector<T>& raR, const std::vector<int>& raC, const std::vector<ProcessorBox<T>>& boxes,
        const std::vector<I>& world)
{
    const int P = static_cast<int>(raC.size());
    std::size_t k = 0;
    for (int q = 0; q < P; q++)
        for (int j = 0; j < raC[q]; j++, k++) {
            const I* ri = &raI[SearchResponseInts * k];
            const T* rr = &raR[SearchResponseReals * k];
            const I p = ri[0];
            if (p < 0 || p >= st.npts) { st.set_error(SearchProtocolBadPoint); continue; }
            const auto it = std::find(st.queried[p].begin(), st.queried[p].end(), q);
            if (it == st.queried[p].end()) { st.set_error(SearchProtocolUnexpected); continue; }
            if (ri[6] != world[q]) { st.set_error(SearchProtocolBadRank); continue; }
            if (ri[1] == 0) {
                if (ri[2] < 0 || ri[2] > 2) { st.set_error(SearchProtocolBadKind); continue; }
                st.summaries[p][it - st.queried[p].begin()]++;
                st.facesForPoint[p] += boxes[q].nfaces;
                if (ri[2] == 2) { st.set_error(SearchProtocolInvalidQuery); st.failed[p] = 1; continue; }
                if (ri[2] == 1) {
                    st.failed[p] = 1;
                    st.failures.record(p, ri[6], ri[3]);
                }
                st.U[p] = std::min(st.U[p], rr[0]);
                if (d3 && ri[2] == 0 && rr[0] < std::numeric_limits<T>::max()) {
                    st.Ufirst[p] = rr[0];
                    st.UfirstKnown[p] = 1;
                }
            }
            else if (ri[1] == 1) {
                if (ri[2] != 0 && ri[2] != 1) { st.set_error(SearchProtocolBadKind); continue; }
                if (ri[2] == 1) {
                    st.failed[p] = 1;
                    st.failures.record(p, ri[6], ri[3]);
                    continue;
                }
                bool ok = ri[3] >= 0 && ri[4] >= 0 && ri[5] >= 0 && ri[5] < boxes[q].nfe;
                ok = ok && rr[0] <= rr[1] && rr[0] >= 0 && rr[4] >= 0 && rr[5] >= 0 && rr[3] >= 0 && rr[3] <= 1;
                if (!ok) { st.set_error(SearchProtocolBadPayload); continue; }
                SearchCandidate<T, I> c;
                c.global = ri[3]; c.elem = ri[4]; c.lface = ri[5]; c.rank = ri[6];
                c.lower = rr[0]; c.upper = rr[1]; c.dist = rr[2]; c.xi = rr[3]; c.hface = rr[4];
                st.cand[p].push_back(c);
            }
            else st.set_error(SearchProtocolBadKind);
        }
    for (I p = 0; p < st.npts; p++)
        for (int count : st.summaries[p])
            if (count != 1) st.set_error(SearchProtocolSummaryCount);
}

// D5 for one point: selection over the returned candidates (shared
// tie rule), with the bound consistency check. Scratch vectors are reused.
template <class T, class I>
inline I select_returned_candidate(SearchOwnerState<T, I>& st, I p, std::vector<T>& lo, std::vector<T>& up,
        std::vector<I>& gl)
{
    std::vector<SearchCandidate<T, I>>& c = st.cand[p];
    std::sort(c.begin(), c.end(), [](const SearchCandidate<T, I>& u, const SearchCandidate<T, I>& v) {
        return std::tie(u.global, u.rank) < std::tie(v.global, v.rank);
    });
    lo.resize(c.size()); up.resize(c.size()); gl.resize(c.size());
    T minUpper = std::numeric_limits<T>::max();
    for (std::size_t m = 0; m < c.size(); m++) {
        lo[m] = c[m].lower; up[m] = c[m].upper; gl[m] = c[m].global;
        minUpper = std::min(minUpper, c[m].upper);
    }
    const I best = select_certified_face(lo.data(), up.data(), gl.data(), static_cast<I>(c.size()));
    if (!st.failed[p]) {
        if (best < 0) st.set_error(SearchProtocolNoCandidate);
        else if (!(minUpper == st.U[p])) st.set_error(SearchProtocolBoundMismatch);
    }
    return best;
}

// Steps D3-D5 for the own points of res (both models at once: every member owns
// points of its model and queries processors of the other model). Fills res as
// donor_search does and returns the owner-side protocol error code (0 if none);
// failing points are identified in failures. Collective on comm_Gamma; never exits
// between collectives except for errors that every member detects identically.
template <class T, class I>
inline int distributed_donor_search(DonorSearchResult<T, I>& res, const InterfaceComm<I>& ic,
        const InterfaceFaceSet<T, I>& own, const T* xpf, I elemtype, I porder, I mpirank, I model,
        const DonorSearchParams& prm, SearchStatistics* stats, SearchFailures<I>* failures,
        I nfe)
{
    const I nd = res.nd, ndf = nd - 1, npts = res.npts;
    const int P = ic.size;
    if (nd != 2 || porder < 1 || porder > prm.maxPorder) {
        std::ostringstream oss;
        oss << "non-matching donor search: supported for nd = 2 and 1 <= porder <= " << prm.maxPorder
            << " (got nd = " << nd << ", porder = " << porder << "); see DonorSearchParams";
        error(oss.str());
    }
    res.r.assign(npts, -1);
    res.e.assign(npts, -1);
    res.f.assign(npts, -1);
    res.a.assign(npts, -1);
    res.dist.assign(npts, std::numeric_limits<T>::max());
    res.xi.assign(static_cast<std::size_t>(npts) * static_cast<std::size_t>(std::max<I>(ndf, 1)), static_cast<T>(0));
    res.hface.assign(npts, static_cast<T>(0));
    res.converged.assign(npts, 0);
    if (stats) stats->ownerPoints += npts;

    // D1-D2.
    const PreparedFaces<T, I> local = prepare_candidate_faces(own, xpf, elemtype, porder);
    const std::vector<ProcessorBox<T>> boxes = exchange_processor_boxes(ic, local, mpirank, model, nfe);
    std::vector<int> opposite;
    for (int q = 0; q < P; q++)
        if (boxes[q].model != model) opposite.push_back(q); // increasing comm rank = increasing world rank
    if (opposite.empty()) // every member sees the same boxes: uniform
        error("non-matching donor search: the other model has no interface faces");

    std::vector<T> xp(static_cast<std::size_t>(npts) * 2);
    for (I p = 0; p < npts; p++)
        for (int d = 0; d < 2; d++)
            xp[2 * p + d] = res.x[static_cast<std::size_t>(p) + static_cast<std::size_t>(npts) * d];

    SearchOwnerState<T, I> st;
    st.init(npts, nfe > 0 ? nfe : std::numeric_limits<I>::max());
    std::vector<int> first(static_cast<std::size_t>(npts), -1);

    std::vector<std::vector<I>> qI(P), aI;
    std::vector<std::vector<T>> qR(P), aR;
    std::vector<I> rqI, raI;
    std::vector<T> rqR, raR;
    std::vector<int> rqC, raC;
    RecordExchangeWorkspace<T, I> ws; // shared by the four exchanges of D3 and D4
    auto round = [&](bool d3) {
        exchange_records(ic, SearchQueryInts, SearchQueryReals, qI, qR, rqI, rqR, rqC,
                stats ? &stats->queryBytes : nullptr, ws);
        answer_queries(local, rqI, rqR, rqC, mpirank, prm, P, aI, aR, stats);
        exchange_records(ic, SearchResponseInts, SearchResponseReals, aI, aR, raI, raR, raC,
                stats ? &stats->responseBytes : nullptr, ws);
        absorb_responses(st, d3, raI, raR, raC, boxes, ic.world);
        for (int q = 0; q < P; q++) { qI[q].clear(); qR[q].clear(); }
    };

    // D3: the closest box (smallest box lower bound; tie: smallest world rank).
    st.begin_round();
    for (I p = 0; p < npts; p++) {
        int best = -1; T bestLB = 0;
        for (int q : opposite) {
            const T lb = box_distance_lower(&xp[2 * p], boxes[q]);
            if (best < 0 || lb < bestLB) { best = q; bestLB = lb; }
        }
        first[p] = best;
        st.expect(p, best);
        qI[best].push_back(p); qI[best].push_back(0);
        qR[best].push_back(xp[2 * p]); qR[best].push_back(xp[2 * p + 1]); qR[best].push_back(0);
    }
    round(true);

    // D4: every other opposite processor with lo(boxLB - G) <= U_first (all of them
    // if the D3 processor failed or returned no usable bound: UfirstKnown false).
    st.begin_round();
    for (I p = 0; p < npts; p++) {
        long long fan = 0;
        for (int q : opposite) {
            if (q == first[p]) continue;
            if (st.UfirstKnown[p] &&
                !d4_must_query(&xp[2 * p], boxes[q], static_cast<int>(porder), st.Ufirst[p])) continue;
            st.expect(p, q);
            qI[q].push_back(p); qI[q].push_back(1);
            qR[q].push_back(xp[2 * p]); qR[q].push_back(xp[2 * p + 1]);
            qR[q].push_back(st.Ufirst[p]);
            fan++;
        }
        if (stats) {
            stats->d4Queries += fan;
            stats->maxD4FanOut = std::max(stats->maxD4FanOut, fan);
        }
    }
    round(false);

    // D5: the shared tie-rule selection over the returned faces, in a fixed order.
    std::vector<T> lo, up;
    std::vector<I> gl;
    for (I p = 0; p < npts; p++) {
        if (stats) {
            stats->facesPerPointSum += st.facesForPoint[p];
            stats->maxFacesPerPoint = std::max(stats->maxFacesPerPoint, st.facesForPoint[p]);
        }
        const I best = select_returned_candidate(st, p, lo, up, gl);
        res.converged[p] = (best >= 0 && !st.failed[p]) ? 1 : 0;
        if (best < 0) continue;
        const SearchCandidate<T, I>& c = st.cand[p][best];
        res.r[p] = c.rank;
        res.e[p] = c.elem;
        res.f[p] = c.global;
        res.a[p] = c.lface;
        res.dist[p] = c.dist;
        res.xi[p] = c.xi;
        res.hface[p] = c.hface;
    }
    if (failures) *failures = st.failures;
    return st.error;
}

// Exhaustive oracle on comm_Gamma: all-gather of the interface faces of both
// models among the members only, then the direct donor_search.
template <class T, class I>
inline void exhaustive_donor_search(DonorSearchResult<T, I>& res, const InterfaceComm<I>& ic,
        const InterfaceFaceSet<T, I>& own, const FaceMasterDescriptor<T, I>& donorMaster, I model,
        const DonorSearchParams& prm, SearchStatistics* stats, SearchFailures<I>* failures = nullptr)
{
    const InterfaceFaceSet<T, I> allFaces = gather_interface_faces(own, ic.comm);
    ic.collective(); ic.collective(); ic.collective();
    const InterfaceFaceSet<T, I> candFaces = select_model_faces(allFaces, static_cast<I>(1 - model));
    if (stats) {
        stats->remoteFaceGeometry += allFaces.size() - own.size();
        stats->ownerPoints += res.npts;
        stats->facesPerPointSum += static_cast<long long>(res.npts) * candFaces.size();
        stats->maxFacesPerPoint = std::max<long long>(stats->maxFacesPerPoint, candFaces.size());
    }
    donor_search(res, candFaces, donorMaster.xpf.data(), donorMaster.elemtype,
                 donorMaster.porder, prm, failures);
}

#endif // HAVE_MPI

} // namespace nonmatching
} // namespace exasim

#endif
