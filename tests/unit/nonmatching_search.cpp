// Phase 7 of the non-matching interface solver: scalable donor search on the
// interface communicator (backend/Discretization/nonmatchingsearch.hpp; design
// Section 3.1 "Interface communicator", Section 3.2 Steps D1-D5).
//
// MPI test, run with 2, 4 or 8 ranks. Cases:
//   equivalence: for every Phase 2-3 mesh fixture and partition (1+1, 2+2, 4+4,
//      and asymmetric 1+3, 3+1, 2+6, 6+2 where the world size allows), the
//      distributed search (production default) and the exhaustive oracle give
//      bitwise identical (r, e, f, a, dist, xi, hface, converged) and bitwise
//      identical Phase 3 caches.
//   cross-processor ties: the Phase 2 near-tie, mirror, shared-vertex and
//      tie-threshold configurations with the competing faces on DIFFERENT
//      donor ranks; distributed == exhaustive, and a test-local no-margin
//      routing calculation selects differently in at least one configuration.
//   processor boxes: enclosure of dense samples and of every Bernstein
//      coefficient interval (p = 1..4, quads and triangles, a p = 4 face that
//      leaves its node box); the box follows the face only, not the element.
//   nonmembers: column partitions in which some ranks (world rank 0 among them)
//      own no interface face; they get MPI_COMM_NULL, enter no interface
//      collective, keep a valid empty type-1 cache, and the donor faces,
//      distances and xi equal those of a partition without nonmembers.
//   failure policy: a failing face on a certifiably pruned processor does not
//      fail the point; a failing face on a queried processor does.
//   rebuild: a motion that changes the search and run-time neighbors gives the
//      same cache as a fresh setup; comm_Gamma creation/free balance is zero.
//   scalability (localized weak scaling, fixed faces per rank): bytes, D4
//      fan-out and faces evaluated per point; distributed stores no remote face
//      geometry and its per-point work stays bounded, exhaustive grows.
//   inherited limitation: scale 1e-3 translated by 1e6, both modes recorded.
//   floating-point diagnostics (test-only; maintainer decision 2026-10-09: the
//      production code has no NaN/infinity validation and is built with
//      -ffast-math): bit-level finite scans of the processor boxes, query
//      responses, donor tuples and complete setup caches of every valid fixture.
//
// Built twice: unit_nonmatching_search (strict) and unit_nonmatching_search_fastmath
// (-ffast-math, NONMATCHING_SEARCH_FAST_MATH), in which only the strict-IEEE
// diagnostics (TwoDiff rounding boundary, subnormal inputs) are skipped.

#define main nonmatching_donorsearch_embedded_main
#include "nonmatching_donorsearch.cpp"
#undef main

#include <chrono>
#include <cstdint>
#include <cstring>

namespace {

using exasim::nonmatching::SearchMode;
using exasim::nonmatching::SearchStatistics;
using exasim::nonmatching::nonmatching_comm_balance;
using exasim::nonmatching::SearchFailures;

// Test-only finite test by bit inspection (exponent field not all ones), so it is
// meaningful also in the -ffast-math build, where std::isfinite may be folded.
static_assert(sizeof(dstype) == sizeof(std::uint64_t), "phase7 finite scan assumes double");
bool test_finite(dstype v)
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return ((bits >> 52) & 0x7ffu) != 0x7ffu;
}

// Infinity and NaN built from their bit patterns (not std::numeric_limits), so the
// fast-math target does not depend on compiler handling of non-finite constants.
dstype test_from_bits(std::uint64_t bits)
{
    dstype v = 0;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}
const std::uint64_t testInfBits = 0x7ff0000000000000ull, testNaNBits = 0x7ff8000000000000ull;

long count_nonfinite(const dstype* x, std::size_t n, long& scanned)
{
    long bad = 0;
    if (n > 0 && x == nullptr) return 1;
    for (std::size_t i = 0; i < n; i++) bad += test_finite(x[i]) ? 0 : 1;
    scanned += static_cast<long>(n);
    return bad;
}

long finiteScanned = 0; // values covered by the test-only finite scans on this rank

template <class T>
void append_bytes(std::vector<unsigned char>& out, const T* data, std::size_t count)
{
    if (count == 0) return;
    check(data != nullptr, "phase7: cache snapshot pointer");
    if (!data) return;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(data);
    out.insert(out.end(), bytes, bytes + count * sizeof(T));
}

// Complete two-direction cache (as in the Phase 6 setup test).
std::vector<unsigned char> cache_snapshot(const nonmatchingdata& nm)
{
    std::vector<unsigned char> out;
    append_bytes(out, &nm.interfacetype, 1);
    for (int direction = 0; direction < 2; ++direction) {
        const nonmatchinginterface& d = nm.dir[direction];
        const Int scalars[] = {d.direction, d.isowner, d.isdonor, d.nd, d.npe, d.npf,
            d.nfe, d.ngf, d.ncu, d.ncq, d.ncu12, d.nownfaces, d.nrecvnb, d.npts,
            d.nslots, d.npairs, d.nsendnb, d.szRi, d.szHi, d.szsendbuf, d.szrecvbuf};
        append_bytes(out, scalars, sizeof(scalars) / sizeof(scalars[0]));
        Int nrecv = 0;
        for (Int q = 0; q < d.nrecvnb; ++q) nrecv += d.recvcount[q];
        append_bytes(out, d.ownface, d.nownfaces);
        append_bytes(out, d.ownfaceglobal, d.nownfaces);
        append_bytes(out, d.recvnb, d.nrecvnb);
        append_bytes(out, d.recvcount, d.nrecvnb);
        append_bytes(out, d.recvface, nrecv);
        append_bytes(out, d.pe, d.npts); append_bytes(out, d.pa, d.npts);
        append_bytes(out, d.pf, d.npts); append_bytes(out, d.pslot, d.npts);
        append_bytes(out, d.pg, d.npts);
        append_bytes(out, d.px, static_cast<std::size_t>(d.npts) * d.nd);
        append_bytes(out, d.pnl, static_cast<std::size_t>(d.npts) * d.nd);
        append_bytes(out, d.pwJ, d.npts);
        append_bytes(out, d.pxi, static_cast<std::size_t>(d.npts) * (d.nd - 1));
        append_bytes(out, d.pzeta, static_cast<std::size_t>(d.npts) * d.nd);
        append_bytes(out, d.pphi, static_cast<std::size_t>(d.npts) * d.npe);
        append_bytes(out, d.pphibar, static_cast<std::size_t>(d.npts) * d.npe);
        append_bytes(out, d.ppsi, static_cast<std::size_t>(d.npts) * d.npf);
        append_bytes(out, d.pd, static_cast<std::size_t>(d.npts) * d.nd);
        if (d.ncq == 0) append_bytes(out, d.pdphibar, static_cast<std::size_t>(d.npts) * d.nd * d.npe);
        append_bytes(out, d.slotrank, d.nslots); append_bytes(out, d.slotface, d.nslots);
        append_bytes(out, d.slotptr, d.nslots > 0 ? d.nslots + 1 : 0);
        append_bytes(out, d.pairslot, d.npairs); append_bytes(out, d.pairelem, d.npairs);
        append_bytes(out, d.pairptr, d.npairs > 0 ? d.npairs + 1 : 0);
        append_bytes(out, d.sendnb, d.nsendnb); append_bytes(out, d.sendcount, d.nsendnb);
        append_bytes(out, d.Ri, d.szRi); append_bytes(out, d.Hi, d.szHi);
        append_bytes(out, d.sendbuf, d.szsendbuf); append_bytes(out, d.recvbuf, d.szrecvbuf);
    }
    return out;
}

// Test-only finite scan of every real array of a complete two-direction cache.
long scan_cache_nonfinite(const nonmatchingdata& nm)
{
    long bad = 0;
    for (int direction = 0; direction < 2; ++direction) {
        const nonmatchinginterface& d = nm.dir[direction];
        const std::size_t n = static_cast<std::size_t>(d.npts);
        bad += count_nonfinite(d.px, n * d.nd, finiteScanned);
        bad += count_nonfinite(d.pnl, n * d.nd, finiteScanned);
        bad += count_nonfinite(d.pwJ, n, finiteScanned);
        bad += count_nonfinite(d.pxi, n * (d.nd - 1), finiteScanned);
        bad += count_nonfinite(d.pzeta, n * d.nd, finiteScanned);
        bad += count_nonfinite(d.pphi, n * d.npe, finiteScanned);
        bad += count_nonfinite(d.pphibar, n * d.npe, finiteScanned);
        bad += count_nonfinite(d.ppsi, n * d.npf, finiteScanned);
        bad += count_nonfinite(d.pd, n * d.nd, finiteScanned);
        if (d.ncq == 0) bad += count_nonfinite(d.pdphibar, n * d.nd * d.npe, finiteScanned);
        bad += count_nonfinite(d.Ri, d.szRi, finiteScanned);
        bad += count_nonfinite(d.Hi, d.szHi, finiteScanned);
        bad += count_nonfinite(d.sendbuf, d.szsendbuf, finiteScanned);
        bad += count_nonfinite(d.recvbuf, d.szrecvbuf, finiteScanned);
    }
    return bad;
}

// Test-only finite scan of the donor tuples (owner points, dist, xi, hface).
long scan_result_nonfinite(const DonorSearchResult<dstype, Int>& res)
{
    long bad = count_nonfinite(res.x.data(), res.x.size(), finiteScanned);
    for (Int p = 0; p < res.npts; p++) {
        if (!res.converged[p]) continue;
        bad += count_nonfinite(&res.dist[p], 1, finiteScanned);
        bad += count_nonfinite(&res.xi[p], 1, finiteScanned);
        bad += count_nonfinite(&res.hface[p], 1, finiteScanned);
    }
    return bad;
}

template <class T>
bool bitwise_equal(const std::vector<T>& a, const std::vector<T>& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// Deterministic value in [0, 1).
double hv_unit(long i)
{
    const double x = std::sin(0.7311 * static_cast<double>(i) + 0.123) * 43758.5453;
    return x - std::floor(x);
}

bool results_equal(const DonorSearchResult<dstype, Int>& a, const DonorSearchResult<dstype, Int>& b)
{
    return bitwise_equal(a.r, b.r) && bitwise_equal(a.e, b.e) && bitwise_equal(a.f, b.f) &&
           bitwise_equal(a.a, b.a) && bitwise_equal(a.dist, b.dist) && bitwise_equal(a.xi, b.xi) &&
           bitwise_equal(a.hface, b.hface) && bitwise_equal(a.converged, b.converged);
}

// Rank structs of the elements accepted by keep(ge) (row or column strips).
void make_rank_structs(RankStructs& rs, const ModelMesh& mm, MasterData& m,
                       const std::function<bool(int)>& keep, int interfacetype)
{
    build_rank_structs(rs, mm, m, 0, 1, interfacetype); // all elements, then filter
    std::vector<Int> elempart, bf;
    std::vector<dstype> xdg;
    std::vector<dstype> xe;
    std::vector<int> bfe;
    for (int ge = 0; ge < mm.ne(); ge++) {
        if (!keep(ge)) continue;
        element_data(xe, bfe, mm, m, ge);
        elempart.push_back(ge);
        xdg.insert(xdg.end(), xe.begin(), xe.end());
        bf.insert(bf.end(), bfe.begin(), bfe.end());
    }
    rs.elempart = elempart; rs.bf = bf; rs.xdg = xdg;
    const int ne = static_cast<int>(elempart.size());
    rs.meshndims[1] = ne;
    rs.mesh.ndims = rs.meshndims.data();
    rs.mesh.bf = rs.bf.data();
    rs.mesh.elempart = rs.elempart.data();
    rs.mesh.szelempart = ne;
    rs.sol.xdg = rs.xdg.data();
    static Int dummyI = 0;
    static dstype dummyD = 0;
    if (ne == 0) { rs.mesh.bf = &dummyI; rs.mesh.elempart = &dummyI; rs.sol.xdg = &dummyD; }
}

std::function<bool(int)> row_strip(const ModelMesh& mm, int part, int nparts)
{
    return [&mm, part, nparts](int ge) { return (mm.row(ge) * nparts) / mm.ny() == part; };
}

// Column strips of a quad mesh: part 0 holds the columns farthest from the interface.
std::function<bool(int)> column_strip(const ModelMesh& mm, int part, int nparts)
{
    return [&mm, part, nparts](int ge) {
        int col = ge % mm.nx();
        if (mm.xiface < mm.xfar) col = mm.nx() - 1 - col; // interface column last
        return (col * nparts) / mm.nx() == part;
    };
}

struct SubWorld {
    MPI_Comm comm = MPI_COMM_NULL;
    explicit SubWorld(int n)
    {
        MPI_Comm_split(MPI_COMM_WORLD, worldRank < n ? 0 : MPI_UNDEFINED, worldRank, &comm);
        if (comm != MPI_COMM_NULL) EXASIM_COMM_WORLD = comm;
    }
    ~SubWorld()
    {
        if (comm != MPI_COMM_NULL) { MPI_Comm_free(&comm); EXASIM_COMM_WORLD = MPI_COMM_WORLD; }
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

struct StatSummary {
    long long points = 0, d4 = 0, maxFan = 0, facesSum = 0, maxFaces = 0, remote = 0;
    long long queryBytes = 0, responseBytes = 0, boxBytes = 0, countBytes = 0;
    double maxGap = 0;
};

StatSummary reduce_stats(const SearchStatistics& s, MPI_Comm comm)
{
    long long sum[7] = {s.ownerPoints, s.d4Queries, s.facesPerPointSum, s.remoteFaceGeometry,
                        s.queryBytes, s.responseBytes, s.countBytes}, gsum[7];
    long long mx[3] = {s.maxD4FanOut, s.maxFacesPerPoint, s.boxBytes}, gmx[3];
    double g = s.maxGapRatio, gg = 0;
    MPI_Allreduce(sum, gsum, 7, MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(mx, gmx, 3, MPI_LONG_LONG, MPI_MAX, comm);
    MPI_Allreduce(&g, &gg, 1, MPI_DOUBLE, MPI_MAX, comm);
    StatSummary r;
    r.points = gsum[0]; r.d4 = gsum[1]; r.facesSum = gsum[2]; r.remote = gsum[3];
    r.queryBytes = gsum[4]; r.responseBytes = gsum[5]; r.countBytes = gsum[6];
    r.maxFan = gmx[0]; r.maxFaces = gmx[1]; r.boxBytes = gmx[2]; r.maxGap = gg;
    return r;
}

// Rank-independent tuples (owner global face, g, f, dist, xi), gathered on world rank 0 of comm.
std::map<std::pair<long, long>, std::vector<double>> gather_tuples(const DonorSearchResult<dstype, Int>& res,
                                                                    MPI_Comm comm)
{
    std::vector<double> mine;
    for (Int l = 0; l < res.nfaces; l++)
        for (Int g = 0; g < res.ngf; g++) {
            const Int p = g + res.ngf * l;
            mine.insert(mine.end(), {double(res.direction), double(res.ownglobal[l]), double(g),
                                     double(res.f[p]), double(res.dist[p]), double(res.xi[p])});
        }
    int n = 0, me = 0;
    MPI_Comm_size(comm, &n);
    MPI_Comm_rank(comm, &me);
    int count = static_cast<int>(mine.size());
    std::vector<int> counts(n), displs(n, 0);
    MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
    int total = 0;
    for (int r = 0; r < n; r++) { displs[r] = total; total += counts[r]; }
    std::vector<double> all(me == 0 ? total : 0);
    MPI_Gatherv(mine.data(), count, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
    std::map<std::pair<long, long>, std::vector<double>> out;
    for (std::size_t i = 0; i + 6 <= all.size(); i += 6)
        out[std::make_pair((long(all[i]) << 40) + long(all[i + 1]), long(all[i + 2]))] =
            std::vector<double>(all.begin() + i + 3, all.begin() + i + 6);
    return out;
}

// Test-only finite scan of the production protocol outputs on a valid fixture: the
// all-gathered processor records of Step D2 (exchange_processor_boxes) and the
// responses of answer_queries of every member to every owner point of the other
// model (D3 mode). Collective on EXASIM_COMM_WORLD (the partition's SubWorld).
long scan_protocol_nonfinite(const RankStructs& rs, MasterData& m, int nprocs, int model, int fileoffset,
                             MPI_Comm comm, long* responseRecords)
{
    using namespace exasim::nonmatching;
    const InterfaceFaceSet<dstype, Int> own = collect_own_interface_faces(rs.app, rs.master, rs.mesh, rs.sol,
            Int(worldRank), Int(fileoffset));
    DonorSearchResult<dstype, Int> res;
    res.direction = model;
    own_quadrature_points(res, own, m.shapfgt.data(), Int(m.ngf));
    std::vector<double> mine;
    for (Int p = 0; p < res.npts; p++)
        mine.insert(mine.end(), {double(model), double(res.x[p]), double(res.x[p + res.npts])});
    int n = 0;
    MPI_Comm_size(comm, &n);
    int count = static_cast<int>(mine.size());
    std::vector<int> counts(n), displs(n, 0);
    MPI_Allgather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
    int total = 0;
    for (int r = 0; r < n; r++) { displs[r] = total; total += counts[r]; }
    std::vector<double> all(total);
    MPI_Allgatherv(mine.data(), count, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);

    long bad = 0;
    InterfaceComm<Int> ic;
    create_interface_comm(ic, own.size() > 0, Int(worldRank), Int(nprocs), Int(model), nullptr);
    if (ic.comm != MPI_COMM_NULL) {
        const PreparedFaces<dstype, Int> local = prepare_candidate_faces(own, m.xpf.data(), Int(m.elemtype),
                Int(m.porder));
        const std::vector<ProcessorBox<dstype>> boxes = exchange_processor_boxes(
                ic, local, Int(worldRank), Int(model), Int(m.nfe));
        for (const ProcessorBox<dstype>& b : boxes) {
            const dstype v[6] = {b.xmin[0], b.xmin[1], b.xmax[0], b.xmax[1], b.cmax, b.beta};
            bad += count_nonfinite(v, 6, finiteScanned);
        }
        std::vector<Int> qI;
        std::vector<dstype> qR;
        for (std::size_t i = 0; i + 3 <= all.size(); i += 3) {
            if (int(all[i]) == model) continue;
            qI.push_back(Int(qI.size() / SearchQueryInts)); qI.push_back(0);
            qR.insert(qR.end(), {dstype(all[i + 1]), dstype(all[i + 2]), dstype(0)});
        }
        const std::vector<int> qc(1, static_cast<int>(qI.size() / SearchQueryInts));
        std::vector<std::vector<Int>> rI;
        std::vector<std::vector<dstype>> rR;
        answer_queries(local, qI, qR, qc, Int(worldRank), DonorSearchParams(), 1, rI, rR,
                       static_cast<SearchStatistics*>(nullptr));
        bad += count_nonfinite(rR[0].data(), rR[0].size(), finiteScanned);
        *responseRecords += static_cast<long>(rR[0].size() / SearchResponseReals);
    }
    ic.release();
    return bad;
}

// One partition: the first k1 + k2 world ranks, k1 ranks of model 1 then k2 of model 2.
// Runs the full setup in Exhaustive and Distributed mode and compares them bitwise.
struct ModeRun {
    std::map<std::pair<long, long>, std::vector<double>> tuples; // distributed, on rank 0
    StatSummary dist, exh;
    double timeDist = 0, timeExh = 0;           // full setup, one run (max over ranks)
    double searchDist = 0, searchExh = 0;       // search only (D1-D5 / gather + exhaustive), median of runs
};

ModeRun compare_modes(const std::string& name, const ModelMesh& m1, const ModelMesh& m2, MasterData& m,
                      int k1, int k2, bool columns1 = false, int timingRuns = 1)
{
    ModeRun run;
    SubWorld sw(k1 + k2);
    if (sw.comm == MPI_COMM_NULL) return run;
    const int model = worldRank < k1 ? 0 : 1;
    const int part = worldRank - model * k1, nparts = model ? k2 : k1;
    const ModelMesh& mm = model ? m2 : m1;
    const std::string tag = name + " (" + std::to_string(k1) + "+" + std::to_string(k2) +
                            (columns1 ? ", column strips" : "") + ")";
    RankStructs rs;
    make_rank_structs(rs, mm, m, (columns1 && model == 0) ? column_strip(mm, part, nparts) : row_strip(mm, part, nparts), 1);

    const long long balance0 = nonmatching_comm_balance();
    DonorSearchParams pe, pd;
    pe.searchMode = SearchMode::Exhaustive;
    pd.searchMode = SearchMode::Distributed;
    nonmatchingdata nmE, nmD;
    DonorSearchResult<dstype, Int> resE, resD;
    SearchStatistics sE, sD;
    auto t0 = std::chrono::steady_clock::now();
    const bool ranE = setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, Int(k1 + k2), Int(worldRank),
            Int(model * k1), nmE, &resE, pe, true, &sE);
    auto t1 = std::chrono::steady_clock::now();
    const bool ranD = setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, Int(k1 + k2), Int(worldRank),
            Int(model * k1), nmD, &resD, pd, true, &sD);
    auto t2 = std::chrono::steady_clock::now();
    check(ranE && ranD, tag + ": both setups ran");
    check(nonmatching_comm_balance() == balance0, tag + ": comm_Gamma created and freed once per setup");
    check(results_equal(resE, resD), tag + ": distributed == exhaustive donor tuples (bitwise)");
    check(cache_snapshot(nmE) == cache_snapshot(nmD), tag + ": distributed == exhaustive setup cache (bitwise)");
    for (Int p = 0; p < resD.npts; p++) check(resD.converged[p] == 1, tag + ": every point converged");
    check(sD.remoteFaceGeometry == 0, tag + ": distributed stores no remote face geometry");
    // Test-only floating-point diagnostics on the outputs of this valid fixture.
    check(scan_result_nonfinite(resD) == 0 && scan_result_nonfinite(resE) == 0,
          tag + ": test-only scan: donor tuples finite");
    check(scan_cache_nonfinite(nmD) == 0 && scan_cache_nonfinite(nmE) == 0,
          tag + ": test-only scan: complete setup caches finite");
    long responseRecords = 0;
    check(scan_protocol_nonfinite(rs, m, k1 + k2, model, model * k1, sw.comm, &responseRecords) == 0,
          tag + ": test-only scan: processor boxes and query responses finite");
    check(nonmatching_comm_balance() == balance0, tag + ": scan communicator released");
    const bool member = resD.nfaces > 0;
    check(sD.member == (member ? 1 : 0), tag + ": member flag");
    if (!member) {
        check(sD.interfaceCollectives == 0 && sE.interfaceCollectives == 0,
              tag + ": nonmember enters no interface collective");
        check(nmD.interfacetype == 1 && nmD.dir[0].pe == nullptr && nmD.dir[1].pe == nullptr &&
              nmD.dir[0].isowner == 0 && nmD.dir[1].isowner == 0 && nmD.dir[0].isdonor == 0 &&
              nmD.dir[1].isdonor == 0, tag + ": nonmember keeps a valid empty type-1 cache");
    }
    run.dist = reduce_stats(sD, sw.comm);
    run.exh = reduce_stats(sE, sw.comm);
    double te = std::chrono::duration<double>(t1 - t0).count(), td = std::chrono::duration<double>(t2 - t1).count();
    MPI_Allreduce(&te, &run.timeExh, 1, MPI_DOUBLE, MPI_MAX, sw.comm);
    MPI_Allreduce(&td, &run.timeDist, 1, MPI_DOUBLE, MPI_MAX, sw.comm);
    // Search-only timings (SearchStatistics::searchSeconds, max over ranks), median of
    // timingRuns repeated setups per mode; indicative only on a shared workstation.
    auto median_search = [&](const DonorSearchParams& prm, double first) {
        std::vector<double> t(1, first);
        for (int r = 1; r < timingRuns; r++) {
            nonmatchingdata nmT;
            SearchStatistics sT;
            setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, Int(k1 + k2), Int(worldRank),
                    Int(model * k1), nmT, static_cast<DonorSearchResult<dstype, Int>*>(nullptr), prm, true, &sT);
            double g = 0;
            MPI_Allreduce(&sT.searchSeconds, &g, 1, MPI_DOUBLE, MPI_MAX, sw.comm);
            t.push_back(g);
            exasim::nonmatching::free_nonmatching_data(nmT);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    };
    double sd0 = 0, se0 = 0;
    MPI_Allreduce(&sD.searchSeconds, &sd0, 1, MPI_DOUBLE, MPI_MAX, sw.comm);
    MPI_Allreduce(&sE.searchSeconds, &se0, 1, MPI_DOUBLE, MPI_MAX, sw.comm);
    run.searchDist = median_search(pd, sd0);
    run.searchExh = median_search(pe, se0);
    run.tuples = gather_tuples(resD, sw.comm);
    int nonmembers = member ? 0 : 1, gnon = 0;
    MPI_Allreduce(&nonmembers, &gnon, 1, MPI_INT, MPI_SUM, sw.comm);
    if (worldRank == 0)
        std::printf("%s: %lld points, %d nonmember ranks, max gap/gapTol %.3f; distributed: D4 queries %lld "
                    "(max fan-out %lld), faces/point mean %.2f max %lld; exhaustive: faces/point mean %.2f max %lld\n",
                    tag.c_str(), run.dist.points, gnon, run.dist.maxGap, run.dist.d4, run.dist.maxFan,
                    run.dist.points ? double(run.dist.facesSum) / run.dist.points : 0.0, run.dist.maxFaces,
                    run.exh.points ? double(run.exh.facesSum) / run.exh.points : 0.0, run.exh.maxFaces);
    exasim::nonmatching::free_nonmatching_data(nmE);
    exasim::nonmatching::free_nonmatching_data(nmD);
    return run;
}

std::vector<std::pair<int, int>> partitions()
{
    std::vector<std::pair<int, int>> p = {{1, 1}};
    if (worldSize >= 4) { p.push_back({2, 2}); p.push_back({1, 3}); p.push_back({3, 1}); }
    if (worldSize >= 8) { p.push_back({4, 4}); p.push_back({2, 6}); p.push_back({6, 2}); }
    return p;
}

ModelMesh transformed(ModelMesh mesh, dstype scale, dstype tx, dstype ty)
{
    for (dstype& x : mesh.xs) x = tx + scale * x;
    for (dstype& y : mesh.ys) y = ty + scale * y;
    mesh.xiface = tx + scale * mesh.xiface;
    mesh.xfar = tx + scale * mesh.xfar;
    mesh.amp *= scale;
    return mesh;
}

void case_equivalence()
{
    MasterData q3 = make_master(1, 3), q2 = make_master(1, 2), t2 = make_master(0, 2), q1 = make_master(1, 1);
    ModelMesh a1, a2;
    matching_meshes(a1, a2);
    ModelMesh n1, n2;
    n1.xs = uniform(0.0, 0.5, 4); n1.ys = uniform(0.0, 1.0, 12); n1.xiface = 0.5; n1.xfar = 0.0;
    n2.xs = uniform(0.51, 1.0, 3); n2.ys = {0.0, 0.11, 0.27, 0.38, 0.55, 0.71, 0.82, 0.9, 1.0};
    n2.xiface = 0.51; n2.xfar = 1.0;
    ModelMesh c2 = n2;
    c2.xiface = 0.5; c2.xs = uniform(0.5, 1.0, 3); c2.amp = 0.01;
    ModelMesh tr1, tr2;
    tr1.tri = true; tr1.xs = uniform(0.0, 0.5, 4); tr1.ys = uniform(0.0, 1.0, 12); tr1.xiface = 0.5; tr1.xfar = 0.0;
    tr2.tri = true; tr2.xs = uniform(0.5, 1.0, 3); tr2.ys = {0.0, 0.11, 0.27, 0.38, 0.55, 0.71, 0.82, 0.9};
    tr2.xiface = 0.5; tr2.xfar = 1.0; tr2.amp = 0.01;
    // Tie mesh of Phase 2 (equidistant points, gap 0.01).
    dstype ystar = 0.0;
    for (int j = 0; j < q1.npf; j++) ystar += q1.shapfgt[0 + q1.ngf * j] * (j == 0 ? 0.0 : 0.25);
    ModelMesh e1, e2;
    e1.xs = uniform(0.0, 0.5, 2); e1.ys = uniform(0.0, 1.0, 4); e1.xiface = 0.5; e1.xfar = 0.0;
    e2.xs = uniform(0.51, 1.0, 2); e2.xiface = 0.51; e2.xfar = 1.0;
    e2.ys = {0.0, std::min(ystar, 0.25 - ystar), std::max(ystar, 0.25 - ystar), 0.6, 1.0};

    struct Fixture { std::string name; const ModelMesh* m1; const ModelMesh* m2; ModelMesh t1, t2; MasterData* m; bool own; };
    std::vector<Fixture> fixtures;
    fixtures.push_back({"matching", &a1, &a2, {}, {}, &q3, false});
    fixtures.push_back({"matching-scale-1e4", nullptr, nullptr, transformed(a1, 1e4, 0, 0), transformed(a2, 1e4, 0, 0), &q3, true});
    fixtures.push_back({"matching-translate-1e4", nullptr, nullptr, transformed(a1, 1, 1e4, 1e4), transformed(a2, 1, 1e4, 1e4), &q3, true});
    fixtures.push_back({"matching-scale-1e-6", nullptr, nullptr, transformed(a1, 1e-6, 0, 0), transformed(a2, 1e-6, 0, 0), &q3, true});
    fixtures.push_back({"nonmatching-quads", &n1, &n2, {}, {}, &q2, false});
    fixtures.push_back({"curved-quads", &n1, &c2, {}, {}, &q2, false});
    fixtures.push_back({"curved-triangles-beyond-edge", &tr1, &tr2, {}, {}, &t2, false});
    fixtures.push_back({"tie-mesh", &e1, &e2, {}, {}, &q1, false});
    for (Fixture& f : fixtures) {
        const ModelMesh& m1 = f.own ? f.t1 : *f.m1;
        const ModelMesh& m2 = f.own ? f.t2 : *f.m2;
        for (const auto& kk : partitions()) {
            if (f.name == "tie-mesh" && (kk.first > 4 || kk.second > 4)) continue; // 4 rows
            compare_modes(f.name, m1, m2, *f.m, kk.first, kk.second);
        }
    }
}

// Nonmembers: model 1 in column strips (part 0 = far columns, world rank 0 is a nonmember).
void case_nonmembers()
{
    if (worldSize < 4) return;
    MasterData q2 = make_master(1, 2);
    ModelMesh n1, n2;
    n1.xs = uniform(0.0, 0.5, 4); n1.ys = uniform(0.0, 1.0, 12); n1.xiface = 0.5; n1.xfar = 0.0;
    n2.xs = uniform(0.51, 1.0, 3); n2.ys = {0.0, 0.11, 0.27, 0.38, 0.55, 0.71, 0.82, 0.9, 1.0};
    n2.xiface = 0.51; n2.xfar = 1.0;
    const ModeRun ref = compare_modes("nonmembers-reference", n1, n2, q2, 1, 1);
    std::vector<std::pair<int, int>> parts = {{2, 2}};
    if (worldSize >= 8) { parts.push_back({4, 4}); parts.push_back({4, 2}); }
    for (const auto& kk : parts) {
        const ModeRun run = compare_modes("nonmembers", n1, n2, q2, kk.first, kk.second, true);
        if (worldRank == 0)
            check(run.tuples == ref.tuples, "nonmembers (" + std::to_string(kk.first) + "+" +
                  std::to_string(kk.second) + "): donor faces, distances and xi equal the 1+1 partition");
    }
}

// Protocol-level harness: rank 0 (model 1) owns a far dummy face and the query
// points; ranks 1 and 2 (model 2) each own one of the given faces.
struct ProtocolOutcome {
    DonorSearchResult<dstype, Int> dist, exh;
    bool sameAsExhaustive = false;
    Int noMarginDonor = -1;
};

InterfaceFaceSet<dstype, Int> one_face(const MasterData& m, const Nodes& y, Int rank, Int model, Int global)
{
    InterfaceFaceSet<dstype, Int> fs;
    fs.nd = m.nd; fs.npf = m.npf;
    fs.rank.push_back(rank); fs.model.push_back(model); fs.elem.push_back(0); fs.lface.push_back(0);
    fs.global.push_back(global);
    fs.nodes = y;
    return fs;
}

// Test-local routing WITHOUT the margin G_q (not a production path): D3 to the
// smaller box bound, D4 only if boxLB <= U_first, then the certified selection.
Int no_margin_donor(const MasterData& m, const Nodes& yA, Int gA, const Nodes& yB, Int gB, const dstype* P)
{
    using namespace exasim::nonmatching;
    InterfaceFaceSet<dstype, Int> fa = one_face(m, yA, 1, 1, gA), fb = one_face(m, yB, 2, 1, gB);
    const auto pa = prepare_candidate_faces(fa, m.xpf.data(), Int(m.elemtype), Int(m.porder));
    const auto pb = prepare_candidate_faces(fb, m.xpf.data(), Int(m.elemtype), Int(m.porder));
    const auto ba = local_processor_box(pa, Int(1), Int(1), Int(m.nfe));
    const auto bb = local_processor_box(pb, Int(2), Int(1), Int(m.nfe));
    const dstype la = box_distance_lower(P, ba), lb = box_distance_lower(P, bb);
    const DonorSearchParams prm;
    const auto ea = evaluate_candidate_face(pa, Int(0), P, prm), eb = evaluate_candidate_face(pb, Int(0), P, prm);
    const bool firstA = la < lb || (la == lb); // tie: smaller world rank (A on rank 1)
    const auto& e1 = firstA ? ea : eb; const auto& e2 = firstA ? eb : ea;
    const dstype l2 = firstA ? lb : la;
    const Int g1 = firstA ? gA : gB, g2 = firstA ? gB : gA;
    std::vector<dstype> lo = {e1.lower}, up = {e1.upper};
    std::vector<Int> gl = {g1};
    if (l2 <= e1.upper) { lo.push_back(e2.lower); up.push_back(e2.upper); gl.push_back(g2); }
    const Int k = select_certified_face(lo.data(), up.data(), gl.data(), Int(lo.size()));
    return gl[k];
}

// Runs one configuration on the first 3 world ranks; returns on world rank 0.
bool protocol_case(const std::string& name, const MasterData& m, const Nodes& yA, Int gA, const Nodes& yB, Int gB,
                   const dstype* P, int expect, bool* sensitive)
{
    using namespace exasim::nonmatching;
    SubWorld sw(3);
    bool ok = true;
    if (sw.comm == MPI_COMM_NULL) return ok;
    const Int model = worldRank == 0 ? 0 : 1;
    Nodes dummy(2 * m.npf);
    for (int j = 0; j < m.npf; j++) { dummy[j] = -100.0; dummy[j + m.npf] = m.xpf[j]; }
    InterfaceFaceSet<dstype, Int> own = worldRank == 0 ? one_face(m, dummy, 0, 0, 1000)
                                     : worldRank == 1 ? one_face(m, yA, 1, 1, gA) : one_face(m, yB, 2, 1, gB);
    DonorSearchResult<dstype, Int> res;
    res.nd = m.nd; res.ngf = 1; res.direction = model;
    res.nfaces = worldRank == 0 ? 1 : 0;
    res.npts = worldRank == 0 ? 1 : 0;
    if (worldRank == 0) res.x = {P[0], P[1]};
    InterfaceComm<Int> ic;
    SearchStatistics st;
    create_interface_comm(ic, true, Int(worldRank), Int(3), model, &st);
    SearchFailures<Int> failures;
    const int code = distributed_donor_search(res, ic, own, m.xpf.data(), Int(m.elemtype), Int(m.porder),
                                              Int(worldRank), model, DonorSearchParams(), &st, &failures,
                                              Int(m.nfe));
    ic.release();
    // Every response passed the owner-side validation (one summary per query, finite
    // ordered payloads, metadata, candidate attaining min U_q).
    check(code == exasim::nonmatching::SearchProtocolOk, name + ": responses validated");
    if (worldRank == 0) {
        InterfaceFaceSet<dstype, Int> cand = one_face(m, yA, 1, 1, gA);
        const InterfaceFaceSet<dstype, Int> fb = one_face(m, yB, 2, 1, gB);
        cand.rank.push_back(2); cand.model.push_back(1); cand.elem.push_back(0); cand.lface.push_back(0);
        cand.global.push_back(gB); cand.nodes.insert(cand.nodes.end(), fb.nodes.begin(), fb.nodes.end());
        DonorSearchResult<dstype, Int> ex;
        ex.nd = m.nd; ex.ngf = 1; ex.nfaces = 1; ex.npts = 1; ex.x = {P[0], P[1]};
        donor_search(ex, cand, m.xpf.data(), Int(m.elemtype), Int(m.porder));
        if (expect == 1) {
            // Failing face on a certifiably pruned processor: the point succeeds with the
            // exhaustive donor tuple, while the exhaustive oracle fails it (policy C3).
            DonorSearchResult<dstype, Int> exOk = ex;
            exOk.converged = res.converged;
            ok = res.converged[0] == 1 && ex.converged[0] == 0 && results_equal(exOk, res);
            check(ok, name + ": pruned failing face does not fail the point");
            std::printf("%s: converged %d with donor %d (exhaustive converged %d, donor %d)\n", name.c_str(),
                        int(res.converged[0]), int(res.f[0]), int(ex.converged[0]), int(ex.f[0]));
        }
        else if (expect == 0) {
            ok = results_equal(ex, res) && res.converged[0] == 1;
            check(ok, name + ": distributed == exhaustive across donor ranks");
            const Int nm = no_margin_donor(m, yA, gA, yB, gB, P);
            if (nm != res.f[0]) *sensitive = true;
            std::printf("%s: donor %d (exhaustive %d, no-margin routing %d), D4 fan-out %lld\n", name.c_str(),
                        int(res.f[0]), int(ex.f[0]), int(nm), st.maxD4FanOut);
        }
        else {
            // Failing face on a queried processor: the point fails and the failure
            // identifies the donor rank and global face (face B on world rank 2).
            ok = res.converged[0] == 0 && failures.donorRank[0] == 2 && failures.face[0] == gB;
            check(ok, name + ": point fails with the failing donor rank and face");
            std::printf("%s: converged %d (exhaustive %d), first failing donor rank %d, face %d\n", name.c_str(),
                        int(res.converged[0]), int(ex.converged[0]), int(failures.donorRank[0]),
                        int(failures.face[0]));
        }
    }
    return ok;
}

void case_cross_processor_ties()
{
    if (worldSize < 4) return;
    bool sensitive = false;
    MasterData m1 = make_master(1, 1);
    std::vector<dstype> xa, ya, xb, yb;
    auto nodes = [](const std::vector<dstype>& x, const std::vector<dstype>& y) {
        Nodes n(x); n.insert(n.end(), y.begin(), y.end()); return n;
    };
    const dstype P[2] = {0.0, 0.5};
    for (dstype delta : {1e-9, 5e-11, 1e-12, 1e-13}) {
        straight(xa, ya, m1, -(0.1 + delta));
        straight(xb, yb, m1, 0.1);
        protocol_case("cross-rank near tie delta=" + std::to_string(delta), m1, nodes(xa, ya), 3, nodes(xb, yb), 7,
                      P, 0, &sensitive);
    }
    straight(xa, ya, m1, -0.1);
    straight(xb, yb, m1, 0.1);
    protocol_case("cross-rank mirror tie (3,7)", m1, nodes(xa, ya), 3, nodes(xb, yb), 7, P, 0, &sensitive);
    protocol_case("cross-rank mirror tie (9,4)", m1, nodes(xa, ya), 9, nodes(xb, yb), 4, P, 0, &sensitive);
    std::vector<dstype> y0(m1.npf), y1(m1.npf);
    for (int j = 0; j < m1.npf; j++) { y0[j] = 0.5 * m1.xpf[j]; y1[j] = 0.5 + 0.5 * m1.xpf[j]; }
    xa.assign(m1.npf, 0.2);
    protocol_case("cross-rank shared vertex", m1, nodes(xa, y1), 5, nodes(xa, y0), 6, P, 0, &sensitive);

    // Tie thresholds: L (global 3, farther by factor * tau) and H (global 7) on different ranks.
    auto thresholds = [&](const std::string& name, const MasterData& mp, const dstype* Q, dstype D,
                          const std::function<Nodes(dstype)>& L, const std::function<Nodes(dstype)>& H) {
        const auto bm = make_bernstein_master(mp.xpf.data(), Int(mp.porder));
        auto bounds = [&](const Nodes& y) {
            dstype xi[2], xb2[2], d;
            ClosestPointOnFace(xi, xb2, &d, Q, y.data(), mp.xpf.data(), mp.nd, mp.npf, mp.elemtype, mp.porder, 100, 1e-13);
            return face_distance_bounds(bm, y.data(), Int(mp.npf), Q, xi[0]);
        };
        const auto bL = bounds(L(0.0)), bH = bounds(H(0.0));
        const dstype tau = (D - bL.lower) + (bH.upper - D);
        for (dstype factor : {0.5, 0.85, 1.15, 2.0})
            protocol_case(name + " " + std::to_string(factor) + " tau", mp, L(factor * tau), 3, H(factor * tau), 7,
                          Q, 0, &sensitive);
    };
    for (int porder : {1, 4}) {
        MasterData mp = make_master(1, porder);
        const dstype up[2] = {0.0, 1.0};
        thresholds("cross-rank threshold straight p=" + std::to_string(porder), mp, P, 0.1,
            [&](dstype dl) { const dstype Q[2] = {-(0.1 + dl), 0.5}; return straight_face(mp, Q, up); },
            [&](dstype) { const dstype Q[2] = {0.1, 0.5}; return straight_face(mp, Q, up); });
    }
    for (int porder : {2, 4}) {
        MasterData mp = make_master(1, porder);
        const dstype s = 1.0 / std::sqrt(1.16);
        const dstype nrm[2] = {-s, -0.4 * s}, tan[2] = {0.4 * s, -s};
        const dstype G[2] = {0.04, 0.3};
        const dstype Q[2] = {G[0] + 0.25 * nrm[0], G[1] + 0.25 * nrm[1]};
        const Nodes curved = parabola_face(mp);
        thresholds("cross-rank threshold curved-farther p=" + std::to_string(porder), mp, Q, 0.25,
            [&](dstype) { return curved; },
            [&](dstype dl) { const dstype R[2] = {Q[0] + (0.25 - dl) * nrm[0], Q[1] + (0.25 - dl) * nrm[1]};
                             return straight_face(mp, R, tan); });
        thresholds("cross-rank threshold curved-closer p=" + std::to_string(porder), mp, Q, 0.25,
            [&](dstype dl) { const dstype R[2] = {Q[0] + (0.25 + dl) * nrm[0], Q[1] + (0.25 + dl) * nrm[1]};
                             return straight_face(mp, R, tan); },
            [&](dstype) { return curved; });
    }
    // Translated tie thresholds: the certified gap of the farther face L is now much
    // larger than that of H, so lower_L <= upper_H < boxLB_L can hold. Routing without
    // the margin then prunes L's processor although the oracle selects L (found by a
    // scan over translations, face lengths, p and delta / tau; see the Phase 7 report).
    for (int porder : {1, 2}) {
        MasterData mp = make_master(1, porder);
        for (dstype T : {1e2, 1e6}) {
            if (porder == 2 && T > 1e2) continue;
            const dstype PT[2] = {T, T + 0.5};
            const dstype up[2] = {0.0, 1.0};
            const auto bm = make_bernstein_master(mp.xpf.data(), Int(mp.porder));
            auto Lf = [&](dstype dl) { const dstype Q[2] = {T - (0.1 + dl), T + 0.5}; return straight_face(mp, Q, up); };
            auto Hf = [&]() { const dstype Q[2] = {T + 0.1, T + 0.5}; return straight_face(mp, Q, up); };
            auto bounds = [&](const Nodes& y) {
                dstype xi[2], xb2[2], d;
                ClosestPointOnFace(xi, xb2, &d, PT, y.data(), mp.xpf.data(), mp.nd, mp.npf, mp.elemtype, mp.porder, 100, 1e-13);
                return face_distance_bounds(bm, y.data(), Int(mp.npf), PT, xi[0]);
            };
            const auto bL = bounds(Lf(0.0)), bH = bounds(Hf());
            const dstype tau = (0.1 - bL.lower) + (bH.upper - 0.1);
            for (dstype factor : {0.5, 0.9})
                protocol_case("cross-rank translated threshold p=" + std::to_string(porder) + " T=" +
                              std::to_string(T) + " " + std::to_string(factor) + " tau", mp, Lf(factor * tau), 3,
                              Hf(), 7, PT, 0, &sensitive);
        }
    }
    int s = sensitive ? 1 : 0, gs = 0;
    MPI_Allreduce(&s, &gs, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    check(gs == 1, "cross-rank ties: routing without the margin G_q selects a different donor in some case");
    if (worldRank == 0)
        std::printf("cross-rank ties: no-margin routing differs from the exhaustive oracle in at least one case: %s\n",
                    gs ? "yes" : "no");

    // Failure policy. A degenerate face (all nodes equal) cannot be projected.
    const Nodes good = [&] { straight(xa, ya, m1, 0.1); return nodes(xa, ya); }();
    Nodes far(2 * m1.npf, 50.0), near(2 * m1.npf);
    for (int j = 0; j < m1.npf; j++) { near[j] = 0.05; near[j + m1.npf] = 0.5; }
    protocol_case("failure on a pruned processor", m1, good, 3, far, 7, P, 1, &sensitive);
    protocol_case("failure on a queried processor", m1, good, 3, near, 7, P, 2, &sensitive);
}

// Phase 2 curved p = 4 adversarial faces (bump and Lebesgue) routed across ranks:
// face A (curved, global 9) on rank 1, face B (straight, global 2) on rank 2. The
// independent dense-sampling inequalities of Phase 2 are kept: the image of A leaves
// its node box (box bound > d_A), d_A < d_B, and in at least one case the old
// 25%-padded node box would also have pruned the correct donor (padded bound > d_B).
void case_curved_adversarial_routing()
{
    if (worldSize < 4) return;
    MasterData m = make_master(1, 4);
    bool paddedDefeated = false, sensitive = false;
    for (int mode = 0; mode < 2; mode++)
        for (dstype c : {1.0, 4.0}) {
            const int ns = 20000;
            std::vector<dstype> psi(m.npf), xa(m.npf, 0.0), ya(m.npf);
            for (int j = 0; j < m.npf; j++) ya[j] = m.xpf[j];
            if (mode == 0) {
                int jbump = -1;
                for (int j = 0; j < m.npf; j++)
                    if (m.xpf[j] > 0.0 && m.xpf[j] < 1.0 &&
                        (jbump < 0 || std::abs(m.xpf[j] - 0.25) < std::abs(m.xpf[jbump] - 0.25)))
                        jbump = j;
                xa[jbump] = c;
            }
            else {
                dstype lmax = -1;
                std::vector<dstype> best(m.npf);
                for (int i = 0; i <= ns; i++) {
                    const dstype xi[2] = {dstype(i) / ns, 0.0};
                    EvaluateFaceShape(psi.data(), xi, m.xpf.data(), m.nd, m.npf, m.elemtype, m.porder);
                    dstype l = 0;
                    for (int j = 0; j < m.npf; j++) l += std::abs(psi[j]);
                    if (l > lmax) { lmax = l; best = psi; }
                }
                for (int j = 0; j < m.npf; j++) xa[j] = (best[j] >= 0 ? -c : c);
            }
            dstype xmin = 1e300, ymin = 0;
            std::vector<dstype> sx(ns + 1), sy(ns + 1);
            for (int i = 0; i <= ns; i++) {
                const dstype xi[2] = {dstype(i) / ns, 0.0};
                EvaluateFaceShape(psi.data(), xi, m.xpf.data(), m.nd, m.npf, m.elemtype, m.porder);
                sx[i] = 0; sy[i] = 0;
                for (int j = 0; j < m.npf; j++) { sx[i] += psi[j] * xa[j]; sy[i] += psi[j] * ya[j]; }
                if (sx[i] < xmin) { xmin = sx[i]; ymin = sy[i]; }
            }
            const dstype P[2] = {xmin - 0.05, ymin};
            dstype dA = 1e300;
            for (int i = 0; i <= ns; i++) dA = std::min(dA, std::hypot(P[0] - sx[i], P[1] - sy[i]));
            dstype bxmin = 1e300, bxmax = -1e300;
            for (int j = 0; j < m.npf; j++) { bxmin = std::min(bxmin, xa[j]); bxmax = std::max(bxmax, xa[j]); }
            const dstype diag = std::hypot(bxmax - bxmin, 1.0);
            const dstype lbNodal = bxmin - P[0];
            const dstype lbPadded = std::max<dstype>(0.0, lbNodal - 0.25 * diag);
            const dstype dB = 0.5 * (dA + (lbPadded > dA ? lbPadded : lbNodal));
            std::vector<dstype> xb, yb;
            straight(xb, yb, m, P[0] - dB);
            const std::string tag = std::string("cross-rank curved p=4 ") + (mode ? "lebesgue" : "bump") +
                                    " c=" + std::to_string(c);
            if (worldRank == 0) {
                check(lbNodal > dA, tag + ": image leaves the node box (node box bound > d_A)");
                check(dA < dB && dB < lbNodal, tag + ": d_A < d_B < node box bound");
                if (lbPadded > dB) paddedDefeated = true;
            }
            Nodes yA(xa); yA.insert(yA.end(), ya.begin(), ya.end());
            Nodes yB(xb); yB.insert(yB.end(), yb.begin(), yb.end());
            protocol_case(tag, m, yA, 9, yB, 2, P, 0, &sensitive);
            if (worldRank == 0)
                std::printf("%s: d_A = %.6f, d_B = %.6f, node box bound %.6f, padded %.6f\n", tag.c_str(),
                            double(dA), double(dB), double(lbNodal), double(lbPadded));
        }
    int pd = paddedDefeated ? 1 : 0, gpd = 0;
    MPI_Allreduce(&pd, &gpd, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    check(gpd == 1, "cross-rank curved: a case where the 25%-padded node box would have pruned the donor");
}

// The gap rule and the pruning margin use one outward-rounded predicate.
void case_gap_predicate()
{
    if (worldRank != 0) return;
    using namespace exasim::nonmatching;
    dstype cert = 0;
    (void)cert;
#ifndef NONMATCHING_SEARCH_FAST_MATH
    // Strict-IEEE diagnostic (skipped in the fast-math target, where the compiler may
    // reassociate the error-free transformation away).
    // (a) Rounding boundary: upper = 1 + 2^-52, lower = 3 * 2^-54. The exact difference
    // is 1 + 2^-54, but the floating-point subtraction rounds to 1.0. The exact value is
    // s + e with the error-free TwoDiff (s = fl(upper - lower), e > 0 here; long double
    // is not wider than double on every platform). The certified bound must exceed s.
    const dstype upper = 1.0 + std::ldexp(1.0, -52), lower = 3.0 * std::ldexp(1.0, -54);
    const dstype sdiff = upper - lower, bb = sdiff - upper;
    const dstype ediff = (upper - (sdiff - bb)) + (-lower - bb);
    check(sdiff == 1.0 && ediff == std::ldexp(1.0, -54), "gap predicate: float subtraction rounds below the exact gap");
    cert = certified_gap_upper(upper, lower);
    check(cert > sdiff && cert - sdiff >= ediff, "gap predicate: outward-rounded gap bound >= exact gap");
#endif
    // (b) Monotone in cmax and beta: the processor margin bounds every face target.
    long violations = 0, belowFloat = 0;
    for (int i = 0; i < 2000; i++) {
        const dstype cm = std::pow(10.0, -6 + 12 * hv_unit(i)), be = std::pow(10.0, -20 + 15 * hv_unit(i + 7));
        const dstype cq = cm * (1 + hv_unit(i + 13)), bq = be * (1 + hv_unit(i + 17));
        const dstype xa = std::pow(10.0, -3 + 9 * hv_unit(i + 23));
        for (int p = 1; p <= 4; p++) {
            const dstype tm = certified_gap_target_upper(cm, be, xa, p), tq = certified_gap_target_upper(cq, bq, xa, p);
            if (!(tm <= tq)) violations++;
            // The floating-point target used by the branch and bound is below the certified one.
            const dstype tf = static_cast<dstype>(4 * (2 * p + 2)) * std::numeric_limits<dstype>::epsilon() *
                              (cm + xa) + 4 * be;
            if (!(tf <= tm)) belowFloat++;
        }
    }
    check(violations == 0, "gap predicate: target bound monotone in cmax and beta");
    check(belowFloat == 0, "gap predicate: outward-rounded target >= floating-point target");
    // (c) Acceptance is exactly gapUpper <= target; boundary cases.
    const dstype t = certified_gap_target_upper(dstype(1), dstype(0), dstype(1), 1);
    check(certified_gap_accepted(dstype(1) + t / 2, dstype(1), dstype(1), dstype(0), dstype(1), 1),
          "gap predicate: gap well inside the target accepted");
    check(!certified_gap_accepted(dstype(2) * t + dstype(1), dstype(1), dstype(1), dstype(0), dstype(1), 1),
          "gap predicate: gap above the target rejected");
    check(certified_gap_accepted(t, dstype(0), dstype(1), dstype(0), dstype(1), 1) ==
          (certified_gap_upper(t, dstype(0)) <= t), "gap predicate: equality boundary follows the certified bound");
    // (d) Zero inputs; subnormal inputs (strict-IEEE diagnostic: fast math may flush
    // subnormals to zero). Non-finite inputs are outside the production precondition
    // (the predicate is a numerical comparison only); the test-only scan detects them.
    check(certified_gap_target_upper(dstype(0), dstype(0), dstype(0), 1) >= 0,
          "gap predicate: zero inputs");
#ifndef NONMATCHING_SEARCH_FAST_MATH
    const dstype tiny = std::numeric_limits<dstype>::denorm_min();
    check(certified_gap_accepted(tiny, dstype(0), dstype(0), tiny, dstype(0), 1) ==
          (certified_gap_upper(tiny, dstype(0)) <= certified_gap_target_upper(dstype(0), tiny, dstype(0), 1)),
          "gap predicate: subnormal inputs");
#endif
    check(!test_finite(test_from_bits(testInfBits)) && !test_finite(test_from_bits(testNaNBits)) &&
          test_finite(std::numeric_limits<dstype>::max()),
          "gap predicate: the test-only finite scan detects infinity and NaN");
    // (e) The production margin is the same predicate's target for the processor maxima.
    ProcessorBox<dstype> b;
    b.cmax = 3.5; b.beta = 1e-17;
    const dstype x[2] = {-2.0, 7.0};
    check(pruning_margin_upper(x, b, 3) == certified_gap_target_upper(dstype(3.5), dstype(1e-17), dstype(7.0), 3),
          "gap predicate: pruning margin uses the shared target bound");
#ifndef NONMATCHING_SEARCH_FAST_MATH
    std::printf("gap predicate: outward-rounded gap %.17g >= exact 1 + 2^-54 (float difference 1.0); monotone "
                "target, subnormal inputs and the test-only finite scan checked\n", double(cert));
#else
    std::printf("gap predicate (fast-math target): monotone target and the test-only finite scan checked; "
                "TwoDiff and subnormal diagnostics skipped\n");
#endif
}

// Owner-side response validation on crafted responses (no MPI).
void case_response_validation()
{
    if (worldRank != 0) return;
    using namespace exasim::nonmatching;
    std::vector<ProcessorBox<dstype>> boxes(3);
    for (int q = 0; q < 3; q++) {
        boxes[q].nfaces = 2;
        boxes[q].nfe = 4;
        boxes[q].world = 10 + q;
    }
    const std::vector<Int> world = {10, 11, 12};
    struct Rec { Int i[SearchResponseInts]; dstype r[SearchResponseReals]; };
    auto run = [&](const std::vector<std::pair<int, Rec>>& recs, bool select) {
        SearchOwnerState<dstype, Int> st;
        st.init(1, 4);
        st.begin_round();
        st.expect(0, 1);
        std::vector<Int> ri;
        std::vector<dstype> rr;
        std::vector<int> cnt(3, 0);
        for (int q = 0; q < 3; q++)
            for (const auto& pr : recs)
                if (pr.first == q) {
                    ri.insert(ri.end(), pr.second.i, pr.second.i + SearchResponseInts);
                    rr.insert(rr.end(), pr.second.r, pr.second.r + SearchResponseReals);
                    cnt[q]++;
                }
        absorb_responses(st, true, ri, rr, cnt, boxes, world);
        if (select) {
            std::vector<dstype> lo, up;
            std::vector<Int> gl;
            select_returned_candidate(st, Int(0), lo, up, gl);
        }
        return st.error;
    };
    const Rec summary = {{0, 0, 0, -1, -1, -1, 11}, {0.5, 0, 0, 0, 0, 0}};
    const Rec face = {{0, 1, 0, 7, 3, 1, 11}, {0.4, 0.5, 0.45, 0.3, 1.0, 1e-17}};
    Rec badKind = face; badKind.i[1] = 5;
    Rec badRank = face; badRank.i[6] = 12;
    Rec unordered = face; unordered.r[0] = 0.6;
    Rec badFace = face; badFace.i[5] = 9;
    Rec otherBound = summary; otherBound.r[0] = 0.4;
    Rec invalidQuery = summary; invalidQuery.i[2] = 2;
    Rec badPoint = summary; badPoint.i[0] = 3;
    struct Case { const char* name; std::vector<std::pair<int, Rec>> recs; int expect; };
    const std::vector<Case> cases = {
        {"valid", {{1, summary}, {1, face}}, SearchProtocolOk},
        {"missing summary", {{1, face}}, SearchProtocolSummaryCount},
        {"duplicate summary", {{1, summary}, {1, summary}, {1, face}}, SearchProtocolSummaryCount},
        {"unexpected processor", {{1, summary}, {1, face}, {2, summary}}, SearchProtocolUnexpected},
        {"bad kind", {{1, summary}, {1, badKind}}, SearchProtocolBadKind},
        {"bad rank", {{1, summary}, {1, badRank}}, SearchProtocolBadRank},
        {"lower > upper", {{1, summary}, {1, unordered}}, SearchProtocolBadPayload},
        {"local face out of range", {{1, summary}, {1, badFace}}, SearchProtocolBadPayload},
        {"bad point", {{1, badPoint}, {1, summary}, {1, face}}, SearchProtocolBadPoint},
        {"invalid query", {{1, invalidQuery}}, SearchProtocolInvalidQuery},
        {"no candidate", {{1, summary}}, SearchProtocolNoCandidate},
        {"bound mismatch", {{1, otherBound}, {1, face}}, SearchProtocolBoundMismatch},
    };
    for (const Case& c : cases) {
        const int got = run(c.recs, true);
        check(got == c.expect, std::string("response validation: ") + c.name);
    }
    // Floating-point classes are not validated by the production parser (maintainer
    // decision): an injected NaN or infinity payload is detected by the test-only scan.
    long scanned = 0;
    Rec nanLower = face; nanLower.r[0] = test_from_bits(testNaNBits);
    Rec infUpper = face; infUpper.r[1] = test_from_bits(testInfBits);
    check(count_nonfinite(face.r, SearchResponseReals, scanned) == 0 &&
          count_nonfinite(nanLower.r, SearchResponseReals, scanned) == 1 &&
          count_nonfinite(infUpper.r, SearchResponseReals, scanned) == 1,
          "response validation: test-only scan detects injected NaN and infinity payloads");
    std::printf("response validation: %d crafted response sets classified; injected NaN/infinity detected by the "
                "test-only scan\n", int(cases.size()));
}

// Box enclosure: every sample of every face and every coefficient interval lies in the
// processor box; a p = 4 face that leaves its node box is enclosed; volume nodes are not used.
void case_boxes()
{
    if (worldRank != 0) return;
    using namespace exasim::nonmatching;
    long checked = 0;
    for (int elemtype : {0, 1})
        for (int porder = 1; porder <= 4; porder++) {
            MasterData m = make_master(elemtype, porder);
            InterfaceFaceSet<dstype, Int> fs;
            fs.nd = 2; fs.npf = m.npf;
            for (int f = 0; f < 3; f++) {
                fs.rank.push_back(0); fs.model.push_back(1); fs.elem.push_back(f); fs.lface.push_back(0);
                fs.global.push_back(f);
                for (int j = 0; j < m.npf; j++)
                    fs.nodes.push_back(0.3 * f + 0.2 * std::sin(7.0 * m.xpf[j] + f) * (porder > 1 ? 1 : 0));
                for (int j = 0; j < m.npf; j++) fs.nodes.push_back(m.xpf[j] + 0.1 * f);
            }
            const auto pf = prepare_candidate_faces(fs, m.xpf.data(), Int(elemtype), Int(porder));
            const auto box = local_processor_box(pf, Int(0), Int(1), Int(m.nfe));
            check(box.valid, "boxes: valid box");
            std::vector<dstype> psi(m.npf);
            for (int f = 0; f < 3; f++) {
                for (int i = 0; i <= 2000; i++) {
                    const dstype xi[2] = {dstype(i) / 2000, 0};
                    EvaluateFaceShape(psi.data(), xi, m.xpf.data(), m.nd, m.npf, m.elemtype, m.porder);
                    for (int d = 0; d < 2; d++) {
                        dstype v = 0;
                        for (int j = 0; j < m.npf; j++) v += psi[j] * fs.nodes[j + m.npf * d + 2 * m.npf * f];
                        check(v >= box.xmin[d] && v <= box.xmax[d], "boxes: sample inside the processor box");
                        checked++;
                    }
                }
                const int n = porder + 1;
                for (int d = 0; d < 2; d++)
                    for (int k = 0; k < n; k++) {
                        const auto& c = pf.bfaces[f].ci[k + n * d];
                        check(c.lo >= box.xmin[d] && c.hi <= box.xmax[d], "boxes: coefficient interval inside");
                    }
            }
        }
    // p = 4 face leaving its node box (Phase 2 curved adversarial, lebesgue shape).
    MasterData m4 = make_master(1, 4);
    std::vector<dstype> psi(m4.npf), best(m4.npf);
    dstype lmax = -1;
    for (int i = 0; i <= 20000; i++) {
        const dstype xi[2] = {dstype(i) / 20000, 0.0};
        EvaluateFaceShape(psi.data(), xi, m4.xpf.data(), m4.nd, m4.npf, m4.elemtype, m4.porder);
        dstype l = 0;
        for (int j = 0; j < m4.npf; j++) l += std::abs(psi[j]);
        if (l > lmax) { lmax = l; best = psi; }
    }
    Nodes y(2 * m4.npf);
    for (int j = 0; j < m4.npf; j++) { y[j] = best[j] >= 0 ? -1.0 : 1.0; y[j + m4.npf] = m4.xpf[j]; }
    const auto fs4 = one_face(m4, y, 0, 1, 0);
    const auto pf4 = prepare_candidate_faces(fs4, m4.xpf.data(), Int(1), Int(4));
    const auto box4 = local_processor_box(pf4, Int(0), Int(1), Int(m4.nfe));
    dstype xminImage = 1e300;
    for (int i = 0; i <= 20000; i++) {
        const dstype xi[2] = {dstype(i) / 20000, 0.0};
        EvaluateFaceShape(psi.data(), xi, m4.xpf.data(), m4.nd, m4.npf, m4.elemtype, m4.porder);
        dstype v = 0;
        for (int j = 0; j < m4.npf; j++) v += psi[j] * y[j];
        xminImage = std::min(xminImage, v);
    }
    check(xminImage < -1.0, "boxes: the p = 4 face leaves its node box");
    check(box4.xmin[0] <= xminImage, "boxes: the processor box encloses the p = 4 face");
    // Narrow face of a long element: the element reaches x = 10, the face lies at x = 0.
    // A point at x = 5 is inside the element box (bound 0) but 5 away from the face box.
    MasterData m1 = make_master(1, 1);
    std::vector<dstype> fx, fy;
    straight(fx, fy, m1, 0.0);
    Nodes yf(fx); yf.insert(yf.end(), fy.begin(), fy.end());
    const auto pfn = prepare_candidate_faces(one_face(m1, yf, 0, 1, 0), m1.xpf.data(), Int(1), Int(1));
    const auto bn = local_processor_box(pfn, Int(0), Int(1), Int(m1.nfe));
    const dstype Pn[2] = {5.0, 0.5};
    exasim::nonmatching::ProcessorBox<dstype> elementBox = bn;
    elementBox.xmax[0] = 10.0; // the box of the element's nodes
    const dstype lbFace = box_distance_lower(Pn, bn), lbElem = box_distance_lower(Pn, elementBox);
    check(bn.xmax[0] < 1e-12 && lbFace > 4.99 && lbElem == 0,
          "boxes: the face box excludes the volume nodes (an element box would force a query)");
    std::printf("boxes: %ld sample coordinates enclosed (p = 1..4, quads and triangles); p = 4 image min x %.4f "
                "vs box %.4f; narrow face bound %.3f vs element box %.3f\n",
                checked, double(xminImage), double(box4.xmin[0]), double(lbFace), double(lbElem));
}

// Rebuild after motion (2+2): cache equals a fresh setup; comm balance zero.
void case_rebuild()
{
    if (worldSize < 4) return;
    SubWorld sw(4);
    if (sw.comm == MPI_COMM_NULL) return;
    MasterData m = make_master(1, 3);
    ModelMesh a1, a2;
    matching_meshes(a1, a2);
    const int model = worldRank < 2 ? 0 : 1, part = worldRank - 2 * model;
    RankStructs rs, fresh;
    make_rank_structs(rs, model ? a2 : a1, m, row_strip(model ? a2 : a1, part, 2), 1);
    const long long balance0 = nonmatching_comm_balance();
    nonmatchingdata nm, nf;
    commonstruct common;
    check(rebuild_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, common, Int(4), Int(worldRank),
          Int(model * 2), nm), "rebuild: initial setup ran");
    Int before = 0, after = 0;
    for (int d = 0; d < 2; d++) before += nm.dir[d].nsendnb + nm.dir[d].nrecvnb;
    if (model == 1)
        for (Int e = 0; e < rs.mesh.ndims[1]; ++e)
            for (Int j = 0; j < m.npe; ++j) rs.sol.xdg[j + m.npe * (1 + m.nd * e)] += 0.2;
    check(rebuild_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, common, Int(4), Int(worldRank),
          Int(model * 2), nm), "rebuild: motion rebuild ran");
    for (int d = 0; d < 2; d++) after += nm.dir[d].nsendnb + nm.dir[d].nrecvnb;
    Int lb = before, la = after, gb = 0, ga = 0;
    MPI_Allreduce(&lb, &gb, 1, mpi_type<Int>(), MPI_SUM, sw.comm);
    MPI_Allreduce(&la, &ga, 1, mpi_type<Int>(), MPI_SUM, sw.comm);
    check(ga != gb, "rebuild: run-time neighbor lists changed");
    check(setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, Int(4), Int(worldRank),
          Int(model * 2), nf), "rebuild: fresh setup on the moved mesh ran");
    check(cache_snapshot(nm) == cache_snapshot(nf), "rebuild: cache after motion == fresh setup (bitwise)");
    exasim::nonmatching::free_nonmatching_data(nm);
    exasim::nonmatching::free_nonmatching_data(nf);
    CPUFREE(common.requests);
    CPUFREE(common.statuses);
    check(nonmatching_comm_balance() == balance0, "rebuild: comm_Gamma creation/free balance");
    if (worldRank == 0) std::printf("rebuild: neighbor entries %d -> %d, cache equals a fresh setup\n", int(gb), int(ga));
}

// Localized weak scaling: fixed faces per rank (6 model-1 rows, 4 model-2 rows per rank).
void case_scaling()
{
    MasterData q2 = make_master(1, 2);
    if (worldRank == 0)
        std::printf("scaling (ranks per model, points | distributed: faces/point mean max, D4 fan-out max, "
                    "query+response bytes per point, box bytes per rank, count bytes, search time median of 7 | exhaustive: faces/point, "
                    "search time median of 7)\n");
    double prevMeanDist = 0;
    for (int k = 1; 2 * k <= worldSize; k *= 2) {
        ModelMesh s1, s2;
        s1.xs = uniform(0.0, 0.5, 2); s1.ys = uniform(0.0, dstype(k), 6 * k); s1.xiface = 0.5; s1.xfar = 0.0;
        s2.xs = uniform(0.51, 1.0, 2); s2.ys.clear();
        for (int i = 0; i <= 4 * k; i++) s2.ys.push_back(dstype(i) / 4 + 0.013 * std::sin(3.0 * i));
        s2.ys.front() = 0; s2.ys.back() = dstype(k);
        s2.xiface = 0.51; s2.xfar = 1.0;
        const ModeRun r = compare_modes("scaling", s1, s2, q2, k, k, false, 7);
        if (worldRank == 0) {
            const double meanD = double(r.dist.facesSum) / std::max<long long>(1, r.dist.points);
            const double meanE = double(r.exh.facesSum) / std::max<long long>(1, r.exh.points);
            std::printf("scaling %d+%d, %lld points | %.2f %lld, %lld, %.1f, %lld, %lld, %.4f s | %.2f, %.4f s\n",
                        k, k, r.dist.points, meanD, r.dist.maxFaces, r.dist.maxFan,
                        double(r.dist.queryBytes + r.dist.responseBytes) / std::max<long long>(1, r.dist.points),
                        r.dist.boxBytes, r.dist.countBytes, r.searchDist, meanE, r.searchExh);
            check(r.dist.remote == 0 && r.exh.remote > 0, "scaling: only the exhaustive oracle stores remote faces");
            check(r.dist.maxFaces <= 3 * 6, "scaling: distributed faces evaluated per point stay bounded");
            if (k > 1) check(meanE > meanD, "scaling: exhaustive work per point exceeds distributed");
            if (k > 1 && prevMeanDist > 0) check(meanD <= 2 * prevMeanDist, "scaling: distributed work per point does not grow");
            prevMeanDist = meanD;
        }
    }
}

// Inherited limitation (Phase 3): geometry of scale 1e-3 translated by 1e6. Both
// drivers are run without the setup error path and their outcomes are compared.
void case_translated_small()
{
    SubWorld sw(2);
    if (sw.comm == MPI_COMM_NULL) return;
    using namespace exasim::nonmatching;
    MasterData q3 = make_master(1, 3);
    ModelMesh a1, a2;
    matching_meshes(a1, a2);
    const ModelMesh t1 = transformed(a1, 1e-3, 1e6, 1e6), t2 = transformed(a2, 1e-3, 1e6, 1e6);
    const int model = worldRank;
    RankStructs rs;
    make_rank_structs(rs, model ? t2 : t1, q3, [](int) { return true; }, 1);
    const InterfaceFaceSet<dstype, Int> own = collect_own_interface_faces(rs.app, rs.master, rs.mesh, rs.sol,
            Int(worldRank), Int(model));
    DonorSearchResult<dstype, Int> rd, re;
    rd.direction = re.direction = model;
    own_quadrature_points(rd, own, q3.shapfgt.data(), Int(q3.ngf));
    own_quadrature_points(re, own, q3.shapfgt.data(), Int(q3.ngf));
    InterfaceComm<Int> ic;
    create_interface_comm(ic, true, Int(worldRank), Int(2), Int(model), nullptr);
    distributed_donor_search(rd, ic, own, q3.xpf.data(), Int(q3.elemtype), Int(q3.porder), Int(worldRank),
                             Int(model), DonorSearchParams(), nullptr,
                             static_cast<SearchFailures<Int>*>(nullptr), Int(q3.nfe));
    FaceMasterDescriptor<dstype, Int> descriptor;
    descriptor.elemtype = Int(q3.elemtype);
    descriptor.porder = Int(q3.porder);
    descriptor.npf = Int(q3.npf);
    descriptor.xpf = q3.xpf;
    exhaustive_donor_search(re, ic, own, descriptor, Int(model), DonorSearchParams(), nullptr);
    ic.release();
    long failD = 0, failE = 0, diff = 0;
    for (Int p = 0; p < rd.npts; p++) {
        failD += rd.converged[p] == 0;
        failE += re.converged[p] == 0;
        if (rd.converged[p] && re.converged[p] && (rd.f[p] != re.f[p] || rd.dist[p] != re.dist[p])) diff++;
    }
    long loc[3] = {failD, failE, diff}, glob[3];
    MPI_Allreduce(loc, glob, 3, MPI_LONG, MPI_SUM, sw.comm);
    check(glob[2] == 0, "translated small geometry: points converged in both modes agree");
    check(glob[0] <= glob[1], "translated small geometry: distributed fails no more points than exhaustive");
    if (worldRank == 0)
        std::printf("translated small geometry (scale 1e-3, translation 1e6, inherited limitation): failed points "
                    "distributed %ld, exhaustive %ld (of %d)\n", glob[0], glob[1], int(2 * rd.npts));
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (worldSize < 2 || (worldSize & (worldSize - 1)) != 0) {
        if (worldRank == 0) std::printf("nonmatching_search needs 2, 4, 8, ... ranks\n");
        MPI_Finalize();
        return 1;
    }
    case_boxes();
    case_gap_predicate();
    case_response_validation();
    case_equivalence();
    case_nonmembers();
    case_cross_processor_ties();
    case_curved_adversarial_routing();
    case_rebuild();
    case_scaling();
    case_translated_small();
    check(nonmatching_comm_balance() == 0, "comm_Gamma creation/free balance is zero at exit");
    long scannedTotal = 0;
    MPI_Allreduce(&finiteScanned, &scannedTotal, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (worldRank == 0)
        std::printf("test-only finite scans: %ld values (processor boxes, responses, donor tuples, setup caches)%s\n",
                    scannedTotal,
#ifdef NONMATCHING_SEARCH_FAST_MATH
                    ", fast-math build"
#else
                    ""
#endif
        );

    int total = 0;
    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (worldRank == 0)
        std::printf("%s nonmatching_search (%d ranks)\n", total == 0 ? "PASS" : "FAIL", worldSize);
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
