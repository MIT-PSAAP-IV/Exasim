/*
    nonmatchingpreprocess.hpp

    Consolidated host preprocessing for the non-conformal, non-matching
    interface solver (docs/nonmatching/nonmatching_multiphysics.tex).

    When interfacetype = 1 (app.problem[34]), every rank that owns faces of
    its model's coupled interface computes, for each quadrature point x_gl of
    each own face, the donor tuple (r, e, f, a) of eq. (56) by the closest-face
    rule (3), with ties broken by the smallest global face index. It also
    records dist(x_gl, F_f) and the reference coordinate xi of the closest
    point (eq. (21)). From the tuples it forms K_l (eqs. (4), (6)), R_l and
    G_{l,r} (eq. (60)) and fills the owner-side fields of the frozen struct
    (nonmatchinginterface.hpp). The same code runs on both models, so both
    directions are covered: a rank of model 1 searches Gamma^2 for its points,
    a rank of model 2 searches Gamma^1.

    Phase 12 uses prepared polynomial face geometry, Bernstein-enclosed boxes,
    deterministic multi-start projection, and scalable D1-D5 routing on one
    setup-only interface communicator. Multi-start is the sole donor-search
    implementation in production and focused tests.

    Execution policy: this is host-only construction/rebuild code. Its plain
    loops intentionally operate on std::vector workspaces and host MPI
    collectives once per setup; residual-time kernels are Kokkos kernels in
    nonmatchingruntimekernels.hpp. Moving setup to a device requires a separate
    design for device-resident containers, deterministic reductions and
    device-aware MPI, rather than mechanically replacing these loops.

    Component boundaries:
      - nonmatchingbernstein.hpp: reusable polynomial/basis utilities;
      - nonmatchingpreprocessgeometry.hpp: prepared curved-face geometry and projection;
      - nonmatchingpreprocesssearch.hpp: local bounding, screening and face selection;
      - nonmatchingpreprocessmpi.hpp: distributed candidate routing only;
      - this file: Exasim data extraction and D6-D7 setup orchestration.

    Requirements: include after Common/common.h and interfacepartition.hpp.
*/

#ifndef __NONMATCHINGPREPROCESS_HPP
#define __NONMATCHINGPREPROCESS_HPP

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "nonmatchinginterface.hpp"
#include "interfacepartition.hpp"
#include "nonmatchinggeometry.hpp"
#include "nonmatchingruntime.hpp" // [nonmatching] phase 8 transactional host/runtime rebuild
#include "nonmatchingmastercompatibility.hpp" // [nonmatching] phase 10 mixed element families
#include "nonmatchingpreprocessmpi.hpp" // [nonmatching] phase 12 multi-start search and MPI routing

namespace exasim {
namespace nonmatching {

// Parameters of the consolidated Phase-12 donor search.
//
struct DonorSearchParams {
    dstype onFaceRelTol = 1.0e-10;    // diagnostics only: point on its donor face if dist <= onFaceRelTol * h_f
    Int maxProjectionIterations = 100;
    dstype projectionTolerance = 1.0e-13; // step norm in reference-face units (eqs. (21)-(22))
    Int maxPorder = 4;                // supported range, see above
    Int maxInverseIterations = 50;
    Int maxProjectedInverseIterations = 100;
    // Forward-error allowance for a degree <= 4 map (at most 25 nodes), its
    // residual norm, and the small Newton solve. The physical tolerance is
    // this factor times eps times max(|x|_inf, donor-element extent).
    dstype inverseToleranceFactor = 64.0;
    // Fallback neighborhood in dimensionless reference-element coordinates.
    // Unrestricted Newton is attempted first; the fallback retains the best
    // finite projected least-squares point when an exact inverse is unavailable.
    dstype inverseProjectionMargin = 2.0;
};

// Lightweight Phase-12 search API used by focused preprocessing tests. The
// production transaction below uses the same prepared faces and distributed
// search directly so there is only one numerical implementation.
namespace preprocess {
template <class T, class I>
struct SearchParameters {
    T projectionTolerance = static_cast<T>(1.0e-13);
    I maximumProjectionIterations = 100;
};

template <class T, class I, class FaceSet>
inline std::vector<DistributedDonor<T,I>> locate_donors(
        const FaceSet& localFaces, const T* xpf, I faceType, I porder,
        const std::vector<T>& ownerPointsColumnMajor, I worldRank, I model,
        const SearchParameters<T,I>& parameters = SearchParameters<T,I>())
{
#ifndef HAVE_MPI
    (void)localFaces; (void)xpf; (void)faceType; (void)porder;
    (void)ownerPointsColumnMajor; (void)worldRank; (void)model; (void)parameters;
    throw std::runtime_error("non-matching preprocessing requires MPI");
#else
    const I nd = localFaces.nd;
    if ((nd != 2 && nd != 3) || ownerPointsColumnMajor.size() % nd != 0)
        throw std::runtime_error("non-matching preprocessing: invalid owner-point dimensions");
    const I npoints = static_cast<I>(ownerPointsColumnMajor.size() / nd);
    std::vector<T> ownerPoints(static_cast<std::size_t>(npoints) * nd);
    for (I point = 0; point < npoints; ++point)
        for (I d = 0; d < nd; ++d)
            ownerPoints[d + static_cast<std::size_t>(nd) * point] =
                ownerPointsColumnMajor[point + static_cast<std::size_t>(npoints) * d];

    auto communicator = make_interface_communicator<I>(localFaces.size() > 0, worldRank, model);
    if (communicator.comm == MPI_COMM_NULL) return {};
    PreparedFaces<T,I> prepared;
    int localFailure = 0;
    try { prepared = prepare_faces<T,I>(localFaces, xpf, faceType, porder); }
    catch (const std::runtime_error&) { localFailure = 1; }
    int anyFailure = 0;
    MPI_Allreduce(&localFailure, &anyFailure, 1, MPI_INT, MPI_MAX, communicator.comm);
    if (anyFailure) {
        I localRank = localFailure ? worldRank : std::numeric_limits<I>::max();
        I failedRank = 0;
        MPI_Allreduce(&localRank, &failedRank, 1, preprocess_mpi_type<I>(), MPI_MIN,
                      communicator.comm);
        std::ostringstream message;
        message << "non-matching preprocessing: face preparation failed on rank " << failedRank;
        throw std::runtime_error(message.str());
    }
    return distributed_search(communicator, prepared, ownerPoints, worldRank, model,
        parameters.projectionTolerance, parameters.maximumProjectionIterations);
#endif
}
} // namespace preprocess

// Interface faces of one rank (own faces) or of all ranks (gathered).
// Per face f: nodes[j + npf*d + npf*nd*f], face nodes in the element-local
// order of mesh.perm, so that xi refers to the same face parameterization as
// chi_a (Phase 1) and as the face Gauss data of the solver.
template <class T = ::dstype, class I = ::Int>
struct InterfaceFaceSet {
    I nd = 0, npf = 0;
    std::vector<I> rank;     // world rank that owns the face
    std::vector<I> model;    // 0: model 1 (fileoffset == 0), 1: model 2
    std::vector<I> elem;     // local element on the owning rank
    std::vector<I> lface;    // local face number in elem (0..nfe-1)
    std::vector<I> global;   // global face index nfe*elempart[elem] + lface (per model)
    std::vector<T> nodes;    // [npf x nd x nfaces]

    I size() const { return static_cast<I>(rank.size()); }
};

// Result of the donor search for the own faces of one rank. Point p = g + ngf*l
// is quadrature point g of own face l (l in the order of mesh.intfaces).
// Coordinate-like arrays are point-first, as in the frozen header.
template <class T = ::dstype, class I = ::Int>
struct DonorSearchResult {
    I nd = 0, ngf = 0, nfaces = 0, npts = 0;
    I direction = -1;            // 0: own faces on Gamma^1 (model 1), 1: on Gamma^2
    std::vector<I> ownglobal;    // [nfaces] global index of own face l
    std::vector<T> x;            // [npts x nd] x_gl, x[p + npts*d]
    std::vector<I> r;            // [npts] donor rank r_gl (world)                eq. (56)
    std::vector<I> e;            // [npts] local donor element on r                eq. (56)
    std::vector<I> f;            // [npts] global donor face index f_gl = k_gl     eqs. (3), (56)
    std::vector<I> a;            // [npts] local face number of f in e             eq. (56)
    std::vector<T> dist;         // [npts] dist(x_gl, F_f)                         eq. (3)
    std::vector<T> xi;           // [npts x (nd-1)] closest point in the reference face, eq. (21)
    std::vector<T> hface;        // [npts] diagonal of the donor face box (local length scale)
    std::vector<char> converged; // [npts] projection onto the donor face converged
    // Phase 3 inverse-map diagnostics, indexed by coupling direction. These
    // remain outside the frozen run-time data structure.
    I inversePoints[2] = {0, 0};
    I projectedExactInversePoints[2] = {0, 0};
    I projectedInexactInversePoints[2] = {0, 0};
    T maxInverseResidual[2] = {0, 0};
    T maxRelativeInverseResidual[2] = {0, 0};
    T maxReferenceExtrapolation[2] = {0, 0};
};

template <class T, class I>
inline void face_point_range(const DonorSearchResult<T, I>& res, I l, I& begin, I& end)
{
    begin = res.ngf * l;
    end = begin + res.ngf;
}

template <class T, class I>
inline void point_face_gauss(const DonorSearchResult<T, I>& res, I p, I& l, I& g)
{
    l = p / res.ngf;
    g = p - res.ngf * l;
}

// Phase 7: optional counters of the search (tests and reports only; no effect on results).
struct SearchStatistics {
    int member = 0;                    // this rank belongs to comm_Gamma
    int interfaceSize = 0;             // |P_Gamma|
    long long interfaceCollectives = 0;// collectives this rank entered on comm_Gamma
    long long boxBytes = 0;            // D2 bytes received
    long long countBytes = 0;          // Alltoall count bytes (sent + received)
    long long queryBytes = 0;          // D3 + D4 query payload sent
    long long responseBytes = 0;       // D3 + D4 response payload sent
    long long ownerPoints = 0;         // own quadrature points searched
    long long d4Queries = 0;           // D4 (point, processor) queries sent
    long long maxD4FanOut = 0;         // max D4 processors of one point
    long long facesPerPointSum = 0;    // sum over own points of faces evaluated for it
    long long maxFacesPerPoint = 0;    // max faces evaluated for one own point
    long long facesEvaluated = 0;      // faces evaluated by this rank (all received points)
    long long facesPruned = 0;         // Phase 12: faces rejected by the local box screen
    long long projectionStarts = 0;    // Phase 12: projected Gauss--Newton starts
    long long projectionIterations = 0;// Phase 12: projected Gauss--Newton iterations
    long long remoteFaceGeometry = 0;  // remote faces whose node coordinates this rank stores
    double maxGapRatio = 0;            // max observed gap ratio gapUpper / gapTarget of accepted faces
    double searchSeconds = 0;          // wall time of D1-D5 search, excluding D6-D7
};

// Setup failures are raised only after participating interface ranks have
// completed the same collectives and selected one uniform diagnostic.
class UniformSetupError : public std::runtime_error {
public:
    explicit UniformSetupError(const std::string& message)
        : std::runtime_error(message) {}
};

// Own coupled interface faces of this rank, in the order of mesh.intfaces
// (faces with bf == problem[30] over e < ne1, element then local face; see
// collect_local_interface_faces and discretization.cpp getinterfacefaces).
// sol.xdg is npe x ncx x ne.
template <class T, class I>
inline InterfaceFaceSet<T, I> collect_own_interface_faces(const appstructT<T, I>& app,
        const masterstructT<T, I>& master, const meshstructT<T, I>& mesh, const solstructT<T, I>& sol,
        I mpirank, I fileoffset)
{
    InterfaceFaceSet<T, I> faces;
    const I bc = app.problem[30];
    const I nd = master.ndims[0];
    const I npe = master.ndims[5];
    const I npf = master.ndims[6];
    const I ne = mesh.ndims[1];
    const I nfe = mesh.ndims[4];
    const I ncx = app.ndims[AppNdims::ncx];
    faces.nd = nd;
    faces.npf = npf;

    I ne1 = ne;
    if (mesh.elempartpts && exasim::interfacepartition::mesh_nsize(mesh, static_cast<I>(10)) >= 2)
        ne1 = std::min(ne, mesh.elempartpts[0] + mesh.elempartpts[1]);

    bool haveFace = false;
    for (I e = 0; e < ne1 && !haveFace; e++)
        for (I lf = 0; lf < nfe; lf++)
            if (mesh.bf[lf + nfe * e] == bc) haveFace = true;
    if (haveFace && (mesh.elempart == nullptr || mesh.szelempart < ne)) {
        std::ostringstream oss;
        oss << "non-matching donor search: mesh.elempart has " << mesh.szelempart
            << " entries on rank " << mpirank << ", need ne = " << ne;
        error(oss.str());
    }

    const I model = (fileoffset > 0) ? 1 : 0;
    for (I e = 0; e < ne1; e++) {
        for (I lf = 0; lf < nfe; lf++) {
            if (mesh.bf[lf + nfe * e] != bc)
                continue;
            faces.rank.push_back(mpirank);
            faces.model.push_back(model);
            faces.elem.push_back(e);
            faces.lface.push_back(lf);
            faces.global.push_back(nfe * mesh.elempart[e] + lf); // Section 3.1, global face numbering
            const std::size_t off = faces.nodes.size();
            faces.nodes.resize(off + static_cast<std::size_t>(npf * nd));
            for (I j = 0; j < npf; j++) {
                const I node = mesh.perm[j + npf * lf];
                for (I d = 0; d < nd; d++)
                    faces.nodes[off + j + npf * d] = sol.xdg[node + npe * d + npe * ncx * e];
            }
        }
    }
    return faces;
}

// Quadrature points x_gl = sum_i psi_i(g) y_i of the own faces, with
// master.shapfgt (ngf x npf, values first): x[p + npts*d], p = g + ngf*l.
template <class T, class I>
inline void own_quadrature_points(DonorSearchResult<T, I>& res, const InterfaceFaceSet<T, I>& own,
        const T* shapfgt, I ngf)
{
    const I nd = own.nd, npf = own.npf, nf = own.size();
    res.nd = nd;
    res.ngf = ngf;
    res.nfaces = nf;
    const long long nptsWide = static_cast<long long>(ngf) * static_cast<long long>(nf);
    if (ngf < 0 || nf < 0 || nptsWide > std::numeric_limits<I>::max())
        error("non-matching setup: owner point count exceeds integer limits");
    res.npts = static_cast<I>(nptsWide);
    res.ownglobal = own.global;
    res.x.assign(static_cast<std::size_t>(res.npts) * static_cast<std::size_t>(nd), static_cast<T>(0));
    for (I l = 0; l < nf; l++)
        for (I g = 0; g < ngf; g++)
            for (I d = 0; d < nd; d++) {
                T s = 0;
                for (I j = 0; j < npf; j++)
                    s += shapfgt[g + ngf * j] * own.nodes[static_cast<std::size_t>(j + npf * d + npf * nd * l)];
                res.x[static_cast<std::size_t>(g + ngf * l) +
                      static_cast<std::size_t>(res.npts) * d] = s;
            }
}

template <class I>
inline void copy_to_host_array(I*& ptr, const std::vector<I>& v)
{
    if (ptr) CPUFREE(ptr);
    ptr = nullptr;
    if (!v.empty()) {
        TemplateMalloc(&ptr, static_cast<I>(v.size()), 0);
        std::copy(v.begin(), v.end(), ptr);
    }
}

template <class T>
inline void copy_to_host_scalar_array(T*& ptr, const std::vector<T>& v)
{
    if (ptr) CPUFREE(ptr);
    ptr = nullptr;
    if (!v.empty()) {
        TemplateMalloc(&ptr, static_cast<Int>(v.size()), 0);
        std::copy(v.begin(), v.end(), ptr);
    }
}

// Owner side of the frozen struct from the donor tuples: K_l, R_l and G_{l,r}
// (eqs. (4), (6), (60)) and the receive lists (eq. (62)): recvnb in increasing
// donor rank; for each donor, the own faces l with that rank in R_l, ordered
// by increasing own global face index (list order L_pr). ownface[l] = l
// (index into mesh.intfaces). Host arrays.
template <class T, class I>
inline void build_owner_side(nonmatchinginterfaceT<T, I>& nmi, const DonorSearchResult<T, I>& res)
{
    const I nf = res.nfaces;
    nmi.nownfaces = nf;
    nmi.isowner = (nf > 0) ? 1 : 0;

    std::vector<I> ownface(static_cast<std::size_t>(nf));
    for (I l = 0; l < nf; l++) ownface[l] = l;
    copy_to_host_array(nmi.ownface, ownface);
    copy_to_host_array(nmi.ownfaceglobal, res.ownglobal);

    // R_l per face, eq. (60).
    std::vector<std::vector<I>> R(static_cast<std::size_t>(nf));
    std::vector<I> allranks;
    for (I l = 0; l < nf; l++) {
        I begin = 0, end = 0;
        face_point_range(res, l, begin, end);
        for (I p = begin; p < end; p++) R[l].push_back(res.r[p]);
        std::sort(R[l].begin(), R[l].end());
        R[l].erase(std::unique(R[l].begin(), R[l].end()), R[l].end());
        allranks.insert(allranks.end(), R[l].begin(), R[l].end());
    }
    std::sort(allranks.begin(), allranks.end());
    allranks.erase(std::unique(allranks.begin(), allranks.end()), allranks.end());

    // Own faces in increasing global index (list order of eq. (62)).
    std::vector<I> order(static_cast<std::size_t>(nf));
    for (I l = 0; l < nf; l++) order[l] = l;
    std::sort(order.begin(), order.end(),
            [&](I u, I v) { return res.ownglobal[u] < res.ownglobal[v]; });

    std::vector<I> recvcount, recvface;
    for (I rnk : allranks) {
        I count = 0;
        for (I l : order)
            if (std::binary_search(R[l].begin(), R[l].end(), rnk)) {
                recvface.push_back(l);
                count++;
            }
        recvcount.push_back(count);
    }
    nmi.nrecvnb = static_cast<I>(allranks.size());
    copy_to_host_array(nmi.recvnb, allranks);
    copy_to_host_array(nmi.recvcount, recvcount);
    copy_to_host_array(nmi.recvface, recvface);
}

// |K_l| (eqs. (4), (6)) and |R_l| (eq. (60)) of own face l.
template <class T, class I>
inline void face_set_sizes(I& nK, I& nR, const DonorSearchResult<T, I>& res, I l)
{
    std::vector<I> K, R;
    I begin = 0, end = 0;
    face_point_range(res, l, begin, end);
    for (I p = begin; p < end; p++) {
        K.push_back(res.f[p]);
        R.push_back(res.r[p]);
    }
    std::sort(K.begin(), K.end());
    std::sort(R.begin(), R.end());
    nK = static_cast<I>(std::unique(K.begin(), K.end()) - K.begin());
    nR = static_cast<I>(std::unique(R.begin(), R.end()) - R.begin());
}

template <class T, class I>
inline void free_nonmatching_interface(nonmatchinginterfaceT<T, I>& nmi)
{
    CPUFREE(nmi.ownface);
    CPUFREE(nmi.ownfaceglobal);
    CPUFREE(nmi.recvnb);
    CPUFREE(nmi.recvcount);
    CPUFREE(nmi.recvface);
    CPUFREE(nmi.pe);
    CPUFREE(nmi.pa);
    CPUFREE(nmi.pf);
    CPUFREE(nmi.pslot);
    CPUFREE(nmi.pg);
    CPUFREE(nmi.px);
    CPUFREE(nmi.pnl);
    CPUFREE(nmi.pwJ);
    CPUFREE(nmi.pxi);
    CPUFREE(nmi.pzeta);
    CPUFREE(nmi.pphi);
    CPUFREE(nmi.pphibar);
    CPUFREE(nmi.ppsi);
    CPUFREE(nmi.pd);
    CPUFREE(nmi.pdphibar);
    CPUFREE(nmi.slotrank);
    CPUFREE(nmi.slotface);
    CPUFREE(nmi.slotptr);
    CPUFREE(nmi.pairslot);
    CPUFREE(nmi.pairelem);
    CPUFREE(nmi.pairptr);
    CPUFREE(nmi.sendnb);
    CPUFREE(nmi.sendcount);
    CPUFREE(nmi.Ri);
    CPUFREE(nmi.Hi);
    CPUFREE(nmi.sendbuf);
    CPUFREE(nmi.recvbuf);
    nmi = nonmatchinginterfaceT<T, I>();
}

// Releases all host arrays allocated by the Phase 2-3 setup.
template <class T, class I>
inline void free_nonmatching_data(nonmatchingdataT<T, I>& nm)
{
    for (int k = 0; k < 2; k++)
        free_nonmatching_interface(nm.dir[k]);
    nm.interfacetype = 0; // [nonmatching] leave a fully inactive cache after release
}

#ifdef HAVE_MPI

// Recompute the owner face geometry needed by Step D6.  This is the same
// node ordering and formulas as ElemFaceGeomBlock/FaceGeom2D/FaceGeom3D, but it runs
// during cpuInit, before sol.elemfaceg is constructed in compGeometry.
template <class T, class I>
inline void owner_point_geometry(std::vector<T>& wJ, std::vector<T>& normal,
        const DonorSearchResult<T, I>& res, const InterfaceFaceSet<T, I>& own,
        const T* shapfgt, const T* gwf, std::string* failure = nullptr)
{
    if ((res.nd != 2 && res.nd != 3) || own.nd != res.nd ||
        shapfgt == nullptr || gwf == nullptr)
        error("non-matching setup: owner geometry requires nd = 2 or 3");
    const I nd = res.nd;
    const I npf = own.npf, ngf = res.ngf, npts = res.npts;
    wJ.assign(static_cast<std::size_t>(npts), static_cast<T>(0));
    normal.assign(static_cast<std::size_t>(npts) * nd, static_cast<T>(0));
    for (I l = 0; l < res.nfaces; l++)
        for (I g = 0; g < ngf; g++) {
            T tangent[6] = {0, 0, 0, 0, 0, 0};
            for (I j = 0; j < npf; j++) {
                const std::size_t off = static_cast<std::size_t>(npf) * nd * l;
                for (I k = 0; k < nd - 1; ++k) {
                    const T dpsi = shapfgt[(k + 1) * ngf * npf + g + ngf * j];
                    for (I d = 0; d < nd; ++d)
                        tangent[d * (nd - 1) + k] += dpsi * own.nodes[off + j + npf * d];
                }
            }
            T unscaled[3] = {0, 0, 0};
            if (nd == 2) {
                unscaled[0] = tangent[1];
                unscaled[1] = -tangent[0];
            }
            else {
                unscaled[0] = tangent[2] * tangent[5] - tangent[4] * tangent[3];
                unscaled[1] = tangent[4] * tangent[1] - tangent[0] * tangent[5];
                unscaled[2] = tangent[0] * tangent[3] - tangent[2] * tangent[1];
            }
            T jac2 = 0;
            for (I d = 0; d < nd; ++d) jac2 += unscaled[d] * unscaled[d];
            const T jac = std::sqrt(jac2);
            if (!(jac > 0)) {
                // D6 callers pass `failure` so the condition can be reported
                // uniformly on comm_Gamma instead of aborting from one rank.
                if (!failure) error("non-matching setup: singular owner face geometry");
                std::ostringstream oss;
                oss << "singular owner face geometry at owner face " << res.ownglobal[l]
                    << ", quadrature point " << g;
                *failure = oss.str();
                return;
            }
            const I p = g + ngf * l;
            wJ[p] = gwf[g] * jac;
            for (I d = 0; d < nd; ++d)
                normal[p + static_cast<std::size_t>(npts) * d] = unscaled[d] / jac;
        }
}

template <class I>
inline I checked_setup_size(std::initializer_list<I> factors, const char* what)
{
    long long value = 1;
    for (I factor : factors) {
        if (factor < 0 || (factor != 0 && value > std::numeric_limits<int>::max() / factor)) {
            std::ostringstream oss;
            oss << "non-matching setup: " << what << " exceeds integer allocation limits";
            error(oss.str());
        }
        value *= static_cast<long long>(factor);
    }
    return static_cast<I>(value);
}

template <class I>
inline std::size_t checked_size_product(std::initializer_list<I> factors,
        const char* what, bool& failure, std::string& message)
{
    std::size_t value = 1;
    for (I factor : factors) {
        if (factor < 0 || (factor != 0 && value >
                std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(factor))) {
            if (!failure) {
                std::ostringstream oss;
                oss << "non-matching setup: " << what << " exceeds size limits";
                message = oss.str();
            }
            failure = true;
            return 0;
        }
        value *= static_cast<std::size_t>(factor);
    }
    return value;
}

template <class I>
inline I checked_size_narrow(std::size_t value, const char* what,
        bool& failure, std::string& message)
{
    if (value > static_cast<std::size_t>(std::numeric_limits<I>::max())) {
        if (!failure) {
            std::ostringstream oss;
            oss << "non-matching setup: " << what << " exceeds index limits";
            message = oss.str();
        }
        failure = true;
        return 0;
    }
    return static_cast<I>(value);
}

template <class T, class I>
inline void allocate_zeroed(T*& ptr, I& storedSize, I size)
{
    if (ptr) CPUFREE(ptr);
    ptr = nullptr;
    storedSize = size;
    if (size > 0) {
        TemplateMalloc(&ptr, size, 0);
        std::fill(ptr, ptr + size, static_cast<T>(0));
    }
}

// Physical scale for the inverse-map forward error.  |x| accounts for
// cancellation on translated meshes, while the donor-element extent keeps the
// tolerance relative on meshes centered near the origin.
template <class T, class I>
inline T inverse_map_scale(const T* x, const T* elementNodes, I nd, I npe)
{
    T scale = 0;
    for (I d = 0; d < nd; ++d) {
        T xmin = elementNodes[static_cast<std::size_t>(npe) * d];
        T xmax = xmin;
        scale = std::max(scale, std::abs(x[d]));
        for (I j = 1; j < npe; ++j) {
            const T value = elementNodes[j + static_cast<std::size_t>(npe) * d];
            xmin = std::min(xmin, value);
            xmax = std::max(xmax, value);
        }
        scale = std::max(scale, xmax - xmin);
    }
    return std::max(scale, std::numeric_limits<T>::min());
}

// [nonmatching] Uniform reporting of a rank-local D6-D7 failure. Every
// comm_Gamma member calls this at the same point of the collective sequence.
// If any rank recorded a failure, the lowest failing world rank is selected,
// its diagnostic is broadcast, and every member raises the same
// UniformSetupError. A rank's own failure is the first one it met in its
// deterministic (sorted) processing order.
template <class I>
inline void raise_uniform_setup_failure(const preprocess::InterfaceCommunicator<I>& ic, I mpirank,
        bool localFailure, const std::string& localMessage)
{
    int local = localFailure ? 1 : 0, any = 0;
    MPI_Allreduce(&local, &any, 1, MPI_INT, MPI_MAX, ic.comm);
    ic.collective();
    if (!any) return;
    const I none = std::numeric_limits<I>::max();
    const I mine = localFailure ? mpirank : none;
    I failedRank = none;
    MPI_Allreduce(&mine, &failedRank, 1, mpi_type<I>(), MPI_MIN, ic.comm);
    ic.collective();
    if (failedRank < 0 || failedRank >= static_cast<I>(ic.worldToComm.size()) ||
        ic.worldToComm[failedRank] < 0)
        error("non-matching setup: uniform failure protocol selected an invalid rank");
    const int root = ic.worldToComm[failedRank];
    int length = mpirank == failedRank ? static_cast<int>(localMessage.size()) : 0;
    MPI_Bcast(&length, 1, MPI_INT, root, ic.comm);
    ic.collective();
    std::string message(static_cast<std::size_t>(length), ' ');
    if (mpirank == failedRank) message = localMessage;
    if (length > 0) MPI_Bcast(&message[0], length, MPI_CHAR, root, ic.comm);
    ic.collective();
    throw UniformSetupError(message);
}

template <class T, class I>
inline void verify_slot_lists_and_extents(
        const preprocess::InterfaceCommunicator<I>& ic,
        const nonmatchinginterfaceT<T, I>& donor,
        const nonmatchinginterfaceT<T, I>& owner,
        const NonmatchingDirectionContract<I>& contract,
        I advertisedExtent, I localExtent)
{
    // Shared-space directions exchange slot counts only.  Independent-space
    // directions additionally exchange the row-block width so that a short
    // MPI receive cannot silently hide an owner/donor contract mismatch.
    const bool independentSpaces = uses_independent_face_spaces(contract);
    const int width = independentSpaces ? 2 : 1;
    const int interfaceSize = ic.size;
    const I mpirank = ic.world[ic.rank];
    bool localFailure = false;
    std::string localFailureMessage;
    auto record_failure = [&](const char* message) {
        if (!localFailure) {
            localFailure = true;
            localFailureMessage = message;
        }
    };

    std::vector<int> slotSendPayload(
            static_cast<std::size_t>(width) * interfaceSize, 0);
    for (I n = 0; n < donor.nsendnb; ++n) {
        const I peer = donor.sendnb[n];
        const int q = peer >= 0 && peer < static_cast<I>(ic.worldToComm.size())
                ? ic.worldToComm[peer] : -1;
        if (q < 0 || donor.sendcount[n] < 0 ||
                donor.sendcount[n] > static_cast<I>(std::numeric_limits<int>::max())) {
            record_failure("non-matching setup: donor sendcount does not match owner recvcount");
            continue;
        }
        slotSendPayload[static_cast<std::size_t>(width) * q] =
                static_cast<int>(donor.sendcount[n]);
        if (independentSpaces && donor.sendcount[n] > 0) {
            if (advertisedExtent < 0 ||
                    advertisedExtent > static_cast<I>(std::numeric_limits<int>::max()))
                record_failure("non-matching setup: donor block extent does not match owner block extent");
            else
                slotSendPayload[static_cast<std::size_t>(width) * q + 1] =
                        static_cast<int>(advertisedExtent);
        }
    }
    std::vector<int> slotReceivePayload(
            static_cast<std::size_t>(width) * interfaceSize, 0);
    MPI_Alltoall(slotSendPayload.data(), width, MPI_INT,
            slotReceivePayload.data(), width, MPI_INT, ic.comm);
    ic.collective();

    std::vector<int> slotSends(static_cast<std::size_t>(interfaceSize), 0);
    std::vector<int> slotReceives(static_cast<std::size_t>(interfaceSize), 0);
    std::size_t nslotSendWide = 0;
    std::size_t nslotRecvWide = 0;
    for (int q = 0; q < interfaceSize; ++q) {
        slotSends[q] = slotSendPayload[static_cast<std::size_t>(width) * q];
        slotReceives[q] = slotReceivePayload[static_cast<std::size_t>(width) * q];
        I expected = 0;
        for (I n = 0; n < owner.nrecvnb; ++n)
            if (owner.recvnb[n] == ic.world[q]) expected = owner.recvcount[n];
        if (expected < 0 || expected > static_cast<I>(std::numeric_limits<int>::max()) ||
                slotReceives[q] != static_cast<int>(expected))
            record_failure("non-matching setup: donor sendcount does not match owner recvcount");
        if (independentSpaces) {
            const int receivedExtent =
                    slotReceivePayload[static_cast<std::size_t>(width) * q + 1];
            const I expectedExtent = expected > 0 ? localExtent : 0;
            if (localExtent < 0 || expectedExtent > static_cast<I>(std::numeric_limits<int>::max()) ||
                    receivedExtent != static_cast<int>(expectedExtent))
                record_failure("non-matching setup: donor block extent does not match owner block extent");
        }
        if (slotSends[q] < 0 || nslotSendWide >
                static_cast<std::size_t>(std::numeric_limits<int>::max() - slotSends[q]))
            record_failure("non-matching setup: donor slot list exceeds MPI integer limits");
        else
            nslotSendWide += static_cast<std::size_t>(slotSends[q]);
        if (slotReceives[q] < 0 || nslotRecvWide >
                static_cast<std::size_t>(std::numeric_limits<int>::max() - slotReceives[q]))
            record_failure("non-matching setup: owner slot list exceeds MPI integer limits");
        else
            nslotRecvWide += static_cast<std::size_t>(slotReceives[q]);
    }
    raise_uniform_setup_failure(ic, mpirank, localFailure, localFailureMessage);

    auto make_displacements = [](const std::vector<int>& counts,
            std::vector<int>& displacements) {
        int extent = 0;
        displacements.resize(counts.size());
        for (std::size_t q = 0; q < counts.size(); ++q) {
            displacements[q] = extent;
            extent += counts[q];
        }
        return extent;
    };
    std::vector<int> slotSendDisp, slotRecvDisp;
    const int nslotSend = make_displacements(slotSends, slotSendDisp);
    const int nslotRecv = make_displacements(slotReceives, slotRecvDisp);
    std::vector<I> slotFaceSend(static_cast<std::size_t>(nslotSend));
    std::vector<I> slotFaceRecv(static_cast<std::size_t>(nslotRecv));
    std::vector<int> slotCursor = slotSendDisp;
    for (I slot = 0; slot < donor.nslots; ++slot) {
        const int q = ic.worldToComm[donor.slotrank[slot]];
        slotFaceSend[static_cast<std::size_t>(slotCursor[q]++)] = donor.slotface[slot];
    }
    MPI_Alltoallv(slotFaceSend.data(), slotSends.data(), slotSendDisp.data(), mpi_type<I>(),
            slotFaceRecv.data(), slotReceives.data(), slotRecvDisp.data(), mpi_type<I>(), ic.comm);
    ic.collective();

    for (int q = 0; q < interfaceSize; ++q) {
        I ownerOffset = 0;
        bool found = false;
        for (I n = 0; n < owner.nrecvnb; ++n) {
            if (owner.recvnb[n] == ic.world[q]) {
                found = true;
                break;
            }
            ownerOffset += owner.recvcount[n];
        }
        if (!found && slotReceives[q] != 0)
            record_failure("non-matching setup: received an unexpected donor slot list");
        for (int j = 0; found && j < slotReceives[q]; ++j) {
            const I ownFace = owner.recvface[ownerOffset + static_cast<I>(j)];
            if (ownFace < 0 || ownFace >= owner.nownfaces ||
                    slotFaceRecv[slotRecvDisp[q] + j] != owner.ownfaceglobal[ownFace])
                record_failure("non-matching setup: donor slotface list does not match owner recvface order");
        }
    }
    raise_uniform_setup_failure(ic, mpirank, localFailure, localFailureMessage);
}

// Distribute owner quadrature records and build the donor/owner slot lists.
// The owner-to-donor record contains the mapped face coordinate, so the donor
// evaluates its basis directly and does not repeat closest-face projection.
template <class T, class I>
inline void distribute_setup_data(nonmatchingdataT<T, I>& nm,
        DonorSearchResult<T, I>& res, const InterfaceFaceSet<T, I>& own,
        appstructT<T, I>& app, masterstructT<T, I>& master,
        meshstructT<T, I>& mesh, solstructT<T, I>& sol,
        I mpiprocs, I mpirank, I ownDirection, const DonorSearchParams& prm,
        const preprocess::InterfaceCommunicator<I>& ic, OwnerTestData<T, I>* ownerTests = nullptr)
{
    const I nd = res.nd, ndf = nd - 1, npe = master.ndims[5], npf = master.ndims[6];
    const I nfe = mesh.ndims[4], elemtype = master.ndims[1], porder = master.ndims[3];
    const I ncx = app.ndims[AppNdims::ncx];
    const I donorDirection = 1 - ownDirection;
    // Rank-local failures are recorded (first one wins) and reported by
    // raise_uniform_setup_failure at fixed points of the collective sequence.
    bool localFailure = false;
    std::string localFailureMessage;
    auto record_failure = [&](const std::string& message) {
        if (!localFailure) { localFailure = true; localFailureMessage = message; }
    };
    SimplexNodalBasis2D simplexBasis2D;
    SimplexNodalBasis3D simplexBasis3D;
    PolynomialNodalBasis tensorBasis3D;
    const PolynomialNodalBasis* simplexBasisPtr = nullptr;
    if (UsesPolynomialSimplexBasis(nd, elemtype) || (nd == 3 && elemtype == 1)) {
        try {
            if (nd == 2) {
                simplexBasis2D.initialize(master.xpe, npe, porder);
                simplexBasisPtr = &simplexBasis2D;
            }
            else if (elemtype == 0) {
                simplexBasis3D.initialize(master.xpe, npe, porder);
                simplexBasisPtr = &simplexBasis3D;
            }
            else {
                tensorBasis3D.initialize(master.xpe, npe, porder, 3, 1);
                simplexBasisPtr = &tensorBasis3D;
            }
        }
        catch (const std::runtime_error& exception) {
            std::ostringstream oss;
            oss << "non-matching setup: volume nodal basis initialization failed on rank "
                << mpirank << " (direction " << donorDirection << ", elemtype "
                << elemtype << ", porder " << porder << "): " << exception.what();
            record_failure(oss.str());
        }
    }
    SurfaceNodalBasis surfaceBasis;
    const SurfaceNodalBasis* surfaceBasisPtr = nullptr;
    if (nd == 3) {
        try {
            surfaceBasis.initialize(master.xpf, npf, porder, elemtype);
            surfaceBasisPtr = &surfaceBasis;
        }
        catch (const std::runtime_error& exception) {
            std::ostringstream oss;
            oss << "non-matching setup: surface nodal basis initialization failed on rank "
                << mpirank << " (direction " << donorDirection << "): " << exception.what();
            record_failure(oss.str());
        }
    }

    std::vector<T> ownerWJ, ownerNormal;
    {
        std::string ownerFailure;
        owner_point_geometry(ownerWJ, ownerNormal, res, own, master.shapfgt, master.gwf,
                &ownerFailure);
        if (!ownerFailure.empty()) {
            std::ostringstream oss;
            oss << "non-matching setup: " << ownerFailure << " on rank " << mpirank
                << " (direction " << ownDirection << ", Gamma^" << ownDirection + 1 << ")";
            record_failure(oss.str());
        }
    }
    const bool ownerTestsRequired = ownerTests && ownerTests->dir[donorDirection].enabled;
    NonmatchingDirectionContract<I> sendContract;
    NonmatchingDirectionContract<I> recvContract;
    if (ownerTests) {
        sendContract = ownerTests->dir[ownDirection].contract;
        recvContract = ownerTests->dir[donorDirection].contract;
    }
    else {
        sendContract.ownerFaceNodes = recvContract.ownerFaceNodes = npf;
        sendContract.ownerFaceQuadraturePoints =
                recvContract.ownerFaceQuadraturePoints = master.ndims[8];
        sendContract.donorVolumeNodes = recvContract.donorVolumeNodes = npe;
        sendContract.donorFaceNodes = recvContract.donorFaceNodes = npf;
        sendContract.donorFacesPerElement =
                recvContract.donorFacesPerElement = nfe;
    }
    const bool independentSpaces = uses_independent_face_spaces(recvContract);
    const I sendOwnerNpf = independentSpaces ? sendContract.ownerFaceNodes : npf;
    const I recvOwnerNpf = independentSpaces ? recvContract.ownerFaceNodes : npf;
    const I recvOwnerNgf = independentSpaces
            ? recvContract.ownerFaceQuadraturePoints : master.ndims[8];
    if (independentSpaces &&
            (sendOwnerNpf != npf || recvOwnerNpf <= 0 || recvOwnerNgf <= 0))
        record_failure("non-matching setup: mixed owner role dimensions are inconsistent");

    // The caller supplies one communicator containing exactly the ranks that
    // own or donate points on this interface. Nonmembers do not enter here.
    if (ic.comm == MPI_COMM_NULL)
        error("non-matching setup: D6-D7 called without the interface communicator");
    const MPI_Comm interfaceComm = ic.comm;
    const int interfaceSize = ic.size;
    const std::vector<I>& interfaceWorld = ic.world;
    const std::vector<int>& worldToInterface = ic.worldToComm;
    if (static_cast<I>(worldToInterface.size()) != mpiprocs)
        error("non-matching setup: interface rank map has the wrong size");
    // Shared-space records retain the established synchronization point.
    // Independent-space records defer it until variable message extents exist.
    if (!independentSpaces)
        raise_uniform_setup_failure(ic, mpirank, localFailure, localFailureMessage);
    std::vector<int> sendPoints(static_cast<std::size_t>(interfaceSize), 0);
    for (I p = 0; p < res.npts; p++) {
        if (res.r[p] < 0 || res.r[p] >= mpiprocs || worldToInterface[res.r[p]] < 0) {
            if (!independentSpaces)
                error("non-matching setup: donor rank does not own interface faces");
            record_failure("non-matching setup: donor rank does not own interface faces");
            continue;
        }
        int& count = sendPoints[worldToInterface[res.r[p]]];
        if (count == std::numeric_limits<int>::max()) {
            if (!independentSpaces)
                error("non-matching setup: MPI point count exceeds integer limits");
            record_failure("non-matching setup: MPI point count exceeds integer limits");
        }
        else
            count++;
    }
    std::vector<int> recvPoints(static_cast<std::size_t>(interfaceSize), 0);
    MPI_Alltoall(sendPoints.data(), 1, MPI_INT, recvPoints.data(), 1, MPI_INT, interfaceComm);
    ic.collective();

    auto make_displacements = [&](const std::vector<int>& counts, std::vector<int>& displs) {
        long long extent = 0;
        displs.resize(counts.size());
        for (std::size_t q = 0; q < counts.size(); q++) {
            if (counts[q] < 0 || extent > std::numeric_limits<int>::max() - counts[q]) {
                if (!independentSpaces)
                    error("non-matching setup: MPI point counts exceed integer limits");
                record_failure("non-matching setup: MPI point counts exceed integer limits");
                std::fill(displs.begin(), displs.end(), 0);
                return 0;
            }
            displs[q] = static_cast<int>(extent);
            extent += counts[q];
        }
        return static_cast<int>(extent);
    };
    std::vector<int> sendDisp, recvDisp;
    const int nsend = make_displacements(sendPoints, sendDisp);
    const int nrecv = make_displacements(recvPoints, recvDisp);

    // owner rank, owner face, g, donor e, donor a, donor f
    const int nmeta = 6;
    // x, n, wJ, xi: nd + nd + 1 + (nd-1), followed by optional owner test values.
    const int sendNreal = 3 * static_cast<int>(nd) +
                          (ownerTestsRequired ? static_cast<int>(sendOwnerNpf) : 0);
    const int recvNreal = 3 * static_cast<int>(nd) +
                          (ownerTestsRequired ? static_cast<int>(recvOwnerNpf) : 0);
    auto scaled_mpi_int = [&](int value, int scale, const char* what) {
        const long long result = static_cast<long long>(value) * scale;
        if (value < 0 || scale < 0 || result > std::numeric_limits<int>::max()) {
            std::ostringstream oss;
            oss << "non-matching setup: " << what << " exceeds MPI integer limits";
            if (!independentSpaces) error(oss.str());
            record_failure(oss.str());
            return 0;
        }
        return static_cast<int>(result);
    };
    const int nsendMeta = scaled_mpi_int(nsend, nmeta, "metadata send extent");
    const int nsendReal = scaled_mpi_int(nsend, sendNreal, "coordinate send extent");
    const int nrecvMeta = scaled_mpi_int(nrecv, nmeta, "metadata receive extent");
    const int nrecvReal = scaled_mpi_int(nrecv, recvNreal, "coordinate receive extent");
    std::vector<I> sendMeta(static_cast<std::size_t>(nsendMeta));
    std::vector<T> sendReal(static_cast<std::size_t>(nsendReal));
    // All local preparation and message extents are valid before any
    // variable-size exchange or buffer write.
    if (independentSpaces)
        raise_uniform_setup_failure(ic, mpirank, localFailure, localFailureMessage);
    std::vector<int> cursor = sendDisp;
    for (I p = 0; p < res.npts; p++) {
        I l = 0, g = 0;
        point_face_gauss(res, p, l, g);
        const int q = worldToInterface[res.r[p]];
        const int k = cursor[q]++;
        I* im = &sendMeta[static_cast<std::size_t>(nmeta * k)];
        im[0] = mpirank; im[1] = res.ownglobal[l]; im[2] = g;
        im[3] = res.e[p]; im[4] = res.a[p]; im[5] = res.f[p];
        T* rm = &sendReal[static_cast<std::size_t>(sendNreal) * k];
        for (I d = 0; d < nd; d++)
            rm[d] = res.x[static_cast<std::size_t>(p) + static_cast<std::size_t>(res.npts) * d];
        for (I d = 0; d < nd; d++)
            rm[nd + d] = ownerNormal[static_cast<std::size_t>(p) + static_cast<std::size_t>(res.npts) * d];
        rm[2 * nd] = ownerWJ[p];
        for (I d = 0; d < ndf; d++)
            rm[2 * nd + 1 + d] = res.xi[static_cast<std::size_t>(p) +
                                        static_cast<std::size_t>(res.npts) * d];
        if (ownerTestsRequired) {
            T* xm = &rm[3 * nd];
            for (I i = 0; i < sendOwnerNpf; i++)
                xm[i] = master.shapfgt[g + static_cast<std::size_t>(res.ngf) * i];
        }
    }

    std::vector<int> smc(interfaceSize), smd(interfaceSize), rmc(interfaceSize), rmd(interfaceSize);
    std::vector<int> src(interfaceSize), srd(interfaceSize), rrc(interfaceSize), rrd(interfaceSize);
    for (int q = 0; q < interfaceSize; q++) {
        smc[q] = scaled_mpi_int(sendPoints[q], nmeta, "metadata send count");
        smd[q] = scaled_mpi_int(sendDisp[q], nmeta, "metadata send displacement");
        rmc[q] = scaled_mpi_int(recvPoints[q], nmeta, "metadata receive count");
        rmd[q] = scaled_mpi_int(recvDisp[q], nmeta, "metadata receive displacement");
        src[q] = scaled_mpi_int(sendPoints[q], sendNreal, "coordinate send count");
        srd[q] = scaled_mpi_int(sendDisp[q], sendNreal, "coordinate send displacement");
        rrc[q] = scaled_mpi_int(recvPoints[q], recvNreal, "coordinate receive count");
        rrd[q] = scaled_mpi_int(recvDisp[q], recvNreal, "coordinate receive displacement");
    }
    std::vector<I> recvMeta(static_cast<std::size_t>(nrecvMeta));
    std::vector<T> recvReal(static_cast<std::size_t>(nrecvReal));
    MPI_Alltoallv(sendMeta.data(), smc.data(), smd.data(), mpi_type<I>(),
            recvMeta.data(), rmc.data(), rmd.data(), mpi_type<I>(), interfaceComm);
    ic.collective();
    MPI_Alltoallv(sendReal.data(), src.data(), srd.data(), mpi_type<T>(),
            recvReal.data(), rrc.data(), rrd.data(), mpi_type<T>(), interfaceComm);
    ic.collective();

    std::vector<I> order(static_cast<std::size_t>(nrecv));
    for (int p = 0; p < nrecv; p++) order[p] = static_cast<I>(p);
    std::sort(order.begin(), order.end(), [&](I u, I v) {
        const I* a = &recvMeta[static_cast<std::size_t>(nmeta * u)];
        const I* b = &recvMeta[static_cast<std::size_t>(nmeta * v)];
        return std::tie(a[0], a[1], a[3], a[2], a[5]) <
               std::tie(b[0], b[1], b[3], b[2], b[5]);
    });

    nonmatchinginterfaceT<T, I>& donor = nm.dir[donorDirection];
    nonmatchinginterfaceT<T, I>& owner = nm.dir[ownDirection];
    donor.ncu = app.ndims[AppNdims::ncu];
    donor.ncq = app.ndims[AppNdims::ncq];
    donor.npts = static_cast<I>(nrecv);
    donor.isdonor = nrecv > 0 ? 1 : 0;

    const I nr = static_cast<I>(nrecv);
    auto mixed_extent = [&](std::initializer_list<I> factors, const char* what) {
        return checked_size_narrow<I>(checked_size_product<I>(
                factors, what, localFailure, localFailureMessage), what,
                localFailure, localFailureMessage);
    };
    const I nrNd = independentSpaces ? mixed_extent({nr, nd}, "point coordinates")
                               : checked_setup_size<I>({nr, nd}, "point coordinates");
    const I nrNdf = independentSpaces ? mixed_extent({nr, ndf}, "face coordinates")
                                : checked_setup_size<I>({nr, ndf}, "face coordinates");
    const I nrNpe = independentSpaces ? mixed_extent({nr, npe}, "volume basis data")
                                : checked_setup_size<I>({nr, npe}, "volume basis data");
    const I nrNpf = independentSpaces ? mixed_extent({nr, npf}, "face basis data")
                                : checked_setup_size<I>({nr, npf}, "face basis data");
    const I nrOwnerNpf = independentSpaces
            ? checked_size_narrow<I>(checked_size_product<I>(
                    {nr, recvOwnerNpf}, "owner test data", localFailure,
                    localFailureMessage), "owner test data", localFailure,
                    localFailureMessage)
            : nrNpf;
    const I nrNdNpe = independentSpaces ? mixed_extent({nr, nd, npe}, "volume-basis gradients")
                                  : checked_setup_size<I>({nr, nd, npe}, "volume-basis gradients");
    std::vector<I> pe(static_cast<std::size_t>(nr)), pa(static_cast<std::size_t>(nr));
    std::vector<I> pf(static_cast<std::size_t>(nr)), pslot(static_cast<std::size_t>(nr));
    std::vector<I> pg(static_cast<std::size_t>(nr));
    std::vector<T> px(static_cast<std::size_t>(nrNd));
    std::vector<T> pnl(static_cast<std::size_t>(nrNd));
    std::vector<T> pwJ(static_cast<std::size_t>(nr));
    std::vector<T> pxi(static_cast<std::size_t>(nrNdf));
    std::vector<T> pzeta(static_cast<std::size_t>(nrNd));
    std::vector<T> pphi(static_cast<std::size_t>(nrNpe));
    std::vector<T> pphibar(static_cast<std::size_t>(nrNpe));
    std::vector<T> ppsi(static_cast<std::size_t>(nrNpf));
    std::vector<T> pd(static_cast<std::size_t>(nrNd));
    std::vector<T> pdphibar;
    if (donor.ncq == 0) pdphibar.resize(static_cast<std::size_t>(nrNdNpe));
    std::vector<I> slotrank, slotface, slotptr(1, 0), pairslot, pairelem, pairptr(1, 0);

    const bool donorOwnerTests = ownerTestsRequired;
    std::vector<T> sideTest;
    if (donorOwnerTests) {
        sideTest.assign(static_cast<std::size_t>(nrOwnerNpf), static_cast<T>(0));
    }

    I currentSlot = -1, lastRank = -1, lastFace = -1, lastElem = -1;
    const I ne = mesh.ndims[1];
    // Point-loop work arrays, allocated once per call and reset per point
    // (review N3). The arithmetic per point is unchanged.
    const I npeNd = independentSpaces ? mixed_extent({npe, nd}, "element-node workspace")
                                : checked_setup_size<I>({npe, nd}, "element-node workspace");
    const I npfNd = independentSpaces ? mixed_extent({npf, nd}, "face-node workspace")
                                : checked_setup_size<I>({npf, nd}, "face-node workspace");
    std::vector<T> elementNodes(static_cast<std::size_t>(npeNd));
    std::vector<T> psi(static_cast<std::size_t>(npf)), zetaBar(static_cast<std::size_t>(nd));
    std::vector<T> faceNodes(static_cast<std::size_t>(npfNd));
    std::vector<T> zeta(static_cast<std::size_t>(nd));
    std::vector<T> shape, dshape;
    if (donor.ncq == 0) {
        shape.resize(static_cast<std::size_t>(npe));
        dshape.resize(static_cast<std::size_t>(npeNd));
    }
    for (I p = 0; p < static_cast<I>(nrecv); p++) {
        if (localFailure) continue;
        const I raw = order[p];
        const I* im = &recvMeta[static_cast<std::size_t>(nmeta * raw)];
        const T* rm = &recvReal[static_cast<std::size_t>(recvNreal) * raw];
        auto point_failure = [&](const char* what) {
            std::ostringstream oss;
            oss << "non-matching setup: " << what << " on rank " << mpirank
                << " (direction " << donorDirection << ", Gamma^" << donorDirection + 1
                << "), owner rank " << im[0] << ", owner face " << im[1]
                << ", quadrature point " << im[2] << ", donor element " << im[3];
            return oss.str();
        };
        if (im[3] < 0 || im[3] >= ne || im[4] < 0 || im[4] >= nfe ||
            im[2] < 0 || im[2] >= recvOwnerNgf) {
            record_failure(point_failure("received donor tuple is out of range"));
            break;
        }
        if (p == 0 || im[0] != lastRank || im[1] != lastFace) {
            if (p > 0) slotptr.push_back(p);
            slotrank.push_back(im[0]);
            slotface.push_back(im[1]);
            currentSlot++;
            lastRank = im[0]; lastFace = im[1]; lastElem = -1;
        }
        if (p == 0 || im[3] != lastElem || pairslot.empty() || pairslot.back() != currentSlot) {
            if (!pairslot.empty()) pairptr.push_back(p);
            pairslot.push_back(currentSlot);
            pairelem.push_back(im[3]);
            lastElem = im[3];
        }
        pe[p] = im[3]; pa[p] = im[4]; pf[p] = im[5]; pslot[p] = currentSlot; pg[p] = im[2];
        for (I d = 0; d < nd; d++) {
            const std::size_t index = static_cast<std::size_t>(p) + static_cast<std::size_t>(nrecv) * d;
            px[index] = rm[d];
            pnl[index] = rm[nd + d];
        }
        pwJ[p] = rm[2 * nd];
        for (I d = 0; d < ndf; d++)
            pxi[static_cast<std::size_t>(p) + static_cast<std::size_t>(nrecv) * d] = rm[2 * nd + 1 + d];
        if (donorOwnerTests) {
            const T* xm = &rm[3 * nd];
            for (I i = 0; i < recvOwnerNpf; i++)
                sideTest[static_cast<std::size_t>(i) + static_cast<std::size_t>(recvOwnerNpf) * p] = xm[i];
        }

        for (I d = 0; d < nd; d++)
            for (I j = 0; j < npe; j++)
                elementNodes[j + npe * d] = sol.xdg[j + npe * d + npe * ncx * pe[p]];
        std::fill(psi.begin(), psi.end(), static_cast<T>(0));
        T faceXi[2] = {pxi[p], nd == 3 ? pxi[p + nrecv] : static_cast<T>(0)};
        FaceToElementReference(zetaBar.data(), psi.data(), faceXi,
            master.xpe, master.xpf, mesh.perm, pa[p], nd, npe, npf,
            elemtype, porder, surfaceBasisPtr);
        for (I j = 0; j < npf; j++)
            for (I d = 0; d < nd; d++)
                faceNodes[j + npf * d] = elementNodes[mesh.perm[j + npf * pa[p]] + npe * d];
        T xbar[3] = {0, 0, 0};
        for (I d = 0; d < nd; d++)
            for (I j = 0; j < npf; j++) xbar[d] += psi[j] * faceNodes[j + npf * d];
        T xpoint[3] = {0, 0, 0};
        for (I d = 0; d < nd; d++)
            xpoint[d] = px[static_cast<std::size_t>(p) + static_cast<std::size_t>(nrecv) * d];
        const T inverseScale = inverse_map_scale(xpoint, elementNodes.data(), nd, npe);
        const T inverseTolerance = static_cast<T>(prm.inverseToleranceFactor) *
                std::numeric_limits<T>::epsilon() * inverseScale;
        if (!(inverseTolerance > 0)) {
            record_failure(point_failure("invalid scale-invariant inverse tolerance"));
            break;
        }
        T inverseResidual = std::numeric_limits<T>::max();
        std::fill(zeta.begin(), zeta.end(), static_cast<T>(0));
        InverseMapStatus inverse = InverseElementMap(zeta.data(), &inverseResidual, xpoint,
                elementNodes.data(), master.xpe, zetaBar.data(), nd, npe, elemtype, porder,
                prm.maxInverseIterations, prm.maxProjectedInverseIterations,
                inverseTolerance, prm.inverseProjectionMargin, simplexBasisPtr);
        if (inverse == InverseMapStatus::Failed) {
            T alternate[3] = {0, 0, 0};
            DefaultReferenceGuess(alternate, nd, elemtype);
            inverse = InverseElementMap(zeta.data(), &inverseResidual, xpoint,
                    elementNodes.data(), master.xpe, alternate, nd, npe, elemtype, porder,
                    prm.maxInverseIterations, prm.maxProjectedInverseIterations,
                    inverseTolerance, prm.inverseProjectionMargin, simplexBasisPtr);
        }
        if (inverse == InverseMapStatus::Failed) {
            std::ostringstream oss;
            oss << point_failure("inverse donor element map failed")
                << ", donor face " << pf[p] << ", x = (" << xpoint[0] << ", " << xpoint[1];
            if (nd == 3) oss << ", " << xpoint[2];
            oss << "), zetabar = (" << zetaBar[0] << ", " << zetaBar[1];
            if (nd == 3) oss << ", " << zetaBar[2];
            oss << "), zeta = (" << zeta[0] << ", " << zeta[1];
            if (nd == 3) oss << ", " << zeta[2];
            oss << "), final residual = " << inverseResidual;
            record_failure(oss.str());
            break;
        }
        res.inversePoints[donorDirection]++;
        if (inverse == InverseMapStatus::ConvergedProjected)
            res.projectedExactInversePoints[donorDirection]++;
        else if (inverse == InverseMapStatus::ProjectedInexact)
            res.projectedInexactInversePoints[donorDirection]++;
        res.maxInverseResidual[donorDirection] =
                std::max(res.maxInverseResidual[donorDirection], inverseResidual);
        res.maxRelativeInverseResidual[donorDirection] =
                std::max(res.maxRelativeInverseResidual[donorDirection], inverseResidual / inverseScale);
        T projectedReference[3] = {0, 0, 0};
        T projectionWork[3] = {0, 0, 0};
        const T referenceDistance = ReferenceElementDistance(
                zeta.data(), projectedReference, projectionWork, nd, elemtype);
        res.maxReferenceExtrapolation[donorDirection] =
                std::max(res.maxReferenceExtrapolation[donorDirection], referenceDistance);
        T* phi = &pphi[static_cast<std::size_t>(npe) * p];
        T* phibar = &pphibar[static_cast<std::size_t>(npe) * p];
        if (nd == 3) {
            std::fill(phibar, phibar + npe, static_cast<T>(0));
            for (I j = 0; j < npf; ++j)
                phibar[mesh.perm[j + npf * pa[p]]] = psi[j];
        }
        else {
            EvaluateNonmatchingVolumeShape(
                    phibar, zetaBar.data(), master.xpe,
                    nd, npe, elemtype, porder, simplexBasisPtr);
        }
        EvaluateNonmatchingVolumeShape(
                phi, zeta.data(), master.xpe,
                nd, npe, elemtype, porder, simplexBasisPtr);
        for (I j = 0; j < npf; j++) ppsi[j + static_cast<std::size_t>(npf) * p] = psi[j];
        for (I d = 0; d < nd; d++) {
            const std::size_t index = static_cast<std::size_t>(p) + static_cast<std::size_t>(nrecv) * d;
            pzeta[index] = zeta[d];
            pd[index] = px[index] - xbar[d];
        }

        if (donor.ncq == 0) {
            std::fill(shape.begin(), shape.end(), static_cast<T>(0));
            std::fill(dshape.begin(), dshape.end(), static_cast<T>(0));
            T mapped[3] = {0, 0, 0}, jacobian[9] = {0};
            MapNonmatchingReferenceToPhysical(
                    mapped, jacobian, shape.data(), dshape.data(), zetaBar.data(),
                    elementNodes.data(), master.xpe, nd, npe, elemtype, porder,
                    simplexBasisPtr);
            T transposed[9] = {0};
            for (I k = 0; k < nd; k++)
                for (I d = 0; d < nd; d++) transposed[k * nd + d] = jacobian[d * nd + k];
            bool singularMap = false;
            for (I j = 0; j < npe && !singularMap; j++) {
                T rhs[3] = {0, 0, 0}, grad[3] = {0, 0, 0};
                for (I k = 0; k < nd; k++) rhs[k] = dshape[k * npe + j];
                if (!SolveScaleInvariantSmallSystem(grad, transposed, rhs, nd)) {
                    singularMap = true;
                    break;
                }
                for (I d = 0; d < nd; d++)
                    pdphibar[j + static_cast<std::size_t>(npe) *
                                   (d + static_cast<std::size_t>(nd) * p)] = grad[d];
            }
            if (singularMap) {
                record_failure(point_failure("singular donor element map at closest point"));
                break;
            }
        }
    }
    I mixedNrecvslots = 0;
    I mixedAdvertisedExtent = -1;
    I mixedLocalExtent = -1;
    if (independentSpaces) {
        donor.nslots = checked_size_narrow<I>(slotrank.size(), "slot count",
                localFailure, localFailureMessage);
        donor.npairs = checked_size_narrow<I>(pairslot.size(), "pair count",
                localFailure, localFailureMessage);
        std::size_t nrecvslots = 0;
        for (I n = 0; n < owner.nrecvnb; ++n) {
            if (owner.recvcount[n] < 0 || nrecvslots >
                    static_cast<std::size_t>(std::numeric_limits<I>::max()) -
                    static_cast<std::size_t>(owner.recvcount[n])) {
                record_failure("non-matching setup: receive-slot count exceeds integer limits");
                break;
            }
            nrecvslots += static_cast<std::size_t>(owner.recvcount[n]);
        }
        mixedNrecvslots = checked_size_narrow<I>(nrecvslots, "receive-slot count",
                localFailure, localFailureMessage);

        (void) mixed_extent({donor.ncu12, recvOwnerNpf, donor.nslots}, "Ri");
        (void) mixed_extent({donor.ncu12, recvOwnerNpf, donor.ncu, npf, nfe,
                             donor.npairs}, "Hi");
        (void) mixed_extent({donor.ncu12, recvOwnerNpf, npe, donor.ncu,
                             donor.npairs}, "K");
        (void) mixed_extent({recvOwnerNpf, npe, donor.npairs, donor.ncu12,
                             donor.ncu}, "G");
        (void) mixed_extent({npe, npe, donor.npairs}, "Cpair");
        (void) mixed_extent({npe, npf, nfe, donor.npairs}, "Epair");
        (void) mixed_extent({donor.ncu12, recvOwnerNpf, donor.npairs},
                            "pair residual");
        (void) mixed_extent({donor.ncu, npf, nfe, donor.npairs}, "vpair");
        (void) mixed_extent({npe, donor.ncu, donor.ncu, npf, nfe,
                             donor.npairs}, "Fpair");
        (void) mixed_extent({owner.ncu12, owner.npf, mixedNrecvslots}, "recvbuf");
        mixedAdvertisedExtent = mixed_extent(
                {donor.ncu12, recvOwnerNpf}, "donor block extent");
        mixedLocalExtent = mixed_extent(
                {owner.ncu12, owner.npf}, "owner block extent");

        bool genericFailure = false;
        std::string genericFailureMessage;
        const std::size_t genericRange = checked_size_product<I>(
                {donor.ncu12, recvOwnerNpf, npe, donor.ncu, donor.npairs},
                "mixed coupling block", genericFailure, genericFailureMessage);
        const std::size_t genericIndex = checked_size_product<I>(
                {recvOwnerNpf, npe, donor.npairs, donor.ncu12, donor.ncu},
                "mixed coupling block", genericFailure, genericFailureMessage);
        const std::size_t genericCpairIndex = checked_size_product<I>(
                {npe, npe, donor.npairs}, "mixed coupling block",
                genericFailure, genericFailureMessage);
        if (genericFailure || genericRange > static_cast<std::size_t>(INT_MAX) ||
                genericIndex > static_cast<std::size_t>(INT_MAX) ||
                genericCpairIndex > static_cast<std::size_t>(INT_MAX))
            record_failure("non-matching setup: mixed coupling block exceeds the generic Schur helper integer range");
    }
    // Synchronization point 2: every received point and every mixed extent has
    // been processed; no rank leaves before slot-list verification.
    raise_uniform_setup_failure(ic, mpirank, localFailure, localFailureMessage);
    if (nrecv > 0) { slotptr.push_back(static_cast<I>(nrecv)); pairptr.push_back(static_cast<I>(nrecv)); }

    if (!independentSpaces) {
        donor.nslots = static_cast<I>(slotrank.size());
        donor.npairs = static_cast<I>(pairslot.size());
    }
    std::vector<I> sendnb, sendcount;
    for (I s = 0; s < donor.nslots; s++) {
        if (s == 0 || slotrank[s] != slotrank[s - 1]) {
            sendnb.push_back(slotrank[s]);
            sendcount.push_back(0);
        }
        sendcount.back()++;
    }
    donor.nsendnb = static_cast<I>(sendnb.size());

    copy_to_host_array(donor.pe, pe); copy_to_host_array(donor.pa, pa);
    copy_to_host_array(donor.pf, pf); copy_to_host_array(donor.pslot, pslot);
    copy_to_host_array(donor.pg, pg); copy_to_host_scalar_array(donor.px, px);
    copy_to_host_scalar_array(donor.pnl, pnl); copy_to_host_scalar_array(donor.pwJ, pwJ);
    copy_to_host_scalar_array(donor.pxi, pxi); copy_to_host_scalar_array(donor.pzeta, pzeta);
    copy_to_host_scalar_array(donor.pphi, pphi); copy_to_host_scalar_array(donor.pphibar, pphibar);
    copy_to_host_scalar_array(donor.ppsi, ppsi); copy_to_host_scalar_array(donor.pd, pd);
    copy_to_host_scalar_array(donor.pdphibar, pdphibar);
    copy_to_host_array(donor.slotrank, slotrank); copy_to_host_array(donor.slotface, slotface);
    copy_to_host_array(donor.slotptr, slotptr); copy_to_host_array(donor.pairslot, pairslot);
    copy_to_host_array(donor.pairelem, pairelem); copy_to_host_array(donor.pairptr, pairptr);
    copy_to_host_array(donor.sendnb, sendnb); copy_to_host_array(donor.sendcount, sendcount);
    if (donorOwnerTests) {
        ownerTests->dir[donorDirection].values = std::move(sideTest);
    }

    if (independentSpaces) {
        allocate_zeroed(donor.Ri, donor.szRi, static_cast<I>(0));
        allocate_zeroed(donor.Hi, donor.szHi, static_cast<I>(0));
        allocate_zeroed(donor.sendbuf, donor.szsendbuf, static_cast<I>(0));
    }
    else {
        const I szRi = checked_setup_size<I>({donor.ncu12, donor.npf, donor.nslots}, "Ri");
        const I szHi = checked_setup_size<I>({donor.ncu12, donor.npf, donor.ncu, donor.npf,
                                              donor.nfe, donor.npairs}, "Hi");
        allocate_zeroed(donor.Ri, donor.szRi, szRi);
        allocate_zeroed(donor.Hi, donor.szHi, szHi);
        allocate_zeroed(donor.sendbuf, donor.szsendbuf, szRi);
    }

    long long nrecvslotsWide = 0;
    if (!independentSpaces)
        for (I q = 0; q < owner.nrecvnb; q++) {
            if (owner.recvcount[q] < 0 ||
                nrecvslotsWide > std::numeric_limits<int>::max() - owner.recvcount[q])
                error("non-matching setup: receive-slot count exceeds integer limits");
            nrecvslotsWide += owner.recvcount[q];
        }
    const I nrecvslots = independentSpaces ? mixedNrecvslots : static_cast<I>(nrecvslotsWide);
    const I szrecv = independentSpaces ? mixed_extent({owner.ncu12, owner.npf, nrecvslots}, "recvbuf")
                                 : checked_setup_size<I>({owner.ncu12, owner.npf, nrecvslots}, "recvbuf");
    allocate_zeroed(owner.recvbuf, owner.szrecvbuf, szrecv);

    verify_slot_lists_and_extents(ic, donor, owner, recvContract,
            mixedAdvertisedExtent, mixedLocalExtent);
}

// Preserve the existing request workspace while adding capacity for the
// asymmetric non-matching donor sends and owner receives.  The workspace is
// still owned and released by commonstructT.
template <class T, class I>
inline I nonmatching_request_capacity(const commonstructT<T, I>& common,
        const nonmatchingdataT<T, I>& nm, const char* negativeNeighborMessage,
        const char* rangeMessage)
{
    long long nsend = 0, nrecv = 0;
    for (int direction = 0; direction < 2; ++direction) {
        if (nm.dir[direction].nsendnb < 0 || nm.dir[direction].nrecvnb < 0)
            error(negativeNeighborMessage);
        nsend += static_cast<long long>(nm.dir[direction].nsendnb);
        nrecv += static_cast<long long>(nm.dir[direction].nrecvnb);
    }
    const long long conformal = 2LL * static_cast<long long>(common.nnbsd) +
                                2LL * static_cast<long long>(common.nnbintf);
    const long long nonmatching = 2LL * static_cast<long long>(common.nnbsd) +
                                  nsend + nrecv;
    const long long capacity = std::max(conformal, nonmatching);
    if (nsend < 0 || nrecv < 0 || conformal < 0 || nonmatching < 0 ||
            capacity > std::numeric_limits<int>::max())
        error(rangeMessage);
    return static_cast<I>(capacity);
}

template <class T, class I>
inline void ensure_nonmatching_request_capacity(commonstructT<T, I>& common,
        const nonmatchingdataT<T, I>& nm)
{
    const I capacity = nonmatching_request_capacity(common, nm,
        "non-matching setup: negative MPI neighbor count",
        "non-matching setup: MPI request count exceeds integer limits");
    if (common.requests) CPUFREE(common.requests);
    if (common.statuses) CPUFREE(common.statuses);
    common.requests = nullptr;
    common.statuses = nullptr;
    if (capacity > 0) {
        common.requests = static_cast<MPI_Request*>(std::malloc(static_cast<std::size_t>(capacity) * sizeof(MPI_Request)));
        common.statuses = static_cast<MPI_Status*>(std::malloc(static_cast<std::size_t>(capacity) * sizeof(MPI_Status)));
        if (!common.requests || !common.statuses) error("non-matching setup: MPI request allocation failed");
    }
}

// Diagnostics of Section 3.2, printed on comm_Gamma rank 0 for both directions:
// faces, points, fraction of faces with |R_l| > 1, average |R_l| and |K_l|,
// max point-to-face distance, points off their donor face. Mixed 3D also
// reports the global count of owner faces whose points select more than one
// donor global face. Collective.
template <class T, class I>
inline void print_donor_diagnostics(const DonorSearchResult<T, I>& res,
        const preprocess::InterfaceCommunicator<I>& ic,
        const DonorSearchParams& prm = DonorSearchParams(), bool mixedTopology3D = false)
{
    // sums[k + 6*dir]: faces, points, faces with |R_l|>1, sum |R_l|, sum |K_l|, off-face points
    double sums[12] = {0}, gsums[12] = {0};
    double maxd[2] = {0, 0}, gmaxd[2] = {0, 0};
    // Per direction: total, projected-exact, projected-inexact.
    double inverseCounts[6] = {0, 0, 0, 0, 0, 0};
    double globalInverseCounts[6] = {0, 0, 0, 0, 0, 0};
    // Per direction: maximum physical residual, relative residual, and distance from E_hat.
    double inverseMax[6] = {0, 0, 0, 0, 0, 0};
    double globalInverseMax[6] = {0, 0, 0, 0, 0, 0};
    I nestingViolations[2] = {0, 0};
    I globalNestingViolations[2] = {0, 0};
    if (res.direction == 0 || res.direction == 1) {
        double* s = &sums[6 * res.direction];
        s[0] = res.nfaces;
        s[1] = res.npts;
        for (I l = 0; l < res.nfaces; l++) {
            I nK = 0, nR = 0;
            face_set_sizes(nK, nR, res, l);
            if (nR > 1) s[2] += 1;
            s[3] += nR;
            s[4] += nK;
            I begin = 0, end = 0;
            face_point_range(res, l, begin, end);
            for (I p = begin + 1; p < end; ++p)
                if (res.f[p] != res.f[begin]) {
                    ++nestingViolations[res.direction];
                    break;
                }
        }
        for (I p = 0; p < res.npts; p++) {
            maxd[res.direction] = std::max(maxd[res.direction], static_cast<double>(res.dist[p]));
            if (res.dist[p] > static_cast<T>(prm.onFaceRelTol) * res.hface[p]) s[5] += 1;
        }
    }
    for (int dir = 0; dir < 2; ++dir) {
        inverseCounts[3 * dir] = static_cast<double>(res.inversePoints[dir]);
        inverseCounts[3 * dir + 1] = static_cast<double>(res.projectedExactInversePoints[dir]);
        inverseCounts[3 * dir + 2] = static_cast<double>(res.projectedInexactInversePoints[dir]);
        inverseMax[3 * dir] = static_cast<double>(res.maxInverseResidual[dir]);
        inverseMax[3 * dir + 1] = static_cast<double>(res.maxRelativeInverseResidual[dir]);
        inverseMax[3 * dir + 2] = static_cast<double>(res.maxReferenceExtrapolation[dir]);
    }
    MPI_Allreduce(sums, gsums, 12, MPI_DOUBLE, MPI_SUM, ic.comm);
    MPI_Allreduce(maxd, gmaxd, 2, MPI_DOUBLE, MPI_MAX, ic.comm);
    MPI_Allreduce(inverseCounts, globalInverseCounts, 6, MPI_DOUBLE, MPI_SUM, ic.comm);
    MPI_Allreduce(inverseMax, globalInverseMax, 6, MPI_DOUBLE, MPI_MAX, ic.comm);
    ic.collective(); ic.collective(); ic.collective(); ic.collective();
    if (mixedTopology3D) {
        MPI_Allreduce(nestingViolations, globalNestingViolations, 2,
                mpi_type<I>(), MPI_SUM, ic.comm);
        ic.collective();
    }
    if (ic.rank == 0) {
        for (int dir = 0; dir < 2; dir++) {
            const double* s = &gsums[6 * dir];
            const double nf = std::max(1.0, s[0]);
            printf("Non-matching donor search, Gamma^%d (owners in model %d): %.0f faces, %.0f points; "
                   "fraction |R_l|>1 = %.6f, average |R_l| = %.6f, average |K_l| = %.6f, "
                   "max distance = %.6e, points off donor face = %.0f\n",
                   dir + 1, dir + 1, s[0], s[1], s[2] / nf, s[3] / nf, s[4] / nf, gmaxd[dir], s[5]);
            if (globalInverseCounts[3 * dir] > 0)
                printf("  inverse maps: %.0f points, %.0f projected exact, %.0f projected inexact; "
                       "max residual = %.6e, max relative residual = %.6e, "
                       "max distance from E_hat = %.6e\n",
                       globalInverseCounts[3 * dir], globalInverseCounts[3 * dir + 1],
                       globalInverseCounts[3 * dir + 2], globalInverseMax[3 * dir],
                       globalInverseMax[3 * dir + 1], globalInverseMax[3 * dir + 2]);
            if (mixedTopology3D)
                printf("  mixed nesting violations, Gamma^%d = %lld\n",
                       dir + 1, static_cast<long long>(globalNestingViolations[dir]));
        }
    }
}


// Low-level setup entry point. Production construction and mesh-motion rebuilds
// use rebuild_nonmatching_interface below so request capacity cannot lag the cache.
// Returns false (and does nothing) unless interfacetype == 1 and the coupled
// guards of build_runtime_interface_partition hold. Otherwise every rank of both
// models takes part (collective on EXASIM_COMM_WORLD). If result is non-null the
// per-point tuples are returned in it (phase 3 consumes them).
template <class T, class I>
inline bool setup_nonmatching_interface(appstructT<T, I>& app, masterstructT<T, I>& master,
        meshstructT<T, I>& mesh, solstructT<T, I>& sol, I mpiprocs, I mpirank, I fileoffset,
        nonmatchingdataT<T, I>& nm, DonorSearchResult<T, I>* result = nullptr,
        const DonorSearchParams& prm = DonorSearchParams(), bool performDistribution = true,
        SearchStatistics* stats = nullptr, OwnerTestData<T, I>* ownerTests = nullptr)
{
    // [nonmatching] Setup owns a complete cache generation.  Discard the old
    // directional data first so mesh-motion rebuilds and active-to-inactive
    // transitions cannot retain stale point locations or work arrays.
    free_nonmatching_data(nm);
    if (ownerTests) ownerTests->clear();

    // Same guards as build_runtime_interface_partition.  Since every world
    // rank enters collectives below, first make pointer/configuration selection
    // rank-uniform: a rank-local early return would deadlock the other ranks.
    if (mpiprocs <= 1)
        return false;
    const int localReady = app.problem && app.nsize && master.ndims && mesh.ndims &&
                           mesh.bf && mesh.perm && sol.xdg;
    int allReady = 0, anyReady = 0;
    MPI_Allreduce(&localReady, &allReady, 1, MPI_INT, MPI_MIN, EXASIM_COMM_WORLD);
    MPI_Allreduce(&localReady, &anyReady, 1, MPI_INT, MPI_MAX, EXASIM_COMM_WORLD);
    if (allReady != anyReady)
        error("non-matching setup: required input arrays are not available on every coupled rank");
    if (!allReady)
        return false;
    const int localActive = app.nsize[2] > 34 && app.problem[28] > 0 &&
                            app.problem[29] > 0 && app.problem[30] > 0 &&
                            app.problem[34] == 1;
    int allActive = 0, anyActive = 0;
    MPI_Allreduce(&localActive, &allActive, 1, MPI_INT, MPI_MIN, EXASIM_COMM_WORLD);
    MPI_Allreduce(&localActive, &anyActive, 1, MPI_INT, MPI_MAX, EXASIM_COMM_WORLD);
    if (allActive != anyActive)
        error("non-matching setup: coupling selection is inconsistent across coupled ranks");
    if (!allActive)
        return false;
    nm.interfacetype = 1;

    const I nd = master.ndims[0], elemtype = master.ndims[1], porder = master.ndims[3];
    const I pgauss = master.ndims[4];
    const I npe = master.ndims[5], npf = master.ndims[6], ngf = master.ndims[8];
    const I nfe = mesh.ndims[4];
    const I own = (fileoffset > 0) ? 1 : 0;

    const InterfaceFaceSet<T, I> ownFaces = collect_own_interface_faces(app, master, mesh, sol, mpirank, fileoffset);
    for (int k = 0; k < 2; k++) {
        nonmatchinginterfaceT<T, I>& d = nm.dir[k];
        d.direction = k;
        d.nd = nd; d.npe = npe; d.npf = npf; d.nfe = nfe; d.ngf = ngf;
        d.ncu12 = (app.nsize[14] > 0) ? app.nsize[14] : 0; // szinterfacefluxmap
    }
    DonorSearchResult<T, I> res;
    res.direction = own;
    own_quadrature_points(res, ownFaces, master.shapfgt, ngf);

    // One setup-only communicator is created per rebuild from the ranks that
    // own a coupled interface face. Nonmembers retain a valid empty cache and
    // enter no further interface collective.
    preprocess::InterfaceCommunicator<I> ic = [&]() {
        try {
            return preprocess::make_interface_communicator<I>(ownFaces.size() > 0, mpirank, own);
        }
        catch (const std::runtime_error& exception) {
            throw UniformSetupError(exception.what());
        }
    }();
    if (stats) {
        stats->member = ic.comm != MPI_COMM_NULL ? 1 : 0;
        stats->interfaceSize = ic.size;
        stats->interfaceCollectives += ic.comm != MPI_COMM_NULL ? 1 : 0;
        ic.collectiveCounter = &stats->interfaceCollectives;
    }
    if (ic.comm == MPI_COMM_NULL) {
        build_owner_side(nm.dir[own], res);
        if (result) *result = std::move(res);
        return true;
    }

    // Establish the topology-specific spaces once.  Downstream setup and
    // runtime algebra consume this contract without re-detecting element
    // families.
    const FaceMasterCompatibility<T, I> compatibility = [&]() {
        try {
            return check_face_master_compatibility<T, I>(
                    master, nfe, own, ic, app.nsize[14]);
        }
        catch (const std::runtime_error& exception) {
            throw UniformSetupError(exception.what());
        }
    }();
    if (compatibility.mixedFamily && !ownerTests) {
        if (compatibility.mixedTopology3D)
            throw UniformSetupError(
                    "non-matching interface: mixed element families require the owner-test sidecar");
        error("non-matching interface: mixed element families require the owner-test sidecar");
    }
    if (ownerTests) {
        for (int k = 0; k < 2; ++k) {
            OwnerTestDirection<T, I>& direction = ownerTests->dir[k];
            const FaceMasterDescriptor<T, I>& owner = compatibility.model[k];
            const FaceMasterDescriptor<T, I>& donor = compatibility.model[1 - k];
            direction.enabled = compatibility.mixedFamily ? 1 : 0;
            direction.contract.relation = compatibility.mixedTopology3D
                    ? IndependentFaceSpaces : SharedFaceSpace;
            direction.contract.ownerElemtype = owner.elemtype;
            direction.contract.ownerFaceNodes = owner.npf;
            direction.contract.ownerFaceQuadraturePoints = owner.ngf;
            direction.contract.donorElemtype = donor.elemtype;
            direction.contract.donorVolumeNodes = donor.npe;
            direction.contract.donorFaceNodes = donor.npf;
            direction.contract.donorFacesPerElement = donor.nfe;
        }
    }

    const double searchStart = MPI_Wtime();
    {
        // [nonmatching] Phase 12: distributed multi-start face search.
        preprocess::PreparedFaces<T, I> prepared;
        bool localPreparationFailure = false;
        std::string localPreparationMessage;
        try {
            prepared = preprocess::prepare_faces<T, I>(
                    ownFaces, master.xpf, elemtype, porder);
        }
        catch (const std::runtime_error&) {
            localPreparationFailure = true;
            std::ostringstream message;
            message << "surface patch preparation failed on rank " << mpirank
                    << ", direction " << own;
            if (!ownFaces.global.empty())
                message << " at global face " << ownFaces.global.front();
            localPreparationMessage = message.str();
        }
        raise_uniform_setup_failure(ic, mpirank, localPreparationFailure,
                localPreparationMessage);

        std::vector<T> ownerPoints(static_cast<std::size_t>(res.npts) * nd);
        for (I point = 0; point < res.npts; ++point)
            for (I d = 0; d < nd; ++d)
                ownerPoints[d + static_cast<std::size_t>(nd) * point] =
                        res.x[point + static_cast<std::size_t>(res.npts) * d];
        const preprocess::InterfaceCommunicatorView communicator =
                {ic.comm, ic.rank, ic.size};
        preprocess::DistributedStatistics preprocessStats;
        std::vector<preprocess::DistributedDonor<T, I>> donors;
        try {
            donors = preprocess::distributed_search(communicator, prepared, ownerPoints,
                        mpirank, own, static_cast<T>(prm.projectionTolerance),
                        prm.maxProjectionIterations, stats ? &preprocessStats : nullptr);
        }
        catch (const std::runtime_error& exception) {
            throw UniformSetupError(exception.what());
        }
        if (stats) {
            stats->ownerPoints += preprocessStats.ownerPoints;
            stats->d4Queries += preprocessStats.d4Queries;
            stats->maxD4FanOut = std::max(stats->maxD4FanOut, preprocessStats.maxD4FanOut);
            stats->facesEvaluated += preprocessStats.facesEvaluated;
            stats->facesPruned += preprocessStats.facesPruned;
            stats->projectionStarts += preprocessStats.projectionStarts;
            stats->projectionIterations += preprocessStats.projectionIterations;
            stats->facesPerPointSum += preprocessStats.facesEvaluated;
            stats->maxFacesPerPoint = std::max(stats->maxFacesPerPoint,
                    preprocessStats.maxFacesPerPoint);
            stats->queryBytes += preprocessStats.queryBytes;
            stats->responseBytes += preprocessStats.responseBytes;
            stats->countBytes += preprocessStats.countBytes;
            stats->boxBytes += preprocessStats.boxBytes;
        }
        const bool donorCountMismatch =
                donors.size() != static_cast<std::size_t>(res.npts);
        raise_uniform_setup_failure(ic, mpirank, donorCountMismatch,
                "non-matching preprocessing: donor result count mismatch");
        res.r.resize(res.npts); res.e.resize(res.npts);
        res.f.resize(res.npts); res.a.resize(res.npts);
        res.dist.resize(res.npts); res.hface.resize(res.npts);
        res.xi.resize(static_cast<std::size_t>(res.npts) * (nd - 1));
        res.converged.assign(res.npts, 1);
        bool localUnconvergedDonor = false;
        for (I point = 0; point < res.npts; ++point) {
            const auto& donor = donors[point];
            res.r[point] = donor.rank; res.e[point] = donor.elem;
            res.f[point] = donor.face; res.a[point] = donor.localFace;
            res.dist[point] = donor.distance; res.hface[point] = donor.hface;
            for (I d = 0; d < nd - 1; ++d)
                res.xi[point + static_cast<std::size_t>(res.npts) * d] = donor.xi[d];
            res.converged[point] = donor.converged ? 1 : 0;
            localUnconvergedDonor = localUnconvergedDonor || !donor.converged;
        }
        raise_uniform_setup_failure(ic, mpirank, localUnconvergedDonor,
                "non-matching preprocessing: distributed search returned an unconverged donor");
    }
    if (stats) stats->searchSeconds += MPI_Wtime() - searchStart;

    build_owner_side(nm.dir[own], res);
    if (performDistribution)
        distribute_setup_data(nm, res, ownFaces, app, master, mesh, sol,
                mpiprocs, mpirank, own, prm, ic, ownerTests);
    print_donor_diagnostics(res, ic, prm, compatibility.mixedTopology3D != 0);

    if (result) *result = std::move(res);
    return true;
}

// [nonmatching] Atomic cache rebuild used by production and mesh-motion paths.
// Resizing requests here makes it impossible for changed donor/owner neighbor
// counts to leave the next compact exchange with stale capacity.
template <class T, class I>
inline bool rebuild_nonmatching_interface(appstructT<T, I>& app, masterstructT<T, I>& master,
        meshstructT<T, I>& mesh, solstructT<T, I>& sol, commonstructT<T, I>& common,
        I mpiprocs, I mpirank, I fileoffset, nonmatchingdataT<T, I>& nm,
        DonorSearchResult<T, I>* result = nullptr,
        const DonorSearchParams& prm = DonorSearchParams(), bool performDistribution = true)
{
    const bool hadTypeOne = nm.interfacetype == 1;
    bool ran = false;
    try {
        ran = setup_nonmatching_interface(app, master, mesh, sol, mpiprocs, mpirank,
                fileoffset, nm, result, prm, performDistribution);
    }
    catch (const UniformSetupError& exception) {
        error(exception.what());
    }
    if (ran || hadTypeOne)
        ensure_nonmatching_request_capacity(common, nm);
    return ran;
}

// [nonmatching] Phase 8 transaction: construct host setup data, backend
// mirrors/workspaces and replacement MPI request storage before publishing any
// of them.  This is the production rebuild API used for construction and mesh
// motion; the lower-level overload above remains for setup-only tests.
template <class T, class I>
inline void validate_nonmatching_runtime_owner_maps(const nonmatchingdataT<T, I>& nm,
        const meshstructT<T, I>& mesh, const appstructT<T, I>& app,
        const commonstructT<T, I>& common)
{
    if (nm.interfacetype != 1) return;
    std::vector<I> pendingInterfaceFaces;
    const I* interfaceFaces = mesh.intfaces;
    I interfaceFaceCount = mesh.szintfaces;
    if (!interfaceFaces) {
        const I boundary = common.couplingparams.coupledboundarycondition;
        if (!mesh.bf || boundary <= 0)
            error("non-matching runtime: coupled boundary-face data are incomplete");
        for (I element = 0; element < common.meshsizes.ne1; ++element)
            for (I localFace = 0; localFace < common.meshsizes.nfe; ++localFace)
                if (mesh.bf[localFace + common.meshsizes.nfe * element] == boundary)
                    pendingInterfaceFaces.push_back(localFace + common.meshsizes.nfe * element);
        interfaceFaces = pendingInterfaceFaces.data();
        interfaceFaceCount = static_cast<I>(pendingInterfaceFaces.size());
    }
    for (int direction = 0; direction < 2; ++direction) {
        const auto& owner = nm.dir[direction];
        if (!owner.isowner || owner.nownfaces == 0) continue;
        if (owner.nfe <= 0 || !owner.ownface || !app.interfacefluxmap)
            error("non-matching runtime: owner mapping data are incomplete");
        for (I own = 0; own < owner.nownfaces; ++own) {
            const I interfaceFace = owner.ownface[own];
            if (interfaceFace < 0 || interfaceFace >= interfaceFaceCount)
                error("non-matching runtime: owner interface-face index is out of range");
            const I encoded = interfaceFaces[interfaceFace];
            const I element = encoded / owner.nfe;
            const I localFace = encoded - owner.nfe * element;
            if (encoded < 0 || element < 0 || element >= common.meshsizes.ne ||
                localFace < 0 || localFace >= owner.nfe)
                error("non-matching runtime: encoded owner element face is out of range");
        }
        for (I component = 0; component < owner.ncu12; ++component)
            if (app.interfacefluxmap[component] < 0 ||
                app.interfacefluxmap[component] >= common.components.ncu)
                error("non-matching runtime: interfacefluxmap is out of range");
    }
}

template <class T, class I>
inline bool rebuild_nonmatching_interface(appstructT<T, I>& app,
        masterstructT<T, I>& master, meshstructT<T, I>& mesh,
        solstructT<T, I>& sol, commonstructT<T, I>& common,
        I mpiprocs, I mpirank, I fileoffset, nonmatchingdataT<T, I>& nm,
        std::unique_ptr<NonmatchingRuntime<T, I>>& runtime, I backend,
        DonorSearchResult<T, I>* result = nullptr,
        const DonorSearchParams& prm = DonorSearchParams(),
        bool performDistribution = true,
        std::unique_ptr<OwnerTestData<T, I>>* ownerTests = nullptr)
{
    const bool hadTypeOne = nm.interfacetype == 1;
    nonmatchingdataT<T, I> candidate;
    std::unique_ptr<NonmatchingRuntime<T, I>> candidateRuntime;
    std::unique_ptr<OwnerTestData<T, I>> candidateOwnerTests;
    if (ownerTests) candidateOwnerTests = std::make_unique<OwnerTestData<T, I>>();
    MPI_Request* candidateRequests = nullptr;
    MPI_Status* candidateStatuses = nullptr;
    bool ran = false;
    try {
        ran = setup_nonmatching_interface(app, master, mesh, sol, mpiprocs, mpirank,
                fileoffset, candidate, result, prm, performDistribution, nullptr,
                candidateOwnerTests.get());
        if (candidate.interfacetype == 1) {
            validate_nonmatching_runtime_owner_maps(candidate, mesh, app, common);
            candidateRuntime = std::make_unique<NonmatchingRuntime<T, I>>();
            refresh_nonmatching_runtime(*candidateRuntime, candidate, app, mesh, common, backend,
                                        candidateOwnerTests.get());
        }

        const I requestCapacity = nonmatching_request_capacity(common, candidate,
            "non-matching runtime: MPI request capacity exceeds integer range",
            "non-matching runtime: MPI request capacity exceeds integer range");
        if (requestCapacity > 0) {
            candidateRequests = static_cast<MPI_Request*>(std::malloc(
                static_cast<std::size_t>(requestCapacity) * sizeof(MPI_Request)));
            candidateStatuses = static_cast<MPI_Status*>(std::malloc(
                static_cast<std::size_t>(requestCapacity) * sizeof(MPI_Status)));
            if (!candidateRequests || !candidateStatuses)
                error("non-matching runtime: MPI request allocation failed");
        }
    }
    catch (const UniformSetupError& exception) {
        free_nonmatching_data(candidate);
        CPUFREE(candidateRequests);
        CPUFREE(candidateStatuses);
        if (!hadTypeOne)
            error(exception.what());
        throw;
    }
    catch (...) {
        free_nonmatching_data(candidate);
        CPUFREE(candidateRequests);
        CPUFREE(candidateStatuses);
        throw;
    }

    std::swap(nm, candidate);
    runtime.swap(candidateRuntime);
    if (ownerTests) ownerTests->swap(candidateOwnerTests);
    std::swap(common.requests, candidateRequests);
    std::swap(common.statuses, candidateStatuses);
    free_nonmatching_data(candidate);
    CPUFREE(candidateRequests);
    CPUFREE(candidateStatuses);
    return ran;
}
#else
template <class T, class I>
inline bool setup_nonmatching_interface(appstructT<T, I>&, masterstructT<T, I>&, meshstructT<T, I>&,
        solstructT<T, I>&, I, I, I, nonmatchingdataT<T, I>&, DonorSearchResult<T, I>* = nullptr,
        const DonorSearchParams& = DonorSearchParams(), bool = true, SearchStatistics* = nullptr,
        OwnerTestData<T, I>* = nullptr)
{
    return false; // no coupling without MPI (phase 0 report, D7)
}
template <class T, class I>
inline bool rebuild_nonmatching_interface(appstructT<T, I>&, masterstructT<T, I>&,
        meshstructT<T, I>&, solstructT<T, I>&, commonstructT<T, I>&,
        I, I, I, nonmatchingdataT<T, I>&, DonorSearchResult<T, I>* = nullptr,
        const DonorSearchParams& = DonorSearchParams(), bool = true)
{
    return false;
}
#endif

} // namespace nonmatching
} // namespace exasim

#endif
