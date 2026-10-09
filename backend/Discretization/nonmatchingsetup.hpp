/*
    nonmatchingsetup.hpp

    Setup of the non-conformal, non-matching interface solver
    (docs/nonmatching/nonmatching_multiphysics.tex). Phase 2: donor search.

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

    First version (phase prompt, task 3): every rank gathers all coupled
    interface faces of both models with MPI_Allgatherv on EXASIM_COMM_WORLD,
    as build_runtime_interface_partition does, and searches the faces of the
    other model locally. Phase 7 replaces the all-gather.

    Requirements: include after Common/common.h and interfacepartition.hpp.
    Host-only setup code.
*/

#ifndef __NONMATCHINGSETUP_HPP
#define __NONMATCHINGSETUP_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "nonmatchinginterface.hpp"
#include "nonmatchinggeometry.hpp"

namespace exasim {
namespace nonmatching {

// Tolerances of the donor search (approved with the phase 2 plan, decision D4).
struct DonorSearchParams {
    dstype tieRelTol = 1.0e-10;       // ties: |d_m - d*| <= tieRelTol * L, L = diagonal of the
                                      //   bounding box of the candidate interface (same on every rank)
    dstype onFaceRelTol = 1.0e-10;    // diagnostics: point on its donor face if dist <= onFaceRelTol * h_f
    dstype curvedBoxPad = 0.25;       // porder > 1: face boxes padded by this fraction of their diagonal
    Int maxProjectionIterations = 100;
    dstype projectionTolerance = 1.0e-13; // step norm in reference-face units (eqs. (21)-(22))
};

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
};

template <class T>
inline T box_distance(const T* x, const T* bmin, const T* bmax, int nd)
{
    T d2 = 0;
    for (int d = 0; d < nd; d++) {
        T t = 0;
        if (x[d] < bmin[d]) t = bmin[d] - x[d];
        else if (x[d] > bmax[d]) t = x[d] - bmax[d];
        d2 += t * t;
    }
    return std::sqrt(d2);
}

// Closest-face rule (3) for the points x[p + npts*d], p < npts, against the
// candidate faces cand (all faces of the other model). Pure, no MPI. For each
// point:
//   1. lb_m = dist(x, B_m), B_m the box of the face nodes (padded if porder > 1);
//   2. project onto the face with the smallest lb (ties: smaller global index): d0;
//   3. project onto every face with lb_m <= d0 + tol; d* = min d_m; the donor is
//      the face of smallest global index among {m : d_m <= d* + tol}.
// Every face that attains or ties the minimum has lb_m <= d_m <= d0 + tol, so
// step 3 sees all of them (when the boxes bound the faces). The result depends
// only on the point and on the set of candidates, not on their order, so it is
// the same on every partition.
template <class T, class I>
inline void donor_search(DonorSearchResult<T, I>& res, const InterfaceFaceSet<T, I>& cand,
        const T* xpf, I elemtype, I porder, const DonorSearchParams& prm = DonorSearchParams())
{
    const I nd = res.nd;
    const I npf = cand.npf;
    const I ndf = nd - 1;
    const I npts = res.npts;
    const I nc = cand.size();

    res.r.assign(npts, -1);
    res.e.assign(npts, -1);
    res.f.assign(npts, -1);
    res.a.assign(npts, -1);
    res.dist.assign(npts, std::numeric_limits<T>::max());
    res.xi.assign(static_cast<std::size_t>(npts * std::max<I>(ndf, 1)), static_cast<T>(0));
    res.hface.assign(npts, static_cast<T>(0));
    res.converged.assign(npts, 0);
    if (npts == 0)
        return;
    if (nc == 0)
        error("non-matching donor search: the other model has no interface faces");

    // Face boxes and the length scale L of the candidate interface.
    std::vector<T> bmin(static_cast<std::size_t>(nd * nc)), bmax(static_cast<std::size_t>(nd * nc));
    std::vector<T> hf(static_cast<std::size_t>(nc));
    std::vector<T> gmin(nd, std::numeric_limits<T>::max()), gmax(nd, -std::numeric_limits<T>::max());
    for (I m = 0; m < nc; m++) {
        const T* y = &cand.nodes[static_cast<std::size_t>(npf * nd * m)];
        T diag2 = 0;
        for (I d = 0; d < nd; d++) {
            T lo = y[npf * d], hi = y[npf * d];
            for (I j = 1; j < npf; j++) {
                lo = std::min(lo, y[j + npf * d]);
                hi = std::max(hi, y[j + npf * d]);
            }
            bmin[d + nd * m] = lo;
            bmax[d + nd * m] = hi;
            gmin[d] = std::min(gmin[d], lo);
            gmax[d] = std::max(gmax[d], hi);
            diag2 += (hi - lo) * (hi - lo);
        }
        hf[m] = std::sqrt(diag2);
        if (porder > 1) {
            const T pad = static_cast<T>(prm.curvedBoxPad) * hf[m];
            for (I d = 0; d < nd; d++) {
                bmin[d + nd * m] -= pad;
                bmax[d + nd * m] += pad;
            }
        }
    }
    T L2 = 0;
    for (I d = 0; d < nd; d++) L2 += (gmax[d] - gmin[d]) * (gmax[d] - gmin[d]);
    const T tieTol = static_cast<T>(prm.tieRelTol) * std::sqrt(L2);

    std::vector<T> lb(static_cast<std::size_t>(nc));
    std::vector<T> dm(static_cast<std::size_t>(nc));
    std::vector<T> xim(static_cast<std::size_t>(2 * nc));
    std::vector<char> done(static_cast<std::size_t>(nc)), conv(static_cast<std::size_t>(nc));
    std::vector<I> visited;
    visited.reserve(16);

    for (I p = 0; p < npts; p++) {
        T xp[3] = {0, 0, 0};
        for (I d = 0; d < nd; d++) xp[d] = res.x[p + npts * d];

        I m0 = -1;
        for (I m = 0; m < nc; m++) {
            lb[m] = box_distance(xp, &bmin[nd * m], &bmax[nd * m], static_cast<int>(nd));
            if (m0 < 0 || lb[m] < lb[m0] || (lb[m] == lb[m0] && cand.global[m] < cand.global[m0]))
                m0 = m;
        }

        auto project = [&](I m) {
            if (done[m]) return;
            T xb[3] = {0, 0, 0};
            T dd = std::numeric_limits<T>::max();
            T xx[2] = {0, 0};
            const bool ok = ClosestPointOnFace(xx, xb, &dd, xp,
                    &cand.nodes[static_cast<std::size_t>(npf * nd * m)], xpf, nd, npf, elemtype, porder,
                    prm.maxProjectionIterations, static_cast<T>(prm.projectionTolerance)); // eqs. (21)-(22)
            dm[m] = dd;
            xim[2 * m] = xx[0];
            xim[2 * m + 1] = xx[1];
            conv[m] = ok ? 1 : 0;
            done[m] = 1;
            visited.push_back(m);
        };

        project(m0);
        const T d0 = dm[m0];
        for (I m = 0; m < nc; m++)
            if (lb[m] <= d0 + tieTol)
                project(m);

        T dstar = std::numeric_limits<T>::max();
        for (I m : visited) dstar = std::min(dstar, dm[m]);
        I best = -1;
        for (I m : visited)
            if (dm[m] <= dstar + tieTol && (best < 0 || cand.global[m] < cand.global[best]))
                best = m; // eq. (3): smallest global face index among the ties

        res.r[p] = cand.rank[best];
        res.e[p] = cand.elem[best];
        res.f[p] = cand.global[best];
        res.a[p] = cand.lface[best];
        res.dist[p] = dm[best];
        for (I k = 0; k < ndf; k++) res.xi[p + npts * k] = xim[2 * best + k];
        res.hface[p] = hf[best];
        res.converged[p] = conv[best];

        for (I m : visited) done[m] = 0;
        visited.clear();
    }
}

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
    res.npts = ngf * nf;
    res.ownglobal = own.global;
    res.x.assign(static_cast<std::size_t>(res.npts * nd), static_cast<T>(0));
    for (I l = 0; l < nf; l++)
        for (I g = 0; g < ngf; g++)
            for (I d = 0; d < nd; d++) {
                T s = 0;
                for (I j = 0; j < npf; j++)
                    s += shapfgt[g + ngf * j] * own.nodes[static_cast<std::size_t>(j + npf * d + npf * nd * l)];
                res.x[(g + ngf * l) + res.npts * d] = s;
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

// Owner side of the frozen struct from the donor tuples: K_l, R_l and G_{l,r}
// (eqs. (4), (6), (60)) and the receive lists (eq. (62)): recvnb in increasing
// donor rank; for each donor, the own faces l with that rank in R_l, ordered
// by increasing own global face index (list order L_pr). ownface[l] = l
// (index into mesh.intfaces). Host arrays.
template <class T, class I>
inline void build_owner_side(nonmatchinginterfaceT<T, I>& nmi, const DonorSearchResult<T, I>& res)
{
    const I nf = res.nfaces, ngf = res.ngf;
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
        for (I g = 0; g < ngf; g++) R[l].push_back(res.r[g + ngf * l]);
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
    for (I g = 0; g < res.ngf; g++) {
        K.push_back(res.f[g + res.ngf * l]);
        R.push_back(res.r[g + res.ngf * l]);
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
    nmi.nownfaces = 0;
    nmi.nrecvnb = 0;
    nmi.isowner = 0;
}

// Releases the arrays allocated by the setup (phase 2: owner side, host).
template <class T, class I>
inline void free_nonmatching_data(nonmatchingdataT<T, I>& nm)
{
    for (int k = 0; k < 2; k++)
        free_nonmatching_interface(nm.dir[k]);
}

#ifdef HAVE_MPI
// All coupled interface faces of all ranks of both models, gathered on every
// rank with MPI_Allgatherv on EXASIM_COMM_WORLD. Every rank must call it.
template <class T, class I>
inline InterfaceFaceSet<T, I> gather_interface_faces(const InterfaceFaceSet<T, I>& own, I mpiprocs)
{
    const I nd = own.nd, npf = own.npf, nlocal = own.size();
    const I nmeta = 5, ncoord = npf * nd;

    std::vector<I> counts(static_cast<std::size_t>(mpiprocs), 0);
    MPI_Allgather(&nlocal, 1, mpi_type<I>(), counts.data(), 1, mpi_type<I>(), EXASIM_COMM_WORLD);

    std::vector<int> mc(mpiprocs, 0), md(mpiprocs, 0), cc(mpiprocs, 0), cd(mpiprocs, 0);
    I total = 0;
    for (I r = 0; r < mpiprocs; r++) {
        if (counts[r] > static_cast<I>(std::numeric_limits<int>::max() / std::max<I>(nmeta, ncoord)))
            error("too many non-matching interface faces for MPI_Allgatherv integer counts");
        mc[r] = static_cast<int>(nmeta * counts[r]);
        cc[r] = static_cast<int>(ncoord * counts[r]);
        if (r > 0) {
            md[r] = md[r - 1] + mc[r - 1];
            cd[r] = cd[r - 1] + cc[r - 1];
        }
        total += counts[r];
    }

    std::vector<I> meta(static_cast<std::size_t>(nmeta * nlocal));
    for (I i = 0; i < nlocal; i++) {
        meta[nmeta * i + 0] = own.rank[i];
        meta[nmeta * i + 1] = own.model[i];
        meta[nmeta * i + 2] = own.elem[i];
        meta[nmeta * i + 3] = own.lface[i];
        meta[nmeta * i + 4] = own.global[i];
    }
    std::vector<I> allmeta(static_cast<std::size_t>(nmeta * total));
    InterfaceFaceSet<T, I> all;
    all.nd = nd;
    all.npf = npf;
    all.nodes.assign(static_cast<std::size_t>(ncoord * total), static_cast<T>(0));

    MPI_Allgatherv(meta.data(), static_cast<int>(meta.size()), mpi_type<I>(),
            allmeta.data(), mc.data(), md.data(), mpi_type<I>(), EXASIM_COMM_WORLD);
    MPI_Allgatherv(own.nodes.data(), static_cast<int>(own.nodes.size()), mpi_type<T>(),
            all.nodes.data(), cc.data(), cd.data(), mpi_type<T>(), EXASIM_COMM_WORLD);

    for (I j = 0; j < total; j++) {
        all.rank.push_back(allmeta[nmeta * j + 0]);
        all.model.push_back(allmeta[nmeta * j + 1]);
        all.elem.push_back(allmeta[nmeta * j + 2]);
        all.lface.push_back(allmeta[nmeta * j + 3]);
        all.global.push_back(allmeta[nmeta * j + 4]);
    }
    return all;
}

// Faces of the given model only.
template <class T, class I>
inline InterfaceFaceSet<T, I> select_model_faces(const InterfaceFaceSet<T, I>& all, I model)
{
    InterfaceFaceSet<T, I> out;
    out.nd = all.nd;
    out.npf = all.npf;
    const std::size_t nc = static_cast<std::size_t>(all.npf * all.nd);
    for (I j = 0; j < all.size(); j++) {
        if (all.model[j] != model) continue;
        out.rank.push_back(all.rank[j]);
        out.model.push_back(all.model[j]);
        out.elem.push_back(all.elem[j]);
        out.lface.push_back(all.lface[j]);
        out.global.push_back(all.global[j]);
        out.nodes.insert(out.nodes.end(), all.nodes.begin() + nc * j, all.nodes.begin() + nc * (j + 1));
    }
    return out;
}

// Diagnostics of Section 3.2, printed on world rank 0 for both directions:
// faces, points, fraction of faces with |R_l| > 1, average |R_l| and |K_l|,
// max point-to-face distance, points off their donor face. Collective.
template <class T, class I>
inline void print_donor_diagnostics(const DonorSearchResult<T, I>& res, I mpirank,
        const DonorSearchParams& prm = DonorSearchParams())
{
    // sums[k + 6*dir]: faces, points, faces with |R_l|>1, sum |R_l|, sum |K_l|, off-face points
    double sums[12] = {0}, gsums[12] = {0};
    double maxd[2] = {0, 0}, gmaxd[2] = {0, 0};
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
        }
        for (I p = 0; p < res.npts; p++) {
            maxd[res.direction] = std::max(maxd[res.direction], static_cast<double>(res.dist[p]));
            if (res.dist[p] > static_cast<T>(prm.onFaceRelTol) * res.hface[p]) s[5] += 1;
        }
    }
    MPI_Allreduce(sums, gsums, 12, MPI_DOUBLE, MPI_SUM, EXASIM_COMM_WORLD);
    MPI_Allreduce(maxd, gmaxd, 2, MPI_DOUBLE, MPI_MAX, EXASIM_COMM_WORLD);
    if (mpirank == 0) {
        for (int dir = 0; dir < 2; dir++) {
            const double* s = &gsums[6 * dir];
            const double nf = std::max(1.0, s[0]);
            printf("Non-matching donor search, Gamma^%d (owners in model %d): %.0f faces, %.0f points; "
                   "fraction |R_l|>1 = %.6f, average |R_l| = %.6f, average |K_l| = %.6f, "
                   "max distance = %.6e, points off donor face = %.0f\n",
                   dir + 1, dir + 1, s[0], s[1], s[2] / nf, s[3] / nf, s[4] / nf, gmaxd[dir], s[5]);
        }
    }
}

// Entry point, called from cpuInit right after build_runtime_interface_partition.
// Returns false (and does nothing) unless interfacetype == 1 and the coupled
// guards of build_runtime_interface_partition hold. Otherwise every rank of both
// models takes part (collective on EXASIM_COMM_WORLD). If result is non-null the
// per-point tuples are returned in it (phase 3 consumes them).
template <class T, class I>
inline bool setup_nonmatching_interface(appstructT<T, I>& app, masterstructT<T, I>& master,
        meshstructT<T, I>& mesh, solstructT<T, I>& sol, I mpiprocs, I mpirank, I fileoffset,
        nonmatchingdataT<T, I>& nm, DonorSearchResult<T, I>* result = nullptr,
        const DonorSearchParams& prm = DonorSearchParams())
{
    // Same guards as build_runtime_interface_partition.
    if (mpiprocs <= 1)
        return false;
    if (!app.problem || !mesh.ndims || !mesh.bf || !mesh.perm || !sol.xdg)
        return false;
    if (!app.nsize || app.nsize[2] <= 30)
        return false;
    if (app.problem[28] <= 0 || app.problem[29] <= 0 || app.problem[30] <= 0)
        return false;
    nm.interfacetype = (app.nsize[2] > 34) ? app.problem[34] : 0;
    if (nm.interfacetype != 1)
        return false;

    const I nd = master.ndims[0], elemtype = master.ndims[1], porder = master.ndims[3];
    const I npe = master.ndims[5], npf = master.ndims[6], ngf = master.ndims[8];
    const I nfe = mesh.ndims[4];
    const I own = (fileoffset > 0) ? 1 : 0;

    // One master element on both models (phase 0 decision); phase 3 adds pgauss.
    {
        const I mine[6] = {nd, elemtype, porder, npf, nfe, ngf};
        std::vector<I> all(static_cast<std::size_t>(6 * mpiprocs));
        MPI_Allgather(mine, 6, mpi_type<I>(), all.data(), 6, mpi_type<I>(), EXASIM_COMM_WORLD);
        for (I r = 0; r < mpiprocs; r++)
            for (int k = 0; k < 6; k++)
                if (all[6 * r + k] != mine[k])
                    error("non-matching interface: both models must use the same master element "
                          "(nd, elemtype, porder, npf, nfe, ngf)");
    }

    const InterfaceFaceSet<T, I> ownFaces = collect_own_interface_faces(app, master, mesh, sol, mpirank, fileoffset);
    const InterfaceFaceSet<T, I> allFaces = gather_interface_faces(ownFaces, mpiprocs);
    const InterfaceFaceSet<T, I> candFaces = select_model_faces(allFaces, static_cast<I>(1 - own));

    DonorSearchResult<T, I> res;
    res.direction = own;
    own_quadrature_points(res, ownFaces, master.shapfgt, ngf);
    donor_search(res, candFaces, master.xpf, elemtype, porder, prm);
    for (I p = 0; p < res.npts; p++)
        if (!res.converged[p]) {
            std::ostringstream oss;
            oss << "non-matching donor search: projection onto donor face " << res.f[p]
                << " did not converge on rank " << mpirank << ", point " << p;
            error(oss.str());
        }

    for (int k = 0; k < 2; k++) {
        nonmatchinginterfaceT<T, I>& d = nm.dir[k];
        d.direction = k;
        d.nd = nd; d.npe = npe; d.npf = npf; d.nfe = nfe; d.ngf = ngf;
        d.ncu12 = (app.nsize[14] > 0) ? app.nsize[14] : 0; // szinterfacefluxmap
    }
    build_owner_side(nm.dir[own], res);
    print_donor_diagnostics(res, mpirank, prm);

    if (result) *result = std::move(res);
    return true;
}
#else
template <class T, class I>
inline bool setup_nonmatching_interface(appstructT<T, I>&, masterstructT<T, I>&, meshstructT<T, I>&,
        solstructT<T, I>&, I, I, I, nonmatchingdataT<T, I>&, DonorSearchResult<T, I>* = nullptr,
        const DonorSearchParams& = DonorSearchParams())
{
    return false; // no coupling without MPI (phase 0 report, D7)
}
#endif

} // namespace nonmatching
} // namespace exasim

#endif
