#ifndef EXASIM_UNIFORMREFINEMENTPAR_HPP
#define EXASIM_UNIFORMREFINEMENTPAR_HPP

/*
 * uniformrefinementpar.hpp
 *
 * MPI driver for `uniformrefinementlevel = k` (see uniformrefinement.hpp).
 *
 * Refinement runs on every rank after callParMetis has distributed the COARSE
 * mesh, so children stay on their parent's rank and no rank ever holds more
 * than its share of the refined mesh. Global ids stay consistent:
 *
 *   - child c of coarse element g gets global id g*nchild + c;
 *   - a new vertex is keyed by the sorted global ids of its supporting parent
 *     vertices; keys that can be shared (edge/face vertices) are numbered by
 *     the rank hashKey(key) % size, so all ranks touching an edge agree on its
 *     midpoint's id without knowing who their neighbours are; cell-interior
 *     vertices are numbered by their own rank.
 *
 * The xdg/udg/vdg/wdg files are indexed by coarse global element id, so they
 * are read here, before the ids are renumbered, and prolongated. initializeDMD
 * and writesol skip their own reads when uniformrefinementlevel > 0.
 */

#include "uniformrefinement.hpp"

namespace exasim_uref {

// Assign global ids to the nnew vertices created on this rank; returns the
// number of new vertices across all ranks.
inline long long numberNewVerticesParallel(std::vector<int>& newgid, const RefinedLevel& L,
                                           int np_global, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    constexpr int K = (int)Key{}.size();
    const int nnew = L.nnew;
    newgid.assign(nnew, -1);

    // 1. send each shareable key to its owner
    std::vector<int> sendcounts(size, 0), dest(nnew, -1);
    int ninterior = 0;
    for (int i = 0; i < nnew; i++) {
        if (L.interior[i]) { ninterior++; continue; }
        dest[i] = (int)(hashKey(L.keys[i]) % (std::uint64_t)size);
        sendcounts[dest[i]]++;
    }
    std::vector<int> sdispl(size + 1, 0);
    for (int r = 0; r < size; r++) sdispl[r+1] = sdispl[r] + sendcounts[r];
    const int nsend = sdispl[size];

    std::vector<int> sendpos(nnew, -1), cursor(sdispl.begin(), sdispl.end() - 1);
    std::vector<int> sendbuf((std::size_t)K*nsend);
    for (int i = 0; i < nnew; i++) {
        if (dest[i] < 0) continue;
        const int pos = cursor[dest[i]]++;
        sendpos[i] = pos;
        std::copy(L.keys[i].begin(), L.keys[i].end(), sendbuf.begin() + (std::size_t)K*pos);
    }

    std::vector<int> recvcounts(size, 0);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    std::vector<int> rdispl(size + 1, 0);
    for (int r = 0; r < size; r++) rdispl[r+1] = rdispl[r] + recvcounts[r];
    const int nrecv = rdispl[size];

    std::vector<int> sc(size), sd(size), rc(size), rd(size);
    for (int r = 0; r < size; r++) {
        sc[r] = K*sendcounts[r]; sd[r] = K*sdispl[r];
        rc[r] = K*recvcounts[r]; rd[r] = K*rdispl[r];
    }
    std::vector<int> recvbuf((std::size_t)K*nrecv);
    MPI_Alltoallv(sendbuf.data(), sc.data(), sd.data(), MPI_INT,
                  recvbuf.data(), rc.data(), rd.data(), MPI_INT, comm);

    // 2. the owner numbers the distinct keys it received (from any rank, itself included)
    auto keyAt = [&](int j) {
        Key k;
        std::copy(recvbuf.begin() + (std::size_t)K*j, recvbuf.begin() + (std::size_t)K*(j + 1), k.begin());
        return k;
    };
    std::vector<int> order(nrecv);
    for (int j = 0; j < nrecv; j++) order[j] = j;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return keyAt(a) < keyAt(b); });
    std::vector<int> recvidx(nrecv);
    int nuniq = 0;
    for (int j = 0; j < nrecv; j++) {
        if (j > 0 && keyAt(order[j]) != keyAt(order[j-1])) nuniq++;
        recvidx[order[j]] = nuniq;
    }
    if (nrecv > 0) nuniq++;

    // 3. contiguous id ranges per rank
    long long mine = (long long)nuniq + ninterior;
    std::vector<long long> counts(size);
    MPI_Allgather(&mine, 1, MPI_LONG_LONG, counts.data(), 1, MPI_LONG_LONG, comm);
    long long offset = 0, total = 0;
    for (int r = 0; r < size; r++) { if (r < rank) offset += counts[r]; total += counts[r]; }
    if ((long long)np_global + total > INT_MAX)
        error("uniformrefinement: refined vertex count overflows the int node index.\n");
    const int base = np_global + (int)offset;

    // 4. reply with the ids and fill in the interior vertices
    std::vector<int> reply(nrecv), answer(nsend);
    for (int j = 0; j < nrecv; j++) reply[j] = base + recvidx[j];
    MPI_Alltoallv(reply.data(), recvcounts.data(), rdispl.data(), MPI_INT,
                  answer.data(), sendcounts.data(), sdispl.data(), MPI_INT, comm);

    int next = base + nuniq;
    for (int i = 0; i < nnew; i++)
        newgid[i] = L.interior[i] ? next++ : answer[sendpos[i]];
    return total;
}

} // namespace exasim_uref

// Call between callParMetis and initializeDMD.
inline void uniformRefineParMesh(Mesh& mesh, PDE& pde, const Master& master, MPI_Comm comm)
{
    using namespace exasim_uref;

    const int nlevel = pde.uniformrefinementlevel;
    if (nlevel <= 0) return;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (!pde.uhatfile.empty())
        error("uniformrefinementlevel > 0 cannot prolongate the face-based uhatfile.\n");

    const bool hasxdg = !pde.xdgfile.empty(), hasudg = !pde.udgfile.empty();
    const bool hasvdg = !pde.vdgfile.empty(), haswdg = !pde.wdgfile.empty();
    if (hasxdg) readParFieldFromBinaryFile(make_path(pde.datapath, pde.xdgfile), mesh.elemGlobalID, mesh.xdg, mesh.xdgdims);
    if (hasudg) readParFieldFromBinaryFile(make_path(pde.datapath, pde.udgfile), mesh.elemGlobalID, mesh.udg, mesh.udgdims);
    if (hasvdg) readParFieldFromBinaryFile(make_path(pde.datapath, pde.vdgfile), mesh.elemGlobalID, mesh.vdg, mesh.vdgdims);
    if (haswdg) readParFieldFromBinaryFile(make_path(pde.datapath, pde.wdgfile), mesh.elemGlobalID, mesh.wdg, mesh.wdgdims);

    const RefinementTemplate T = makeRefinementTemplate(mesh.nd, mesh.elemtype, mesh.nve);
    const RefinementProlongation R = makeRefinementProlongation(T, master.xpe, master.porder, master.npe);

    long long ne_local = mesh.ne, ne_global = 0;
    MPI_Allreduce(&ne_local, &ne_global, 1, MPI_LONG_LONG, MPI_SUM, comm);
    checkRefinedSize(ne_global, T.nchild, nlevel);
    int maxgid = -1;
    for (int g : mesh.nodeGlobalID) maxgid = std::max(maxgid, g);
    int np_global = 0;
    MPI_Allreduce(&maxgid, &np_global, 1, MPI_INT, MPI_MAX, comm);
    np_global = std::max(np_global + 1, mesh.np_global);
    const long long ne0 = ne_global;
    const int np0 = np_global;

    RefinedLevel L;
    std::vector<int> newgid, elemgid;
    for (int level = 0; level < nlevel; level++) {
        refineConnectivity(L, T, R, mesh.p, mesh.t, mesh.nodeGlobalID.data(), mesh.np, mesh.ne,
                           hasxdg ? mesh.xdg.data() : nullptr);
        const long long nnew_global = numberNewVerticesParallel(newgid, L, np_global, comm);
        mesh.nodeGlobalID.insert(mesh.nodeGlobalID.end(), newgid.begin(), newgid.end());

        elemgid.resize((std::size_t)mesh.ne*T.nchild);
        for (int e = 0; e < mesh.ne; e++)
            for (int c = 0; c < T.nchild; c++)
                elemgid[(std::size_t)e*T.nchild + c] = mesh.elemGlobalID[e]*T.nchild + c;
        mesh.elemGlobalID.swap(elemgid);

        if (hasxdg) prolongField(mesh.xdg, R.npe, mesh.ne, T.nchild, R.P, "xdg");
        if (hasudg) prolongField(mesh.udg, R.npe, mesh.ne, T.nchild, R.P, "udg");
        if (hasvdg) prolongField(mesh.vdg, R.npe, mesh.ne, T.nchild, R.P, "vdg");
        if (haswdg) prolongField(mesh.wdg, R.npe, mesh.ne, T.nchild, R.P, "wdg");

        mesh.p.swap(L.p);
        mesh.t.swap(L.t);
        mesh.np += L.nnew;
        mesh.ne *= T.nchild;
        np_global += (int)nnew_global;
        ne_global *= T.nchild;
    }
    setFieldDims(mesh.xdgdims, mesh.ne);
    setFieldDims(mesh.udgdims, mesh.ne);
    setFieldDims(mesh.vdgdims, mesh.ne);
    setFieldDims(mesh.wdgdims, mesh.ne);
    mesh.np_global = np_global;
    mesh.ne_global = (int)ne_global;
    pde.np = np_global;
    pde.ne = (int)ne_global;

    // callParMetis built the element neighbours of the coarse mesh.
    mesh.t2t.assign((std::size_t)mesh.nfe*mesh.ne, -1);
    mke2e_global(mesh.t2t.data(), mesh.t.data(), mesh.localfaces.data(),
                 mesh.elemGlobalID.data(), mesh.ne, mesh.nve, mesh.nvf, mesh.nfe, rank);

    if (rank == 0)
        std::printf("uniformrefinementlevel = %d: ne %lld -> %lld, np %d -> %d (global, %d ranks)\n",
                    nlevel, ne0, ne_global, np0, np_global, size);
}

#endif
