#ifndef __INTERFACEPARTITION_HPP
#define __INTERFACEPARTITION_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace exasim {
namespace interfacepartition {

template <class T, class I>
struct LocalInterfaceFace {
    I localFace = 0;
    std::vector<T> center;
    std::vector<T> nodes;
};

template <class T>
inline T point_distance2(const T* a, const T* b, const int nd)
{
    T d2 = 0;
    for (int d = 0; d < nd; d++) {
        const T diff = a[d] - b[d];
        d2 += diff * diff;
    }
    return d2;
}

template <class T>
inline bool same_point(const T* a, const T* b, const int nd, const T tol2)
{
    return point_distance2(a, b, nd) < tol2;
}

template <class T, class I>
inline bool same_face_nodes(const T* localNodes, const T* remoteNodes,
        const I npf, const I nd, const T tol2)
{
    std::vector<char> used(static_cast<std::size_t>(npf), 0);
    for (I n = 0; n < npf; n++) {
        I matchNode = -1;
        for (I m = 0; m < npf; m++) {
            if (used[static_cast<std::size_t>(m)])
                continue;
            bool equal = true;
            for (I d = 0; d < nd; d++) {
                const T* a = localNodes + n + d * npf;
                const T* b = remoteNodes + m + d * npf;
                if (!same_point(a, b, 1, tol2)) {
                    equal = false;
                    break;
                }
            }
            if (!equal)
                continue;
            matchNode = m;
            break;
        }
        if (matchNode < 0)
            return false;
        used[static_cast<std::size_t>(matchNode)] = 1;
    }
    return true;
}

template <class T, class I>
inline I mesh_nsize(const meshstructT<T, I>& mesh, const I slot)
{
    return (mesh.nsize && mesh.lsize && mesh.lsize[0] > slot) ? mesh.nsize[slot] : 0;
}

template <class T, class I>
inline void require_mesh_nsize_slots(meshstructT<T, I>& mesh, const I minSlots)
{
    if (mesh.nsize && mesh.lsize && mesh.lsize[0] >= minSlots)
        return;

    if (!mesh.lsize)
        error("runtime interface partition requires mesh.lsize to be allocated");

    const I oldSlots = (mesh.lsize[0] > 0) ? mesh.lsize[0] : 0;
    I* newNsize = nullptr;
    TemplateMalloc(&newNsize, minSlots, 0);
    for (I i = 0; i < minSlots; i++)
        newNsize[i] = (mesh.nsize && i < oldSlots) ? mesh.nsize[i] : 0;

    if (mesh.nsize)
        CPUFREE(mesh.nsize);
    mesh.nsize = newNsize;
    mesh.lsize[0] = minSlots;
}

template <class I>
inline void verify_existing_array(const char* name, const I rank, const I* stored,
        const I storedSize, const std::vector<I>& generated)
{
    if (storedSize == 0)
        return;

    if (stored == nullptr)
        error(std::string("runtime interface partition reference array is null: ") + name);

    if (storedSize != static_cast<I>(generated.size())) {
        std::ostringstream oss;
        oss << "runtime interface partition size mismatch for " << name
            << " on rank " << rank << ": stored size = " << storedSize
            << ", generated size = " << generated.size();
        error(oss.str());
    }

    for (I i = 0; i < storedSize; i++) {
        if (stored[i] != generated[static_cast<std::size_t>(i)]) {
            std::ostringstream oss;
            oss << "runtime interface partition value mismatch for " << name
                << " on rank " << rank << " at index " << i
                << ": stored = " << stored[i]
                << ", generated = " << generated[static_cast<std::size_t>(i)];
            error(oss.str());
        }
    }
}

template <class I>
inline void replace_mesh_array(I*& ptr, I& sizeField, I* nsize, const I slot,
        const std::vector<I>& values)
{
    if (ptr)
        CPUFREE(ptr);

    ptr = nullptr;
    const I n = static_cast<I>(values.size());
    if (n > 0) {
        TemplateMalloc(&ptr, n, 0);
        for (I i = 0; i < n; i++)
            ptr[i] = values[static_cast<std::size_t>(i)];
    }

    sizeField = n;
    nsize[slot] = n;
}

template <class T, class I>
inline std::vector<LocalInterfaceFace<T, I>> collect_local_interface_faces(
        const appstructT<T, I>& app, const masterstructT<T, I>& master,
        const meshstructT<T, I>& mesh, const solstructT<T, I>& sol)
{
    std::vector<LocalInterfaceFace<T, I>> faces;

    const I bc = app.problem[30];
    const I nd = master.ndims[0];
    const I npe = master.ndims[5];
    const I npf = master.ndims[6];
    const I ne = mesh.ndims[1];
    const I nfe = mesh.ndims[4];

    I ne1 = ne;
    if (mesh.elempartpts && mesh_nsize(mesh, static_cast<I>(10)) >= 2)
        ne1 = std::min(ne, mesh.elempartpts[0] + mesh.elempartpts[1]);

    I faceOrdinal = 0;
    for (I e = 0; e < ne1; e++) {
        for (I lf = 0; lf < nfe; lf++) {
            if (mesh.bf[lf + nfe * e] != bc)
                continue;

            LocalInterfaceFace<T, I> face;
            face.localFace = faceOrdinal++;
            face.center.assign(static_cast<std::size_t>(nd), static_cast<T>(0));
            face.nodes.assign(static_cast<std::size_t>(npf * nd), static_cast<T>(0));

            for (I n = 0; n < npf; n++) {
                const I node = mesh.perm[n + npf * lf];
                for (I d = 0; d < nd; d++) {
                    const T x = sol.xdg[node + d * npe + e * npe * nd];
                    face.nodes[static_cast<std::size_t>(n + npf * d)] = x;
                    face.center[static_cast<std::size_t>(d)] += x / static_cast<T>(npf);
                }
            }
            faces.push_back(std::move(face));
        }
    }

    return faces;
}

#ifdef HAVE_MPI
template <class T, class I>
inline void build_runtime_interface_partition(appstructT<T, I>& app,
        masterstructT<T, I>& master, meshstructT<T, I>& mesh, solstructT<T, I>& sol,
        I mpiprocs, I mpirank, I fileoffset)
{
    if (mpiprocs <= 1)
        return;
    if (!app.problem || !mesh.ndims || !mesh.bf || !mesh.perm || !sol.xdg)
        return;
    if (!app.nsize || app.nsize[2] <= 30)
        return;
    if (app.problem[28] <= 0 || app.problem[29] <= 0 || app.problem[30] <= 0)
        return;

    const I nd = master.ndims[0];
    const I npf = master.ndims[6];
    const T tol2 = static_cast<T>(1.0e-12);

    const auto localFaces = collect_local_interface_faces(app, master, mesh, sol);
    const I nlocal = static_cast<I>(localFaces.size());

    std::vector<I> faceCounts(static_cast<std::size_t>(mpiprocs), 0);
    MPI_Allgather(&nlocal, 1, mpi_type<I>(), faceCounts.data(), 1, mpi_type<I>(), EXASIM_COMM_WORLD);

    std::vector<int> metaCounts(static_cast<std::size_t>(mpiprocs), 0);
    std::vector<int> metaDispls(static_cast<std::size_t>(mpiprocs), 0);
    std::vector<int> coordCounts(static_cast<std::size_t>(mpiprocs), 0);
    std::vector<int> coordDispls(static_cast<std::size_t>(mpiprocs), 0);

    I totalFaces = 0;
    for (I r = 0; r < mpiprocs; r++) {
        if (faceCounts[static_cast<std::size_t>(r)] >
            static_cast<I>(std::numeric_limits<int>::max() / std::max<I>(1, npf * nd + nd)))
            error("too many coupled interface faces for MPI_Allgatherv integer counts");

        metaCounts[static_cast<std::size_t>(r)] = static_cast<int>(3 * faceCounts[static_cast<std::size_t>(r)]);
        coordCounts[static_cast<std::size_t>(r)] = static_cast<int>((nd + npf * nd) * faceCounts[static_cast<std::size_t>(r)]);
        if (r > 0) {
            metaDispls[static_cast<std::size_t>(r)] =
                metaDispls[static_cast<std::size_t>(r - 1)] + metaCounts[static_cast<std::size_t>(r - 1)];
            coordDispls[static_cast<std::size_t>(r)] =
                coordDispls[static_cast<std::size_t>(r - 1)] + coordCounts[static_cast<std::size_t>(r - 1)];
        }
        totalFaces += faceCounts[static_cast<std::size_t>(r)];
    }

    std::vector<I> localMeta(static_cast<std::size_t>(3 * nlocal), 0);
    std::vector<T> localCoords(static_cast<std::size_t>((nd + npf * nd) * nlocal), static_cast<T>(0));
    for (I i = 0; i < nlocal; i++) {
        localMeta[static_cast<std::size_t>(3 * i + 0)] = mpirank;
        localMeta[static_cast<std::size_t>(3 * i + 1)] = fileoffset;
        localMeta[static_cast<std::size_t>(3 * i + 2)] = localFaces[static_cast<std::size_t>(i)].localFace;

        const I offset = i * (nd + npf * nd);
        for (I d = 0; d < nd; d++)
            localCoords[static_cast<std::size_t>(offset + d)] =
                localFaces[static_cast<std::size_t>(i)].center[static_cast<std::size_t>(d)];
        for (I k = 0; k < npf * nd; k++)
            localCoords[static_cast<std::size_t>(offset + nd + k)] =
                localFaces[static_cast<std::size_t>(i)].nodes[static_cast<std::size_t>(k)];
    }

    std::vector<I> allMeta(static_cast<std::size_t>(3 * totalFaces), 0);
    std::vector<T> allCoords(static_cast<std::size_t>((nd + npf * nd) * totalFaces), static_cast<T>(0));

    MPI_Allgatherv(localMeta.data(), static_cast<int>(localMeta.size()), mpi_type<I>(),
            allMeta.data(), metaCounts.data(), metaDispls.data(), mpi_type<I>(), EXASIM_COMM_WORLD);
    MPI_Allgatherv(localCoords.data(), static_cast<int>(localCoords.size()), mpi_type<T>(),
            allCoords.data(), coordCounts.data(), coordDispls.data(), mpi_type<T>(), EXASIM_COMM_WORLD);

    std::map<I, std::vector<std::pair<I, I>>> peerFaces;
    std::vector<I> generatedFaceperm;
    bool haveFaceperm = false;

    for (I i = 0; i < nlocal; i++) {
        const auto& local = localFaces[static_cast<std::size_t>(i)];
        const T* localCenter = local.center.data();
        const T* localNodes = local.nodes.data();

        I matchedGlobal = -1;
        for (I j = 0; j < totalFaces; j++) {
            const I remoteRank = allMeta[static_cast<std::size_t>(3 * j + 0)];
            const I remoteDomain = allMeta[static_cast<std::size_t>(3 * j + 1)];
            if (remoteRank == mpirank || remoteDomain == fileoffset)
                continue;

            const T* remoteCenter = allCoords.data() + j * (nd + npf * nd);
            if (!same_point(localCenter, remoteCenter, static_cast<int>(nd), tol2))
                continue;
            const T* remoteNodes = allCoords.data() + j * (nd + npf * nd) + nd;
            if (!same_face_nodes(localNodes, remoteNodes, npf, nd, tol2))
                continue;

            if (matchedGlobal >= 0) {
                std::ostringstream oss;
                oss << "multiple coupled interface matches for rank " << mpirank
                    << ", local face " << local.localFace;
                error(oss.str());
            }
            matchedGlobal = j;
        }

        if (matchedGlobal < 0) {
            std::ostringstream oss;
            oss << "no coupled interface match for rank " << mpirank
                << ", local face " << local.localFace;
            error(oss.str());
        }

        const I remoteRank = allMeta[static_cast<std::size_t>(3 * matchedGlobal + 0)];
        const I remoteDomain = allMeta[static_cast<std::size_t>(3 * matchedGlobal + 1)];
        const I remoteLocalFace = allMeta[static_cast<std::size_t>(3 * matchedGlobal + 2)];
        const T* remoteNodes = allCoords.data() + matchedGlobal * (nd + npf * nd) + nd;

        std::vector<I> thisFaceperm(static_cast<std::size_t>(npf), -1);
        for (I n = 0; n < npf; n++) {
            const T* localNode = localNodes + n;
            I matchNode = -1;
            for (I m = 0; m < npf; m++) {
                bool equal = true;
                for (I d = 0; d < nd; d++) {
                    const T* a = localNode + d * npf;
                    const T* b = remoteNodes + m + d * npf;
                    if (!same_point(a, b, 1, tol2)) {
                        equal = false;
                        break;
                    }
                }
                if (!equal)
                    continue;
                if (matchNode >= 0) {
                    std::ostringstream oss;
                    oss << "multiple node matches in coupled interface face permutation on rank "
                        << mpirank << ", local face " << local.localFace;
                    error(oss.str());
                }
                matchNode = m;
            }
            if (matchNode < 0) {
                std::ostringstream oss;
                oss << "missing node match in coupled interface face permutation on rank "
                    << mpirank << ", local face " << local.localFace;
                error(oss.str());
            }
            thisFaceperm[static_cast<std::size_t>(n)] = matchNode;
        }

        if (!haveFaceperm) {
            generatedFaceperm = thisFaceperm;
            haveFaceperm = true;
        }
        else if (generatedFaceperm != thisFaceperm) {
            std::ostringstream oss;
            oss << "inconsistent coupled interface face permutation on rank " << mpirank
                << ", local face " << local.localFace;
            error(oss.str());
        }

        // The MATLAB reference orders both domains by the lower-fileoffset side's face ordinal.
        // That canonical key preserves the same send/receive face order on both sides.
        const I canonicalFace = (fileoffset < remoteDomain) ? local.localFace : remoteLocalFace;
        peerFaces[remoteRank].push_back(std::make_pair(canonicalFace, local.localFace));
    }

    std::vector<I> generatedNbintf;
    std::vector<I> generatedFaces;
    std::vector<I> generatedPts;
    for (auto& peer : peerFaces) {
        std::sort(peer.second.begin(), peer.second.end(),
                [](const std::pair<I, I>& a, const std::pair<I, I>& b) {
                    return (a.first < b.first) || (a.first == b.first && a.second < b.second);
                });
        generatedNbintf.push_back(peer.first);
        generatedPts.push_back(static_cast<I>(peer.second.size()));
        for (const auto& item : peer.second)
            generatedFaces.push_back(item.second);
    }

    require_mesh_nsize_slots(mesh, static_cast<I>(45));

    verify_existing_array("mesh.faceperm", mpirank, mesh.faceperm, mesh.szfaceperm, generatedFaceperm);
    verify_existing_array("mesh.nbintf", mpirank, mesh.nbintf, mesh.sznbintf, generatedNbintf);
    verify_existing_array("mesh.facesend", mpirank, mesh.facesend, mesh.szfacesend, generatedFaces);
    verify_existing_array("mesh.facesendpts", mpirank, mesh.facesendpts, mesh.szfacesendpts, generatedPts);
    verify_existing_array("mesh.facerecv", mpirank, mesh.facerecv, mesh.szfacerecv, generatedFaces);
    verify_existing_array("mesh.facerecvpts", mpirank, mesh.facerecvpts, mesh.szfacerecvpts, generatedPts);

    replace_mesh_array(mesh.faceperm, mesh.szfaceperm, mesh.nsize, static_cast<I>(39), generatedFaceperm);
    replace_mesh_array(mesh.nbintf, mesh.sznbintf, mesh.nsize, static_cast<I>(40), generatedNbintf);
    replace_mesh_array(mesh.facesend, mesh.szfacesend, mesh.nsize, static_cast<I>(41), generatedFaces);
    replace_mesh_array(mesh.facesendpts, mesh.szfacesendpts, mesh.nsize, static_cast<I>(42), generatedPts);
    replace_mesh_array(mesh.facerecv, mesh.szfacerecv, mesh.nsize, static_cast<I>(43), generatedFaces);
    replace_mesh_array(mesh.facerecvpts, mesh.szfacerecvpts, mesh.nsize, static_cast<I>(44), generatedPts);
}
#else
template <class T, class I>
inline void build_runtime_interface_partition(appstructT<T, I>&,
        masterstructT<T, I>&, meshstructT<T, I>&, solstructT<T, I>&, I, I, I)
{
}
#endif

} // namespace interfacepartition
} // namespace exasim

#endif
