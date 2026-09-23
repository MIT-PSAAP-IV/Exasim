#ifndef __MESHDIST_HPP__
#define __MESHDIST_HPP__

#include <cmath>
#include <limits>
#include <vector>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef HAVE_MPI
using MeshDistanceComm = MPI_Comm;
#else
using MeshDistanceComm = int;
#endif

#ifdef HAVE_MPI
template <class T> inline MPI_Datatype meshdist_mpi_type();
template <> inline MPI_Datatype meshdist_mpi_type<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype meshdist_mpi_type<float>()  { return MPI_FLOAT;  }
#endif

template <class I>
inline int meshdist_has_boundary(const I* boundary_ids, I n_boundary_ids, I ib)
{
    for (I i = 0; i < n_boundary_ids; ++i)
        if (boundary_ids[i] == ib) return 1;
    return 0;
}

template <class T, class I>
inline void computeMeshDistanceToBoundary(T* dist, const T* xdg, const I* bf, const I* perm,
        const I* boundary_ids, I n_boundary_ids, I npe, I npf, I nd, I nfe,
        I ne_local, MeshDistanceComm mpi_comm)
{
    if (n_boundary_ids <= 0)
        error("distanceboundaryconditions must contain at least one boundary-condition ID.");
    std::vector<T> p;
    for (I e = 0; e < ne_local; ++e)
        for (I f = 0; f < nfe; ++f)
            if (meshdist_has_boundary(boundary_ids, n_boundary_ids, bf[f + nfe*e]))
                for (I a = 0; a < npf; ++a) {
                    I j = perm[a + npf*f];
                    for (I d = 0; d < nd; ++d)
                        p.push_back(xdg[j + npe*(d + nd*e)]);
                }

#ifdef HAVE_MPI
    if (mpi_comm != MPI_COMM_NULL) {
        int nproc = 1;
        MPI_Comm_size(mpi_comm, &nproc);
        if (nproc > 1) {
            int nsend = static_cast<int>(p.size());
            std::vector<int> counts(nproc), displs(nproc);
            MPI_Allgather(&nsend, 1, MPI_INT, counts.data(), 1, MPI_INT, mpi_comm);
            int nrecv = 0;
            for (int r = 0; r < nproc; ++r) {
                displs[r] = nrecv;
                nrecv += counts[r];
            }
            std::vector<T> pg(nrecv);
            MPI_Allgatherv(p.data(), nsend, meshdist_mpi_type<T>(),
                           pg.data(), counts.data(), displs.data(), meshdist_mpi_type<T>(),
                           mpi_comm);
            p.swap(pg);
        }
    }
#endif

    I np = static_cast<I>(p.size()/nd);
    if (np <= 0)
        error("No mesh faces match distanceboundaryconditions.");
    for (I e = 0; e < ne_local; ++e)
        for (I j = 0; j < npe; ++j) {
            T dmin = std::numeric_limits<T>::max();
            for (I k = 0; k < np; ++k) {
                T d2 = 0;
                for (I m = 0; m < nd; ++m) {
                    T dx = xdg[j + npe*(m + nd*e)] - p[m + nd*k];
                    d2 += dx*dx;
                }
                if (d2 < dmin) dmin = d2;
            }
            dist[j + npe*e] = std::sqrt(dmin);
        }
}

template <class T, class I>
inline void computeNearestBoundaryNodeDistance(T* dist, const T* xdg,
        const T* boundaryCoordinates, I numberBoundaryNodes,
        I npe, I ncx, I nd, I ne)
{
    if (numberBoundaryNodes <= 0)
        error("No mesh faces match distanceboundaryconditions.");
    const I numberNodes = npe*ne;
    Kokkos::parallel_for(
        "MeshAdaptNearestBoundaryDistance",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<I>>(0, numberNodes),
        KOKKOS_LAMBDA(const I index) {
            const I i = index % npe;
            const I e = index / npe;
            T minimum = std::numeric_limits<T>::max();
            for (I b = 0; b < numberBoundaryNodes; ++b) {
                T squaredDistance = 0;
                for (I d = 0; d < nd; ++d) {
                    const T difference = xdg[i + npe*d + npe*ncx*e]
                                       - boundaryCoordinates[d + nd*b];
                    squaredDistance += difference*difference;
                }
                if (squaredDistance < minimum) minimum = squaredDistance;
            }
            dist[index] = Kokkos::sqrt(minimum);
        });
}

#endif
