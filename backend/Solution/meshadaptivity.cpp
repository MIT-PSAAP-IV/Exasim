#ifndef __MESHADAPTIVITY
#define __MESHADAPTIVITY

#include "../Preprocessing/makemaster.hpp"
#include "../Preprocessing/meshdist.hpp"
#include "distributedradixquantiles.hpp"

namespace exasim_meshadapt {

KOKKOS_INLINE_FUNCTION
dstype determinant(const dstype J[3][3], Int nd)
{
    if (nd == 2) return J[0][0]*J[1][1] - J[0][1]*J[1][0];
    return J[0][0]*J[1][1]*J[2][2] - J[0][0]*J[1][2]*J[2][1]
         + J[0][1]*J[1][2]*J[2][0] - J[0][1]*J[1][0]*J[2][2]
         + J[0][2]*J[1][0]*J[2][1] - J[0][2]*J[1][1]*J[2][0];
}

inline std::vector<dstype> invert(const std::vector<dstype>& columnMajor, Int n)
{
    std::vector<dstype> a(n*2*n, 0.0);
    for (Int i = 0; i < n; ++i) {
        for (Int j = 0; j < n; ++j) a[i*(2*n) + j] = columnMajor[i + n*j];
        a[i*(2*n) + n+i] = 1.0;
    }
    for (Int k = 0; k < n; ++k) {
        Int pivot = k;
        for (Int i = k+1; i < n; ++i)
            if (std::abs(a[i*(2*n)+k]) > std::abs(a[pivot*(2*n)+k])) pivot = i;
        if (std::abs(a[pivot*(2*n)+k]) < 1.0e-14)
            error("Mesh-adaptivity modal Vandermonde matrix is singular.");
        if (pivot != k)
            for (Int j = 0; j < 2*n; ++j) std::swap(a[k*(2*n)+j], a[pivot*(2*n)+j]);
        const dstype diagonal = a[k*(2*n)+k];
        for (Int j = 0; j < 2*n; ++j) a[k*(2*n)+j] /= diagonal;
        for (Int i = 0; i < n; ++i) if (i != k) {
            const dstype factor = a[i*(2*n)+k];
            for (Int j = 0; j < 2*n; ++j) a[i*(2*n)+j] -= factor*a[k*(2*n)+j];
        }
    }
    std::vector<dstype> inverse(n*n);
    for (Int i = 0; i < n; ++i)
        for (Int j = 0; j < n; ++j) inverse[i + n*j] = a[i*(2*n)+n+j];
    return inverse;
}

inline void writeVerificationField(const string& prefix, const string& name,
    const std::vector<dstype>& values, Int rank)
{
    if (values.empty()) return;
    const string filename = prefix + "_meshadapt_" + name + "_np" +
        NumberToString(rank) + ".bin";
    writearray2file(filename, const_cast<dstype*>(values.data()),
                    static_cast<Int>(values.size()));
}

template <class D>
void exchangeElementField(D& disc, dstype* field, const Int* sendIndices,
                          const Int* receiveIndices, Int blockSize);

template <class D>
void exchangeElementField(D& disc, dstype* field, Int blockSize);

template <class D>
void smoothDG2CG2(D& disc, dstype* field, dstype* scratch, Int components,
                  Int passes, Int backend)
{
    for (Int pass = 0; pass < passes; ++pass) {
        disc.DG2CG2(field, field, scratch, components, components, components, backend);
        // exchangeElementField(disc, field, disc.common.grid.npe*components);
    }
}

template <class D>
void exchangeElementField(D& disc, dstype* field, const Int* sendIndices,
                          const Int* receiveIndices, Int blockSize)
{
#ifdef HAVE_MPI
    if (disc.common.mpiProcs <= 1) return;

    GetArrayAtIndex(disc.tmp.buffsend, field, sendIndices,
                    blockSize*disc.common.nelemsend);
    Kokkos::fence();

    Int sendOffset = 0, receiveOffset = 0, requestCount = 0;
    for (Int n = 0; n < disc.common.nnbsd; ++n) {
        const Int count = disc.common.elemsendpts[n]*blockSize;
        if (count > 0) {
            MPI_Isend(&disc.tmp.buffsend[sendOffset], count, mpi_type<dstype>(),
                      disc.common.nbsd[n], 0, EXASIM_COMM_LOCAL,
                      &disc.common.requests[requestCount++]);
            sendOffset += count;
        }
    }
    for (Int n = 0; n < disc.common.nnbsd; ++n) {
        const Int count = disc.common.elemrecvpts[n]*blockSize;
        if (count > 0) {
            MPI_Irecv(&disc.tmp.buffrecv[receiveOffset], count, mpi_type<dstype>(),
                      disc.common.nbsd[n], 0, EXASIM_COMM_LOCAL,
                      &disc.common.requests[requestCount++]);
            receiveOffset += count;
        }
    }
    MPI_Waitall(requestCount, disc.common.requests, disc.common.statuses);
    PutArrayAtIndex(field, disc.tmp.buffrecv, receiveIndices,
                    blockSize*disc.common.nelemrecv);
#else
    (void)disc;
    (void)field;
    (void)sendIndices;
    (void)receiveIndices;
    (void)blockSize;
#endif
}

// Exchange an element field stored as [npe, components, ne].  Unlike
// elemsendudg/elemsendodg, elemsend identifies element columns and therefore
// remains valid for any component count represented by blockSize.
template <class D>
void exchangeElementField(D& disc, dstype* field, Int blockSize)
{
#ifdef HAVE_MPI
    if (disc.common.mpiProcs <= 1) return;

    const Int sendCount = blockSize*disc.common.nelemsend;
    const Int receiveCount = blockSize*disc.common.nelemrecv;
    if ((sendCount > 0 &&
         (disc.tmp.buffsend == nullptr || disc.tmp.szbuffsend < sendCount)) ||
        (receiveCount > 0 &&
         (disc.tmp.buffrecv == nullptr || disc.tmp.szbuffrecv < receiveCount)))
        error("MPI element buffers are too small for DG2CG2 smoothing.");

    GetCollumnAtIndex(disc.tmp.buffsend, field, disc.mesh.elemsend,
                      blockSize, disc.common.nelemsend);
    Kokkos::fence();

    Int sendOffset = 0, receiveOffset = 0, requestCount = 0;
    for (Int n = 0; n < disc.common.nnbsd; ++n) {
        const Int count = disc.common.elemsendpts[n]*blockSize;
        if (count > 0) {
            MPI_Isend(&disc.tmp.buffsend[sendOffset], count, mpi_type<dstype>(),
                      disc.common.nbsd[n], 0, EXASIM_COMM_LOCAL,
                      &disc.common.requests[requestCount++]);
            sendOffset += count;
        }
    }
    for (Int n = 0; n < disc.common.nnbsd; ++n) {
        const Int count = disc.common.elemrecvpts[n]*blockSize;
        if (count > 0) {
            MPI_Irecv(&disc.tmp.buffrecv[receiveOffset], count, mpi_type<dstype>(),
                      disc.common.nbsd[n], 0, EXASIM_COMM_LOCAL,
                      &disc.common.requests[requestCount++]);
            receiveOffset += count;
        }
    }
    MPI_Waitall(requestCount, disc.common.requests, disc.common.statuses);
    PutCollumnAtIndex(field, disc.tmp.buffrecv, disc.mesh.elemrecv,
                      blockSize, disc.common.nelemrecv);
#else
    (void)disc;
    (void)field;
    (void)blockSize;
#endif
}

template <class D>
void exchangeElementUDG(D& disc)
{
    exchangeElementField(disc, disc.sol.udg, disc.mesh.elemsendudg,
                         disc.mesh.elemrecvudg,
                         disc.common.grid.npe*disc.common.components.nc);
}

template <class D>
void rebuildGeometry(D& disc, const dstype* xdg, Int backend)
{
    if (disc.sol.xdg != xdg)
        ArrayCopy(disc.sol.xdg, xdg, disc.sol.szxdg);
    TemplateFree(disc.sol.elemg, backend); disc.sol.elemg = nullptr; disc.sol.szelemg = 0;
    TemplateFree(disc.sol.faceg, backend); disc.sol.faceg = nullptr; disc.sol.szfaceg = 0;
    TemplateFree(disc.sol.elemfaceg, backend); disc.sol.elemfaceg = nullptr; disc.sol.szelemfaceg = 0;
    disc.compGeometry(backend);

    // HDG gradient recovery stores M^{-1}C and M^{-1}E, both of which depend
    // on the current element and face geometry. Rebuild them after moving xdg.
    if (disc.common.spatialScheme > 0 && disc.common.components.ncq > 0) {
        TemplateFree(disc.res.C, backend); disc.res.C = nullptr; disc.res.szC = 0;
        TemplateFree(disc.res.E, backend); disc.res.E = nullptr; disc.res.szE = 0;
        qEquation(disc.sol, disc.res, disc.app, disc.master, disc.mesh, disc.tmp,
                  disc.common, backend);
        TemplateFree(disc.res.Mass2, backend); disc.res.Mass2 = nullptr; disc.res.szMass2 = 0;
        TemplateFree(disc.res.Minv2, backend); disc.res.Minv2 = nullptr; disc.res.szMinv2 = 0;
    }
}

inline void evaluateSensorError(dstype* sensor, const dstype* scalar,
    const dstype* lowScalar, const dstype* xdg, const dstype* shapegt,
    const dstype* gwe, Int npe, Int nge, Int ncx, Int ne, Int nd)
{
    Kokkos::parallel_for(
        "MeshAdaptEvaluateSensorError",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<Int>>(0, ne),
        KOKKOS_LAMBDA(const Int e) {
            dstype numerator = 0.0;
            dstype denominator = 0.0;
            for (Int g = 0; g < nge; ++g) {
                dstype highValue = 0.0;
                dstype lowValue = 0.0;
                dstype J[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
                for (Int i = 0; i < npe; ++i) {
                    const dstype shape = shapegt[g+nge*i];
                    highValue += shape*scalar[i+npe*e];
                    lowValue += shape*lowScalar[i+npe*e];
                    for (Int r = 0; r < nd; ++r)
                        for (Int d = 0; d < nd; ++d)
                            J[r][d] += shapegt[g+nge*i+nge*npe*(r+1)]
                                     * xdg[i+npe*d+npe*ncx*e];
                }
                const dstype weight = gwe[g]*determinant(J, nd);
                const dstype ratio = highValue/lowValue - 1.0;
                numerator += weight*ratio*ratio;
                denominator += weight;
            }
            const dstype value = Kokkos::sqrt(numerator/denominator);
            for (Int i = 0; i < npe; ++i) sensor[i+npe*e] = value;
        });
}

inline void limitSensor(dstype* sensor, dstype upper, Int count)
{
    const dstype alpha = 1.0e3;
    const dstype pi = 3.141592653589793238462643383279502884;
    const dstype offset = -Kokkos::atan(alpha)/pi + 0.5;
    Kokkos::parallel_for(
        "MeshAdaptLimitSensor",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<Int>>(0, count),
        KOKKOS_LAMBDA(const Int i) {
            const dstype value = sensor[i];
            const dstype lower = value*(Kokkos::atan(alpha*value)/pi + 0.5) + offset;
            const dstype shifted = lower-upper;
            const dstype capped = shifted*(Kokkos::atan(alpha*shifted)/pi + 0.5) + offset;
            sensor[i] = lower-capped;
        });
}

inline void nodalJacobian(dstype* jac, const dstype* xdg, const dstype* shapent,
    Int npe, Int ncx, Int ne, Int nd)
{
    Kokkos::parallel_for(
        "MeshAdaptNodalJacobian",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<Int>>(0, npe*ne),
        KOKKOS_LAMBDA(const Int index) {
            const Int g = index % npe;
            const Int e = index / npe;
            dstype J[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
            for (Int r = 0; r < nd; ++r)
                for (Int d = 0; d < nd; ++d)
                    for (Int i = 0; i < npe; ++i)
                        J[r][d] += shapent[g+npe*i+npe*npe*(r+1)]
                                 * xdg[i+npe*d+npe*ncx*e];
            jac[index] = determinant(J, nd);
        });
}

inline void elementSizeAndMeans(dstype* currentSize, dstype* means,
    const dstype* jac, Int npe, Int ne, Int ne1, Int nd)
{
    Kokkos::parallel_for(
        "MeshAdaptElementSizeAndMeans",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<Int>>(0, ne),
        KOKKOS_LAMBDA(const Int e) {
            dstype mean = 0.0;
            for (Int i = 0; i < npe; ++i) {
                const dstype value = Kokkos::pow(jac[i+npe*e], 1.0/static_cast<dstype>(nd));
                currentSize[i+npe*e] = value;
                mean += value/static_cast<dstype>(npe);
            }
            if (e < ne1) means[e] = mean;
        });
}

inline void scaleSquareRoot(dstype* values, dstype coefficient, Int count)
{
    Kokkos::parallel_for(
        "MeshAdaptScaleSquareRoot",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<Int>>(0, count),
        KOKKOS_LAMBDA(const Int i) {
            values[i] = coefficient*Kokkos::sqrt(values[i]);
        });
}

inline void targetLameAndHelmholtzInput(dstype* elasticityInput,
    dstype* helmholtzInput, const dstype* indicator, const dstype* currentSize,
    dstype hmin, dstype hmax, dstype helmholtzCoefficient, dstype targetExponent,
    dstype youngModulus, dstype minimumYoungModulus, dstype poissonRatio,
    Int npe, Int ne, Int nd)
{
    Kokkos::parallel_for(
        "MeshAdaptTargetLameAndHelmholtzInput",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<Int>>(0, npe*ne),
        KOKKOS_LAMBDA(const Int index) {
            const Int i = index % npe;
            const Int e = index / npe;
            dstype bounded = indicator[index];
            if (bounded < 0.0) bounded = 0.0;
            else if (bounded > 1.0) bounded = 1.0;
            const dstype target = hmin + (hmax-hmin)*Kokkos::pow(1.0-bounded, targetExponent);
            const dstype unboundedE = youngModulus*Kokkos::pow(target/hmax, 2.0);
            const dstype E = unboundedE > minimumYoungModulus ? unboundedE : minimumYoungModulus;
            const dstype mu = E/(2.0*(1.0+poissonRatio));
            const dstype lambda = E*poissonRatio/((1.0+poissonRatio)*(1.0-2.0*poissonRatio));
            elasticityInput[i+npe*0+npe*(2+nd)*e] = mu;
            elasticityInput[i+npe*1+npe*(2+nd)*e] = lambda;
            helmholtzInput[i+npe*0+npe*2*e] = indicator[index];
            helmholtzInput[i+npe*1+npe*2*e] = helmholtzCoefficient*currentSize[index];
        });
}

inline void packElasticityForce(dstype* elasticityInput, const dstype* helmholtzUdg,
    Int npe, Int ne, Int nd)
{
    Kokkos::parallel_for(
        "MeshAdaptPackElasticityForce",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<Int>>(0, npe*nd*ne),
        KOKKOS_LAMBDA(const Int index) {
            const Int i = index % npe;
            const Int q = index / npe;
            const Int d = q % nd;
            const Int e = q / nd;
            elasticityInput[i+npe*(2+d)+npe*(2+nd)*e] =
                -helmholtzUdg[i+npe*(1+d)+npe*(1+nd)*e];
        });
}

inline dstype candidateMinimumJacobian(const dstype* xdg, const dstype* displacement,
    const dstype* shapent, dstype beta, Int npe, Int ncx, Int ne, Int nd)
{
    dstype minimum = std::numeric_limits<dstype>::max();
    const dstype maximumFinite = std::numeric_limits<dstype>::max();
    Kokkos::parallel_reduce(
        "MeshAdaptCandidateMinimumJacobian",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<Int>>(0, npe*ne),
        KOKKOS_LAMBDA(const Int index, dstype& local) {
            const Int g = index % npe;
            const Int e = index / npe;
            dstype J[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
            for (Int r = 0; r < nd; ++r)
                for (Int d = 0; d < nd; ++d)
                    for (Int i = 0; i < npe; ++i) {
                        const dstype coordinate = xdg[i+npe*d+npe*ncx*e]
                            + beta*displacement[i+npe*d+npe*nd*e];
                        J[r][d] += shapent[g+npe*i+npe*npe*(r+1)]*coordinate;
                    }
            const dstype value = determinant(J, nd);
            const dstype mapped = (value > 0.0 && value <= maximumFinite)
                ? value : -maximumFinite;
            if (mapped < local) local = mapped;
        },
        Kokkos::Min<dstype>(minimum));
    return minimum;
}

} // namespace exasim_meshadapt

template <class M>
void CSolution<M>::InitializeWallDistanceWorkspace(Int backend)
{
    using namespace exasim_meshadapt;
    auto& workspace = meshAdaptWorkspace;
    if (workspace.boundaryInitialized) return;

    // Distance boundaries are fixed by the mesh-elasticity boundary conditions,
    // so gather their coordinates once and reuse them after every mesh movement.

    const Int count = disc.common.szdistanceboundaryconditions;
    if (count <= 0)
        error("distanceboundaryconditions is required when AVdistfunction is enabled.");

    const Int npe = disc.common.grid.npe;
    const Int npf = disc.common.grid.npf;
    const Int nd = disc.common.grid.nd;
    const Int nfe = disc.common.meshsizes.nfe;
    const Int ne = disc.common.meshsizes.ne;
    if (disc.common.components.nco <= 0)
        error("Wall distance requires at least one odg component.");

    std::vector<Int> perm(npf*nfe);
    std::vector<Int> indices;
    TemplateCopytoHost(perm.data(), disc.mesh.perm, static_cast<Int>(perm.size()), backend);
    for (Int e = 0; e < ne; ++e)
        for (Int f = 0; f < nfe; ++f)
            if (meshdist_has_boundary(disc.common.distanceboundaryconditions, count,
                                      disc.mesh.bf[f+nfe*e]))
                for (Int a = 0; a < npf; ++a) {
                    const Int node = perm[a+npf*f];
                    for (Int d = 0; d < nd; ++d)
                        indices.push_back(node + npe*d + npe*disc.common.components.ncx*e);
                }

    const Int localBoundaryCoordinateCount = static_cast<Int>(indices.size());
    std::vector<int> boundaryCoordinateCounts;
    std::vector<int> boundaryCoordinateOffsets;
#ifdef HAVE_MPI
    int communicatorSize = 1;
    MPI_Comm_size(EXASIM_COMM_LOCAL, &communicatorSize);
    boundaryCoordinateCounts.resize(communicatorSize);
    boundaryCoordinateOffsets.resize(communicatorSize, 0);
    const int localCount = static_cast<int>(localBoundaryCoordinateCount);
    MPI_Allgather(&localCount, 1, MPI_INT, boundaryCoordinateCounts.data(),
                  1, MPI_INT, EXASIM_COMM_LOCAL);
    for (int rank = 0; rank < communicatorSize; ++rank) {
        boundaryCoordinateOffsets[rank] = workspace.globalBoundaryCoordinateCount;
        workspace.globalBoundaryCoordinateCount += boundaryCoordinateCounts[rank];
    }
#else
    workspace.globalBoundaryCoordinateCount = localBoundaryCoordinateCount;
#endif
    if (workspace.globalBoundaryCoordinateCount <= 0)
        error("No mesh faces match distanceboundaryconditions.");

    Int *boundaryCoordinateIndices = nullptr;
    dstype *localBoundaryCoordinates = nullptr;
    if (localBoundaryCoordinateCount > 0) {
        TemplateMalloc(&boundaryCoordinateIndices, localBoundaryCoordinateCount, backend);
        TemplateMalloc(&localBoundaryCoordinates, localBoundaryCoordinateCount, backend);
        TemplateCopytoDevice(boundaryCoordinateIndices, indices.data(),
                             localBoundaryCoordinateCount, backend);
        GetArrayAtIndex(localBoundaryCoordinates, disc.sol.xdg,
                        boundaryCoordinateIndices, localBoundaryCoordinateCount);
    }
    TemplateMalloc(&workspace.globalBoundaryCoordinates,
                   workspace.globalBoundaryCoordinateCount, backend);
    Kokkos::fence();
#ifdef HAVE_MPI
    MPI_Allgatherv(localBoundaryCoordinates,
                   localBoundaryCoordinateCount, mpi_type<dstype>(),
                   workspace.globalBoundaryCoordinates,
                   boundaryCoordinateCounts.data(), boundaryCoordinateOffsets.data(),
                   mpi_type<dstype>(),
                   EXASIM_COMM_LOCAL);
#else
    ArrayCopy(workspace.globalBoundaryCoordinates, localBoundaryCoordinates,
              localBoundaryCoordinateCount);
#endif

    TemplateFree(boundaryCoordinateIndices, backend);
    TemplateFree(localBoundaryCoordinates, backend);
    workspace.boundaryInitialized = true;
}

template <class M>
void CSolution<M>::UpdateWallDistance(Int continuationIteration, Int backend)
{
    using namespace exasim_meshadapt;
    auto& workspace = meshAdaptWorkspace;
    if (!workspace.boundaryInitialized)
        error("Wall-distance workspace was not initialized.");

    const Int npe = disc.common.grid.npe;
    const Int nd = disc.common.grid.nd;
    const Int ne = disc.common.meshsizes.ne;
    const Int nco = disc.common.components.nco;
    if (nco <= 0) error("Wall distance requires at least one odg component.");

    computeNearestBoundaryNodeDistance(disc.res.Ru, disc.sol.xdg,
        workspace.globalBoundaryCoordinates, workspace.globalBoundaryCoordinateCount/nd,
        npe, disc.common.components.ncx, nd, ne);
    ArrayInsert(disc.sol.odg, disc.res.Ru, npe, nco, ne,
                0, npe, 0, 1, 0, ne);

    const char *verificationEnvironment = std::getenv("EXASIM_MESHADAPT_VERIFY");
    if (verificationEnvironment != nullptr && string(verificationEnvironment) != "0" &&
        string(verificationEnvironment) != "") {
        const Int outputRank = disc.common.mpiRank-disc.common.outputparams.fileoffset;
        std::vector<dstype> distance(npe*ne);
        TemplateCopytoHost(distance.data(), disc.res.Ru, npe*ne, backend);
        writeVerificationField(disc.common.fileout,
            "aviter" + NumberToString(continuationIteration) + "_wall_distance",
            distance, outputRank);
    }
}

template <class M>
void CSolution<M>::computeMeshIndicator(dstype* indicator)
{
    using namespace exasim_meshadapt;
    if (indicator == nullptr)
        error("Mesh-indicator output storage is not initialized.");

    const Int backend = disc.common.backend;
    const auto& cfg = disc.common.meshadaptparams;
    const Int npe = disc.common.grid.npe;
    const Int nge = disc.common.grid.nge;
    const Int nd = disc.common.grid.nd;
    const Int ncx = disc.common.components.ncx;
    const Int ne = disc.common.meshsizes.ne;
    const Int ncAV = disc.common.physicsparams.ncAV;
    const Int nodeCount = npe*ne;
    const Int avFieldSize = ncAV*nodeCount;

    if (nd != 2 && nd != 3)
        error("Backend mesh adaptivity supports only 2D and 3D.");
    if (ncAV <= 0)
        error("Mesh adaptation requires avfield outputs.");
    if (cfg.avComponent < 1 || cfg.avComponent > ncAV)
        error("meshadaptavcomponent does not identify an available avfield component.");
    if (cfg.scalarField < 1 || cfg.scalarField > ncAV)
        error("meshadaptfield does not identify an available avfield component.");
    if (disc.res.Ru == nullptr || disc.res.szRu < nodeCount)
        error("Ru workspace is too small for mesh-adaptivity field1.");
    if (disc.res.Rq == nullptr || disc.res.szRq < avFieldSize+nodeCount)
        error("Rq workspace is too small for avfield outputs and field2.");

    auto& workspace = meshAdaptWorkspace;
    if ((workspace.lowModalBasis == nullptr) !=
        (workspace.lowModalInverse == nullptr))
        error("Mesh-adaptivity modal projection workspace is inconsistent.");
    if (workspace.lowModalBasis == nullptr) {
        std::vector<dstype> xpe(disc.master.szxpe);
        TemplateCopytoHost(xpe.data(), disc.master.xpe, disc.master.szxpe, backend);
        std::vector<double> nodes(npe*nd), basisDouble(npe*npe);
        for (Int i = 0; i < npe*nd; ++i)
            nodes[i] = static_cast<double>(xpe[i]);
        if (disc.common.grid.elemtype == 0)
            koornwinder(basisDouble.data(), nodes.data(), npe,
                        disc.common.grid.porder, nd, 0);
        else
            tensorproduct(basisDouble.data(), nodes.data(), npe,
                          disc.common.grid.porder, nd, 0);

        std::vector<dstype> basis(npe*npe);
        for (Int i = 0; i < npe*npe; ++i)
            basis[i] = static_cast<dstype>(basisDouble[i]);
        const std::vector<dstype> inverse = invert(basis, npe);
        workspace.lowModeCount =
            disc.common.grid.elemtype == 0 ? nd+1 : 1 << nd;
        std::vector<dstype> lowBasis(npe*workspace.lowModeCount);
        std::vector<dstype> lowInverse(workspace.lowModeCount*npe);
        for (Int a = 0; a < workspace.lowModeCount; ++a)
            for (Int i = 0; i < npe; ++i) {
                lowBasis[i+npe*a] = basis[i+npe*a];
                lowInverse[a+workspace.lowModeCount*i] = inverse[a+npe*i];
            }

        TemplateMalloc(&workspace.lowModalBasis,
                       static_cast<Int>(lowBasis.size()), backend);
        TemplateMalloc(&workspace.lowModalInverse,
                       static_cast<Int>(lowInverse.size()), backend);
        TemplateCopytoDevice(workspace.lowModalBasis, lowBasis.data(),
                             static_cast<Int>(lowBasis.size()), backend);
        TemplateCopytoDevice(workspace.lowModalInverse, lowInverse.data(),
                             static_cast<Int>(lowInverse.size()), backend);
    }

    if (workspace.lowModeCount <= 0 || workspace.lowModeCount > npe)
        error("Invalid mesh-adaptivity low-mode count.");

    dstype* field1 = disc.res.Ru;
    dstype* field2 = disc.res.Rq+avFieldSize;
    residual.evalAVfield(disc.res.Rq, backend);

    // Temporarily use field1 for the raw scalar that defines the modal sensor.
    ArrayExtract(field1, disc.res.Rq, npe, ncAV, ne,
                 0, npe, cfg.scalarField-1, cfg.scalarField, 0, ne);

    // Temporarily use field2 for the retained low-order modal coefficients.
    Node2Gauss(disc.common.cublasHandle, field2, field1,
               workspace.lowModalInverse, workspace.lowModeCount,
               npe, ne, backend);

    // Temporarily use indicator for the corresponding low-order nodal field.
    Node2Gauss(disc.common.cublasHandle, indicator, field2,
               workspace.lowModalBasis, npe, workspace.lowModeCount,
               ne, backend);

    // The modal coefficients are no longer needed; replace them with the
    // elementwise error between the original and low-order fields.
    evaluateSensorError(field2, field1, indicator, disc.sol.xdg,
                        disc.master.shapegt, disc.master.gwe,
                        npe, nge, ncx, ne, nd);

    const dstype rawMaximum = PArrayMax(field2, nodeCount);
    limitSensor(field2, 0.5*rawMaximum, nodeCount);

    // The low-order nodal field is no longer needed; reuse indicator as scratch.
    smoothDG2CG2(disc, field2, indicator, 1, 3, backend);
    const dstype field2Maximum = PArrayMaxAbs(field2, nodeCount);

    // Replace the temporary raw scalar with the actual AV component.
    ArrayExtract(field1, disc.res.Rq, npe, ncAV, ne,
                 0, npe, cfg.avComponent-1, cfg.avComponent, 0, ne);
    const dstype field1Maximum = PArrayMaxAbs(field1, nodeCount);

    const dstype field1Scale =
        field1Maximum > 0.0 ? cfg.alpha/field1Maximum : 0.0;
    const dstype field2Scale =
        field2Maximum > 0.0 ? (1.0-cfg.alpha)/field2Maximum : 0.0;

    ArrayAXPBY(indicator, field1, field2,
               field1Scale, field2Scale, nodeCount);
}

template <class M>
bool CSolution<M>::updateMeshCoordinatesFromIndicator(
    dstype* xdg, const dstype* indicator, Int movementIterations)
{
    using namespace exasim_meshadapt;
    if (xdg == nullptr)
        error("Mesh-coordinate output storage is not initialized.");
    if (indicator == nullptr)
        error("Mesh-adaptivity indicator storage is not initialized.");
    if (movementIterations <= 0)
        error("Mesh movement requires at least one iteration.");
    if (!helmholtz || !elasticity)
        error("Mesh-adaptivity auxiliary solvers were not constructed.");
    if (disc.common.spatialScheme != 1)
        error("Backend mesh adaptivity requires HDG (spatialScheme == 1).");

    const Int backend = disc.common.backend;
    const auto& cfg = disc.common.meshadaptparams;
    const Int nd = disc.common.grid.nd;
    const Int npe = disc.common.grid.npe;
    const Int ne = disc.common.meshsizes.ne;
    const Int ne1 = disc.common.meshsizes.ne1;
    const Int ncx = disc.common.components.ncx;
    const Int nodeCount = npe*ne;
    if (nd != 2 && nd != 3)
        error("Backend mesh adaptivity supports only 2D and 3D.");
    if (ne1 > nodeCount)
        error("Element-mean workspace exceeds mesh-adaptivity vector storage.");
    ArraySetValue(helmholtz->disc.app.tau, cfg.helmholtzTau,
                  helmholtz->disc.app.sztau);

    const Int vectorBufferSize = std::max(nd*nodeCount, nodeCount+ne1);
    if (disc.res.Ru == nullptr || disc.res.szRu < nodeCount)
        error("Ru workspace is too small for mesh-adaptivity scalar storage.");
    if (disc.res.Rq == nullptr || disc.res.szRq < vectorBufferSize)
        error("Rq workspace is too small for mesh-adaptivity vector storage.");

    dstype* scalarBuffer = disc.res.Ru;
    dstype* vectorBuffer = disc.res.Rq;
    if (xdg != disc.sol.xdg)
        ArrayCopy(xdg, disc.sol.xdg, disc.sol.szxdg);

    bool meshAccepted = true;
    std::ofstream auxiliaryOutput;
    for (Int iteration = 0; iteration < movementIterations; ++iteration) {
        dstype* jac = scalarBuffer;
        dstype* smoothScratch = vectorBuffer;
        nodalJacobian(jac, xdg, disc.master.shapent, npe, ncx, ne, nd);
        if (!PArrayAllPositiveFinite(jac, nodeCount)) {
            meshAccepted = false;
            break;
        }
        smoothDG2CG2(disc, jac, smoothScratch, 1, 2, backend);

        dstype reference = PArrayMin(jac, nodeCount);
#ifdef HAVE_MPI
        dstype globalReference = reference;
        MPI_Allreduce(&reference, &globalReference, 1, mpi_type<dstype>(),
                      MPI_MIN, EXASIM_COMM_LOCAL);
        reference = globalReference;
#endif

        dstype* currentSize = vectorBuffer;
        dstype* means = vectorBuffer + nodeCount;
        elementSizeAndMeans(currentSize, means, jac, npe, ne, ne1, nd);

        dstype hmin = 0.0, hmax = 0.0;
        DistributedRadixQuantiles(hmin, hmax, means, ne1, cfg.qmin, cfg.qmax);

        // Auxiliary geometry is synchronized on entry. Later movement
        // iterations must refresh it from the current output coordinates.
        if (iteration > 0)
            rebuildGeometry(helmholtz->disc, xdg, backend);
        targetLameAndHelmholtzInput(elasticity->disc.sol.odg,
            helmholtz->disc.sol.odg, indicator, currentSize, hmin, hmax,
            cfg.helmholtzCoeff, cfg.targetExponent, cfg.youngModulus,
            cfg.minimumYoungModulus, cfg.poissonRatio, npe, ne, nd);

        ArraySetValue(helmholtz->disc.sol.udg, zero, helmholtz->disc.sol.szudg);
        ArraySetValue(helmholtz->solv.sys.u, zero, helmholtz->solv.sys.szu);
        ArraySetValue(helmholtz->solv.sys.x, zero, helmholtz->solv.sys.szx);
        if (helmholtz->disc.sol.szuh > 0)
            ArraySetValue(helmholtz->disc.sol.uh, zero, helmholtz->disc.sol.szuh);
        const SolveStatus helmholtzStatus =
            helmholtz->SteadyProblem(auxiliaryOutput, backend, true);
        if (!helmholtzStatus.finite) {
            meshAccepted = false;
            break;
        }

        packElasticityForce(elasticity->disc.sol.odg,
                            helmholtz->disc.sol.udg, npe, ne, nd);
        smoothDG2CG2(elasticity->disc, elasticity->disc.sol.odg, scalarBuffer,
                     2+nd, cfg.smoothingPasses, backend);

        if (iteration > 0)
            rebuildGeometry(elasticity->disc, xdg, backend);
        ArraySetValue(elasticity->disc.sol.udg, zero, elasticity->disc.sol.szudg);
        ArraySetValue(elasticity->solv.sys.u, zero, elasticity->solv.sys.szu);
        ArraySetValue(elasticity->solv.sys.x, zero, elasticity->solv.sys.szx);
        if (elasticity->disc.sol.szuh > 0)
            ArraySetValue(elasticity->disc.sol.uh, zero, elasticity->disc.sol.szuh);
        const SolveStatus elasticityStatus =
            elasticity->SteadyProblem(auxiliaryOutput, backend, true);
        if (!elasticityStatus.finite) {
            meshAccepted = false;
            break;
        }

        // The HDG solve updates owned elements. Refresh ghost-element values
        // before the local DG-to-CG average so interface nodes see the same
        // displacement data on neighboring MPI ranks.
        exchangeElementUDG(elasticity->disc);
        dstype* continuous = vectorBuffer;
        elasticity->disc.DG2CG(continuous, elasticity->disc.sol.udg, scalarBuffer, nd,
                               elasticity->disc.common.components.nc, nd, backend);

#ifdef HAVE_MPI
        if (elasticity->disc.common.mpiProcs > 1) {
            // Keep owner and ghost copies of the continuous displacement equal
            // before rebuilding geometry on the moved mesh.
            const Int elasticityNc = elasticity->disc.common.components.nc;
            ArrayInsert(elasticity->disc.sol.udg, continuous, npe, elasticityNc, ne,
                        0, npe, 0, nd, 0, ne);
            exchangeElementField(elasticity->disc, elasticity->disc.sol.udg,
                elasticity->disc.mesh.elemsendind,
                elasticity->disc.mesh.elemrecvind, npe*nd);
            ArrayExtract(continuous, elasticity->disc.sol.udg, npe, elasticityNc, ne,
                         0, npe, 0, nd, 0, ne);
        }
#endif

        dstype beta = cfg.damping;
        while (true) {
            dstype minimum = candidateMinimumJacobian(xdg, continuous,
                disc.master.shapent, beta, npe, ncx, ne, nd);
#ifdef HAVE_MPI
            dstype globalMinimum = minimum;
            MPI_Allreduce(&minimum, &globalMinimum, 1, mpi_type<dstype>(),
                          MPI_MIN, EXASIM_COMM_LOCAL);
            minimum = globalMinimum;
#endif
            if (minimum > cfg.minimumJacobianRatio*reference) break;
            beta *= 0.5;
            if (beta < 1.0e-8) {
                meshAccepted = false;
                break;
            }
        }
        if (!meshAccepted)
            break;

        ArrayAXPBY(xdg, xdg, continuous, one, beta, npe*nd*ne);
    }

    if (meshAccepted) {
        rebuildGeometry(disc, xdg, backend);
        rebuildGeometry(helmholtz->disc, xdg, backend);
        rebuildGeometry(elasticity->disc, xdg, backend);
    }
    return meshAccepted;
}

template <class M>
bool CSolution<M>::AdaptMesh(Int backend, Int continuationIteration)
{
    using namespace exasim_meshadapt;
    if (!disc.common.meshadaptparams.enabled) return true;
    if (!helmholtz || !elasticity) error("Mesh-adaptivity auxiliary solvers were not constructed.");
    if (disc.common.spatialScheme != 1)
        error("Backend mesh adaptivity requires HDG (spatialScheme == 1).");
    const auto& cfg = disc.common.meshadaptparams;
    const Int nd = disc.common.grid.nd;
    const Int npe = disc.common.grid.npe;
    const Int ne = disc.common.meshsizes.ne;
    const Int ne1 = disc.common.meshsizes.ne1;
    if (nd != 2 && nd != 3) error("Backend mesh adaptivity supports only 2D and 3D.");

    const Int nodeCount = npe*ne;

    if (ne1 > nodeCount)
        error("Element-mean workspace exceeds mesh-adaptivity vector storage.");

    const Int ncAV = disc.common.physicsparams.ncAV;
    if (ncAV <= 0)
        error("Mesh adaptation requires avfield outputs.");

    const Int vectorBufferSize = std::max(nd*nodeCount, nodeCount+ne1);
    const Int avWorkspaceSize = (ncAV+1)*nodeCount;
    const Int indicatorOffset = std::max(vectorBufferSize, avWorkspaceSize);
    const Int requiredRqSize = indicatorOffset+nodeCount;
    if (disc.res.Rq == nullptr || disc.res.szRq < requiredRqSize)
        error("Rq workspace is too small for mesh-adaptivity vector and indicator storage.");

    dstype* indicator = disc.res.Rq+indicatorOffset;
    computeMeshIndicator(indicator);

    const Int movementIterations = continuationIteration > 0 ? 1 : cfg.movementIterations;
    return updateMeshCoordinatesFromIndicator(
        disc.sol.xdg, indicator, movementIterations);
}

#endif
