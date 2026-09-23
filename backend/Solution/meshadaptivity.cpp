#ifndef __MESHADAPTIVITY
#define __MESHADAPTIVITY

#include "../Preprocessing/makemaster.hpp"
#include "../Preprocessing/meshdist.hpp"
#include <Kokkos_Sort.hpp>

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

inline void writeVerificationDeviceField(const string& prefix, const string& name,
    const dstype* values, Int count, Int rank, Int backend)
{
    if (count <= 0) return;
    std::vector<dstype> host(count);
    TemplateCopytoHost(host.data(), const_cast<dstype*>(values), count, backend);
    writeVerificationField(prefix, name, host, rank);
}

template <class D>
void smoothDG2CG2(D& disc, dstype* field, dstype* scratch, Int components,
                  Int passes, Int backend)
{
    for (Int pass = 0; pass < passes; ++pass)
        disc.DG2CG2(field, field, scratch, components, components, components, backend);
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

inline void targetLameAndHelmholtzInput(dstype* targetSize, dstype* elasticityInput,
    dstype* helmholtzInput, const dstype* eta, const dstype* currentSize,
    dstype hmin, dstype hmax, dstype helmholtzCoefficient, dstype targetExponent,
    dstype youngModulus, dstype minimumYoungModulus, dstype poissonRatio,
    Int npe, Int ne, Int nd, bool saveTarget)
{
    Kokkos::parallel_for(
        "MeshAdaptTargetLameAndHelmholtzInput",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<Int>>(0, npe*ne),
        KOKKOS_LAMBDA(const Int index) {
            const Int i = index % npe;
            const Int e = index / npe;
            dstype bounded = eta[index];
            if (bounded < 0.0) bounded = 0.0;
            else if (bounded > 1.0) bounded = 1.0;
            const dstype target = hmin + (hmax-hmin)*Kokkos::pow(1.0-bounded, targetExponent);
            const dstype unboundedE = youngModulus*Kokkos::pow(target/hmax, 2.0);
            const dstype E = unboundedE > minimumYoungModulus ? unboundedE : minimumYoungModulus;
            const dstype mu = E/(2.0*(1.0+poissonRatio));
            const dstype lambda = E*poissonRatio/((1.0+poissonRatio)*(1.0-2.0*poissonRatio));
            if (saveTarget) targetSize[index] = target;
            elasticityInput[i+npe*0+npe*(2+nd)*e] = mu;
            elasticityInput[i+npe*1+npe*(2+nd)*e] = lambda;
            helmholtzInput[i+npe*0+npe*2*e] = eta[index];
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
void CSolution<M>::AdaptMesh(Int backend, Int continuationIteration)
{
    using namespace exasim_meshadapt;
    if (!disc.common.meshadaptparams.enabled) return;
    if (!helmholtz || !elasticity) error("Mesh-adaptivity auxiliary solvers were not constructed.");
    const auto& cfg = disc.common.meshadaptparams;
    const Int nd = disc.common.grid.nd;
    const Int npe = disc.common.grid.npe;
    const Int nge = disc.common.grid.nge;
    const Int ne = disc.common.meshsizes.ne;
    const Int ne1 = disc.common.meshsizes.ne1;
    const Int ncx = disc.common.components.ncx;
    if (nd != 2 && nd != 3) error("Backend mesh adaptivity supports only 2D and 3D.");
    ArraySetValue(helmholtz->disc.app.tau, cfg.helmholtzTau,
                  helmholtz->disc.app.sztau);
    const char *verificationEnvironment = std::getenv("EXASIM_MESHADAPT_VERIFY");
    const bool writeVerification = verificationEnvironment != nullptr &&
        string(verificationEnvironment) != "0" && string(verificationEnvironment) != "";

    if (ncx != nd)
        error("GPU mesh adaptivity currently requires ncx to equal the spatial dimension.");

    auto& workspace = meshAdaptWorkspace;
    if (workspace.lowModalBasis == nullptr) {
        std::vector<dstype> xpe(disc.master.szxpe);
        TemplateCopytoHost(xpe.data(), disc.master.xpe, disc.master.szxpe, backend);
        std::vector<double> nodes(npe*nd), basisDouble(npe*npe);
        for (Int i = 0; i < npe*nd; ++i) nodes[i] = static_cast<double>(xpe[i]);
        if (disc.common.grid.elemtype == 0)
            koornwinder(basisDouble.data(), nodes.data(), npe,
                        disc.common.grid.porder, nd, 0);
        else
            tensorproduct(basisDouble.data(), nodes.data(), npe,
                          disc.common.grid.porder, nd, 0);
        std::vector<dstype> basis(npe*npe);
        for (Int i = 0; i < npe*npe; ++i) basis[i] = static_cast<dstype>(basisDouble[i]);
        const std::vector<dstype> inverse = invert(basis, npe);
        workspace.lowModeCount = disc.common.grid.elemtype == 0 ? nd+1 : 1 << nd;
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

    const Int nodeCount = npe*ne;
    dstype *field1 = nullptr, *field2 = nullptr, *eta = nullptr;
    dstype *scalar = nullptr, *lowScalar = nullptr, *coefficients = nullptr;
    dstype *smoothScratch = nullptr;
    TemplateMalloc(&field1, nodeCount, backend);
    TemplateMalloc(&field2, nodeCount, backend);
    TemplateMalloc(&eta, nodeCount, backend);
    TemplateMalloc(&scalar, nodeCount, backend);
    TemplateMalloc(&lowScalar, nodeCount, backend);
    TemplateMalloc(&coefficients, workspace.lowModeCount*ne, backend);
    TemplateMalloc(&smoothScratch, nodeCount, backend);
    ArraySetValue(field1, zero, nodeCount);

    dstype field1Maximum = 0.0;
    if (cfg.alpha > 0.0) {
        const Int ncAV = disc.common.physicsparams.ncAV;
        if (ncAV <= 0 || cfg.avComponent > ncAV)
            error("meshadaptavcomponent does not identify an available AV field.");
        residual.evalAVfield(disc.res.Rq, backend);
        const Int component = cfg.avComponent - 1;
        ArrayExtract(field1, disc.res.Rq, npe, ncAV, ne,
                     0, npe, component, component+1, 0, ne);
        field1Maximum = PArrayMaxAbs(field1, nodeCount);
    }

    const Int nsca = disc.common.qoiparams.nsca;
    if (cfg.scalarField > nsca) error("meshadaptfield exceeds the number of VisScalars outputs.");
    const Int nc = disc.common.components.nc;
    const Int nco = disc.common.components.nco;
    const Int ncw = disc.common.components.ncw;
    dstype *packedXdg = nullptr, *packedUdg = nullptr, *packedOdg = nullptr;
    dstype *packedWdg = nullptr, *allScalars = nullptr;
    TemplateMalloc(&packedXdg, npe*ncx*ne, backend);
    TemplateMalloc(&packedUdg, npe*nc*ne, backend);
    if (nco > 0) TemplateMalloc(&packedOdg, npe*nco*ne, backend);
    if (ncw > 0) TemplateMalloc(&packedWdg, npe*ncw*ne, backend);
    TemplateMalloc(&allScalars, npe*nsca*ne, backend);
    GetElemNodes(packedXdg, disc.sol.xdg, npe, ncx, 0, ncx, 0, ne);
    GetElemNodes(packedUdg, disc.sol.udg, npe, nc, 0, nc, 0, ne);
    if (nco > 0) GetElemNodes(packedOdg, disc.sol.odg, npe, nco, 0, nco, 0, ne);
    if (ncw > 0) GetElemNodes(packedWdg, disc.sol.wdg, npe, ncw, 0, ncw, 0, ne);
    EXASIM_DRIVER_CALL(VisScalarsDriver, allScalars, packedXdg, packedUdg, packedOdg,
        packedWdg, disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common,
        npe, 0, ne, backend);
    ArrayExtract(scalar, allScalars, nodeCount, nsca, 1,
                 0, nodeCount, cfg.scalarField-1, cfg.scalarField, 0, 1);
    TemplateFree(packedXdg, backend);
    TemplateFree(packedUdg, backend);
    if (packedOdg) TemplateFree(packedOdg, backend);
    if (packedWdg) TemplateFree(packedWdg, backend);
    TemplateFree(allScalars, backend);

    Node2Gauss(disc.common.cublasHandle, coefficients, scalar,
               workspace.lowModalInverse, workspace.lowModeCount, npe, ne, backend);
    Node2Gauss(disc.common.cublasHandle, lowScalar, coefficients,
               workspace.lowModalBasis, npe, workspace.lowModeCount, ne, backend);
    evaluateSensorError(field2, scalar, lowScalar, disc.sol.xdg,
                        disc.master.shapegt, disc.master.gwe,
                        npe, nge, ncx, ne, nd);

    const Int outputRank = disc.common.mpiRank-disc.common.outputparams.fileoffset;
    const string continuationName = continuationIteration > 0 ?
        "aviter" + NumberToString(continuationIteration) + "_" : "";
    if (writeVerification)
        writeVerificationDeviceField(disc.common.fileout,
            continuationName + "sensor_raw", field2, nodeCount, outputRank, backend);

    const dstype rawSensorMaximum = PArrayMax(field2, nodeCount);
    limitSensor(field2, 0.5*rawSensorMaximum, nodeCount);
    smoothDG2CG2(disc, field2, smoothScratch, 1, 3, backend);
    const dstype field2Maximum = PArrayMaxAbs(field2, nodeCount);
    const dstype field1Scale = field1Maximum > 0.0 ? cfg.alpha/field1Maximum : 0.0;
    const dstype field2Scale = field2Maximum > 0.0 ? (1.0-cfg.alpha)/field2Maximum : 0.0;
    ArrayAXPBY(eta, field1, field2, field1Scale, field2Scale, nodeCount);

    if (writeVerification) {
        writeVerificationDeviceField(disc.common.fileout,
            continuationName + "sensor_scalar", scalar, nodeCount, outputRank, backend);
        ArrayCopy(lowScalar, field1, nodeCount);
        if (field1Maximum > 0.0) ArrayMultiplyScalar(lowScalar, one/field1Maximum, nodeCount);
        writeVerificationDeviceField(disc.common.fileout,
            continuationName + "field1", lowScalar, nodeCount, outputRank, backend);
        ArrayCopy(lowScalar, field2, nodeCount);
        if (field2Maximum > 0.0) ArrayMultiplyScalar(lowScalar, one/field2Maximum, nodeCount);
        writeVerificationDeviceField(disc.common.fileout,
            continuationName + "field2", lowScalar, nodeCount, outputRank, backend);
        writeVerificationDeviceField(disc.common.fileout,
            continuationName + "eta", eta, nodeCount, outputRank, backend);
    }

    std::ofstream auxiliaryOutput;
    const Int movementIterations = continuationIteration > 0 ? 1 : cfg.movementIterations;
    for (Int iteration = 0; iteration < movementIterations; ++iteration) {
        dstype *jac = nullptr, *currentSize = nullptr, *means = nullptr;
        TemplateMalloc(&jac, nodeCount, backend);
        TemplateMalloc(&currentSize, nodeCount, backend);
        TemplateMalloc(&means, ne1, backend);
        nodalJacobian(jac, disc.sol.xdg, disc.master.shapent, npe, ncx, ne, nd);
        if (!PArrayAllPositiveFinite(jac, nodeCount))
            error("Mesh adaptation encountered an invalid Jacobian.");
        smoothDG2CG2(disc, jac, smoothScratch, 1, 2, backend);
        elementSizeAndMeans(currentSize, means, jac, npe, ne, ne1, nd);

        dstype hmin = 0.0, hmax = 0.0;
#ifdef HAVE_MPI
        std::vector<int> counts(disc.common.mpiProcs), offsets(disc.common.mpiProcs, 0);
        const int localCount = static_cast<int>(ne1);
        MPI_Gather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT,
                   0, EXASIM_COMM_WORLD);
        int total = 0;
        if (disc.common.mpiRank == 0)
            for (Int r = 0; r < disc.common.mpiProcs; ++r) {
                offsets[r] = total;
                total += counts[r];
            }
        dstype *globalMeans = nullptr;
        if (disc.common.mpiRank == 0) TemplateMalloc(&globalMeans, total, backend);
        Kokkos::fence();
        MPI_Gatherv(means, localCount, mpi_type<dstype>(), globalMeans,
                    counts.data(), offsets.data(), mpi_type<dstype>(),
                    0, EXASIM_COMM_WORLD);
        if (disc.common.mpiRank == 0) {
            using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
            using unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
            Kokkos::View<dstype*, memory_space, unmanaged> sorted(globalMeans, total);
            Kokkos::sort(sorted);
            Int imin = static_cast<Int>(std::round(cfg.qmin*total))-1;
            Int imax = static_cast<Int>(std::round(cfg.qmax*total))-1;
            imin = std::max<Int>(0, std::min<Int>(imin, total-1));
            imax = std::max<Int>(0, std::min<Int>(imax, total-1));
            TemplateCopytoHost(&hmin, &globalMeans[imin], 1, backend);
            TemplateCopytoHost(&hmax, &globalMeans[imax], 1, backend);
            TemplateFree(globalMeans, backend);
        }
        MPI_Bcast(&hmin, 1, mpi_type<dstype>(), 0, EXASIM_COMM_WORLD);
        MPI_Bcast(&hmax, 1, mpi_type<dstype>(), 0, EXASIM_COMM_WORLD);
#else
        using memory_space = typename Kokkos::DefaultExecutionSpace::memory_space;
        using unmanaged = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
        Kokkos::View<dstype*, memory_space, unmanaged> sorted(means, ne1);
        Kokkos::sort(sorted);
        Int imin = static_cast<Int>(std::round(cfg.qmin*ne1))-1;
        Int imax = static_cast<Int>(std::round(cfg.qmax*ne1))-1;
        imin = std::max<Int>(0, std::min<Int>(imin, ne1-1));
        imax = std::max<Int>(0, std::min<Int>(imax, ne1-1));
        TemplateCopytoHost(&hmin, &means[imin], 1, backend);
        TemplateCopytoHost(&hmax, &means[imax], 1, backend);
#endif

        dstype *targetSize = nullptr;
        if (writeVerification) TemplateMalloc(&targetSize, nodeCount, backend);

        // Auxiliary geometry is synchronized on entry and after AdaptMesh returns.
        // Only later movement iterations need an in-loop refresh.
        if (iteration > 0)
            rebuildGeometry(helmholtz->disc, disc.sol.xdg, backend);
        targetLameAndHelmholtzInput(targetSize, elasticity->disc.sol.odg,
            helmholtz->disc.sol.odg, eta, currentSize, hmin, hmax,
            cfg.helmholtzCoeff, cfg.targetExponent, cfg.youngModulus,
            cfg.minimumYoungModulus, cfg.poissonRatio, npe, ne, nd,
            writeVerification);
        ArraySetValue(helmholtz->disc.sol.udg, zero, helmholtz->disc.sol.szudg);
        ArraySetValue(helmholtz->solv.sys.u, zero, helmholtz->solv.sys.szu);
        ArraySetValue(helmholtz->solv.sys.x, zero, helmholtz->solv.sys.szx);
        if (helmholtz->disc.sol.szuh > 0)
            ArraySetValue(helmholtz->disc.sol.uh, zero, helmholtz->disc.sol.szuh);
        helmholtz->SteadyProblem(auxiliaryOutput, backend);
        packElasticityForce(elasticity->disc.sol.odg,
                            helmholtz->disc.sol.udg, npe, ne, nd);
        smoothDG2CG2(disc, elasticity->disc.sol.odg, smoothScratch,
                     2+nd, cfg.smoothingPasses, backend);

        if (iteration > 0)
            rebuildGeometry(elasticity->disc, disc.sol.xdg, backend);
        ArraySetValue(elasticity->disc.sol.udg, zero, elasticity->disc.sol.szudg);
        ArraySetValue(elasticity->solv.sys.u, zero, elasticity->solv.sys.szu);
        ArraySetValue(elasticity->solv.sys.x, zero, elasticity->solv.sys.szx);
        if (elasticity->disc.sol.szuh > 0) ArraySetValue(elasticity->disc.sol.uh, zero, elasticity->disc.sol.szuh);
        elasticity->SteadyProblem(auxiliaryOutput, backend);
        dstype *continuous = nullptr, *scratch = nullptr;
        TemplateMalloc(&continuous, npe*nd*ne, backend);
        TemplateMalloc(&scratch, npe*ne, backend);
        elasticity->disc.DG2CG(continuous, elasticity->disc.sol.udg, scratch, nd,
                               elasticity->disc.common.components.nc, nd, backend);
        TemplateFree(scratch, backend);

        dstype beta = cfg.damping;
        const dstype reference = PArrayMin(jac, nodeCount);
        while (true) {
            dstype minimum = candidateMinimumJacobian(disc.sol.xdg, continuous,
                disc.master.shapent, beta, npe, ncx, ne, nd);
#ifdef HAVE_MPI
            dstype globalMinimum = minimum;
            MPI_Allreduce(&minimum, &globalMinimum, 1, mpi_type<dstype>(),
                          MPI_MIN, EXASIM_COMM_WORLD);
            minimum = globalMinimum;
#endif
            if (minimum > cfg.minimumJacobianRatio*reference) break;
            beta *= 0.5;
            if (beta < 1.0e-8) error("Mesh-adaptivity backtracking could not produce a valid mesh.");
        }
        ArrayAXPBY(disc.sol.xdg, disc.sol.xdg, continuous,
                   one, beta, npe*nd*ne);

        const string iterationName = continuationName + "iter" + NumberToString(iteration+1) + "_";
        if (writeVerification) {
            writeVerificationDeviceField(disc.common.fileout, iterationName + "h",
                                         targetSize, nodeCount, outputRank, backend);
            writeVerificationField(disc.common.fileout, iterationName + "hmin",
                                   std::vector<dstype>{hmin}, outputRank);
            writeVerificationField(disc.common.fileout, iterationName + "hmax",
                                   std::vector<dstype>{hmax}, outputRank);
            ArrayExtract(lowScalar, elasticity->disc.sol.odg, npe, 2+nd, ne,
                         0, npe, 0, 1, 0, ne);
            writeVerificationDeviceField(disc.common.fileout, iterationName + "mu",
                                         lowScalar, nodeCount, outputRank, backend);
            ArrayExtract(lowScalar, elasticity->disc.sol.odg, npe, 2+nd, ne,
                         0, npe, 1, 2, 0, ne);
            writeVerificationDeviceField(disc.common.fileout, iterationName + "lambda",
                                         lowScalar, nodeCount, outputRank, backend);
            writeVerificationDeviceField(disc.common.fileout, iterationName + "helmholtz",
                helmholtz->disc.sol.udg, helmholtz->disc.sol.szudg, outputRank, backend);
            dstype *force = nullptr;
            TemplateMalloc(&force, npe*nd*ne, backend);
            ArrayExtract(force, elasticity->disc.sol.odg, npe, 2+nd, ne,
                         0, npe, 2, 2+nd, 0, ne);
            writeVerificationDeviceField(disc.common.fileout, iterationName + "force",
                                         force, npe*nd*ne, outputRank, backend);
            TemplateFree(force, backend);
            writeVerificationDeviceField(disc.common.fileout, iterationName + "displacement",
                                         continuous, npe*nd*ne, outputRank, backend);
            writeVerificationDeviceField(disc.common.fileout, iterationName + "xdg",
                                         disc.sol.xdg, disc.sol.szxdg, outputRank, backend);
        }
        TemplateFree(continuous, backend);
        TemplateFree(targetSize, backend);
        TemplateFree(jac, backend);
        TemplateFree(currentSize, backend);
        TemplateFree(means, backend);
    }

    rebuildGeometry(disc, disc.sol.xdg, backend);
    rebuildGeometry(helmholtz->disc, disc.sol.xdg, backend);
    rebuildGeometry(elasticity->disc, disc.sol.xdg, backend);
    const string filename = disc.common.fileout + "_meshadapt_xdg_np" +
        NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";
    writearray2file(filename, disc.sol.xdg, disc.sol.szxdg, backend);
    TemplateFree(field1, backend);
    TemplateFree(field2, backend);
    TemplateFree(eta, backend);
    TemplateFree(scalar, backend);
    TemplateFree(lowScalar, backend);
    TemplateFree(coefficients, backend);
    TemplateFree(smoothScratch, backend);
}

#endif
