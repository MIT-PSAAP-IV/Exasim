#ifndef __MESHADAPTIVITY
#define __MESHADAPTIVITY

#include "../Preprocessing/makemaster.hpp"

namespace exasim_meshadapt {

inline dstype determinant(const dstype J[3][3], Int nd)
{
    if (nd == 2) return J[0][0]*J[1][1] - J[0][1]*J[1][0];
    return J[0][0]*J[1][1]*J[2][2] - J[0][0]*J[1][2]*J[2][1]
         + J[0][1]*J[1][2]*J[2][0] - J[0][1]*J[1][0]*J[2][2]
         + J[0][2]*J[1][0]*J[2][1] - J[0][2]*J[1][1]*J[2][0];
}

inline std::vector<dstype> nodalJacobian(const std::vector<dstype>& xdg,
    const std::vector<dstype>& shapent, Int npe, Int ncx, Int ne, Int nd)
{
    std::vector<dstype> jac(npe*ne);
    for (Int e = 0; e < ne; ++e)
        for (Int g = 0; g < npe; ++g) {
            dstype J[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
            for (Int r = 0; r < nd; ++r)
                for (Int d = 0; d < nd; ++d)
                    for (Int i = 0; i < npe; ++i)
                        J[r][d] += shapent[g + npe*i + npe*npe*(r + 1)]
                                 * xdg[i + npe*d + npe*ncx*e];
            jac[g + npe*e] = determinant(J, nd);
        }
    return jac;
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

inline dstype globalMaximum(dstype value)
{
#ifdef HAVE_MPI
    dstype result = value;
    MPI_Allreduce(&value, &result, 1, mpi_type<dstype>(), MPI_MAX, EXASIM_COMM_WORLD);
    return result;
#else
    return value;
#endif
}

inline dstype globalMinimum(dstype value)
{
#ifdef HAVE_MPI
    dstype result = value;
    MPI_Allreduce(&value, &result, 1, mpi_type<dstype>(), MPI_MIN, EXASIM_COMM_WORLD);
    return result;
#else
    return value;
#endif
}

inline void normalize(std::vector<dstype>& field)
{
    dstype maximum = 0.0;
    for (dstype value : field) maximum = std::max(maximum, std::abs(value));
    maximum = globalMaximum(maximum);
    if (maximum > 0.0)
        for (dstype& value : field) value /= maximum;
}

inline dstype smoothMaximum(dstype x, dstype alpha)
{
    const dstype pi = std::acos(-1.0);
    return x*(std::atan(alpha*x)/pi + 0.5) - std::atan(alpha)/pi + 0.5;
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
void smoothDG2CG2(D& disc, std::vector<dstype>& field, Int passes, Int backend)
{
    const Int count = disc.common.grid.npe*disc.common.meshsizes.ne;
    dstype *values = nullptr, *scratch = nullptr;
    TemplateMalloc(&values, count, backend);
    TemplateMalloc(&scratch, count, backend);
    TemplateCopytoDevice(values, field.data(), count, backend);
    for (Int pass = 0; pass < passes; ++pass)
        disc.DG2CG2(values, values, scratch, 1, 1, 1, backend);
    TemplateCopytoHost(field.data(), values, count, backend);
    TemplateFree(values, backend);
    TemplateFree(scratch, backend);
}

template <class D>
void rebuildGeometry(D& disc, const std::vector<dstype>& xdg, Int backend)
{
    TemplateCopytoDevice(disc.sol.xdg, xdg.data(), disc.sol.szxdg, backend);
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

inline std::vector<dstype> discontinuityIndicator(const std::vector<dstype>& scalar,
    const std::vector<dstype>& xdg, const std::vector<dstype>& xpe,
    const std::vector<dstype>& shapegt, const std::vector<dstype>& gwe,
    Int npe, Int nge, Int ncx, Int ne, Int nd, Int elemtype, Int porder,
    std::vector<dstype>* rawSensor = nullptr)
{
    std::vector<double> basis(npe*npe);
    std::vector<double> nodes(xpe.begin(), xpe.begin() + npe*nd);
    if (elemtype == 0) koornwinder(basis.data(), nodes.data(), npe, porder, nd, 0);
    else tensorproduct(basis.data(), nodes.data(), npe, porder, nd, 0);
    const std::vector<dstype> inverse = invert(basis, npe);
    const Int lowModes = (elemtype == 0) ? (nd + 1) : (1 << nd);
    std::vector<dstype> sensor(npe*ne, 0.0), coefficients(npe), low(npe);
    for (Int e = 0; e < ne; ++e) {
        for (Int a = 0; a < npe; ++a) {
            coefficients[a] = 0.0;
            for (Int i = 0; i < npe; ++i)
                coefficients[a] += inverse[a + npe*i]*scalar[i + npe*e];
            if (a >= lowModes) coefficients[a] = 0.0;
        }
        for (Int i = 0; i < npe; ++i) {
            low[i] = 0.0;
            for (Int a = 0; a < npe; ++a) low[i] += basis[i + npe*a]*coefficients[a];
        }
        dstype numerator = 0.0, denominator = 0.0;
        for (Int g = 0; g < nge; ++g) {
            dstype highg = 0.0, lowg = 0.0;
            dstype J[3][3] = {{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
            for (Int i = 0; i < npe; ++i) {
                const dstype shape = shapegt[g + nge*i];
                highg += shape*scalar[i + npe*e];
                lowg += shape*low[i];
                for (Int r = 0; r < nd; ++r)
                    for (Int d = 0; d < nd; ++d)
                        J[r][d] += shapegt[g + nge*i + nge*npe*(r+1)]
                                 * xdg[i + npe*d + npe*ncx*e];
            }
            const dstype weight = gwe[g]*determinant(J, nd);
            const dstype ratio = highg/lowg - 1.0;
            numerator += weight*ratio*ratio;
            denominator += weight;
        }
        const dstype value = std::sqrt(numerator/denominator);
        for (Int i = 0; i < npe; ++i) sensor[i + npe*e] = value;
    }
    if (rawSensor) *rawSensor = sensor;
    dstype maximum = 0.0;
    for (dstype value : sensor) maximum = std::max(maximum, value);
    maximum = globalMaximum(maximum);
    const dstype upper = 0.5*maximum;
    for (dstype& value : sensor) {
        const dstype lowerLimited = smoothMaximum(value, 1.0e3);
        value = lowerLimited - smoothMaximum(lowerLimited - upper, 1.0e3);
    }
    return sensor;
}

} // namespace exasim_meshadapt

template <class M>
void CSolution<M>::AdaptMesh(Int backend)
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

    std::vector<dstype> xdg(disc.sol.szxdg), xpe(disc.master.szxpe), shapent(disc.master.szshapent);
    std::vector<dstype> shapegt(disc.master.szshapegt), gwe(disc.master.szgwe);
    TemplateCopytoHost(xdg.data(), disc.sol.xdg, disc.sol.szxdg, backend);
    TemplateCopytoHost(xpe.data(), disc.master.xpe, disc.master.szxpe, backend);
    TemplateCopytoHost(shapent.data(), disc.master.shapent, disc.master.szshapent, backend);
    TemplateCopytoHost(shapegt.data(), disc.master.shapegt, disc.master.szshapegt, backend);
    TemplateCopytoHost(gwe.data(), disc.master.gwe, disc.master.szgwe, backend);

    std::vector<dstype> field1(npe*ne, 0.0);
    if (cfg.alpha > 0.0) {
        const Int ncAV = disc.common.physicsparams.ncAV;
        if (ncAV <= 0 || cfg.avComponent > ncAV)
            error("meshadaptavcomponent does not identify an available AV field.");
        std::vector<dstype> odg(disc.sol.szodg);
        TemplateCopytoHost(odg.data(), disc.sol.odg, disc.sol.szodg, backend);
        const Int component = disc.common.components.nco - ncAV + cfg.avComponent - 1;
        for (Int e = 0; e < ne; ++e)
            for (Int i = 0; i < npe; ++i)
                field1[i+npe*e] = odg[i + npe*component + npe*disc.common.components.nco*e];
        normalize(field1);
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
    std::vector<dstype> scalarAll(npe*nsca*ne), scalar(npe*ne);
    TemplateCopytoHost(scalarAll.data(), allScalars, npe*nsca*ne, backend);
    TemplateFree(packedXdg, backend);
    TemplateFree(packedUdg, backend);
    if (packedOdg) TemplateFree(packedOdg, backend);
    if (packedWdg) TemplateFree(packedWdg, backend);
    TemplateFree(allScalars, backend);
    for (Int e = 0; e < ne; ++e)
        for (Int i = 0; i < npe; ++i)
            scalar[i+npe*e] = scalarAll[i + npe*e + npe*ne*(cfg.scalarField-1)];
    std::vector<dstype> rawSensor;
    std::vector<dstype> field2 = discontinuityIndicator(scalar, xdg, xpe, shapegt, gwe,
        npe, nge, ncx, ne, nd, disc.common.grid.elemtype, disc.common.grid.porder,
        writeVerification ? &rawSensor : nullptr);
    smoothDG2CG2(disc, field2, 3, backend);
    normalize(field2);

    std::vector<dstype> eta(npe*ne);
    for (Int i = 0; i < npe*ne; ++i)
        eta[i] = cfg.alpha*field1[i] + (1.0-cfg.alpha)*field2[i];

    const Int outputRank = disc.common.mpiRank-disc.common.outputparams.fileoffset;
    if (writeVerification) {
        writeVerificationField(disc.common.fileout, "sensor_scalar", scalar, outputRank);
        writeVerificationField(disc.common.fileout, "sensor_raw", rawSensor, outputRank);
        writeVerificationField(disc.common.fileout, "field1", field1, outputRank);
        writeVerificationField(disc.common.fileout, "field2", field2, outputRank);
        writeVerificationField(disc.common.fileout, "eta", eta, outputRank);
    }

    std::ofstream auxiliaryOutput;
    for (Int iteration = 0; iteration < cfg.movementIterations; ++iteration) {
        std::vector<dstype> jac = nodalJacobian(xdg, shapent, npe, ncx, ne, nd);
        for (dstype value : jac)
            if (!(value > 0.0) || !std::isfinite(value)) error("Mesh adaptation encountered an invalid Jacobian.");
        smoothDG2CG2(disc, jac, 2, backend);
        std::vector<dstype> currentSize(npe*ne), means(ne1);
        for (Int e = 0; e < ne; ++e) {
            dstype mean = 0.0;
            for (Int i = 0; i < npe; ++i) {
                currentSize[i+npe*e] = std::pow(jac[i+npe*e], 1.0/static_cast<dstype>(nd));
                mean += currentSize[i+npe*e]/npe;
            }
            if (e < ne1) means[e] = mean;
        }
#ifdef HAVE_MPI
        Int localCount = ne1;
        std::vector<Int> counts(disc.common.mpiProcs), offsets(disc.common.mpiProcs, 0);
        MPI_Allgather(&localCount, 1, mpi_type<Int>(), counts.data(), 1, mpi_type<Int>(), EXASIM_COMM_WORLD);
        Int total = 0;
        for (Int r = 0; r < disc.common.mpiProcs; ++r) { offsets[r] = total; total += counts[r]; }
        std::vector<dstype> globalMeans(total);
        MPI_Allgatherv(means.data(), localCount, mpi_type<dstype>(), globalMeans.data(), counts.data(),
                       offsets.data(), mpi_type<dstype>(), EXASIM_COMM_WORLD);
#else
        std::vector<dstype> globalMeans = means;
#endif
        std::sort(globalMeans.begin(), globalMeans.end());
        auto quantile = [&](dstype q) {
            Int index = static_cast<Int>(std::round(q*globalMeans.size())) - 1;
            index = std::max<Int>(0, std::min<Int>(index, globalMeans.size()-1));
            return globalMeans[index];
        };
        const dstype hmin = quantile(cfg.qmin), hmax = quantile(cfg.qmax);
        std::vector<dstype> targetSize(npe*ne), mu(npe*ne), lambda(npe*ne), force(npe*nd*ne);
        for (Int i = 0; i < npe*ne; ++i) {
            const dstype boundedEta = std::min<dstype>(1.0, std::max<dstype>(0.0, eta[i]));
            const dstype target = hmin + (hmax-hmin)*std::pow(1.0-boundedEta, cfg.targetExponent);
            targetSize[i] = target;
            const dstype E = std::max(cfg.youngModulus*std::pow(target/hmax, 2.0),
                                      cfg.minimumYoungModulus);
            mu[i] = E/(2.0*(1.0+cfg.poissonRatio));
            lambda[i] = E*cfg.poissonRatio/((1.0+cfg.poissonRatio)*(1.0-2.0*cfg.poissonRatio));
        }

        rebuildGeometry(helmholtz->disc, xdg, backend);
        std::vector<dstype> hodg(npe*2*ne, 0.0), hudg(npe*(1+nd)*ne, 0.0);
        for (Int e = 0; e < ne; ++e)
            for (Int i = 0; i < npe; ++i) {
                const Int p = i+npe*e;
                hodg[i + npe*0 + npe*2*e] = eta[p];
                hodg[i + npe*1 + npe*2*e] = cfg.helmholtzCoeff*currentSize[p];
        }
        TemplateCopytoDevice(helmholtz->disc.sol.odg, hodg.data(), hodg.size(), backend);
        TemplateCopytoDevice(helmholtz->disc.sol.udg, hudg.data(), hudg.size(), backend);
        ArraySetValue(helmholtz->solv.sys.u, zero, helmholtz->solv.sys.szu);
        ArraySetValue(helmholtz->solv.sys.x, zero, helmholtz->solv.sys.szx);
        if (helmholtz->disc.sol.szuh > 0)
            ArraySetValue(helmholtz->disc.sol.uh, zero, helmholtz->disc.sol.szuh);
        helmholtz->SteadyProblem(auxiliaryOutput, backend);
        TemplateCopytoHost(hudg.data(), helmholtz->disc.sol.udg, hudg.size(), backend);
        for (Int e = 0; e < ne; ++e)
            for (Int d = 0; d < nd; ++d)
                for (Int i = 0; i < npe; ++i)
                    force[i + npe*d + npe*nd*e] = -hudg[i + npe*(1+d) + npe*(1+nd)*e];
        smoothDG2CG2(disc, mu, cfg.smoothingPasses, backend);
        smoothDG2CG2(disc, lambda, cfg.smoothingPasses, backend);
        for (Int d = 0; d < nd; ++d) {
            std::vector<dstype> component(npe*ne);
            for (Int e = 0; e < ne; ++e)
                for (Int i = 0; i < npe; ++i) component[i+npe*e] = force[i+npe*d+npe*nd*e];
            smoothDG2CG2(disc, component, cfg.smoothingPasses, backend);
            for (Int e = 0; e < ne; ++e)
                for (Int i = 0; i < npe; ++i) force[i+npe*d+npe*nd*e] = component[i+npe*e];
        }

        rebuildGeometry(elasticity->disc, xdg, backend);
        std::vector<dstype> eodg(npe*(2+nd)*ne, 0.0);
        for (Int e = 0; e < ne; ++e)
            for (Int i = 0; i < npe; ++i) {
                const Int p = i+npe*e;
                eodg[i + npe*0 + npe*(2+nd)*e] = mu[p];
                eodg[i + npe*1 + npe*(2+nd)*e] = lambda[p];
                for (Int d = 0; d < nd; ++d)
                    eodg[i + npe*(2+d) + npe*(2+nd)*e] = force[i+npe*d+npe*nd*e];
            }
        TemplateCopytoDevice(elasticity->disc.sol.odg, eodg.data(), eodg.size(), backend);
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
        std::vector<dstype> displacement(npe*nd*ne);
        TemplateCopytoHost(displacement.data(), continuous, displacement.size(), backend);
        TemplateFree(continuous, backend); TemplateFree(scratch, backend);

        dstype beta = cfg.damping;
        const dstype originalMinimum = *std::min_element(jac.begin(), jac.end());
        std::vector<dstype> candidate(xdg.size());
        while (true) {
            candidate = xdg;
            for (Int e = 0; e < ne; ++e)
                for (Int d = 0; d < nd; ++d)
                    for (Int i = 0; i < npe; ++i)
                        candidate[i+npe*d+npe*ncx*e] += beta*displacement[i+npe*d+npe*nd*e];
            const std::vector<dstype> candidateJac = nodalJacobian(candidate, shapent, npe, ncx, ne, nd);
            dstype minimum = *std::min_element(candidateJac.begin(), candidateJac.end());
            minimum = globalMinimum(minimum);
            const dstype reference = globalMinimum(originalMinimum);
            if (minimum > cfg.minimumJacobianRatio*reference) break;
            beta *= 0.5;
            if (beta < 1.0e-8) error("Mesh-adaptivity backtracking could not produce a valid mesh.");
        }
        xdg.swap(candidate);

        const string iterationName = "iter" + NumberToString(iteration+1) + "_";
        if (writeVerification) {
            writeVerificationField(disc.common.fileout, iterationName + "h", targetSize, outputRank);
            writeVerificationField(disc.common.fileout, iterationName + "hmin",
                                   std::vector<dstype>{hmin}, outputRank);
            writeVerificationField(disc.common.fileout, iterationName + "hmax",
                                   std::vector<dstype>{hmax}, outputRank);
            writeVerificationField(disc.common.fileout, iterationName + "mu", mu, outputRank);
            writeVerificationField(disc.common.fileout, iterationName + "lambda", lambda, outputRank);
            writeVerificationField(disc.common.fileout, iterationName + "helmholtz", hudg, outputRank);
            writeVerificationField(disc.common.fileout, iterationName + "force", force, outputRank);
            writeVerificationField(disc.common.fileout, iterationName + "displacement", displacement, outputRank);
            writeVerificationField(disc.common.fileout, iterationName + "xdg", xdg, outputRank);
        }
    }

    rebuildGeometry(disc, xdg, backend);
    const string filename = disc.common.fileout + "_meshadapt_xdg_np" +
        NumberToString(disc.common.mpiRank-disc.common.outputparams.fileoffset) + ".bin";
    writearray2file(filename, disc.sol.xdg, disc.sol.szxdg, backend);
}

#endif
