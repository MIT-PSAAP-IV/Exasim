/*
    CPU run-time kernels for non-matching multiphysics coupling.

    Setup-owned arrays and layouts are defined in the frozen
    nonmatchinginterface.hpp.  This header evaluates the donor fields and
    coupling flux, forms the partial residual/matrices, condenses them with the
    donor local solver, and exchanges slot residuals with their owners.
*/

#ifndef __NONMATCHINGCOUPLING_HPP
#define __NONMATCHINGCOUPLING_HPP

#include "wequation.hpp"
#include "nonmatchinginterface.hpp"
#include "nonmatchingruntime.hpp" // [nonmatching] phase 8 persistent backend cache
#include "nonmatchingruntimekernels.hpp" // [nonmatching] backend-portable runtime kernels

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

namespace exasim {
namespace nonmatching {

template <class T, class I>
inline void reject_rectangular_host_path(const OwnerTestData<T, I>* ownerTests,
        const char* where)
{
    if (ownerTests &&
            (ownerTests->dir[0].rectangular || ownerTests->dir[1].rectangular)) {
        std::ostringstream message;
        message << "non-matching host path: rectangular tet/hex coupling requires the "
                   "NonmatchingRuntime path (" << where << ")";
        throw std::runtime_error(message.str());
    }
}

#ifdef HAVE_MPI
// Tag base of the coupling-slot exchange (one message per donor/owner pair and
// direction). The same exchange carries the partial residuals (eq. (63)) and
// the partial matrix-vector products (eq. (68)); the two never overlap in time.
constexpr int CouplingSlotTagBase = 27100;
#endif

template <class T, class I>
struct CouplingPointData {
    std::vector<T> x, udg, odg, wdg, uhg, nl, wsrc;
    std::vector<T> flux, fluxUdg, fluxWdg, fluxUhg, wdgUdg;
};

template <class I>
inline std::size_t checked_runtime_size(std::initializer_list<I> factors, const char* what)
{
    std::size_t size = 1;
    for (I factor : factors) {
        if (factor < 0 || (factor != 0 && size > std::numeric_limits<std::size_t>::max() /
                                               static_cast<std::size_t>(factor))) {
            std::ostringstream oss;
            oss << "non-matching coupling: " << what << " size overflow";
            error(oss.str());
        }
        size *= static_cast<std::size_t>(factor);
    }
    return size;
}

template <class T, class I>
inline T interpolate_element_field(const T* field, const T* basis, I point,
        I element, I component, I npts, I npe, I ncomp)
{
    if (field == nullptr) return static_cast<T>(0);
    T value = 0;
    for (I j = 0; j < npe; ++j)
        value += basis[j + static_cast<std::size_t>(npe) * point] *
                 field[j + static_cast<std::size_t>(npe) *
                           (component + static_cast<std::size_t>(ncomp) * element)];
    return value;
}

template <class I>
inline I q_component(I component, I direction, I ncu)
{
    return ncu + component + ncu * direction;
}

template <class T, class I>
inline T trace_value(const T* trace, I encodedNode, I component,
        I npf, I ncu, I spatialScheme)
{
    if (spatialScheme == 0) {
        const I node = encodedNode % npf;
        const I face = encodedNode / npf;
        return trace[node + static_cast<std::size_t>(npf) *
                              (component + static_cast<std::size_t>(ncu) * face)];
    }
    return trace[component + static_cast<std::size_t>(ncu) * encodedNode];
}

// Equations (31), (32), (33), and (42).  Coordinate-like arrays use the
// point-first layout expected by FintDriver.
template <class T, class I>
inline void build_coupling_point_data(CouplingPointData<T, I>& data,
        const nonmatchinginterfaceT<T, I>& donor, const solstructT<T, I>& sol,
        const meshstructT<T, I>& mesh, const commonstructT<T, I>& common)
{
    const I npts = donor.npts, nd = donor.nd, npe = donor.npe;
    const I npf = donor.npf, nfe = donor.nfe, ncu = donor.ncu, ncq = donor.ncq;
    const I nc = common.components.nc, ncx = common.components.ncx;
    const I nco = common.components.nco, ncw = common.components.ncw;
    const I ndf = npf * nfe;
    if (nc != ncu + ncq)
        error("non-matching coupling: donor nc is inconsistent with ncu + ncq");

    data.x.assign(checked_runtime_size<I>({npts, ncx}, "point coordinates"), 0);
    data.udg.assign(checked_runtime_size<I>({npts, nc}, "point udg"), 0);
    data.odg.assign(checked_runtime_size<I>({npts, nco}, "point odg"), 0);
    data.wdg.assign(checked_runtime_size<I>({npts, ncw}, "point wdg"), 0);
    data.uhg.assign(checked_runtime_size<I>({npts, ncu}, "point uhg"), 0);
    data.nl.assign(checked_runtime_size<I>({npts, nd}, "point normals"), 0);
    data.wsrc.assign(checked_runtime_size<I>({npts, ncw}, "point wsrc"), 0);

    for (I p = 0; p < npts; ++p) {
        const I e = donor.pe[p], a = donor.pa[p];
        for (I d = 0; d < ncx; ++d)
            data.x[p + static_cast<std::size_t>(npts) * d] =
                d < nd ? donor.px[p + static_cast<std::size_t>(npts) * d] : 0;
        for (I d = 0; d < nd; ++d)
            data.nl[p + static_cast<std::size_t>(npts) * d] =
                -donor.pnl[p + static_cast<std::size_t>(npts) * d];
        for (I c = 0; c < nc; ++c)
            data.udg[p + static_cast<std::size_t>(npts) * c] =
                interpolate_element_field(sol.udg, donor.pphi, p, e, c,
                                          npts, npe, nc);
        for (I c = 0; c < nco; ++c)
            data.odg[p + static_cast<std::size_t>(npts) * c] =
                interpolate_element_field(sol.odg, donor.pphi, p, e, c,
                                          npts, npe, nco);
        for (I c = 0; c < ncw; ++c) {
            data.wdg[p + static_cast<std::size_t>(npts) * c] =
                interpolate_element_field(sol.wdg, donor.pphi, p, e, c,
                                          npts, npe, ncw);
            data.wsrc[p + static_cast<std::size_t>(npts) * c] =
                interpolate_element_field(sol.wsrc, donor.pphi, p, e, c,
                                          npts, npe, ncw);
        }

        for (I c = 0; c < ncu; ++c) {
            T trace = 0;
            for (I j = 0; j < npf; ++j) {
                const I traceNode = mesh.elemcon[j + npf * a + ndf * e];
                trace += donor.ppsi[j + static_cast<std::size_t>(npf) * p] *
                         trace_value(sol.uh, traceNode, c, npf, ncu,
                                     common.spatialScheme);
            }
            if (ncq > 0) {
                for (I d = 0; d < nd; ++d) {
                    T qbar = 0;
                    for (I j = 0; j < npe; ++j)
                        qbar += donor.pphibar[j + static_cast<std::size_t>(npe) * p] *
                                sol.udg[j + static_cast<std::size_t>(npe) *
                                    (q_component(c, d, ncu) +
                                     static_cast<std::size_t>(nc) * e)];
                    trace -= donor.pd[p + static_cast<std::size_t>(npts) * d] * qbar;
                }
            }
            else {
                // q = -grad(u) in Exasim.  Equation (42) therefore changes
                // -q.d in equation (32) into +grad(u).d.
                for (I d = 0; d < nd; ++d) {
                    T gradient = 0;
                    for (I j = 0; j < npe; ++j)
                        gradient += donor.pdphibar[j + static_cast<std::size_t>(npe) *
                                                       (d + static_cast<std::size_t>(nd) * p)] *
                                    sol.udg[j + static_cast<std::size_t>(npe) *
                                                (c + static_cast<std::size_t>(nc) * e)];
                    trace += donor.pd[p + static_cast<std::size_t>(npts) * d] * gradient;
                }
            }
            data.uhg[p + static_cast<std::size_t>(npts) * c] = trace;
        }
    }
}

template <class M, class T, class I>
inline void evaluate_coupling_flux(CouplingPointData<T, I>& data,
        nonmatchinginterfaceT<T, I>& donor, appstructT<T, I>& app,
        masterstructT<T, I>& master, meshstructT<T, I>& mesh,
        solstructT<T, I>& sol, tempstructT<T, I>& tmp,
        commonstructT<T, I>& common, bool withJacobian, I backend)
{
    const I npts = donor.npts, ncu12 = donor.ncu12;
    const I ncu = donor.ncu, nc = common.components.nc;
    const I ncw = common.components.ncw;
    data.flux.assign(checked_runtime_size<I>({npts, ncu12}, "flux"), 0);
    if (npts == 0) return;
    if (backend > 1)
        error("non-matching coupling: CUDA/HIP run-time kernels are deferred to Phase 8");

    if (withJacobian) {
        data.fluxUdg.assign(checked_runtime_size<I>({npts, ncu12, nc}, "flux udg Jacobian"), 0);
        data.fluxUhg.assign(checked_runtime_size<I>({npts, ncu12, ncu}, "flux uhg Jacobian"), 0);
        data.fluxWdg.assign(checked_runtime_size<I>({npts, ncu12, ncw}, "flux wdg Jacobian"), 0);
    }

    if (withJacobian && ncw > 0 && common.timeparams.wave == 0) {
        data.wdgUdg.assign(checked_runtime_size<I>({npts, ncw, nc}, "wdg Jacobian"), 0);
        std::vector<T> equationUdg = data.udg;
        for (I c = 0; c < ncu; ++c)
            for (I p = 0; p < npts; ++p)
                equationUdg[p + static_cast<std::size_t>(npts) * c] =
                    data.uhg[p + static_cast<std::size_t>(npts) * c];

        const I nprop = app.materialdb_nprop;
        const I nstate = app.materialdb_nstate;
        const I np = app.materialdb_porder + 1;
        const I ncwa = ncw - nprop;
        if (nprop < 0 || ncwa < 0)
            error("non-matching coupling: invalid material property count");
        std::size_t scratchPerPoint = static_cast<std::size_t>(ncw) +
            static_cast<std::size_t>(ncw) * ncw + 1;
        if (nprop > 0) {
            scratchPerPoint += static_cast<std::size_t>(ncwa) * ncwa +
                               static_cast<std::size_t>(ncw) * nc +
                               static_cast<std::size_t>(ncwa) * nc;
            scratchPerPoint += static_cast<std::size_t>(nstate) * (1 + nc + ncw + nc) +
                               static_cast<std::size_t>(nprop) * nstate +
                               static_cast<std::size_t>(nstate) *
                                   (2 + static_cast<std::size_t>(3) * np);
        }
        if (scratchPerPoint > std::numeric_limits<std::size_t>::max() /
                                  static_cast<std::size_t>(npts))
            error("non-matching coupling: differentiated w equation scratch size overflow");
        std::vector<T> wScratch(static_cast<std::size_t>(npts) * scratchPerPoint, 0);
        const I integerFields = nprop > 0 ? nstate + 1 : 1;
        std::vector<I> iScratch(
            checked_runtime_size<I>({npts, integerFields}, "material integer scratch"), 0);
        wEquation<M>(data.wdg.data(), data.wdgUdg.data(), data.x.data(),
                equationUdg.data(), data.odg.data(), data.wsrc.data(), wScratch.data(),
                app, common, npts, backend, iScratch.data());
    }

    if (withJacobian) {
        EXASIM_DRIVER_CALL(FintDriver, data.flux.data(), data.fluxUdg.data(),
                data.fluxWdg.data(), data.fluxUhg.data(), data.x.data(),
                data.udg.data(), data.odg.data(), data.wdg.data(), data.uhg.data(),
                data.nl.data(), mesh, master, app, sol, tmp, common, npts,
                common.couplingparams.coupledcondition, backend);

        if (ncw > 0 && common.timeparams.wave == 0) {
            // Match the existing conformal chain rule: w(uh,q), not w(u,q).
            for (I p = 0; p < npts; ++p)
                for (I n = 0; n < ncu12; ++n) {
                    for (I c = 0; c < ncu; ++c)
                        for (I w = 0; w < ncw; ++w)
                            data.fluxUhg[p + static_cast<std::size_t>(npts) * (n + ncu12 * c)] +=
                                data.fluxWdg[p + static_cast<std::size_t>(npts) * (n + ncu12 * w)] *
                                data.wdgUdg[p + static_cast<std::size_t>(npts) * (w + ncw * c)];
                    for (I c = ncu; c < nc; ++c)
                        for (I w = 0; w < ncw; ++w)
                            data.fluxUdg[p + static_cast<std::size_t>(npts) * (n + ncu12 * c)] +=
                                data.fluxWdg[p + static_cast<std::size_t>(npts) * (n + ncu12 * w)] *
                                data.wdgUdg[p + static_cast<std::size_t>(npts) * (w + ncw * c)];
                }
        }
    }
    else {
        if (ncw > 0 && common.timeparams.wave == 0) {
            std::vector<T> equationUdg = data.udg;
            for (I c = 0; c < ncu; ++c)
                for (I p = 0; p < npts; ++p)
                    equationUdg[p + static_cast<std::size_t>(npts) * c] =
                        data.uhg[p + static_cast<std::size_t>(npts) * c];
            const I nprop = app.materialdb_nprop;
            const I nstate = app.materialdb_nstate;
            const I np = app.materialdb_porder + 1;
            const I ncwa = ncw - nprop;
            if (nprop < 0 || ncwa < 0)
                error("non-matching coupling: invalid material property count");

            // The nonlinear ordinary-w workspace and material interpolation
            // workspace are reused by wEquation, so allocate their maximum.
            const std::size_t ordinaryPerPoint = static_cast<std::size_t>(ncw) +
                static_cast<std::size_t>(ncw) * ncw +
                static_cast<std::size_t>(ncwa) * ncwa;
            if (npts > 0 && ordinaryPerPoint >
                    std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(npts))
                error("non-matching coupling: w equation scratch size overflow");
            const std::size_t ordinaryScratch =
                static_cast<std::size_t>(npts) * ordinaryPerPoint;
            std::size_t materialScratch = 0;
            if (nprop > 0) {
                const std::size_t materialPerPoint = static_cast<std::size_t>(nstate) *
                                                     (3 + static_cast<std::size_t>(2) * np);
                if (npts > 0 && materialPerPoint >
                        std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(npts))
                    error("non-matching coupling: material property scratch size overflow");
                materialScratch = static_cast<std::size_t>(npts) * materialPerPoint;
            }
            std::vector<T> scratch(std::max(ordinaryScratch, materialScratch), 0);
            const I integerFields = nprop > 0 ? nstate + 1 : 1;
            std::vector<I> iscratch(
                checked_runtime_size<I>({npts, integerFields}, "material integer scratch"), 0);
            wEquation<M>(data.wdg.data(), data.x.data(), equationUdg.data(),
                    data.odg.data(), data.wsrc.data(), scratch.data(), app, common,
                    npts, backend, iscratch.data());
        }
        EXASIM_DRIVER_CALL(FintDriver, data.flux.data(), data.x.data(), data.udg.data(),
                data.odg.data(), data.wdg.data(), data.uhg.data(), data.nl.data(),
                mesh, master, app, sol, tmp, common, npts,
                common.couplingparams.coupledcondition, backend);
    }
}

template <class T, class I>
inline T weighted_test(const nonmatchinginterfaceT<T, I>& donor,
        const masterstructT<T, I>& master, I point, I testNode,
        const OwnerTestView<T, I>& ownerTests = OwnerTestView<T, I>())
{
    if (ownerTests.values)
        return donor.pwJ[point] * ownerTests.values[
            testNode + static_cast<std::size_t>(ownerTests.npf) * point];
    return donor.pwJ[point] *
           master.shapfgt[donor.pg[point] + static_cast<std::size_t>(donor.ngf) * testNode];
}

// Equation (40): residual-only assembly, prior to static condensation.
template <class T, class I>
inline void form_uncondensed_slot_residual(nonmatchinginterfaceT<T, I>& donor,
        const masterstructT<T, I>& master, const CouplingPointData<T, I>& data,
        const OwnerTestView<T, I>& ownerTests = OwnerTestView<T, I>())
{
    std::fill(donor.Ri, donor.Ri + donor.szRi, static_cast<T>(0));
    for (I p = 0; p < donor.npts; ++p) {
        const I slot = donor.pslot[p];
        for (I i = 0; i < donor.npf; ++i) {
            const T W = weighted_test(donor, master, p, i, ownerTests);
            for (I n = 0; n < donor.ncu12; ++n)
                donor.Ri[n + donor.ncu12 * (i + donor.npf * slot)] -=
                    W * data.flux[p + static_cast<std::size_t>(donor.npts) * n];
        }
    }
}

// Equations (39), (41), (42), and (49), grouped by (slot, donor element).
template <class T, class I>
inline void form_and_condense_pairs(nonmatchinginterfaceT<T, I>& donor,
        const masterstructT<T, I>& master, const CouplingPointData<T, I>& data,
        const resstructT<T, I>& res, const commonstructT<T, I>& common,
        cublasHandle_t handle, I backend,
        const OwnerTestView<T, I>& ownerTests = OwnerTestView<T, I>())
{
    const I npe = donor.npe, npf = donor.npf, nfe = donor.nfe;
    const I ncu = donor.ncu, ncu12 = donor.ncu12, ncq = donor.ncq;
    const I nd = donor.nd, nc = ncu + ncq;
    const I n = npe * ncu, m = npf * nfe * ncu, m12 = npf * ncu12;
    const I ne = common.meshsizes.ne;
    std::fill(donor.Ri, donor.Ri + donor.szRi, static_cast<T>(0));
    std::fill(donor.Hi, donor.Hi + donor.szHi, static_cast<T>(0));

    // The flux part is naturally accumulated once per slot (eq. 39).
    form_uncondensed_slot_residual(donor, master, data, ownerTests);

    std::vector<T> K(static_cast<std::size_t>(m12) * n);
    std::vector<T> H(static_cast<std::size_t>(m12) * m);
    std::vector<T> G(static_cast<std::size_t>(m12) * npe * ncu);
    std::vector<T> pairResidual(static_cast<std::size_t>(m12));

    T scalar = 1.0;
    if (common.timeparams.wave == 1) scalar = 1.0 / common.timestate.dtfactor;

    for (I pair = 0; pair < donor.npairs; ++pair) {
        std::fill(K.begin(), K.end(), static_cast<T>(0));
        std::fill(H.begin(), H.end(), static_cast<T>(0));
        const I e = donor.pairelem[pair];
        const I slot = donor.pairslot[pair];

        for (I pindex = donor.pairptr[pair]; pindex < donor.pairptr[pair + 1]; ++pindex) {
            const I p = pindex;
            const I a = donor.pa[p];
            for (I i = 0; i < npf; ++i) {
                const T W = weighted_test(donor, master, p, i, ownerTests);
                for (I out = 0; out < ncu12; ++out) {
                    for (I c = 0; c < ncu; ++c) {
                        const T fu = data.fluxUdg[p + static_cast<std::size_t>(donor.npts) *
                                                       (out + ncu12 * c)];
                        const T fuh = data.fluxUhg[p + static_cast<std::size_t>(donor.npts) *
                                                        (out + ncu12 * c)];
                        for (I j = 0; j < npe; ++j) {
                            T derivative = fu * donor.pphi[j + static_cast<std::size_t>(npe) * p];
                            if (ncq == 0)
                                for (I d = 0; d < nd; ++d)
                                    derivative += fuh * donor.pd[p + static_cast<std::size_t>(donor.npts) * d] *
                                        donor.pdphibar[j + static_cast<std::size_t>(npe) *
                                                            (d + static_cast<std::size_t>(nd) * p)];
                            K[(out + ncu12 * i) + static_cast<std::size_t>(m12) *
                                  (j + npe * c)] += W * derivative;
                        }
                        for (I j = 0; j < npf; ++j) {
                            const I column = c + ncu * (j + npf * a);
                            H[(out + ncu12 * i) + static_cast<std::size_t>(m12) * column] +=
                                W * fuh * donor.ppsi[j + static_cast<std::size_t>(npf) * p];
                        }
                    }
                }
            }
        }

        if (ncq > 0) {
            for (I d = 0; d < nd; ++d) {
                std::fill(G.begin(), G.end(), static_cast<T>(0));
                for (I p = donor.pairptr[pair]; p < donor.pairptr[pair + 1]; ++p)
                    for (I i = 0; i < npf; ++i) {
                        const T W = weighted_test(donor, master, p, i, ownerTests);
                        for (I out = 0; out < ncu12; ++out)
                            for (I c = 0; c < ncu; ++c) {
                                const T fq = data.fluxUdg[p + static_cast<std::size_t>(donor.npts) *
                                    (out + ncu12 * q_component(c, d, ncu))];
                                const T fuh = data.fluxUhg[p + static_cast<std::size_t>(donor.npts) *
                                    (out + ncu12 * c)];
                                for (I j = 0; j < npe; ++j) {
                                    const T derivative = fq * donor.pphi[j + static_cast<std::size_t>(npe) * p] -
                                        fuh * donor.pd[p + static_cast<std::size_t>(donor.npts) * d] *
                                        donor.pphibar[j + static_cast<std::size_t>(npe) * p];
                                    // One-face G layout expected by the existing condensation kernels.
                                    G[i + npf * j + static_cast<std::size_t>(npf) * npe *
                                        (out + ncu12 * c)] += W * derivative;
                                }
                            }
                    }
                const T* C = &res.C[static_cast<std::size_t>(npe) * npe * (e + ne * d)];
                const T* E = &res.E[static_cast<std::size_t>(npe) * npf * nfe * (e + ne * d)];
                schurMatrixGMinvC(K.data(), G.data(), C, scalar,
                                  npe, ncu12, ncu, npf, 1, 1);
                schurMatrixGintMinvE(H.data(), G.data(), E, scalar,
                                     npe, ncu12, ncu, npf, nfe, 1);
            }
        }

        // res.F and res.Ru contain D^{-1}F with its sign and D^{-1}Ru after
        // uEquationSchurBlock, so these are the existing eq. (49) updates.
        PGEMNMStridedBached(handle, m12, m, n, one, K.data(), m12,
                &res.F[static_cast<std::size_t>(n) * m * e], n, one,
                H.data(), m12, 1, backend);
        std::fill(pairResidual.begin(), pairResidual.end(), static_cast<T>(0));
        PGEMNMStridedBached(handle, m12, 1, n, minusone, K.data(), m12,
                &res.Ru[static_cast<std::size_t>(n) * e], n, zero,
                pairResidual.data(), m12, 1, backend);

        T* pairH = &donor.Hi[static_cast<std::size_t>(m12) * m * pair];
        std::copy(H.begin(), H.end(), pairH);
        T* slotR = &donor.Ri[static_cast<std::size_t>(m12) * slot];
        for (I i = 0; i < m12; ++i) slotR[i] += pairResidual[i];
    }
}

template <class M, class T, class I>
inline void compute_nonmatching_coupling(nonmatchingdataT<T, I>& nm,
        solstructT<T, I>& sol, resstructT<T, I>& res, appstructT<T, I>& app,
        masterstructT<T, I>& master, meshstructT<T, I>& mesh,
        tempstructT<T, I>& tmp, commonstructT<T, I>& common,
        cublasHandle_t handle, bool withJacobian, I backend,
        const OwnerTestData<T, I>* ownerTests)
{
    reject_rectangular_host_path(ownerTests, "compute_nonmatching_coupling");
    if (nm.interfacetype != 1) return;
    if (!ownerTests)
        for (int direction = 0; direction < 2; ++direction)
            if (nm.dir[direction].isdonor && nm.dir[direction].nslots > 0 &&
                    nm.dir[direction].Ri == nullptr)
                error("non-matching host path: missing frozen donor work arrays");
    for (int direction = 0; direction < 2; ++direction) {
        nonmatchinginterfaceT<T, I>& donor = nm.dir[direction];
        if (!donor.isdonor || donor.npts == 0) continue;
        CouplingPointData<T, I> data;
        build_coupling_point_data(data, donor, sol, mesh, common);
        evaluate_coupling_flux<M>(data, donor, app, master, mesh, sol, tmp,
                                  common, withJacobian, backend);
        const OwnerTestView<T, I> view = owner_test_view(ownerTests, direction, donor.npf);
        if (withJacobian)
            form_and_condense_pairs(donor, master, data, res, common, handle, backend, view);
        else
            form_uncondensed_slot_residual(donor, master, data, view);
    }
}

template <class M, class T, class I>
inline void compute_nonmatching_coupling(NonmatchingRuntime<T, I>& runtime,
        solstructT<T, I>& sol, resstructT<T, I>& res, appstructT<T, I>& app,
        masterstructT<T, I>& master, meshstructT<T, I>& mesh,
        tempstructT<T, I>& tmp, commonstructT<T, I>& common,
        cublasHandle_t handle, bool withJacobian, I backend)
{
    if (runtime.interfacetype != 1) return;
    for (int direction = 0; direction < 2; ++direction) {
        auto& donor = runtime.dir[direction];
        if (!donor.view.isdonor || donor.view.npts == 0) continue;
        build_runtime_point_data(donor, sol, mesh, common);
        evaluate_runtime_flux<M>(donor, app, master, mesh, sol, tmp, common, withJacobian, backend);
        form_runtime_slot_residual(donor, master);
        if (withJacobian) form_runtime_pairs(donor, master, res, common, handle, backend);
    }
}

// Phase 5: the one predicate for active non-matching coupling. Every type-1
// branch (hdgMatVec, residual and linear-system assembly) is guarded by it, so
// type-1 setup data is never applied while coupling is disabled.
template <class T, class I>
inline bool nonmatching_coupling_active(const commonstructT<T, I>& common)
{
    return common.couplingparams.coupledcondition > 0 &&
           common.couplingparams.coupledinterface > 0 &&
           common.couplingparams.interfacetype == 1;
}

#ifdef HAVE_MPI
constexpr bool CouplingHaveMPI = true;
#else
constexpr bool CouplingHaveMPI = false;
#endif

// Checks the coupling selection and returns nonmatching_coupling_active(common).
// - Setup data of type 1 requires common.couplingparams.interfacetype == 1.
// - Active coupling requires the non-matching data, a completed setup, MPI,
//   and a CPU backend (CUDA/HIP run-time kernels are Phase 8).
// The checks use only rank-uniform inputs (common, the presence of the data,
// backend), so every participating rank fails at the same call, before any
// donor/owner branching or MPI communication. Coupling is never omitted
// silently. haveMPI is a parameter only so that tests can exercise the
// no-MPI branch; production uses validate_coupling_selection.
template <class T, class I>
inline bool check_coupling_selection(const commonstructT<T, I>& common,
        const nonmatchingdataT<T, I>* nm, I backend, const char* where, bool haveMPI)
{
    const bool active = nonmatching_coupling_active(common);
    const I fromApp = common.couplingparams.interfacetype;
    const I fromSetup = nm ? nm->interfacetype : 0;
    std::ostringstream oss;
    if (fromSetup == 1 && fromApp != 1)
        oss << "non-matching coupling (" << where << "): setup data has interfacetype 1 but "
            << "common.couplingparams.interfacetype = " << fromApp;
    else if (active && nm == nullptr)
        oss << "non-matching coupling (" << where << "): interfacetype 1 requires the "
            << "non-matching data, but none was passed";
    else if (active && fromSetup != 1)
        oss << "non-matching coupling (" << where << "): interfacetype 1 is selected but the "
            << "non-matching setup did not run (nonmatching->interfacetype = " << fromSetup << ")";
    else if (active && !haveMPI)
        oss << "non-matching coupling (" << where << "): interfacetype 1 requires MPI";
    else if (active && backend > 1)
        oss << "non-matching coupling (" << where << "): interfacetype 1 runs on the CPU only "
            << "(backend = " << backend << "); CUDA/HIP run-time kernels are deferred to Phase 8";
    if (!oss.str().empty()) error(oss.str());
    return active;
}

template <class T, class I>
inline bool validate_coupling_selection(const commonstructT<T, I>& common,
        const nonmatchingdataT<T, I>* nm, I backend, const char* where)
{
    return check_coupling_selection(common, nm, backend, where, CouplingHaveMPI);
}

template <class T, class I>
inline bool validate_coupling_selection(const commonstructT<T, I>& common,
        const NonmatchingRuntime<T, I>* runtime, I backend, const char* where)
{
    const bool active = nonmatching_coupling_active(common);
    const I fromApp = common.couplingparams.interfacetype;
    const I fromRuntime = runtime ? runtime->interfacetype : 0;
    int failure = 0;
    if (fromRuntime == 1 && fromApp != 1) failure = 1;
    else if (active && runtime == nullptr) failure = 2;
    else if (active && fromRuntime != 1) failure = 3;
    else if (active && !CouplingHaveMPI) failure = 4;
    else if (active && backend > 1) failure = 5;
    if (failure) {
        std::ostringstream oss;
        oss << "non-matching coupling (" << where << "): ";
        if (failure == 1) oss << "runtime data has interfacetype 1 but application selection is " << fromApp;
        else if (failure == 2) oss << "interfacetype 1 requires a run-time cache";
        else if (failure == 3) oss << "interfacetype 1 is selected but the run-time cache was not built";
        else if (failure == 4) oss << "interfacetype 1 requires MPI";
        else oss << "CUDA/HIP execution awaits device validation (backend = " << backend << ")";
        error(oss.str());
    }
    return active;
}

// Partial matrix-vector products of eq. (68) for every direction in which this
// rank is a donor: donor.Ri(:, slot) = sum over the pairs q of the slot of
// Hbar^{le}_12(q) * ve(:, pairelem[q]) (eq. (67) matrices stored by Phase 4).
// ve is the element trace vector of this rank (ncu x npf x nfe per element,
// m = ncu*npf*nfe, ne1 elements), gathered by hdgMatVec from the trace vector.
// One strided batched GEMM over all pairs, then a segmented sum per slot.
// Host workspaces (CPU-first; a persistent device workspace is Phase 8).
template <class T, class I>
inline void nonmatching_matvec_products(nonmatchingdataT<T, I>& nm, const T* ve,
        I m, I ne1, cublasHandle_t handle, I backend,
        const OwnerTestData<T, I>* ownerTests)
{
    reject_rectangular_host_path(ownerTests, "nonmatching_matvec_products");
    if (nm.interfacetype != 1) return;
    if (!ownerTests)
        for (int direction = 0; direction < 2; ++direction)
            if (nm.dir[direction].isdonor && nm.dir[direction].nslots > 0 &&
                    nm.dir[direction].Ri == nullptr)
                error("non-matching host path: missing frozen donor work arrays");
    if (backend > 1)
        error("non-matching coupling: CUDA/HIP run-time kernels are deferred to Phase 8");
    for (int direction = 0; direction < 2; ++direction) {
        nonmatchinginterfaceT<T, I>& d = nm.dir[direction];
        if (!d.isdonor || d.npairs == 0) continue;
        const I m12 = d.ncu12 * d.npf;
        if (m != d.ncu * d.npf * d.nfe)
            error("non-matching coupling: element trace size differs from the pair matrix columns");
        const std::size_t szHi = checked_runtime_size<I>({m12, m, d.npairs}, "pair matrices");
        const std::size_t szV = checked_runtime_size<I>({d.npairs, m}, "pair trace workspace");
        const std::size_t szR = checked_runtime_size<I>({d.npairs, m12}, "pair product workspace");
        const std::size_t szRi = checked_runtime_size<I>({d.nslots, m12}, "slot products");
        if (szHi != static_cast<std::size_t>(d.szHi) || szRi != static_cast<std::size_t>(d.szRi))
            error("non-matching coupling: pair or slot work arrays have unexpected sizes");
        if (d.pairptr[0] != 0 || d.pairptr[d.npairs] != d.npts)
            error("non-matching coupling: pair point ranges do not cover the donor points");

        std::vector<T> vpair(szV), rpair(szR);
        for (I q = 0; q < d.npairs; ++q) {
            const I e = d.pairelem[q];
            if (e < 0 || e >= ne1)
                error("non-matching coupling: pair element is out of range");
            if (d.pairptr[q + 1] < d.pairptr[q])
                error("non-matching coupling: pair point ranges are not monotone");
            std::copy(ve + static_cast<std::size_t>(m) * e,
                      ve + static_cast<std::size_t>(m) * e + m,
                      vpair.begin() + static_cast<std::size_t>(m) * q);
        }
        PGEMNMStridedBached(handle, m12, 1, m, one, d.Hi, m12, vpair.data(), m, zero,
                            rpair.data(), m12, d.npairs, backend);
        std::fill(d.Ri, d.Ri + d.szRi, static_cast<T>(0));
        for (I q = 0; q < d.npairs; ++q) {
            const I s = d.pairslot[q];
            if (s < 0 || s >= d.nslots)
                error("non-matching coupling: pair slot is out of range");
            T* slot = d.Ri + static_cast<std::size_t>(m12) * s;
            const T* r = rpair.data() + static_cast<std::size_t>(m12) * q;
            for (I i = 0; i < m12; ++i) slot[i] += r[i];
        }
    }
}

#ifdef HAVE_MPI
template <class I>
inline int checked_mpi_count(I blocks, I ncu12, I npf, const char* what)
{
    const std::size_t count = checked_runtime_size<I>({blocks, ncu12, npf}, what);
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        std::ostringstream oss;
        oss << "non-matching coupling: " << what << " exceeds MPI int count range";
        error(oss.str());
    }
    return static_cast<int>(count);
}

// Coupling-slot exchange: append the compact donor-to-owner traffic of donor.Ri
// (a partial residual, eq. (63), or a partial product, eq. (68)) to the normal
// element-neighbor requests.
template <class T, class I>
inline void begin_coupling_slot_exchange(nonmatchingdataT<T, I>& nm,
        commonstructT<T, I>& common, I& requestCounter,
        const OwnerTestData<T, I>* ownerTests)
{
    reject_rectangular_host_path(ownerTests, "begin_coupling_slot_exchange");
    if (nm.interfacetype != 1) return;
    if (!ownerTests)
        for (int direction = 0; direction < 2; ++direction)
            if (nm.dir[direction].isdonor && nm.dir[direction].nslots > 0 &&
                    nm.dir[direction].Ri == nullptr)
                error("non-matching host path: missing frozen donor work arrays");
    for (int direction = 0; direction < 2; ++direction) {
        nonmatchinginterfaceT<T, I>& d = nm.dir[direction];
        if (d.isdonor && d.nslots > 0) {
            std::copy(d.Ri, d.Ri + d.szRi, d.sendbuf);
            std::size_t offset = 0;
            for (I q = 0; q < d.nsendnb; ++q) {
                const int count = checked_mpi_count(d.sendcount[q], d.ncu12, d.npf,
                                                    "donor send count");
                if (count > 0) {
                    MPI_Isend(&d.sendbuf[offset], count, mpi_type<T>(), d.sendnb[q],
                              CouplingSlotTagBase + direction, EXASIM_COMM_WORLD,
                              &common.requests[requestCounter++]);
                    offset += count;
                }
            }
            if (offset != static_cast<std::size_t>(d.szsendbuf))
                error("non-matching coupling: packed donor send extent is inconsistent");
        }
        if (d.isowner && d.nrecvnb > 0) {
            std::size_t offset = 0;
            for (I q = 0; q < d.nrecvnb; ++q) {
                const int count = checked_mpi_count(d.recvcount[q], d.ncu12, d.npf,
                                                    "owner receive count");
                if (count > 0) {
                    MPI_Irecv(&d.recvbuf[offset], count, mpi_type<T>(), d.recvnb[q],
                              CouplingSlotTagBase + direction, EXASIM_COMM_WORLD,
                              &common.requests[requestCounter++]);
                    offset += count;
                }
            }
            if (offset != static_cast<std::size_t>(d.szrecvbuf))
                error("non-matching coupling: packed owner receive extent is inconsistent");
        }
    }
}

template <class T, class I>
inline void begin_coupling_slot_exchange(NonmatchingRuntime<T, I>& runtime,
        commonstructT<T, I>& common, I& requestCounter)
{
    if (runtime.interfacetype != 1) return;
    // Ri is already contiguous in send-neighbor/slot order.  Fence only ranks
    // that expose a device-produced send buffer to GPU-aware MPI.
    bool hasSend = false;
    for (int direction = 0; direction < 2; ++direction)
        hasSend = hasSend || (runtime.dir[direction].view.isdonor &&
                              runtime.dir[direction].view.nslots > 0);
    if (hasSend) Kokkos::fence();
    for (int direction = 0; direction < 2; ++direction) {
        auto& d = runtime.dir[direction];
        const auto& v = d.view;
        if (v.isdonor && v.nslots > 0) {
            std::size_t offset = 0;
            for (std::size_t q = 0; q < d.sendnb.size(); ++q) {
                const I rowNpf = v.rectangular ? v.ownerNpf : v.npf;
                const int count = checked_mpi_count(d.sendcount[q], v.ncu12, rowNpf,
                                                    "donor send count");
                if (count > 0) {
                    MPI_Isend(&d.Ri.data[offset], count, mpi_type<T>(), d.sendnb[q],
                              CouplingSlotTagBase + direction, EXASIM_COMM_WORLD,
                              &common.requests[requestCounter++]);
                    offset += count;
                }
            }
            if (offset != static_cast<std::size_t>(d.Ri.size))
                error("non-matching coupling: direct donor send extent is inconsistent");
        }
        if (v.isowner && !d.recvnb.empty()) {
            std::size_t offset = 0;
            for (std::size_t q = 0; q < d.recvnb.size(); ++q) {
                const int count = checked_mpi_count(d.recvcount[q], v.ncu12, v.npf,
                                                    "owner receive count");
                if (count > 0) {
                    MPI_Irecv(&d.recvbuf.data[offset], count, mpi_type<T>(), d.recvnb[q],
                              CouplingSlotTagBase + direction, EXASIM_COMM_WORLD,
                              &common.requests[requestCounter++]);
                    offset += count;
                }
            }
            if (offset != static_cast<std::size_t>(d.recvbuf.size))
                error("non-matching coupling: owner receive extent is inconsistent");
        }
    }
}

// Owner assembly of the coupling-slot exchange, eq. (64) (residual) or eq. (68)
// (product), with identity node order. recvnb and recvface already encode
// increasing donor rank and L_pr list order.
template <class T, class I>
inline void assemble_coupling_slot_owner(T* elementResidual,
        const nonmatchingdataT<T, I>& nm, const meshstructT<T, I>& mesh,
        const appstructT<T, I>& app, const commonstructT<T, I>& common)
{
    if (nm.interfacetype != 1) return;
    const I ncu = common.components.ncu;
    for (int direction = 0; direction < 2; ++direction) {
        const nonmatchinginterfaceT<T, I>& owner = nm.dir[direction];
        if (!owner.isowner) continue;
        I block = 0;
        if (owner.nfe <= 0 || owner.npf <= 0 || owner.ncu12 < 0 ||
            (owner.nrecvnb > 0 && (!owner.recvcount || !owner.recvnb)) ||
            (owner.nownfaces > 0 && !owner.ownface) ||
            (owner.szrecvbuf > 0 && !owner.recvbuf) || !mesh.intfaces || !app.interfacefluxmap)
            error("non-matching coupling: owner assembly data is incomplete");
        I expectedBlocks = 0;
        for (I q = 0; q < owner.nrecvnb; ++q) {
            if (owner.recvcount[q] < 0 || expectedBlocks > std::numeric_limits<I>::max() - owner.recvcount[q])
                error("non-matching coupling: owner receive-block count is invalid");
            expectedBlocks += owner.recvcount[q];
        }
        if (expectedBlocks > 0 && !owner.recvface)
            error("non-matching coupling: owner receive-face list is missing");
        const std::size_t expectedValues = checked_runtime_size<I>(
                {owner.ncu12, owner.npf, expectedBlocks}, "owner receive buffer");
        if (expectedValues != static_cast<std::size_t>(owner.szrecvbuf))
            error("non-matching coupling: owner receive-buffer extent is inconsistent");
        for (I q = 0; q < owner.nrecvnb; ++q)
            for (I entry = 0; entry < owner.recvcount[q]; ++entry, ++block) {
                const I ownIndex = owner.recvface[block];
                if (ownIndex < 0 || ownIndex >= owner.nownfaces)
                    error("non-matching coupling: owner receive-face index is out of range");
                const I intfaceIndex = owner.ownface[ownIndex];
                if (intfaceIndex < 0 || intfaceIndex >= mesh.szintfaces)
                    error("non-matching coupling: owner interface-face index is out of range");
                const I encoded = mesh.intfaces[intfaceIndex];
                const I e = encoded / owner.nfe;
                const I lf = encoded - owner.nfe * e;
                if (encoded < 0 || e < 0 || e >= common.meshsizes.ne || lf < 0 || lf >= owner.nfe)
                    error("non-matching coupling: encoded owner element face is out of range");
                const T* source = &owner.recvbuf[static_cast<std::size_t>(owner.ncu12) *
                                                        owner.npf * block];
                for (I i = 0; i < owner.npf; ++i)
                    for (I c = 0; c < owner.ncu12; ++c) {
                        const I mapped = app.interfacefluxmap[c];
                        if (mapped < 0 || mapped >= ncu)
                            error("non-matching coupling: interfacefluxmap is out of range");
                        elementResidual[mapped + static_cast<std::size_t>(ncu) *
                            (i + owner.npf * (lf + owner.nfe * e))] +=
                                source[c + owner.ncu12 * i];
                    }
            }
        if (block != expectedBlocks)
            error("non-matching coupling: owner receive traversal is incomplete");
    }
}


#endif

} // namespace nonmatching
} // namespace exasim

#endif
