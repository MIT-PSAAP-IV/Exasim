/*
    Backend-portable runtime kernels for non-matching multiphysics coupling.

    These Kokkos and batched-BLAS kernels are dimension-parameterized and are
    shared by the supported 2D line and 3D surface interface paths.
*/

#ifndef __NONMATCHINGRUNTIMEKERNELS_HPP
#define __NONMATCHINGRUNTIMEKERNELS_HPP

#include "nonmatchingruntime.hpp"
#include "nonmatchingrectangular.hpp"

namespace exasim {
namespace nonmatching {

// Phase 8 production path.  Every Kokkos iteration owns one output entry and
// performs reductions in the accepted point/node order, avoiding atomics and
// preserving CPU summation order while remaining device-callable.
template <class T, class I>
inline void build_runtime_point_data(NonmatchingRuntimeDirection<T, I>& runtime,
        const solstructT<T, I>& sol, const meshstructT<T, I>& mesh,
        const commonstructT<T, I>& common)
{
    const auto v = runtime.view;
    const I npts = v.npts, nd = v.nd, npe = v.npe, npf = v.npf, nfe = v.nfe;
    const I ncu = v.ncu, ncq = v.ncq, nc = common.components.nc;
    const I ncx = common.components.ncx, nco = common.components.nco;
    const I ncw = common.components.ncw, ndf = npf * nfe;
    const I spatialScheme = common.spatialScheme;
    T* x = runtime.x.data; T* udg = runtime.udg.data; T* odg = runtime.odg.data;
    T* wdg = runtime.wdg.data; T* wsrc = runtime.wsrc.data;
    T* nl = runtime.nl.data; T* uhg = runtime.uhg.data;
    const T* solUdg = sol.udg; const T* solOdg = sol.odg;
    const T* solWdg = sol.wdg; const T* solWsrc = sol.wsrc; const T* solUh = sol.uh;
    const I* elemcon = mesh.elemcon;

    Kokkos::parallel_for("NonmatchingPointCoordinates",
        static_cast<std::size_t>(npts) * ncx, KOKKOS_LAMBDA(const std::size_t idx) {
            const I p = static_cast<I>(idx % npts);
            const I d = static_cast<I>(idx / npts);
            x[idx] = d < nd ? v.px[p + static_cast<std::size_t>(npts) * d] : static_cast<T>(0);
        });
    Kokkos::parallel_for("NonmatchingPointNormals",
        static_cast<std::size_t>(npts) * nd, KOKKOS_LAMBDA(const std::size_t idx) {
            nl[idx] = -v.pnl[idx];
        });
    Kokkos::parallel_for("NonmatchingInterpolateUdg",
        static_cast<std::size_t>(npts) * nc, KOKKOS_LAMBDA(const std::size_t idx) {
            const I p = static_cast<I>(idx % npts);
            const I c = static_cast<I>(idx / npts);
            const I e = v.pe[p];
            T value = 0;
            for (I j = 0; j < npe; ++j)
                value += v.pphi[j + static_cast<std::size_t>(npe) * p] *
                    solUdg[j + static_cast<std::size_t>(npe) *
                        (c + static_cast<std::size_t>(nc) * e)];
            udg[idx] = value;
        });
    if (nco > 0)
        Kokkos::parallel_for("NonmatchingInterpolateOdg",
            static_cast<std::size_t>(npts) * nco, KOKKOS_LAMBDA(const std::size_t idx) {
                const I p = static_cast<I>(idx % npts);
                const I c = static_cast<I>(idx / npts);
                const I e = v.pe[p];
                T value = 0;
                if (solOdg)
                    for (I j = 0; j < npe; ++j)
                        value += v.pphi[j + static_cast<std::size_t>(npe) * p] *
                            solOdg[j + static_cast<std::size_t>(npe) *
                                (c + static_cast<std::size_t>(nco) * e)];
                odg[idx] = value;
            });
    if (ncw > 0)
        Kokkos::parallel_for("NonmatchingInterpolateW",
            static_cast<std::size_t>(npts) * ncw, KOKKOS_LAMBDA(const std::size_t idx) {
                const I p = static_cast<I>(idx % npts);
                const I c = static_cast<I>(idx / npts);
                const I e = v.pe[p];
                T value = 0, source = 0;
                for (I j = 0; j < npe; ++j) {
                    const T phi = v.pphi[j + static_cast<std::size_t>(npe) * p];
                    if (solWdg)
                        value += phi * solWdg[j + static_cast<std::size_t>(npe) *
                            (c + static_cast<std::size_t>(ncw) * e)];
                    if (solWsrc)
                        source += phi * solWsrc[j + static_cast<std::size_t>(npe) *
                            (c + static_cast<std::size_t>(ncw) * e)];
                }
                wdg[idx] = value;
                wsrc[idx] = source;
            });
    Kokkos::parallel_for("NonmatchingCorrectedTrace",
        static_cast<std::size_t>(npts) * ncu, KOKKOS_LAMBDA(const std::size_t idx) {
            const I p = static_cast<I>(idx % npts);
            const I c = static_cast<I>(idx / npts);
            const I e = v.pe[p], a = v.pa[p];
            T trace = 0;
            for (I j = 0; j < npf; ++j) {
                const I encoded = elemcon[j + npf * a + ndf * e];
                T nodalTrace;
                if (spatialScheme == 0) {
                    const I node = encoded % npf, face = encoded / npf;
                    nodalTrace = solUh[node + static_cast<std::size_t>(npf) *
                        (c + static_cast<std::size_t>(ncu) * face)];
                }
                else nodalTrace = solUh[c + static_cast<std::size_t>(ncu) * encoded];
                trace += v.ppsi[j + static_cast<std::size_t>(npf) * p] * nodalTrace;
            }
            if (ncq > 0) {
                for (I d = 0; d < nd; ++d) {
                    T qbar = 0;
                    const I qc = ncu + c + ncu * d;
                    for (I j = 0; j < npe; ++j)
                        qbar += v.pphibar[j + static_cast<std::size_t>(npe) * p] *
                            solUdg[j + static_cast<std::size_t>(npe) *
                                (qc + static_cast<std::size_t>(nc) * e)];
                    trace -= v.pd[p + static_cast<std::size_t>(npts) * d] * qbar;
                }
            }
            else {
                for (I d = 0; d < nd; ++d) {
                    T gradient = 0;
                    for (I j = 0; j < npe; ++j)
                        gradient += v.pdphibar[j + static_cast<std::size_t>(npe) *
                            (d + static_cast<std::size_t>(nd) * p)] *
                            solUdg[j + static_cast<std::size_t>(npe) *
                                (c + static_cast<std::size_t>(nc) * e)];
                    trace += v.pd[p + static_cast<std::size_t>(npts) * d] * gradient;
                }
            }
            uhg[idx] = trace;
        });
}

template <class M, class T, class I>
inline void evaluate_runtime_flux(NonmatchingRuntimeDirection<T, I>& runtime,
        appstructT<T, I>& app, masterstructT<T, I>& master, meshstructT<T, I>& mesh,
        solstructT<T, I>& sol, tempstructT<T, I>& tmp,
        commonstructT<T, I>& common, bool withJacobian, I backend)
{
    const auto v = runtime.view;
    const I npts = v.npts, ncu12 = v.ncu12, ncu = v.ncu;
    const I nc = common.components.nc, ncw = common.components.ncw;
    ArraySetValue(runtime.flux.data, zero, runtime.flux.size);
    if (withJacobian) {
        ArraySetValue(runtime.fluxUdg.data, zero, runtime.fluxUdg.size);
        ArraySetValue(runtime.fluxUhg.data, zero, runtime.fluxUhg.size);
        if (runtime.fluxWdg.size > 0) ArraySetValue(runtime.fluxWdg.data, zero, runtime.fluxWdg.size);
    }

    if (ncw > 0 && common.timeparams.wave == 0) {
        const T* udg = runtime.udg.data; const T* uhg = runtime.uhg.data;
        T* equationUdg = runtime.equationUdg.data;
        Kokkos::parallel_for("NonmatchingEquationUdg",
            static_cast<std::size_t>(npts) * nc, KOKKOS_LAMBDA(const std::size_t idx) {
                const I p = static_cast<I>(idx % npts), c = static_cast<I>(idx / npts);
                equationUdg[idx] = c < ncu ? uhg[p + static_cast<std::size_t>(npts) * c] : udg[idx];
            });
        if (withJacobian) {
            ArraySetValue(runtime.wdgUdg.data, zero, runtime.wdgUdg.size);
            wEquation<M>(runtime.wdg.data, runtime.wdgUdg.data, runtime.x.data,
                runtime.equationUdg.data, runtime.odg.data, runtime.wsrc.data,
                runtime.wScratch.data, app, common, npts, backend, runtime.iScratch.data);
        }
        else {
            wEquation<M>(runtime.wdg.data, runtime.x.data, runtime.equationUdg.data,
                runtime.odg.data, runtime.wsrc.data, runtime.wScratch.data,
                app, common, npts, backend, runtime.iScratch.data);
        }
    }

    if (withJacobian) {
        EXASIM_DRIVER_CALL(FintDriver, runtime.flux.data, runtime.fluxUdg.data,
            runtime.fluxWdg.data, runtime.fluxUhg.data, runtime.x.data,
            runtime.udg.data, runtime.odg.data, runtime.wdg.data, runtime.uhg.data,
            runtime.nl.data, mesh, master, app, sol, tmp, common, npts,
            common.couplingparams.coupledcondition, backend);
        if (ncw > 0 && common.timeparams.wave == 0) {
            T* fluxUdg = runtime.fluxUdg.data; T* fluxUhg = runtime.fluxUhg.data;
            const T* fluxWdg = runtime.fluxWdg.data; const T* wdgUdg = runtime.wdgUdg.data;
            Kokkos::parallel_for("NonmatchingWChainRule",
                static_cast<std::size_t>(npts) * ncu12 * nc,
                KOKKOS_LAMBDA(const std::size_t idx) {
                    const I p = static_cast<I>(idx % npts);
                    const I q = static_cast<I>(idx / npts);
                    const I out = q % ncu12, c = q / ncu12;
                    if (c < ncu) {
                        const std::size_t target = p + static_cast<std::size_t>(npts) *
                                                   (out + ncu12 * c);
                        T value = fluxUhg[target];
                        for (I w = 0; w < ncw; ++w)
                            value += fluxWdg[p + static_cast<std::size_t>(npts) * (out + ncu12 * w)] *
                                     wdgUdg[p + static_cast<std::size_t>(npts) * (w + ncw * c)];
                        fluxUhg[target] = value;
                    }
                    else {
                        const std::size_t target = p + static_cast<std::size_t>(npts) *
                                                   (out + ncu12 * c);
                        T value = fluxUdg[target];
                        for (I w = 0; w < ncw; ++w)
                            value += fluxWdg[p + static_cast<std::size_t>(npts) * (out + ncu12 * w)] *
                                     wdgUdg[p + static_cast<std::size_t>(npts) * (w + ncw * c)];
                        fluxUdg[target] = value;
                    }
                });
        }
    }
    else {
        EXASIM_DRIVER_CALL(FintDriver, runtime.flux.data, runtime.x.data, runtime.udg.data,
            runtime.odg.data, runtime.wdg.data, runtime.uhg.data, runtime.nl.data,
            mesh, master, app, sol, tmp, common, npts,
            common.couplingparams.coupledcondition, backend);
    }
}

template <class T, class I>
inline void form_runtime_slot_residual(NonmatchingRuntimeDirection<T, I>& runtime,
        const masterstructT<T, I>& master)
{
    const auto v = runtime.view;
    if (v.rectangular) {
        form_rectangular_slot_residual(runtime);
        return;
    }
    const I m12 = v.ncu12 * v.npf, npts = v.npts;
    const T* flux = runtime.flux.data; const T* shapfgt = master.shapfgt;
    T* Ri = runtime.Ri.data;
    Kokkos::parallel_for("NonmatchingSlotResidual",
        static_cast<std::size_t>(v.nslots) * m12, KOKKOS_LAMBDA(const std::size_t idx) {
            const I row = static_cast<I>(idx % m12), slot = static_cast<I>(idx / m12);
            const I out = row % v.ncu12, i = row / v.ncu12;
            T value = 0;
            for (I p = v.slotptr[slot]; p < v.slotptr[slot + 1]; ++p)
                value -= v.pwJ[p] * runtime_owner_test(v, shapfgt, v.npf, i, p) *
                         flux[p + static_cast<std::size_t>(npts) * out];
            Ri[idx] = value;
        });
}

template <class T, class I>
inline void form_runtime_pairs(NonmatchingRuntimeDirection<T, I>& runtime,
        const masterstructT<T, I>& master, const resstructT<T, I>& res,
        const commonstructT<T, I>& common, cublasHandle_t handle, I backend)
{
    const auto v = runtime.view;
    if (v.rectangular) {
        form_rectangular_pairs(runtime, res, common, handle, backend);
        return;
    }
    const I npe = v.npe, npf = v.npf, nfe = v.nfe, ncu = v.ncu;
    const I ncu12 = v.ncu12, ncq = v.ncq, nd = v.nd, npts = v.npts;
    const I n = npe * ncu, m = npf * nfe * ncu, m12 = npf * ncu12;
    const I ne = common.meshsizes.ne;
    T* K = runtime.K.data; T* H = runtime.Hi.data;
    const T* fluxUdg = runtime.fluxUdg.data; const T* fluxUhg = runtime.fluxUhg.data;
    const T* shapfgt = master.shapfgt;
    const T* C = res.C; const T* E = res.E; const T* F = res.F; const T* Ru = res.Ru;
    const T scalar = common.timeparams.wave == 1 ? static_cast<T>(1) / common.timestate.dtfactor : static_cast<T>(1);

    Kokkos::parallel_for("NonmatchingPairK",
        static_cast<std::size_t>(v.npairs) * m12 * n, KOKKOS_LAMBDA(const std::size_t idx) {
            const I row = static_cast<I>(idx % m12);
            const I q = static_cast<I>(idx / m12);
            const I col = q % n, pair = q / n;
            const I out = row % ncu12, i = row / ncu12;
            const I j = col % npe, c = col / npe;
            T value = 0;
            for (I p = v.pairptr[pair]; p < v.pairptr[pair + 1]; ++p) {
                const T W = v.pwJ[p] * runtime_owner_test(v, shapfgt, npf, i, p);
                T derivative = fluxUdg[p + static_cast<std::size_t>(npts) * (out + ncu12 * c)] *
                               v.pphi[j + static_cast<std::size_t>(npe) * p];
                if (ncq == 0)
                    for (I d = 0; d < nd; ++d)
                        derivative += fluxUhg[p + static_cast<std::size_t>(npts) * (out + ncu12 * c)] *
                            v.pd[p + static_cast<std::size_t>(npts) * d] *
                            v.pdphibar[j + static_cast<std::size_t>(npe) *
                                (d + static_cast<std::size_t>(nd) * p)];
                value += W * derivative;
            }
            K[idx] = value;
        });

    Kokkos::parallel_for("NonmatchingPairH",
        static_cast<std::size_t>(v.npairs) * m12 * m, KOKKOS_LAMBDA(const std::size_t idx) {
            const I row = static_cast<I>(idx % m12);
            const I q = static_cast<I>(idx / m12);
            const I col = q % m, pair = q / m;
            const I out = row % ncu12, i = row / ncu12;
            const I c = col % ncu;
            const I faceNode = col / ncu, jface = faceNode % npf, a = faceNode / npf;
            T value = 0;
            for (I p = v.pairptr[pair]; p < v.pairptr[pair + 1]; ++p)
                if (v.pa[p] == a) {
                    const T W = v.pwJ[p] * runtime_owner_test(v, shapfgt, npf, i, p);
                    value += W * fluxUhg[p + static_cast<std::size_t>(npts) *
                        (out + ncu12 * c)] * v.ppsi[jface + static_cast<std::size_t>(npf) * p];
                }
            H[idx] = value;
        });

    if (ncq > 0) {
        T* G = runtime.G.data; T* Cpair = runtime.Cpair.data; T* Epair = runtime.Epair.data;
        const I* pairelem = v.pairelem;
        const std::size_t gSize = static_cast<std::size_t>(m12) * n * v.npairs;
        const I traceNodes = npf * nfe;
        for (I d = 0; d < nd; ++d) {
            Kokkos::parallel_for("NonmatchingPairG", gSize,
                KOKKOS_LAMBDA(const std::size_t idx) {
                    std::size_t q = idx;
                    const I i = static_cast<I>(q % npf); q /= npf;
                    const I j = static_cast<I>(q % npe); q /= npe;
                    const I pair = static_cast<I>(q % v.npairs); q /= v.npairs;
                    const I out = static_cast<I>(q % ncu12);
                    const I c = static_cast<I>(q / ncu12);
                    T value = 0;
                    for (I p = v.pairptr[pair]; p < v.pairptr[pair + 1]; ++p) {
                        const T W = v.pwJ[p] * runtime_owner_test(v, shapfgt, npf, i, p);
                        const I qc = ncu + c + ncu * d;
                        value += W *
                            (fluxUdg[p + static_cast<std::size_t>(npts) *
                                (out + ncu12 * qc)] *
                                    v.pphi[j + static_cast<std::size_t>(npe) * p] -
                             fluxUhg[p + static_cast<std::size_t>(npts) *
                                (out + ncu12 * c)] *
                                    v.pd[p + static_cast<std::size_t>(npts) * d] *
                                    v.pphibar[j + static_cast<std::size_t>(npe) * p]);
                    }
                    G[idx] = value;
                });
            Kokkos::parallel_for("NonmatchingGatherPairC",
                static_cast<std::size_t>(v.npairs) * npe * npe,
                KOKKOS_LAMBDA(const std::size_t idx) {
                    const I local = static_cast<I>(idx % (static_cast<std::size_t>(npe) * npe));
                    const I pair = static_cast<I>(idx / (static_cast<std::size_t>(npe) * npe));
                    Cpair[idx] = C[local + static_cast<std::size_t>(npe) * npe *
                        (pairelem[pair] + ne * d)];
                });
            Kokkos::parallel_for("NonmatchingGatherPairE",
                static_cast<std::size_t>(v.npairs) * npe * traceNodes,
                KOKKOS_LAMBDA(const std::size_t idx) {
                    const I local = static_cast<I>(idx %
                        (static_cast<std::size_t>(npe) * traceNodes));
                    const I pair = static_cast<I>(idx /
                        (static_cast<std::size_t>(npe) * traceNodes));
                    Epair[idx] = E[local + static_cast<std::size_t>(npe) * traceNodes *
                        (pairelem[pair] + ne * d)];
                });
            schurMatrixGMinvC(K, G, Cpair, scalar,
                npe, ncu12, ncu, npf, 1, v.npairs);
            schurMatrixGintMinvE(H, G, Epair, scalar,
                npe, ncu12, ncu, npf, nfe, v.npairs);
        }
    }

    const I* pairelem = v.pairelem; T* Fpair = runtime.Fpair.data; T* Rupair = runtime.Rupair.data;
    Kokkos::parallel_for("NonmatchingGatherPairF",
        static_cast<std::size_t>(v.npairs) * n * m, KOKKOS_LAMBDA(const std::size_t idx) {
            const I local = static_cast<I>(idx % (static_cast<std::size_t>(n) * m));
            const I pair = static_cast<I>(idx / (static_cast<std::size_t>(n) * m));
            Fpair[idx] = F[local + static_cast<std::size_t>(n) * m * pairelem[pair]];
        });
    Kokkos::parallel_for("NonmatchingGatherPairRu",
        static_cast<std::size_t>(v.npairs) * n, KOKKOS_LAMBDA(const std::size_t idx) {
            const I local = static_cast<I>(idx % n), pair = static_cast<I>(idx / n);
            Rupair[idx] = Ru[local + static_cast<std::size_t>(n) * pairelem[pair]];
        });
    PGEMNMStridedBached(handle, m12, m, n, one, K, m12, Fpair, n, one,
                        H, m12, v.npairs, backend);
    PGEMNMStridedBached(handle, m12, 1, n, minusone, K, m12, Rupair, n, zero,
                        runtime.pairResidual.data, m12, v.npairs, backend);
    T* Ri = runtime.Ri.data; const T* pairResidual = runtime.pairResidual.data;
    const I* pairSlotPtr = v.pairSlotPtr;
    Kokkos::parallel_for("NonmatchingCondensedSlotResidual",
        static_cast<std::size_t>(v.nslots) * m12, KOKKOS_LAMBDA(const std::size_t idx) {
            const I row = static_cast<I>(idx % m12), slot = static_cast<I>(idx / m12);
            T value = Ri[idx];
            for (I pair = pairSlotPtr[slot]; pair < pairSlotPtr[slot + 1]; ++pair)
                value += pairResidual[row + static_cast<std::size_t>(m12) * pair];
            Ri[idx] = value;
        });
}

template <class T, class I>
inline void nonmatching_matvec_products(NonmatchingRuntime<T, I>& runtime,
        const T* ve, I m, I ne1, cublasHandle_t handle, I backend)
{
    if (runtime.interfacetype != 1) return;
    for (int direction = 0; direction < 2; ++direction) {
        auto& d = runtime.dir[direction];
        const auto v = d.view;
        if (!v.isdonor || v.npairs == 0) continue;
        const I rowNpf = v.rectangular ? v.ownerNpf : v.npf;
        const I donorNpf = v.rectangular ? v.donorNpf : v.npf;
        const I donorNfe = v.rectangular ? v.donorNfe : v.nfe;
        const I rows = runtime_count<I>({v.ncu12, rowNpf}, "matvec rows");
        const I columns = runtime_count<I>(
                {v.ncu, donorNpf, donorNfe}, "matvec columns");
        if (m != columns)
            error("non-matching coupling: element trace size differs from pair matrix columns");
        const I* pairelem = v.pairelem;
        T* vpair = d.vpair.data;
        Kokkos::parallel_for("NonmatchingGatherPairTrace",
            static_cast<std::size_t>(runtime_count<I>(
                {v.npairs, columns}, "matvec pair trace")),
            KOKKOS_LAMBDA(const std::size_t idx) {
                const I local = static_cast<I>(idx % columns), pair = static_cast<I>(idx / columns);
                vpair[idx] = ve[local + static_cast<std::size_t>(columns) * pairelem[pair]];
            });
        PGEMNMStridedBached(handle, rows, 1, columns, one, d.Hi.data, rows,
                            d.vpair.data, columns, zero, d.rpair.data, rows,
                            v.npairs, backend);
        const I* pairSlotPtr = v.pairSlotPtr;
        const T* rpair = d.rpair.data; T* Ri = d.Ri.data;
        Kokkos::parallel_for("NonmatchingReducePairProducts",
            static_cast<std::size_t>(runtime_count<I>(
                {v.nslots, rows}, "matvec slot residual")),
            KOKKOS_LAMBDA(const std::size_t idx) {
                const I row = static_cast<I>(idx % rows), slot = static_cast<I>(idx / rows);
                T value = 0;
                for (I pair = pairSlotPtr[slot]; pair < pairSlotPtr[slot + 1]; ++pair)
                    value += rpair[row + static_cast<std::size_t>(rows) * pair];
                Ri[idx] = value;
            });
    }
    (void) ne1;
}

#ifdef HAVE_MPI
template <class T, class I>
inline void assemble_coupling_slot_owner(T* elementResidual,
        const NonmatchingRuntime<T, I>& runtime, const meshstructT<T, I>& mesh,
        const appstructT<T, I>& app, const commonstructT<T, I>& common)
{
    if (runtime.interfacetype != 1) return;
    const I ncu = common.components.ncu;
    for (int direction = 0; direction < 2; ++direction) {
        const auto& d = runtime.dir[direction];
        const auto v = d.view;
        if (!v.isowner || v.nownfaces == 0) continue;
        const I* fluxmap = app.interfacefluxmap;
        const I* ownerTarget = v.ownerTarget;
        const I* faceBlockPtr = v.faceBlockPtr; const I* faceBlocks = v.faceBlocks;
        const T* received = d.recvbuf.data;
        Kokkos::parallel_for("NonmatchingOwnerAssembly",
            static_cast<std::size_t>(v.nownfaces) * v.npf,
            KOKKOS_LAMBDA(const std::size_t idx) {
                const I i = static_cast<I>(idx % v.npf);
                const I own = static_cast<I>(idx / v.npf);
                // Preserve the accepted block-then-component accumulation
                // order exactly.  One (face,node) writer also makes duplicate
                // interfacefluxmap destinations deterministic and race-free.
                for (I entry = faceBlockPtr[own]; entry < faceBlockPtr[own + 1]; ++entry) {
                    const I block = faceBlocks[entry];
                    for (I c = 0; c < v.ncu12; ++c)
                        elementResidual[ownerTarget[own] + fluxmap[c] +
                            static_cast<std::size_t>(ncu) * i] +=
                            received[c + v.ncu12 *
                                (i + static_cast<std::size_t>(v.npf) * block)];
                }
            });
    }
    (void) mesh;
}
#endif

} // namespace nonmatching
} // namespace exasim

#endif
