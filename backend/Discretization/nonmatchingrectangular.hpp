/*
    Rectangular mixed-topology kernels for non-matching 3D coupling.

    Owner test rows and donor trace/volume columns have independent sizes.
    These kernels implement the IndependentFaceSpaces contract and leave the
    optimized SharedFaceSpace kernels unchanged.
*/

#ifndef __NONMATCHINGRECTANGULAR_HPP
#define __NONMATCHINGRECTANGULAR_HPP

#include "nonmatchingruntime.hpp"

namespace exasim {
namespace nonmatching {

template <class T, class I>
inline void form_rectangular_slot_residual(NonmatchingRuntimeDirection<T, I>& runtime)
{
    const auto v = runtime.view;
    const I ownerNpf = v.contract.ownerFaceNodes;
    const I rows = runtime_count<I>({v.ncu12, ownerNpf}, "rectangular residual rows");
    const std::size_t ro = static_cast<std::size_t>(rows);
    const T* flux = runtime.flux.data;
    T* Ri = runtime.Ri.data;
    Kokkos::parallel_for("NonmatchingRectangularSlotResidual",
        Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0,
            static_cast<std::size_t>(runtime_count<I>(
                {v.nslots, rows}, "rectangular residual"))),
        KOKKOS_LAMBDA(const std::size_t idx) {
            const I row = static_cast<I>(idx % ro);
            const I slot = static_cast<I>(idx / ro);
            const I out = row % v.ncu12;
            const I i = row / v.ncu12;
            T value = 0;
            for (I p = v.slotptr[slot]; p < v.slotptr[slot + 1]; ++p)
                value -= v.pwJ[p] *
                         v.ptest[i + static_cast<std::size_t>(ownerNpf) * p] *
                         flux[p + static_cast<std::size_t>(v.npts) * out];
            Ri[idx] = value;
        });
}

template <class Ty>
inline void schurMatrixGintMinvERect(Ty* H, const Ty* G, const Ty* MinvE,
        noDeduce_t<Ty> scalar, std::size_t npe, std::size_t ncu12,
        std::size_t ncu, std::size_t ownerNpf, std::size_t donorNpf,
        std::size_t donorNfe, std::size_t npairs, std::size_t total)
{
    const std::size_t ndf = donorNpf * donorNfe;
    const std::size_t Q = npe * ndf;
    const std::size_t M = ownerNpf * npe;
    const std::size_t L = M * npairs;
    const std::size_t P = L * ncu12;
    Kokkos::parallel_for("schurMatrixGintMinvERect",
        Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, total),
        KOKKOS_LAMBDA(const std::size_t idx) {
            std::size_t t = idx;
            const std::size_t m = t % ncu12; t /= ncu12;
            const std::size_t i = t % ownerNpf; t /= ownerNpf;
            const std::size_t n = t % ncu; t /= ncu;
            const std::size_t j = t % ndf; t /= ndf;
            const std::size_t e = t;
            Ty sum = 0;
            for (std::size_t k = 0; k < npe; ++k)
                sum += scalar * G[i + ownerNpf*k + M*e + L*m + P*n] *
                       MinvE[k + npe*j + Q*e];
            H[idx] -= sum;
        });
}

template <class T, class I>
inline void form_rectangular_pairs(NonmatchingRuntimeDirection<T, I>& runtime,
        const resstructT<T, I>& res, const commonstructT<T, I>& common,
        cublasHandle_t handle, I backend)
{
    const auto v = runtime.view;
    const I npe = v.contract.donorVolumeNodes;
    const I npf = v.contract.donorFaceNodes;
    const I nfe = v.contract.donorFacesPerElement;
    const I ownerNpf = v.contract.ownerFaceNodes;
    const I ncu = v.ncu, ncu12 = v.ncu12;
    const I ncq = v.ncq, nd = v.nd, npts = v.npts;
    const I necols = runtime_count<I>({npe, ncu}, "rectangular volume columns");
    const I traceNodes = runtime_count<I>({npf, nfe}, "rectangular trace nodes");
    const I cd = runtime_count<I>({traceNodes, ncu}, "rectangular trace columns");
    const I ro = runtime_count<I>({ownerNpf, ncu12}, "rectangular rows");
    const I ne = common.meshsizes.ne;
    T* K = runtime.K.data;
    T* H = runtime.Hi.data;
    const T* fluxUdg = runtime.fluxUdg.data;
    const T* fluxUhg = runtime.fluxUhg.data;
    const T* C = res.C;
    const T* E = res.E;
    const T* F = res.F;
    const T* Ru = res.Ru;
    const T scalar = common.timeparams.wave == 1
            ? static_cast<T>(1) / common.timestate.dtfactor
            : static_cast<T>(1);

    Kokkos::parallel_for("NonmatchingRectangularPairK",
        Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0,
            static_cast<std::size_t>(runtime_count<I>(
                {v.npairs, ro, necols}, "rectangular K"))),
        KOKKOS_LAMBDA(const std::size_t idx) {
            std::size_t t = idx;
            const I row = static_cast<I>(t % ro); t /= ro;
            const I col = static_cast<I>(t % necols); t /= necols;
            const I pair = static_cast<I>(t);
            const I out = row % ncu12, i = row / ncu12;
            const I j = col % npe, c = col / npe;
            T value = 0;
            for (I p = v.pairptr[pair]; p < v.pairptr[pair + 1]; ++p) {
                const T W = v.pwJ[p] *
                        v.ptest[i + static_cast<std::size_t>(ownerNpf) * p];
                T derivative = fluxUdg[p + static_cast<std::size_t>(npts) *
                        (out + ncu12*c)] *
                        v.pphi[j + static_cast<std::size_t>(npe) * p];
                if (ncq == 0)
                    for (I d = 0; d < nd; ++d)
                        derivative += fluxUhg[p + static_cast<std::size_t>(npts) *
                                (out + ncu12*c)] *
                                v.pd[p + static_cast<std::size_t>(npts) * d] *
                                v.pdphibar[j + static_cast<std::size_t>(npe) *
                                    (d + static_cast<std::size_t>(nd) * p)];
                value += W * derivative;
            }
            K[idx] = value;
        });

    Kokkos::parallel_for("NonmatchingRectangularPairH",
        Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0,
            static_cast<std::size_t>(runtime_count<I>(
                {v.npairs, ro, cd}, "rectangular H"))),
        KOKKOS_LAMBDA(const std::size_t idx) {
            std::size_t t = idx;
            const I row = static_cast<I>(t % ro); t /= ro;
            const I col = static_cast<I>(t % cd); t /= cd;
            const I pair = static_cast<I>(t);
            const I out = row % ncu12, i = row / ncu12;
            const I c = col % ncu;
            const I faceNode = col / ncu;
            const I j = faceNode % npf, a = faceNode / npf;
            T value = 0;
            for (I p = v.pairptr[pair]; p < v.pairptr[pair + 1]; ++p)
                if (v.pa[p] == a)
                    value += v.pwJ[p] *
                        v.ptest[i + static_cast<std::size_t>(ownerNpf) * p] *
                        fluxUhg[p + static_cast<std::size_t>(npts) *
                            (out + ncu12*c)] *
                        v.ppsi[j + static_cast<std::size_t>(npf) * p];
            H[idx] = value;
        });

    if (ncq > 0) {
        T* G = runtime.G.data;
        T* Cpair = runtime.Cpair.data;
        T* Epair = runtime.Epair.data;
        const I* pairelem = v.pairelem;
        const std::size_t gSize = static_cast<std::size_t>(runtime_count<I>(
                {ownerNpf, npe, v.npairs, ncu12, ncu}, "rectangular G"));
        for (I d = 0; d < nd; ++d) {
            Kokkos::parallel_for("NonmatchingRectangularPairG",
                Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, gSize),
                KOKKOS_LAMBDA(const std::size_t idx) {
                    std::size_t t = idx;
                    const I i = static_cast<I>(t % ownerNpf); t /= ownerNpf;
                    const I j = static_cast<I>(t % npe); t /= npe;
                    const I pair = static_cast<I>(t % v.npairs); t /= v.npairs;
                    const I out = static_cast<I>(t % ncu12);
                    const I c = static_cast<I>(t / ncu12);
                    T value = 0;
                    for (I p = v.pairptr[pair]; p < v.pairptr[pair + 1]; ++p) {
                        const T W = v.pwJ[p] *
                            v.ptest[i + static_cast<std::size_t>(ownerNpf) * p];
                        const I qc = ncu + c + ncu*d;
                        value += W *
                            (fluxUdg[p + static_cast<std::size_t>(npts) *
                                (out + ncu12*qc)] *
                                    v.pphi[j + static_cast<std::size_t>(npe) * p] -
                             fluxUhg[p + static_cast<std::size_t>(npts) *
                                (out + ncu12*c)] *
                                    v.pd[p + static_cast<std::size_t>(npts) * d] *
                                    v.pphibar[j + static_cast<std::size_t>(npe) * p]);
                    }
                    G[idx] = value;
                });
            Kokkos::parallel_for("NonmatchingRectangularGatherPairC",
                Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0,
                    static_cast<std::size_t>(runtime_count<I>(
                        {v.npairs, npe, npe}, "rectangular Cpair"))),
                KOKKOS_LAMBDA(const std::size_t idx) {
                    const I local = static_cast<I>(idx %
                            (static_cast<std::size_t>(npe) * npe));
                    const I pair = static_cast<I>(idx /
                            (static_cast<std::size_t>(npe) * npe));
                    Cpair[idx] = C[local + static_cast<std::size_t>(npe) * npe *
                            (pairelem[pair] + ne*d)];
                });
            Kokkos::parallel_for("NonmatchingRectangularGatherPairE",
                Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0,
                    static_cast<std::size_t>(runtime_count<I>(
                        {v.npairs, npe, traceNodes}, "rectangular Epair"))),
                KOKKOS_LAMBDA(const std::size_t idx) {
                    const I local = static_cast<I>(idx %
                            (static_cast<std::size_t>(npe) * traceNodes));
                    const I pair = static_cast<I>(idx /
                            (static_cast<std::size_t>(npe) * traceNodes));
                    Epair[idx] = E[local + static_cast<std::size_t>(npe) * traceNodes *
                            (pairelem[pair] + ne*d)];
                });
            schurMatrixGMinvC(K, G, Cpair, scalar,
                    npe, ncu12, ncu, ownerNpf, 1, v.npairs);
            const std::size_t hTotal = static_cast<std::size_t>(runtime_count<I>(
                    {ncu12, ownerNpf, ncu, traceNodes, v.npairs},
                    "rectangular G-E product"));
            schurMatrixGintMinvERect(H, G, Epair, scalar,
                    static_cast<std::size_t>(npe), static_cast<std::size_t>(ncu12),
                    static_cast<std::size_t>(ncu), static_cast<std::size_t>(ownerNpf),
                    static_cast<std::size_t>(npf), static_cast<std::size_t>(nfe),
                    static_cast<std::size_t>(v.npairs), hTotal);
        }
    }

    const I* pairelem = v.pairelem;
    T* Fpair = runtime.Fpair.data;
    T* Rupair = runtime.Rupair.data;
    Kokkos::parallel_for("NonmatchingRectangularGatherPairF",
        Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0,
            static_cast<std::size_t>(runtime_count<I>(
                {v.npairs, necols, cd}, "rectangular Fpair"))),
        KOKKOS_LAMBDA(const std::size_t idx) {
            const I local = static_cast<I>(idx %
                    (static_cast<std::size_t>(necols) * cd));
            const I pair = static_cast<I>(idx /
                    (static_cast<std::size_t>(necols) * cd));
            Fpair[idx] = F[local + static_cast<std::size_t>(necols) * cd *
                    pairelem[pair]];
        });
    Kokkos::parallel_for("NonmatchingRectangularGatherPairRu",
        Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0,
            static_cast<std::size_t>(runtime_count<I>(
                {v.npairs, necols}, "rectangular Rupair"))),
        KOKKOS_LAMBDA(const std::size_t idx) {
            const I local = static_cast<I>(idx % necols);
            const I pair = static_cast<I>(idx / necols);
            Rupair[idx] = Ru[local + static_cast<std::size_t>(necols) *
                    pairelem[pair]];
        });
    PGEMNMStridedBached(handle, ro, cd, necols, one, K, ro, Fpair, necols,
                        one, H, ro, v.npairs, backend);
    PGEMNMStridedBached(handle, ro, 1, necols, minusone, K, ro, Rupair,
                        necols, zero, runtime.pairResidual.data, ro,
                        v.npairs, backend);
    const I* pairSlotPtr = v.pairSlotPtr;
    const T* pairResidual = runtime.pairResidual.data;
    T* Ri = runtime.Ri.data;
    Kokkos::parallel_for("NonmatchingRectangularCondensedSlotResidual",
        Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0,
            static_cast<std::size_t>(runtime_count<I>(
                {v.nslots, ro}, "rectangular condensed residual"))),
        KOKKOS_LAMBDA(const std::size_t idx) {
            const I row = static_cast<I>(idx % ro);
            const I slot = static_cast<I>(idx / ro);
            T value = Ri[idx];
            for (I pair = pairSlotPtr[slot]; pair < pairSlotPtr[slot + 1]; ++pair)
                value += pairResidual[row + static_cast<std::size_t>(ro) * pair];
            Ri[idx] = value;
        });
}

} // namespace nonmatching
} // namespace exasim

#endif
