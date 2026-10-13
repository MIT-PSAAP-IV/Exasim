/*
    nonmatchingmastercompatibility.hpp

    [nonmatching] Phase 10 compatibility data for mixed triangle/quad models.
    The two models share one 1D trace space and quadrature rule, but retain
    their own volume element family and face interpolation nodes.
*/

#ifndef __NONMATCHINGMASTERCOMPATIBILITY_HPP
#define __NONMATCHINGMASTERCOMPATIBILITY_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace exasim {
namespace nonmatching {

template <class T, class I>
struct FaceMasterDescriptor {
    I elemtype = 0;
    I porder = 0;
    I npf = 0;
    std::vector<T> xpf;
};

template <class T, class I>
struct FaceMasterCompatibility {
    FaceMasterDescriptor<T, I> model[2];
    I mixedFamily = 0;
};

template <class T>
inline bool face_master_value_equal(T a, T b)
{
    const T scale = std::max(static_cast<T>(1), std::max(std::abs(a), std::abs(b)));
    return std::abs(a - b) <= static_cast<T>(64) * std::numeric_limits<T>::epsilon() * scale;
}

#ifdef HAVE_MPI

template <class T, class I, class InterfaceCommType>
inline FaceMasterCompatibility<T, I> check_face_master_compatibility(
        const masterstructT<T, I>& master, I nfe, I model,
        const InterfaceCommType& ic)
{
    constexpr int nmeta = 9;
    const I mine[nmeta] = {
        model, master.ndims[0], master.ndims[1], master.ndims[3],
        master.ndims[4], master.ndims[5], master.ndims[6], nfe,
        master.ndims[8]
    };
    std::vector<I> allMeta(static_cast<std::size_t>(nmeta) * ic.size);
    MPI_Allgather(mine, nmeta, mpi_type<I>(), allMeta.data(), nmeta,
                  mpi_type<I>(), ic.comm);
    ic.collective();

    const I nd = mine[1], porder = mine[3], pgauss = mine[4];
    const I npf = mine[6], ngf = mine[8];
    bool incompatible = (model != 0 && model != 1) || npf < 1 || ngf < 1;
    int representative[2] = {-1, -1};
    for (int q = 0; q < ic.size; ++q) {
        const I* m = &allMeta[static_cast<std::size_t>(nmeta) * q];
        if (m[0] != 0 && m[0] != 1) {
            incompatible = true;
            continue;
        }
        if (representative[m[0]] < 0) representative[m[0]] = q;
        if (m[1] != nd || m[3] != porder || m[4] != pgauss ||
            m[6] != npf || m[8] != ngf)
            incompatible = true;
    }
    if (representative[0] < 0 || representative[1] < 0)
        incompatible = true;

    // Within a model, all interface ranks must use one volume family/master.
    for (int q = 0; q < ic.size; ++q) {
        const I* m = &allMeta[static_cast<std::size_t>(nmeta) * q];
        if (m[0] != 0 && m[0] != 1) continue;
        const I* r = &allMeta[static_cast<std::size_t>(nmeta) * representative[m[0]]];
        if (m[2] != r[2] || m[5] != r[5] || m[7] != r[7])
            incompatible = true;
    }

    if (incompatible)
        error("non-matching interface: incompatible face spaces; both models must use the same "
              "nd, porder, pgauss, npf, ngf, gpf and gwf (elemtype, npe, nfe and xpf may differ)");

    const int nreal = static_cast<int>(npf + 2 * ngf);
    std::vector<T> mineReal(static_cast<std::size_t>(nreal));
    for (I j = 0; j < npf; ++j) mineReal[j] = master.xpf[j];
    for (I g = 0; g < ngf; ++g) {
        mineReal[npf + g] = master.gpf[g];
        mineReal[npf + ngf + g] = master.gwf[g];
    }
    std::vector<T> allReal(static_cast<std::size_t>(nreal) * ic.size);
    MPI_Allgather(mineReal.data(), nreal, mpi_type<T>(), allReal.data(), nreal,
                  mpi_type<T>(), ic.comm);
    ic.collective();

    for (int q = 0; q < ic.size; ++q) {
        const I qmodel = allMeta[static_cast<std::size_t>(nmeta) * q];
        const int r = representative[qmodel];
        const T* a = &allReal[static_cast<std::size_t>(nreal) * q];
        const T* b = &allReal[static_cast<std::size_t>(nreal) * r];
        for (I j = 0; j < npf; ++j)
            if (!face_master_value_equal(a[j], b[j])) incompatible = true;
        for (I g = 0; g < ngf; ++g)
            if (!face_master_value_equal(a[npf + g], b[npf + g]) ||
                !face_master_value_equal(a[npf + ngf + g], b[npf + ngf + g]))
                incompatible = true;
    }
    const T* rule0 = &allReal[static_cast<std::size_t>(nreal) * representative[0]];
    const T* rule1 = &allReal[static_cast<std::size_t>(nreal) * representative[1]];
    for (I g = 0; g < ngf; ++g) {
        if (!face_master_value_equal(rule0[npf + g], rule1[npf + g]) ||
            !face_master_value_equal(rule0[npf + ngf + g], rule1[npf + ngf + g]))
            incompatible = true;
    }
    if (incompatible)
        error("non-matching interface: incompatible face spaces; both models must use the same "
              "nd, porder, pgauss, npf, ngf, gpf and gwf (elemtype, npe, nfe and xpf may differ)");

    FaceMasterCompatibility<T, I> out;
    for (int k = 0; k < 2; ++k) {
        const int r = representative[k];
        const I* m = &allMeta[static_cast<std::size_t>(nmeta) * r];
        const T* values = &allReal[static_cast<std::size_t>(nreal) * r];
        out.model[k].elemtype = m[2];
        out.model[k].porder = m[3];
        out.model[k].npf = m[6];
        out.model[k].xpf.assign(values, values + npf);
    }
    out.mixedFamily = out.model[0].elemtype != out.model[1].elemtype ? 1 : 0;
    return out;
}

#endif

} // namespace nonmatching
} // namespace exasim

#endif
