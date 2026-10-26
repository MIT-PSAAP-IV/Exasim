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
#include <climits>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace exasim {
namespace nonmatching {

template <class T, class I>
struct FaceMasterDescriptor {
    I elemtype = 0;
    I porder = 0;
    I pgauss = 0;
    I npe = 0;
    I npf = 0;
    I nfe = 0;
    I ngf = 0;
    std::vector<T> xpf;
};

template <class T, class I>
struct FaceMasterCompatibility {
    FaceMasterDescriptor<T, I> model[2];
    I mixedFamily = 0;
    I mixedTopology3D = 0;
};

template <class T>
inline bool face_master_value_equal(T a, T b)
{
    const T scale = std::max(static_cast<T>(1), std::max(std::abs(a), std::abs(b)));
    return std::abs(a - b) <= static_cast<T>(64) * std::numeric_limits<T>::epsilon() * scale;
}

#ifdef HAVE_MPI

inline std::runtime_error mixed_face_space_error(const std::string& reason)
{
    return std::runtime_error(
            "non-matching interface: incompatible face spaces: " + reason);
}

template <class T, class I>
inline void validate_mixed_face_rule(const T* values, I elemtype, I npf,
        I ngf, I pgauss)
{
    const I ndf = 2;
    const T* gpf = values + static_cast<std::size_t>(ndf) * npf;
    const T* gwf = gpf + static_cast<std::size_t>(ndf) * ngf;
    const T eps = std::numeric_limits<T>::epsilon();
    const T tau = static_cast<T>(64) * eps;
    const T area = elemtype == 0 ? static_cast<T>(0.5) : static_cast<T>(1);
    T weightSum = 0;
    for (I g = 0; g < ngf; ++g) {
        if (!(gwf[g] > static_cast<T>(0)))
            throw mixed_face_space_error(
                    "mixed tet/hex face rule has a nonpositive weight");
        weightSum += gwf[g];
    }
    for (I g = 0; g < ngf; ++g) {
        const T x = gpf[g];
        const T y = gpf[g + ngf];
        const bool inside = elemtype == 0
                ? x >= -tau && y >= -tau && x + y <= static_cast<T>(1) + tau
                : x >= -tau && x <= static_cast<T>(1) + tau &&
                  y >= -tau && y <= static_cast<T>(1) + tau;
        if (!inside)
            throw mixed_face_space_error(
                    "mixed tet/hex face rule has a node outside the reference face");
    }
    const T tolerance = static_cast<T>(64) * eps * static_cast<T>(ngf) * area;
    if (std::abs(weightSum - area) > tolerance)
        throw mixed_face_space_error(
                "mixed tet/hex face rule fails the reference measure");

    for (I total = 0; total <= pgauss; ++total)
        for (I a = 0; a <= total; ++a) {
            const I b = total - a;
            T numerical = 0;
            for (I g = 0; g < ngf; ++g)
                numerical += gwf[g] * std::pow(gpf[g], a) *
                             std::pow(gpf[g + ngf], b);
            long double exact = 0;
            if (elemtype == 0) {
                long double numerator = 1, denominator = 1;
                for (I k = 2; k <= a; ++k) numerator *= k;
                for (I k = 2; k <= b; ++k) numerator *= k;
                for (I k = 2; k <= a + b + 2; ++k) denominator *= k;
                exact = numerator / denominator;
            }
            else {
                exact = static_cast<long double>(1) /
                        (static_cast<long double>(a + 1) *
                         static_cast<long double>(b + 1));
            }
            if (std::abs(numerical - static_cast<T>(exact)) > tolerance)
                throw mixed_face_space_error(
                        "mixed tet/hex face rule fails monomial exactness");
        }
}

template <class T, class I, class InterfaceCommType>
inline FaceMasterCompatibility<T, I> check_mixed3d_face_master_compatibility(
        const masterstructT<T, I>& master, I model, I ncu12,
        const InterfaceCommType& ic, const std::vector<I>& allMeta,
        const int representative[2])
{
    constexpr int nmeta = 9;
    // Keep compatibility diagnostics deterministic across ranks: validate
    // polynomial orders first, then rule structure, topology and components.
    for (int k = 0; k < 2; ++k)
        if (representative[k] < 0)
            throw mixed_face_space_error("mixed tet/hex pair requires both models");
    for (int q = 0; q < ic.size; ++q) {
        const I family = allMeta[static_cast<std::size_t>(nmeta) * q + 2];
        if (family != 0 && family != 1)
            throw mixed_face_space_error("unsupported element family");
    }
    if (allMeta[static_cast<std::size_t>(nmeta) * representative[0] + 2] ==
        allMeta[static_cast<std::size_t>(nmeta) * representative[1] + 2])
        throw mixed_face_space_error("mixed tet/hex pair requires different element families");

    for (int q = 0; q < ic.size; ++q) {
        const I* m = &allMeta[static_cast<std::size_t>(nmeta) * q];
        if (m[0] != 0 && m[0] != 1)
            throw mixed_face_space_error("mixed tet/hex pair requires both models");
        const I* r = &allMeta[static_cast<std::size_t>(nmeta) * representative[m[0]]];
        for (int j = 1; j < nmeta; ++j)
            if (m[j] != r[j]) {
                std::ostringstream message;
                message << "inconsistent face master metadata within model " << m[0];
                throw mixed_face_space_error(message.str());
            }
    }

    const I* m0 = &allMeta[static_cast<std::size_t>(nmeta) * representative[0]];
    const I* m1 = &allMeta[static_cast<std::size_t>(nmeta) * representative[1]];
    if (m0[3] != m1[3])
        throw mixed_face_space_error("mixed tet/hex pair requires equal porder");
    if (m0[3] < 1 || m0[3] > 4)
        throw mixed_face_space_error("mixed tet/hex pair supports porder from 1 through 4");
    if (m0[4] != m1[4])
        throw mixed_face_space_error("mixed tet/hex pair requires equal pgauss");
    for (int k = 0; k < 2; ++k) {
        const I* m = &allMeta[static_cast<std::size_t>(nmeta) * representative[k]];
        if (m[4] != 2 * m[3]) {
            std::ostringstream message;
            message << "mixed tet/hex pair requires pgauss = 2*porder; unsupported pgauss = "
                    << m[4];
            throw mixed_face_space_error(message.str());
        }
    }

    if (ncu12 >= 0) {
        std::vector<I> allNcu12(static_cast<std::size_t>(ic.size));
        MPI_Allgather(&ncu12, 1, mpi_type<I>(), allNcu12.data(), 1, mpi_type<I>(), ic.comm);
        ic.collective();
        I representativeNcu12[2] = {-1, -1};
        bool representativeNcu12Set[2] = {false, false};
        for (int q = 0; q < ic.size; ++q) {
            const I qmodel = allMeta[static_cast<std::size_t>(nmeta) * q];
            if (!representativeNcu12Set[qmodel]) {
                representativeNcu12[qmodel] = allNcu12[q];
                representativeNcu12Set[qmodel] = true;
            }
            if (allNcu12[q] != representativeNcu12[qmodel]) {
                std::ostringstream message;
                message << "inconsistent ncu12 within model " << qmodel;
                throw mixed_face_space_error(message.str());
            }
        }
        if (!representativeNcu12Set[0] || !representativeNcu12Set[1] ||
                representativeNcu12[0] != representativeNcu12[1])
            throw mixed_face_space_error("mixed tet/hex pair requires equal ncu12");
    }
    for (int k = 0; k < 2; ++k) {
        const I* m = &allMeta[static_cast<std::size_t>(nmeta) * representative[k]];
        const I expectedNpf = m[2] == 0 ? (m[3] + 1) * (m[3] + 2) / 2
                                        : (m[3] + 1) * (m[3] + 1);
        const I expectedNfe = m[2] == 0 ? 4 : 6;
        if (m[6] != expectedNpf || m[7] != expectedNfe)
            throw mixed_face_space_error("mixed tet/hex family dimensions are inconsistent");
    }
    for (int k = 0; k < 2; ++k) {
        const I* m = &allMeta[static_cast<std::size_t>(nmeta) * representative[k]];
        if (m[8] < 1)
            throw mixed_face_space_error("mixed tet/hex face rule has no quadrature points");
    }

    std::vector<int> counts(static_cast<std::size_t>(ic.size));
    std::vector<int> displacements(static_cast<std::size_t>(ic.size));
    std::size_t total = 0;
    for (int q = 0; q < ic.size; ++q) {
        const I* m = &allMeta[static_cast<std::size_t>(nmeta) * q];
        const std::size_t count = static_cast<std::size_t>(2) * m[6] +
                                  static_cast<std::size_t>(3) * m[8];
        if (count > static_cast<std::size_t>(INT_MAX) ||
                total > static_cast<std::size_t>(INT_MAX) - count)
            throw mixed_face_space_error("mixed tet/hex master exchange exceeds MPI int range");
        counts[q] = static_cast<int>(count);
        displacements[q] = static_cast<int>(total);
        total += count;
    }
    const int localCount = counts[ic.rank];
    std::vector<T> mineReal(static_cast<std::size_t>(localCount));
    const I localNpf = master.ndims[6], localNgf = master.ndims[8];
    for (I j = 0; j < 2 * localNpf; ++j) mineReal[j] = master.xpf[j];
    for (I g = 0; g < 2 * localNgf; ++g)
        mineReal[2 * localNpf + g] = master.gpf[g];
    for (I g = 0; g < localNgf; ++g)
        mineReal[2 * localNpf + 2 * localNgf + g] = master.gwf[g];
    std::vector<T> allReal(total);
    MPI_Allgatherv(mineReal.data(), localCount, mpi_type<T>(), allReal.data(),
            counts.data(), displacements.data(), mpi_type<T>(), ic.comm);
    ic.collective();

    for (int q = 0; q < ic.size; ++q) {
        const I qmodel = allMeta[static_cast<std::size_t>(nmeta) * q];
        const int r = representative[qmodel];
        const T* a = allReal.data() + displacements[q];
        const T* b = allReal.data() + displacements[r];
        for (int j = 0; j < counts[q]; ++j)
            if (!face_master_value_equal(a[j], b[j])) {
                std::ostringstream message;
                message << "inconsistent face master within model " << qmodel;
                throw mixed_face_space_error(message.str());
            }
    }
    for (int k = 0; k < 2; ++k) {
        const int r = representative[k];
        const I* m = &allMeta[static_cast<std::size_t>(nmeta) * r];
        validate_mixed_face_rule<T, I>(allReal.data() + displacements[r],
                m[2], m[6], m[8], m[4]);
    }

    FaceMasterCompatibility<T, I> out;
    for (int k = 0; k < 2; ++k) {
        const int r = representative[k];
        const I* m = &allMeta[static_cast<std::size_t>(nmeta) * r];
        const T* values = allReal.data() + displacements[r];
        out.model[k].elemtype = m[2];
        out.model[k].porder = m[3];
        out.model[k].pgauss = m[4];
        out.model[k].npe = m[5];
        out.model[k].npf = m[6];
        out.model[k].nfe = m[7];
        out.model[k].ngf = m[8];
        out.model[k].xpf.assign(values, values + static_cast<std::size_t>(2) * m[6]);
    }
    out.mixedFamily = 1;
    out.mixedTopology3D = 1;
    return out;
}

template <class T, class I, class InterfaceCommType>
inline FaceMasterCompatibility<T, I> check_face_master_compatibility(
        const masterstructT<T, I>& master, I nfe, I model,
        const InterfaceCommType& ic, I receivedWidth = static_cast<I>(-1),
        I donatedWidth = static_cast<I>(-1))
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

    if (donatedWidth >= 0) {
        const I mineWidths[2] = {receivedWidth, donatedWidth};
        std::vector<I> allWidths(static_cast<std::size_t>(2) * ic.size);
        MPI_Allgather(mineWidths, 2, mpi_type<I>(), allWidths.data(), 2,
                      mpi_type<I>(), ic.comm);
        ic.collective();
        I widths[2][2] = {{-1, -1}, {-1, -1}};
        bool haveWidths[2] = {false, false};
        for (int q = 0; q < ic.size; ++q) {
            const I qmodel = allMeta[static_cast<std::size_t>(nmeta) * q];
            if (qmodel < 0 || qmodel > 1)
                throw mixed_face_space_error("non-matching interface: invalid endpoint side");
            const I* qwidth = &allWidths[static_cast<std::size_t>(2) * q];
            if (!haveWidths[qmodel]) {
                widths[qmodel][0] = qwidth[0]; widths[qmodel][1] = qwidth[1];
                haveWidths[qmodel] = true;
            }
            else if (widths[qmodel][0] != qwidth[0] || widths[qmodel][1] != qwidth[1])
                throw mixed_face_space_error(
                        "non-matching interface: inconsistent directional widths within side");
        }
        if (!haveWidths[0] || !haveWidths[1] ||
                widths[0][0] != widths[1][1] || widths[1][0] != widths[0][1])
            throw mixed_face_space_error(
                    "non-matching interface: donor and owner directional widths differ");
    }

    int mixedRepresentative[2] = {-1, -1};
    bool candidateMixed3D = true;
    for (int q = 0; q < ic.size; ++q) {
        const I* m = &allMeta[static_cast<std::size_t>(nmeta) * q];
        if (m[0] != 0 && m[0] != 1) candidateMixed3D = false;
        else if (mixedRepresentative[m[0]] < 0) mixedRepresentative[m[0]] = q;
        if (m[1] != 3) candidateMixed3D = false;
    }
    if (candidateMixed3D && mixedRepresentative[0] >= 0 && mixedRepresentative[1] >= 0 &&
            allMeta[static_cast<std::size_t>(nmeta) * mixedRepresentative[0] + 2] !=
            allMeta[static_cast<std::size_t>(nmeta) * mixedRepresentative[1] + 2])
        return check_mixed3d_face_master_compatibility<T, I>(
                master, model, donatedWidth >= 0 ? static_cast<I>(-1) : receivedWidth,
                ic, allMeta, mixedRepresentative);

    const I nd = mine[1], ndf = nd - 1, porder = mine[3], pgauss = mine[4];
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

    const int nreal = static_cast<int>(ndf * npf + ndf * ngf + ngf);
    std::vector<T> mineReal(static_cast<std::size_t>(nreal));
    for (I j = 0; j < ndf * npf; ++j) mineReal[j] = master.xpf[j];
    for (I g = 0; g < ndf * ngf; ++g) mineReal[ndf * npf + g] = master.gpf[g];
    for (I g = 0; g < ngf; ++g)
        mineReal[ndf * (npf + ngf) + g] = master.gwf[g];
    std::vector<T> allReal(static_cast<std::size_t>(nreal) * ic.size);
    MPI_Allgather(mineReal.data(), nreal, mpi_type<T>(), allReal.data(), nreal,
                  mpi_type<T>(), ic.comm);
    ic.collective();

    for (int q = 0; q < ic.size; ++q) {
        const I qmodel = allMeta[static_cast<std::size_t>(nmeta) * q];
        const int r = representative[qmodel];
        const T* a = &allReal[static_cast<std::size_t>(nreal) * q];
        const T* b = &allReal[static_cast<std::size_t>(nreal) * r];
        for (I j = 0; j < ndf * npf; ++j)
            if (!face_master_value_equal(a[j], b[j])) incompatible = true;
        for (I g = 0; g < ndf * ngf; ++g)
            if (!face_master_value_equal(a[ndf * npf + g], b[ndf * npf + g]))
                incompatible = true;
        for (I g = 0; g < ngf; ++g)
            if (!face_master_value_equal(
                    a[ndf * (npf + ngf) + g], b[ndf * (npf + ngf) + g]))
                incompatible = true;
    }
    const T* rule0 = &allReal[static_cast<std::size_t>(nreal) * representative[0]];
    const T* rule1 = &allReal[static_cast<std::size_t>(nreal) * representative[1]];
    for (I g = 0; g < ndf * ngf; ++g)
        if (!face_master_value_equal(rule0[ndf * npf + g], rule1[ndf * npf + g]))
            incompatible = true;
    for (I g = 0; g < ngf; ++g)
        if (!face_master_value_equal(
                rule0[ndf * (npf + ngf) + g], rule1[ndf * (npf + ngf) + g]))
            incompatible = true;
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
        out.model[k].pgauss = m[4];
        out.model[k].npe = m[5];
        out.model[k].npf = m[6];
        out.model[k].nfe = m[7];
        out.model[k].ngf = m[8];
        out.model[k].xpf.assign(values, values + ndf * npf);
    }
    out.mixedFamily = out.model[0].elemtype != out.model[1].elemtype ? 1 : 0;
    return out;
}

#endif

} // namespace nonmatching
} // namespace exasim

#endif
