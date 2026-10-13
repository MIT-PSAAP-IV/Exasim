#pragma once

#include "basis.hpp"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>

namespace exasim::meshtransfer::detail {

template <class Scalar, class Index>
KOKKOS_INLINE_FUNCTION
bool solve_small_system(Scalar* x, const Scalar* a, const Scalar* b, Index nd)
{
    Scalar scale = Scalar(0);
    for (Index i = 0; i < nd * nd; ++i)
        scale = fmax(scale, fabs(a[i]));
    const Scalar tolerance =
        Scalar(64) * std::numeric_limits<Scalar>::epsilon() * fmax(Scalar(1), scale);
    if (nd == 1) {
        if (fabs(a[0]) <= tolerance) return false;
        x[0] = b[0] / a[0];
        return true;
    }
    if (nd == 2) {
        const Scalar det = a[0] * a[3] - a[1] * a[2];
        if (fabs(det) <= tolerance * fmax(Scalar(1), scale)) return false;
        x[0] = (a[3] * b[0] - a[1] * b[1]) / det;
        x[1] = (-a[2] * b[0] + a[0] * b[1]) / det;
        return true;
    }
    const Scalar det =
        a[0] * (a[4] * a[8] - a[5] * a[7]) -
        a[1] * (a[3] * a[8] - a[5] * a[6]) +
        a[2] * (a[3] * a[7] - a[4] * a[6]);
    if (fabs(det) <= tolerance * fmax(Scalar(1), scale * scale)) return false;
    const Scalar inverse[9] = {
        (a[4] * a[8] - a[5] * a[7]) / det,
        -(a[1] * a[8] - a[2] * a[7]) / det,
        (a[1] * a[5] - a[2] * a[4]) / det,
        -(a[3] * a[8] - a[5] * a[6]) / det,
        (a[0] * a[8] - a[2] * a[6]) / det,
        -(a[0] * a[5] - a[2] * a[3]) / det,
        (a[3] * a[7] - a[4] * a[6]) / det,
        -(a[0] * a[7] - a[1] * a[6]) / det,
        (a[0] * a[4] - a[1] * a[3]) / det
    };
    for (Index i = 0; i < 3; ++i) {
        x[i] = Scalar(0);
        for (Index j = 0; j < 3; ++j)
            x[i] += inverse[i * 3 + j] * b[j];
    }
    return true;
}

template <class Scalar, class Index>
KOKKOS_INLINE_FUNCTION
bool inside_reference(const Scalar* xi, Index elemtype, Index nd, Scalar tolerance)
{
    if (elemtype == 0) {
        Scalar sum = 0;
        for (Index d = 0; d < nd; ++d) {
            if (xi[d] < -tolerance) return false;
            sum += xi[d];
        }
        return sum <= Scalar(1) + tolerance;
    }
    for (Index d = 0; d < nd; ++d)
        if (xi[d] < -tolerance || xi[d] > Scalar(1) + tolerance)
            return false;
    return true;
}

template <class Scalar, class Index>
KOKKOS_INLINE_FUNCTION
void select_exit_face(
    Scalar violation, Index candidate, Scalar& largest, Index& face)
{
    if (violation > largest) {
        largest = violation;
        face = candidate;
    }
}

template <class Scalar, class Index>
KOKKOS_INLINE_FUNCTION
Index exit_face(const Scalar* xi, Index elemtype, Index nd)
{
    Scalar largest = Scalar(-1);
    Index face = -1;
    if (elemtype == 0) {
        Scalar sum = 0;
        for (Index d = 0; d < nd; ++d) sum += xi[d];
        select_exit_face(sum - Scalar(1), Index(0), largest, face);
        for (Index d = 0; d < nd; ++d)
            select_exit_face(-xi[d], d + 1, largest, face);
        return face;
    }
    if (nd == 1) {
        select_exit_face(-xi[0], Index(0), largest, face);
        select_exit_face(xi[0] - Scalar(1), Index(1), largest, face);
    }
    else if (nd == 2) {
        select_exit_face(-xi[1], Index(0), largest, face);
        select_exit_face(xi[0] - Scalar(1), Index(1), largest, face);
        select_exit_face(xi[1] - Scalar(1), Index(2), largest, face);
        select_exit_face(-xi[0], Index(3), largest, face);
    }
    else {
        select_exit_face(-xi[2], Index(0), largest, face);
        select_exit_face(xi[2] - Scalar(1), Index(1), largest, face);
        select_exit_face(-xi[1], Index(2), largest, face);
        select_exit_face(xi[1] - Scalar(1), Index(3), largest, face);
        select_exit_face(xi[0] - Scalar(1), Index(4), largest, face);
        select_exit_face(-xi[0], Index(5), largest, face);
    }
    return face;
}

} // namespace exasim::meshtransfer::detail
