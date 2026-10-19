#ifndef __NONMATCHINGPREPROCESSSEARCH_HPP
#define __NONMATCHINGPREPROCESSSEARCH_HPP

#include "nonmatchingpreprocessgeometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

namespace exasim {
namespace nonmatching {
namespace preprocess {

template <class T, class I>
inline T box_distance(const PreparedFaces<T, I> &faces, I face, const T *x) {
    T value = 0;
    for (I d = 0; d < faces.master.nd; ++d) {
        T gap = 0;
        if (x[d] < faces.boxMin[d + faces.master.nd * face])
            gap = faces.boxMin[d + faces.master.nd * face] - x[d];
        else if (x[d] > faces.boxMax[d + faces.master.nd * face])
            gap = x[d] - faces.boxMax[d + faces.master.nd * face];
        value += gap * gap;
    }
    return down(std::sqrt(value));
}

template <class T> inline T screen_margin(T cmax, T beta, T xabs, int nd, int porder) {
    const T eps = std::numeric_limits<T>::epsilon();
    const T coefficient = nd == 2 ? static_cast<T>(8 * (porder + 1)) : static_cast<T>(4096);
    const T scale = nd == 2 ? cmax + xabs : std::max<T>(1, cmax + xabs);
    return up(coefficient * eps * scale + static_cast<T>(4) * beta);
}

template <class T, class I>
inline T face_margin(const PreparedFaces<T, I> &faces, I face, const T *x) {
    T xabs = 0;
    for (I d = 0; d < faces.master.nd; ++d)
        xabs = std::max(xabs, std::abs(x[d]));
    return screen_margin(faces.cmax[face], faces.beta[face], xabs, faces.master.nd,
                         faces.master.porder);
}

template <class T, class I> struct LocalSearchResult {
    I face = -1;
    I failedFace = -1;
    FaceProjection<T, I> projection;
    T feasibleUpper = std::numeric_limits<T>::max();
    I projectedFaces = 0, prunedFaces = 0;
    long long projectionStarts = 0, projectionIterations = 0;
    bool failed = false;
};

template <class T, class I>
inline LocalSearchResult<T, I> search_local_faces(const PreparedFaces<T, I> &faces, const T *x,
                                                  T tolerance, I maxIterations,
                                                  ProjectionWorkspace<T, I> &work) {
    LocalSearchResult<T, I> result;
    if (faces.count == 0) {
        result.failed = true;
        return result;
    }
    const long long startsBefore = work.projectionStarts;
    const long long iterationsBefore = work.projectionIterations;
    work.faceOrder.resize(static_cast<std::size_t>(faces.count));
    std::iota(work.faceOrder.begin(), work.faceOrder.end(), 0);
    work.faceLower.resize(static_cast<std::size_t>(faces.count));
    for (I m = 0; m < faces.count; ++m)
        work.faceLower[m] = box_distance(faces, m, x) - face_margin(faces, m, x);
    std::sort(work.faceOrder.begin(), work.faceOrder.end(), [&](I a, I b) {
        return std::tie(work.faceLower[a], faces.globalFace[a]) <
               std::tie(work.faceLower[b], faces.globalFace[b]);
    });
    for (I m : work.faceOrder) {
        if (result.face >= 0) {
            const T window =
                tie_window(result.feasibleUpper, faces.cmax[m], faces.cmax[result.face]);
            if (work.faceLower[m] > up(result.feasibleUpper + window)) {
                ++result.prunedFaces;
                continue;
            }
        }
        const auto candidate = project_face_multistart(faces, m, x, tolerance, maxIterations, work);
        ++result.projectedFaces;
        if (!candidate.converged) {
            result.failed = true;
            if (result.failedFace < 0 || faces.globalFace[m] < faces.globalFace[result.failedFace])
                result.failedFace = m;
            continue;
        }
        result.feasibleUpper = std::min(result.feasibleUpper, up(candidate.distance));
        const bool tie = result.face >= 0 && tied(candidate.distance, result.projection.distance,
                                                  faces.cmax[m], faces.cmax[result.face]);
        if (result.face < 0 || (!tie && candidate.distance < result.projection.distance) ||
            (tie && faces.globalFace[m] < faces.globalFace[result.face])) {
            result.face = m;
            result.projection = candidate;
        }
    }
    result.projectionStarts = work.projectionStarts - startsBefore;
    result.projectionIterations = work.projectionIterations - iterationsBefore;
    return result;
}

} // namespace preprocess
} // namespace nonmatching
} // namespace exasim

#endif
