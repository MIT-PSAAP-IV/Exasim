/*
    nonmatchingquadrature.hpp

    [nonmatching] Phase 9 A1 (docs/nonmatching/reports/phase9_plan_amendment_r1.md,
    revision 3): opt-in composite owner-face quadrature on coincident 2D interfaces.

    The frozen nonmatchinginterface.hpp is not modified. This RAII sidecar holds the
    extra setup data of the composite rule and is owned beside nonmatchingdataT and
    NonmatchingRuntime (CDiscretization::nonmatchingQuadrature). It is built by the
    rebuild transaction and published together with nm, the runtime and the request
    storage.

    A direction is "applied" when the option is on (rank-uniform), the direction is
    coincident (every forward-search point lies on its donor face and every donor-face
    endpoint that projects into the interior of an owner face lies on it, both within
    onFaceRelTol * h_F), and at least one owner face has an interior breakpoint.
    On an applied direction the meaning of three frozen-header fields is narrowed:
      pg   Gauss index inside the point's piece of its owner face (0..ngf-1);
      pwJ  omega_g (b - a) |J(s)| for a piece [a, b] of the owner reference interval;
      the owner test values psi_i(s_p) come from ptest (not master.shapfgt[pg]).
    Owner faces without interior breakpoints use the uniform rule, and their ptest
    entries are copied from master.shapfgt, so their contributions are bitwise those
    of the uniform path. A direction that is not applied has empty vectors here and
    is bitwise the uniform path.
*/

#ifndef __NONMATCHINGQUADRATURE_HPP
#define __NONMATCHINGQUADRATURE_HPP

#include <vector>

namespace exasim {
namespace nonmatching {

template <class T, class I>
struct CompositeQuadratureDirection {
    I applied = 0;        // rank-uniform: coincident and some owner face has a breakpoint
    I ownerTests = 0;     // [nonmatching] Phase 10: D6 transports the owner face test basis
    I coincident = 0;     // rank-uniform result of the coincidence test (diagnostics)
    I breakpointsTotal = 0;   // global number of interior breakpoints (diagnostics)
    I maxPiecesPerFace = 0;   // global maximum (diagnostics)

    // Owner side: own faces of this rank in this direction, in own-face order l.
    std::vector<I> facebreakptr;  // [nownfaces + 1] when applied, else empty
    std::vector<T> breakpoints;   // sorted owner reference coordinates in (0, 1) per face

    // Donor side, aligned with the point order of nm.dir[k] after distribution.
    std::vector<T> ptest;         // [npf x npts] owner test values psi_i(s_p), ptest[i + npf*p]
    std::vector<T> ps;            // [npts] owner reference coordinate of the point
    std::vector<I> ppiece;        // [npts] piece index of the point on its owner face

    void clear()
    {
        applied = ownerTests = coincident = breakpointsTotal = maxPiecesPerFace = 0;
        facebreakptr.clear(); breakpoints.clear();
        ptest.clear(); ps.clear(); ppiece.clear();
    }
};

template <class T, class I>
struct CompositeOwnerQuadrature {
    I requested = 0;      // rank-uniform option value
    CompositeQuadratureDirection<T, I> dir[2];

    CompositeOwnerQuadrature() = default;
    CompositeOwnerQuadrature(const CompositeOwnerQuadrature&) = delete;
    CompositeOwnerQuadrature& operator=(const CompositeOwnerQuadrature&) = delete;
    CompositeOwnerQuadrature(CompositeOwnerQuadrature&&) noexcept = default;
    CompositeOwnerQuadrature& operator=(CompositeOwnerQuadrature&&) noexcept = default;

    void clear()
    {
        requested = 0;
        dir[0].clear();
        dir[1].clear();
    }

    // Host persistent bytes of the sidecar (measurements, plan A.10).
    long long host_bytes() const
    {
        long long bytes = 0;
        for (int k = 0; k < 2; ++k)
            bytes += static_cast<long long>(sizeof(I)) * (dir[k].facebreakptr.size() + dir[k].ppiece.size()) +
                     static_cast<long long>(sizeof(T)) *
                         (dir[k].breakpoints.size() + dir[k].ptest.size() + dir[k].ps.size());
        return bytes;
    }
};

// Non-owning view for the host coupling kernels: ptest == nullptr selects the
// master rule (pwJ[p] * shapfgt[pg[p] + ngf*i]).
template <class T, class I>
struct CompositeQuadratureView {
    const T* ptest = nullptr;
    I npf = 0;
};

template <class T, class I>
inline CompositeQuadratureView<T, I> composite_view(const CompositeOwnerQuadrature<T, I>* quad,
        int direction, I npf)
{
    CompositeQuadratureView<T, I> view;
    if (quad && !quad->dir[direction].ptest.empty()) {
        view.ptest = quad->dir[direction].ptest.data();
        view.npf = npf;
    }
    return view;
}

} // namespace nonmatching
} // namespace exasim

#endif
