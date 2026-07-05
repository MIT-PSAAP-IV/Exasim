/**
 * @class CWallModel
 * @brief Builds the wall-model sampling data (off-wall sample points / shape functions) for a
 *        discretization's wall-boundary conditions.
 *
 * Turbulence wall-model concern, distinct from the function space (CDiscretization). Behaviour
 * re-home: CWallModel holds a CDiscretization& and runs the host-only point-locator build,
 * copying the result into disc.wallmodel. The wallmodelstruct data itself still lives on
 * CDiscretization for now (read by the wall-BC kernels); only the build logic moves here.
 */
#ifndef __WALLMODELBUILD_H__
#define __WALLMODELBUILD_H__

template <class, class> class CDiscretizationT; using CDiscretization = CDiscretizationT<::dstype, ::Int>;  // forward declaration (CWallModel only holds a reference)

class CWallModel {
public:
    CDiscretization& disc;

    CWallModel(CDiscretization& disc_) : disc(disc_) {}

    // build the wall-model sampling data for boundary condition ibc at off-wall distance y1,
    // storing it in disc.wallmodel. Host-only (the point-locator builder is host-only).
    bool build(Int ibc, dstype y1);
};

#endif
