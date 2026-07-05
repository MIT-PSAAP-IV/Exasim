/*
    CWallModel -- build the wall-model sampling data, extracted from CDiscretization
    (cohesion win). Body unchanged from the former CDiscretization::BuildWallModelData;
    it runs the host-only point-locator builder and copies the result into disc.wallmodel.
*/
#ifndef __WALLMODELBUILD
#define __WALLMODELBUILD

#include "wallmodelbuild.h"

bool CWallModel::build(Int ibc, dstype y1)
{
    if (disc.common.backend > 1)
        error("CWallModel::build is not implemented for GPU/HIP backends because the point locator wall-model builder is host-only.");

    CPointLocator locator;
    const bool success = locator.BuildWallModelSamplingData(disc, ibc, y1);
    if (!success)
        error("CWallModel::build failed while building wall-model sampling data.");

    CopyWallModelSamplingData(disc.wallmodel, locator.wm, disc.common.backend);
    return true;
}

#endif
