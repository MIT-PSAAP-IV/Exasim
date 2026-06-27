/**
 * @class CInterfaceSampler
 * @brief Samples field data (coords, normals, solution traces, fluxes, boundary averages) on
 *        a set of interface/boundary faces of a discretization.
 *
 * Coupling/sampling I/O concern, distinct from the function space (CDiscretization). Holds a
 * CDiscretization& and reads its sol/mesh/master; consumed by the coupling interface exchange
 * (ExasimSolver), boundary averaging (run.hpp), and wall-model sampling.
 */
#ifndef __INTERFACESAMPLER_H__
#define __INTERFACESAMPLER_H__

class CDiscretization;  // forward declaration (CInterfaceSampler only holds a reference)

class CInterfaceSampler {
public:
    CDiscretization& disc;

    CInterfaceSampler(CDiscretization& disc_) : disc(disc_) {}

    Int  getFacesOnInterface(Int **faces, const Int boundarycondition);
    void getDGNodesOnInterface(dstype* xdgint, const Int* faces, const Int nfaces);
    void getUDGOnInterface(dstype* udgint, const Int* faces, const Int nfaces);
    void getWDGOnInterface(dstype* wdgint, const Int* faces, const Int nfaces);
    void getODGOnInterface(dstype* odgint, const Int* faces, const Int nfaces);
    void getUHATOnInterface(dstype* uhint, const Int* faces, const Int nfaces);
    void getNormalVectorOnInterface(dstype* nlint, dstype* xdgint, const Int nfaces);
    void getFieldsAtGaussPointsOnInterface(dstype* xdggint, dstype* xdgint, const Int nfaces, const Int ncx);
    void getInterfaceFluxesAtNodalPoints(dstype *flux, dstype* xdgint, dstype* nlint, const Int* faces, const Int nfaces);
    void getInterfaceFluxesAtGaussPoints(dstype *flux, dstype* xdggint, dstype* nlgint, const Int* faces, const Int nfaces);
    void computeAverageSolutionsOnBoundary();
};

#endif
