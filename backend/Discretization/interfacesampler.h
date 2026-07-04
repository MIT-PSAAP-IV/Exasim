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

template <class, class> class CDiscretizationT; using CDiscretization = CDiscretizationT<::dstype, ::Int>;  // forward declaration (CInterfaceSampler only holds a reference)

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

    // Concrete-model (templated) interface-flux extraction — the model-generic analog of
    // the two ABI-only methods above, threaded on the user Model type M. This is the Gap 2
    // half of the concrete coupling path: the coupled D-N driver reads the interface flux
    // every iteration, so flux EXTRACTION must also route through the pure template when the
    // solve is pure template. For M == exasim::detail::AbiAdapter the body is byte-identical
    // to the non-templated methods (which now delegate here) — the value-only interface flux
    // still goes through the loaded libpdemodel ABI (FintDriver + driver_abi -> HdgFintonly).
    // For a concrete Model M it routes through the templated exasim::FintDriver<M> /
    // fint_only_kernel<M> with NO loaded ABI. Only the final FintDriver call is
    // model-dependent; all the field sampling (getUDGOnInterface, ... at Gauss points) is
    // model-free and shared verbatim with the ABI path. Defined out-of-line in
    // interfacesampler.cpp (included after the driver + drivers.hpp are visible).
    template <class M>
    void getInterfaceFluxesAtNodalPointsFor(dstype *flux, dstype* xdgint, dstype* nlint, const Int* faces, const Int nfaces);
    template <class M>
    void getInterfaceFluxesAtGaussPointsFor(dstype *flux, dstype* xdggint, dstype* nlgint, const Int* faces, const Int nfaces);

    void computeAverageSolutionsOnBoundary();
};

#endif
