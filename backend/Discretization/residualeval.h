/**
 * @class CResidual
 * @brief Evaluates the discretized PDE residual R(u) and the auxiliary flux q.
 *
 * The residual is the *equation* discretized onto the function space -- a distinct object
 * from the space (CDiscretization) it is discretized on, and from its linearization
 * (CAssembler: the Jacobian assemble + apply). CResidual is model-driven (it evaluates the
 * flux/source/boundary kernels via the discretization's driver_abi). Full-split step 1:
 * behaviour re-home -- CResidual holds a CDiscretization& and operates on disc's data
 * (res/sol/driver_abi); data ownership moves to the operator in a later step.
 */
#ifndef __RESIDUALEVAL_H__
#define __RESIDUALEVAL_H__

class CDiscretization;  // forward declaration (CResidual only holds a reference)

class CResidual {
public:
    CDiscretization& disc;

    CResidual(CDiscretization& disc_) : disc(disc_) {}

    // evaluate the residual vector R(u) (current sol.udg, or at a given u)
    void evalResidual(Int backend);
    void evalResidual(dstype* Ru, dstype* u, Int backend);

    // evaluate the auxiliary flux q (store in sol.udg, or at a given u)
    void evalQ(Int backend);
    void evalQSer(Int backend);
    void evalQ(dstype* q, dstype* u, Int backend);

    // insert u into sol.udg (and recompute q)
    void updateUDG(dstype* u, Int backend);
    void updateU(dstype* u, Int backend);

    // evaluate the artificial-viscosity field
    void evalAVfield(dstype* avField, dstype* u, Int backend);
    void evalAVfield(dstype* avField, Int backend);
};

#endif
