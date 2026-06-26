/**
 * @class CAssembler
 * @brief Assembles the HDG global linear system from a discretization.
 *
 * Stage 3 of the internal separation: forming the global linear system is an assembly
 * concern, distinct from the discretization's spatial operators (geometry / residual /
 * flux / matvec). CAssembler holds a reference to the CDiscretization it assembles from
 * and writes the global system into disc.res (H / Rh / Ru / F + preconditioner K). The
 * solver consumes that system. The res buffers and the K scratch arena remain owned by
 * CDiscretization for now (a deeper ownership split is left for later); CAssembler is a
 * behaviour re-home, not a data move.
 */
#ifndef __ASSEMBLER_H__
#define __ASSEMBLER_H__

class CDiscretization;  // forward declaration (CAssembler only holds a reference)

class CAssembler {
public:
    CDiscretization& disc;

    CAssembler(CDiscretization& disc_) : disc(disc_) {}

    // assemble the HDG global linear system (writes disc.res.H / Rh / Ru / F and the
    // preconditioner matrix disc.res.K)
    void hdgAssembleLinearSystem(dstype *b, Int backend);

    // assemble the HDG global residual (writes disc.res.Rh / Ru)
    void hdgAssembleResidual(dstype *b, Int backend);

    // apply the discrete operator: Jv = J(u)*v (LDG = matrix-free FD; HDG = apply res.H)
    void evalMatVec(dstype* Jv, dstype* v, dstype* u, dstype* Ru, Int backend);
    void evalMatVec(dstype* Jv, dstype* v, dstype* u, dstype* Ru, Int spatialScheme, Int backend);
};

#endif
