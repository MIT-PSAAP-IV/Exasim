// SPDX-License-Identifier: see LICENSE
//
// <exasim/export.hpp> — composable helpers for the operator-export path.
//
// An external app (e.g. a PETSc driver) that wants Exasim's discretization +
// operators in memory should NOT pull in the high-level ExasimSolver facade
// (that bundles the whole CSolution solve). These free helpers do only the
// setup an external driver needs, in small pieces it can compose:
//
//   PDE pde = exasim::default_pde<MyModel>();   // HDG-friendly config for the model
//   pde.porder = 3; pde.physicsparam = {1.0};   // override any field directly
//
//   exasim::MeshSpec mesh{p, t, np, ne, /*nve=*/4};
//   mesh.add_boundary(1, [](const double* x){ return std::abs(x[1]) < 1e-8; });
//   ...
//
//   CDiscretization disc(exasim::make_preprocessed<MyModel>(pde, mesh),
//                        "", pde.exasimpath, 1, 0, 0, 0, backend, pde.builtinmodelID);
//   CResidual<MyModel> r(disc); CAssembler<MyModel> a(disc); CPreconditioner<MyModel> p(disc, backend);
//
// No solver is constructed; the driver owns the solve. The same `default_pde<M>`
// is what ExasimSolver uses for its defaults, so the two paths cannot drift.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <exasim/operators.hpp>   // PDE / InputParams / ParsedSpec / BoundaryPred /
                                  // CPreprocessing / meshFromArrays / Preprocessed / CDiscretization
#include <exasim/model.hpp>       // exasim::is_model_v

namespace exasim {

// HDG-friendly default PDE config for a hand-written model M. Every field can be
// overridden on the returned struct before preprocessing. This is the single
// source of the curated defaults (ExasimSolver applies the same ones).
template <class M>
inline PDE default_pde()
{
    static_assert(is_model_v<M>, "default_pde<M>: M must satisfy the Model contract.");
    PDE pde;
    pde.discretization = "hdg";
    pde.platform       = "cpu";
    pde.gendatain      = 1;
    pde.builtinmodelID = 1;
    pde.saveOutputs    = 0;        // in-memory by default
    pde.porder         = 1;
    pde.pgauss         = 2;
    pde.torder         = 1;
    pde.nstage         = 1;
    pde.tdep           = 0;
    pde.NewtonIter     = 20;
    pde.NewtonTol      = 1e-6;
    pde.GMRESiter      = 200;
    pde.GMRESrestart   = 50;
    pde.GMREStol       = 1e-8;
    pde.tau            = {1.0};
    pde.dt             = {0.0};
    pde.neb            = 4096;
    pde.nfb            = 8192;
    pde.ibs            = 1;

    // Compile-time dimensions from the model.
    pde.nd  = M::nd;
    pde.ncu = M::ncu;
    pde.ncw = M::ncw;
    pde.nc  = M::ncu * (1 + M::nd);
    pde.physicsparam.assign(std::max(1, M::nparam), 0.0);

    // exasimpath: $EXASIM_DIR if set, else "." — points at the tree holding
    // backend/Preprocessing/{master,gauss}nodes.bin (or $EXASIM_DATA_DIR overrides).
    if (const char* d = std::getenv("EXASIM_DIR")) pde.exasimpath = d;
    else                                           pde.exasimpath = ".";
    pde.datapath    = ".";
    pde.datainpath  = "./datain";
    pde.dataoutpath = "./dataout";
    pde.modelfile   = "";
    pde.meshfile    = "";
    return pde;
}

// In-memory mesh + boundary description.
//   p   : nd x np doubles, column-major (p[d + nd*j] = coord d of vertex j)
//   t   : nve x ne ints, column-major, 0-based (corners of each element)
//   nve : 4 for 2D quads / 8 for 3D hexes / 3 for tris / 4 for tets
// Boundaries are tagged by a typed predicate on the vertex coordinates; the order
// they are added is the 1-based `ib` the model's fbou_hdg / fbou see.
struct MeshSpec {
    const double* p = nullptr;
    const int*    t = nullptr;
    int np = 0, ne = 0, nve = 0;
    std::vector<int>          boundary_tags;
    std::vector<BoundaryPred> boundary_preds;

    MeshSpec() = default;
    MeshSpec(const double* p_, const int* t_, int np_, int ne_, int nve_)
        : p(p_), t(t_), np(np_), ne(ne_), nve(nve_) {}

    void add_boundary(int tag, BoundaryPred pred) {
        boundary_tags.push_back(tag);
        boundary_preds.push_back(std::move(pred));
    }
};

// Build the in-memory Preprocessed bundle (mesh + master element + runtime structs)
// from a PDE config + mesh. No datain files are written. Serial (mpiprocs == 1).
template <class M>
inline Preprocessed make_preprocessed(const PDE& pde_in, const MeshSpec& mesh,
                                      int mpirank = 0, int mpiprocs = 1)
{
    static_assert(is_model_v<M>, "make_preprocessed<M>: M must satisfy the Model contract.");
    InputParams params;
    ParsedSpec  spec;
    for (std::size_t i = 0; i < mesh.boundary_tags.size(); ++i) {
        params.boundaryConditions.push_back(mesh.boundary_tags[i]);
        params.boundaryPreds.push_back(mesh.boundary_preds[i]);
        params.curvedBoundaries.push_back(0);
        params.curvedBoundaryExprs.push_back("");
    }
    PDE pde = pde_in;
    CPreprocessing preproc(pde, params, spec, mpirank, mpiprocs);
    preproc.mesh = meshFromArrays(mesh.p, mesh.t, mesh.np, mesh.ne, mesh.nve, M::nd,
                                  preproc.params, preproc.pde);
    Preprocessed pre = preproc.take();
    pre.save_outputs = (pde.saveOutputs != 0);
    return pre;
}

// Recover the volume state udg from a converged HDG trace uh (static-condensation
// back-substitution), using the element factors from the most recent
// hdgAssembleLinearSystem. Replaces the open-coded hdgGetDUDG + UpdateUDG (+ hdgGetQ)
// sequence. `scratch` is an ndofuhat-sized buffer in the SAME memory space as the
// operators (e.g. a sysstruct's sys.x); it receives the trace increment uh - sol.uh.
// All ops are backend-dispatched, so this works on CPU and GPU (uh/scratch may be
// device pointers when backend>=2).
inline void recover_volume(CDiscretization& disc, const dstype* uh, dstype* scratch, int backend)
{
    auto& c = disc.common;
    const Int N = c.sizes.ndofuhat;
    ArrayAXPBY(scratch, const_cast<dstype*>(uh), disc.sol.uh, 1.0, -1.0, N);  // duh = uh - sol.uh
    ArrayCopy(disc.sol.uh, const_cast<dstype*>(uh), N);                       // sol.uh <- uh
    hdgGetDUDG(disc.res.Ru, disc.res.F, scratch, disc.res.Rq, disc.mesh, c, backend);
    UpdateUDG(disc.sol.udg, disc.res.Ru, 1.0, c.grid.npe, c.components.nc, c.meshsizes.ne1,
              0, c.grid.npe, 0, c.components.ncu, 0, c.meshsizes.ne1);
    if (c.components.ncq > 0)
        hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, c, backend);
}

// Evaluate the model's volume quantities of interest on the current state, integrated
// with quadrature (the exported QoIvolumeDriver<M> + qoiElement scaffolding). `pde.nvqoi`
// must have been set before construction so the QoI buffers were allocated. Returns the
// integrated values (host). E.g. a model whose qoi_volume[0] = (u - u_exact)^2 yields
// qoi[0] = the squared L2 error -- a proper quadrature norm, not a nodal sum.
template <class M>
inline std::vector<dstype> eval_qoi(CDiscretization& disc)
{
    const int nv = disc.common.qoiparams.nvqoi;
    if (nv <= 0) return {};
    qoiElement<M>(disc.sol, disc.res, disc.app, disc.master, disc.mesh, disc.tmp, disc.common);
    return std::vector<dstype>(disc.common.qoiparams.qoivolume,
                               disc.common.qoiparams.qoivolume + nv);
}

// One-shot sugar: build the in-memory discretization from a PDE config + mesh.
// Returned by unique_ptr because CDiscretization owns malloc'd C arrays (non-copyable);
// the operator classes (CResidual<M>/... ) bind a reference to *disc.
template <class M>
inline std::unique_ptr<CDiscretization>
make_discretization(const PDE& pde, const MeshSpec& mesh, int backend,
                    int mpirank = 0, int mpiprocs = 1)
{
    return std::make_unique<CDiscretization>(
        make_preprocessed<M>(pde, mesh, mpirank, mpiprocs),
        std::string{}, pde.exasimpath, mpiprocs, mpirank, /*fileoffset=*/0, /*omprank=*/0,
        backend, pde.builtinmodelID);
}

} // namespace exasim
