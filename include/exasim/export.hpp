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

#if defined(HAVE_MPI) && defined(HAVE_PARMETIS)
// Distributed-mesh spec for the scalable MPI path: each rank supplies ONLY its slice --
// a contiguous block of global nodes (their coords in p_local) and a contiguous block of
// global elements (t_local, using GLOBAL node indices in [0,np_global)). No rank holds the
// whole mesh. ParMETIS (inside takeParallel) repartitions for locality regardless of the
// initial split, and the node-request protocol fetches each rank's needed coordinates.
struct MeshSpecDistributed {
    const double* p_local = nullptr;   // nd x np_local (this rank's node-slice coords)
    const int*    t_local = nullptr;   // nve x ne_local (this rank's elements, GLOBAL node ids)
    int np_local = 0, ne_local = 0, np_global = 0, ne_global = 0, nve = 0;
    std::vector<int>          boundary_tags;
    std::vector<BoundaryPred> boundary_preds;

    MeshSpecDistributed() = default;
    MeshSpecDistributed(const double* p, const int* t, int npl, int nel,
                        int npg, int neg, int nve_)
        : p_local(p), t_local(t), np_local(npl), ne_local(nel),
          np_global(npg), ne_global(neg), nve(nve_) {}

    void add_boundary(int tag, BoundaryPred pred) {
        boundary_tags.push_back(tag);
        boundary_preds.push_back(std::move(pred));
    }
};

// Scalable MPI counterpart of make_preprocessed: meshFromArraysDistributed (each rank's
// slice) + CPreprocessing::takeParallel(comm) (ParMETIS repartition + DMD + per-rank
// app/master/mesh/sol). Hand the result to the MPI in-memory ctor
// CDiscretization(Preprocessed&&, backend, mpiprocs, mpirank).
template <class M>
inline Preprocessed make_preprocessed_distributed(const PDE& pde_in,
                                                  const MeshSpecDistributed& mesh,
                                                  MPI_Comm comm)
{
    static_assert(is_model_v<M>, "make_preprocessed_distributed<M>: M must satisfy the Model contract.");
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    InputParams params;
    ParsedSpec  spec;
    for (std::size_t i = 0; i < mesh.boundary_tags.size(); ++i) {
        params.boundaryConditions.push_back(mesh.boundary_tags[i]);
        params.boundaryPreds.push_back(mesh.boundary_preds[i]);
        params.curvedBoundaries.push_back(0);
        params.curvedBoundaryExprs.push_back("");
    }
    PDE pde = pde_in;
    CPreprocessing preproc(pde, params, spec, rank, nprocs);
    preproc.mesh = meshFromArraysDistributed(mesh.p_local, mesh.t_local, mesh.np_local,
                                             mesh.ne_local, mesh.nve, M::nd,
                                             mesh.np_global, mesh.ne_global,
                                             preproc.params, preproc.pde);
    Preprocessed pre = preproc.takeParallel(comm);
    pre.save_outputs = (pde.saveOutputs != 0);
    return pre;
}
#endif // HAVE_MPI && HAVE_PARMETIS

// Recover the volume state udg from a converged HDG trace uh (static-condensation
// back-substitution), using the element factors from the most recent
// hdgAssembleLinearSystem. Replaces the open-coded hdgGetDUDG + UpdateUDG (+ hdgGetQ)
// sequence. `scratch` is an ndofuhat-sized buffer in the SAME memory space as the
// operators (e.g. a sysstruct's sys.x); it receives the trace increment uh - sol.uh.
// All ops are backend-dispatched, so this works on CPU and GPU (uh/scratch may be
// device pointers when backend>=2).
inline void recover_volume(CDiscretization& disc, const dstype* uh, dstype* scratch)
{
    auto& c = disc.common;
    const int backend = c.backend;
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
//
// M is the QoI capability only -- a struct carrying the size config + qoi_volume/qoi_boundary
// (e.g. Poisson2D::QoI), NOT the full physics model; its sizes must match the model the
// discretization was built from.
template <class M>
inline std::vector<dstype> eval_qoi(CDiscretization& disc)
{
    static_assert(is_qoi_model_v<M>,
        "eval_qoi: M must expose the QoI surface (size config + qoi_volume + qoi_boundary), "
        "e.g. Poisson2D::QoI -- not the whole physics model.");
    const int nv = disc.common.qoiparams.nvqoi;
    if (nv <= 0) return {};
    qoiElement<M>(disc.sol, disc.res, disc.app, disc.master, disc.mesh, disc.tmp, disc.common);
    return std::vector<dstype>(disc.common.qoiparams.qoivolume,
                               disc.common.qoiparams.qoivolume + nv);
}

// Write the current solution to a ParaView .vtu file, using Exasim's existing vis
// pipeline (CVisualization + vtuwrite). The model's vis_scalars/vis_vectors/vis_tensors
// supply the fields; the CG vis geometry (sol.xcg / mesh.cgelcon) is built in-memory by
// buildConn. pde.nsca/nvec/nten must have been set before construction (so the counts
// and CG geometry are wired). basename gets a ".vtu" suffix (serial) or per-rank .vtu +
// .pvtu (MPI). Mirrors CSolutionWriter::SaveParaview's field computation.
//
// M is the visualization capability only -- a struct carrying the size config +
// vis_scalars/vis_vectors (and vis_tensors only if pde.nten>0), e.g. Poisson2D::Vis, NOT
// the full physics model; its sizes must match the model the discretization was built from.
template <class M>
inline void write_vtk(CDiscretization& disc, const std::string& basename)
{
    auto& c = disc.common;
    const int backend = c.backend;
    const int nc=c.components.nc, ncx=c.components.ncx, nco=c.components.nco, ncw=c.components.ncw;
    const int nsca=c.qoiparams.nsca, nvec=c.qoiparams.nvec, nten=c.qoiparams.nten;
    const int npe=c.grid.npe, ne=c.meshsizes.ne1, ndg=npe*ne;

    CVisualization vis(disc, backend);
    const int ncg = vis.npoints;
    // default field names if the pdeapp vis keys did not supply them
    if ((int)vis.scalar_names.size() < nsca) { vis.scalar_names.clear();
        for (int i=0;i<nsca;++i) vis.scalar_names.push_back("scalar" + std::to_string(i)); }
    if ((int)vis.vector_names.size() < nvec) { vis.vector_names.clear();
        for (int i=0;i<nvec;++i) vis.vector_names.push_back("vector" + std::to_string(i)); }
    if ((int)vis.tensor_names.size() < nten) { vis.tensor_names.clear();
        for (int i=0;i<nten;++i) vis.tensor_names.push_back("tensor" + std::to_string(i)); }

    dstype* udg = disc.res.Rq;
    dstype* wdg = disc.res.Ru;
    const int nvis  = std::max(std::max(nsca, 3*nvec), vis.ntc*nten);
    const int szvis = npe*(ncx+nco+nvis)*ne;
    dstype* tempn = disc.tmp.tempn;
    bool owns = false;
    if (disc.tmp.sztempn + disc.tmp.sztempg < szvis) { TemplateMalloc(&tempn, szvis, backend); owns = true; }
    dstype* xdg = &tempn[0];
    dstype* vdg = &tempn[npe*ncx*ne];
    dstype* f   = &tempn[npe*(ncx+nco)*ne];

    GetElemNodes(xdg, disc.sol.xdg, npe, ncx, 0, ncx, 0, ne);
    GetElemNodes(udg, disc.sol.udg, npe, nc, 0, nc, 0, ne);
    if (nco > 0) GetElemNodes(vdg, disc.sol.odg, npe, nco, 0, nco, 0, ne);
    if (ncw > 0) GetElemNodes(wdg, disc.sol.wdg, npe, ncw, 0, ncw, 0, ne);

    if (nsca > 0) {
        EXASIM_DRIVER_CALL(VisScalarsDriver, f, xdg, udg, vdg, wdg, disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common, npe, 0, ne, backend);
        VisDG2CG(vis.scafields, f, disc.mesh.cgent2dgent, disc.mesh.colent2elem, disc.mesh.rowent2elem, ne, ncg, ndg, 1, 1, nsca);
    }
    if (nvec > 0) {
        EXASIM_DRIVER_CALL(VisVectorsDriver, f, xdg, udg, vdg, wdg, disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common, npe, 0, ne, backend);
        VisDG2CG(vis.vecfields, f, disc.mesh.cgent2dgent, disc.mesh.colent2elem, disc.mesh.rowent2elem, ne, ncg, ndg, 3, ncx, nvec);
    }
    if (nten > 0) {
        EXASIM_DRIVER_CALL(VisTensorsDriver, f, xdg, udg, vdg, wdg, disc.mesh, disc.master, disc.app, disc.sol, disc.tmp, disc.common, npe, 0, ne, backend);
        VisDG2CG(vis.tenfields, f, disc.mesh.cgent2dgent, disc.mesh.colent2elem, disc.mesh.rowent2elem, ne, ncg, ndg, vis.ntc, vis.ntc, nten);
    }

    if (c.mpiProcs == 1) vis.vtuwrite(basename, vis.scafields, vis.vecfields, vis.tenfields);
    else                 vis.vtuwrite_parallel(basename, c.mpiRank, c.mpiProcs, vis.scafields, vis.vecfields, vis.tenfields);
    if (owns) TemplateFree(tempn, backend);
}

// ---- Operator exports (style A: opaque apply-callbacks) --------------------------------
// Exasim already computes the full HDG element-local operator (mass M, q-recovery C/E, the
// u-equation blocks D/F/G, residuals) and Schur-reduces to the condensed trace H + K. These
// helpers surface the pieces besides H/K that a PETSc driver may want. The raw per-element
// block views (Mass/Minv/C/E/D/F/G) are left for a later "style B" export.

// Recover the auxiliary flux q from the current (u on volume, uh on trace) into sol.udg's q
// components: q = -M^{-1}(C u + E uh). Standalone form of the recovery hdgGetQ that
// recover_volume runs as its last step.
template <class M>
inline void recover_q(CDiscretization& disc)
{
    if (disc.common.components.ncq > 0)
        hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common,
                disc.common.backend);
}

// The full (possibly nonlinear) DG residual R(u): sets the volume unknown to u, runs the
// residual assembly, and writes Ru (length common.sizes.ndof1). Thin pass-through to
// CResidual<M>::evalResidual -- the operator PETSc's SNES FormFunction drives for a general model.
template <class M>
inline void eval_residual(CResidual<M>& residual, dstype* Ru, dstype* u, int backend)
{
    residual.evalResidual(Ru, u, backend);
}

// Apply the inverse VOLUME mass matrix: y = M^{-1} x, element block-diagonal (LDG mass, res.Minv;
// requires it to have been computed, e.g. disc.compMassInverse()). ncr is the field width (M::ncu
// for u, M::ncq for q). NB: HDG's q-equation uses its own internal mass (res.Minv2) inside
// recover_q; this is the plain volume mass inverse.
template <class M>
inline void apply_mass_inverse(CDiscretization& disc, const dstype* x, dstype* y, int ncr = M::ncu)
{
    auto& c = disc.common;
    const Int npe = c.grid.npe, ne = c.meshsizes.ne1;
    ApplyMinv(y, disc.res.Minv, const_cast<dstype*>(x), 1.0, c.grid.curvedMesh, npe, ncr, 0, (int)ne);
}

// ---- Style-B: raw per-element block VIEWS ----------------------------------------------
// Read-only handles to the element-local HDG operator blocks Exasim assembles (device pointers
// when backend>=2), for a consumer that wants to build custom operators / assemble per block.
// Per-element, column-major. NB: C/E already fold in M^{-1} (they are the recovery coefficients
// q = -(C u + E uh)), so these are the *recovery* form, not a raw saddle-point. `nullptr` fields
// were not allocated for this discretization/model.
struct ElementBlocks {
    const dstype *H = nullptr;     // condensed trace block AE: (ncu*npf*nfe)^2 per elem
    const dstype *K = nullptr;     // preconditioner block
    const dstype *Mass = nullptr;  // q-mass M           : npe^2 per elem  (res.Mass2)
    const dstype *Minv = nullptr;  // M^{-1}             : npe^2 per elem  (res.Minv2)
    const dstype *C = nullptr;     // u->q recovery      : npe^2 * nd
    const dstype *E = nullptr;     // uh->q recovery
    const dstype *D = nullptr;     // u-equation block
    const dstype *F = nullptr, *G = nullptr;  // couplings
    int npe=0, npf=0, nfe=0, ne=0, ncu=0, ncq=0, nd=0;   // dims to interpret the blocks
    Int szH=0, szK=0, szMass=0, szMinv=0, szC=0, szE=0, szD=0;
};

inline ElementBlocks element_blocks(CDiscretization& disc)
{
    auto& r = disc.res; auto& c = disc.common;
    ElementBlocks b;
    b.H=r.H; b.K=r.K; b.Mass=r.Mass2; b.Minv=r.Minv2; b.C=r.C; b.E=r.E; b.D=r.D; b.F=r.F; b.G=r.G;
    b.npe=c.grid.npe; b.npf=c.grid.npf; b.nfe=(int)c.meshsizes.nfe; b.ne=(int)c.meshsizes.ne1;
    b.ncu=c.components.ncu; b.ncq=c.components.ncq; b.nd=c.grid.nd;
    b.szH=r.szH; b.szK=r.szK; b.szMass=r.szMass2; b.szMinv=r.szMinv2; b.szC=r.szC; b.szE=r.szE; b.szD=r.szD;
    return b;
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
