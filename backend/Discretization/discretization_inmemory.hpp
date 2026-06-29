// <backend/Discretization/discretization_inmemory.hpp>
//
// In-memory (no-ABI) construction of CDiscretization from an already-preprocessed mesh
// (exasim::Preprocessed, produced by ExasimSolver<M>::set_mesh -> meshFromArrays -> take()).
// This is the files-free / driver_abi-free path that lets a concrete model M build the
// discretization + operators in memory and hand them to an external driver (e.g. PETSc).
//
// CONSUMER-ONLY: include this AFTER the backend FEM aggregation (so CDiscretization, cpuInitSetup
// and buildConn are defined) and after buildstructs.hpp (so exasim::Preprocessed is complete). The
// backend unity build (ExasimSolver.cpp) never includes it -- the file build uses the ABI ctor and
// has no Preprocessed available -- so the member definition is `inline` and lives here, while
// discretization.h only forward-declares exasim::Preprocessed + declares the constructor.
#ifndef __DISCRETIZATION_INMEMORY_HPP__
#define __DISCRETIZATION_INMEMORY_HPP__

#include <backend/Preprocessing/buildstructs.hpp>   // exasim::Preprocessed (complete type)
#include "discretization.h"                          // CDiscretization (ctor declared, Preprocessed fwd-declared)

// cpuInitSetup: the post-read setup factored out of cpuInit (allocate res/tmp, derive common, build
// shape products + face maps, allocate sol scratch). Defined in setstructs.cpp (already in the TU via
// the backend aggregation). Forward-declared here so this header is order-robust.
void cpuInitSetup(solstruct &sol, resstruct &res, appstruct &app, masterstruct &master,
        meshstruct &mesh, tempstruct &tmp, commonstruct &common,
        std::string filein, std::string fileout, Int mpiprocs, Int mpirank, Int fileoffset, Int omprank);

// In-memory analogue of cpuInit: instead of readInput()'ing datain binaries, transfer the structs
// already populated by the preprocessor, build the element/face connectivity from the raw ti (the one
// post-read step readInput does), then run the shared cpuInitSetup. Pointer ownership of the struct
// arrays moves to the destination structs; `pre` is a transient (std::move'd in by the caller) whose
// POD structs are never freed, so there is no double free.
inline void cpuInitInMemory(exasim::Preprocessed& pre, solstruct &sol, resstruct &res, appstruct &app,
        masterstruct &master, meshstruct &mesh, tempstruct &tmp, commonstruct &common,
        std::string filein, std::string fileout, Int mpiprocs, Int mpirank, Int fileoffset, Int omprank)
{
    app    = pre.app;
    master = pre.master;
    mesh   = pre.mesh;
    sol    = pre.sol;

    // Build element/face connectivity from the raw connectivity ti (readInput does this from file,
    // readbinaryfiles.hpp: buildConn when ti exists and facecon was not precomputed).
    if (mesh.nsize[26] > 0 && mesh.nsize[27] > 0 && mesh.szfacecon == 0) {
        buildConn(mesh, sol, app, master, pre.ti.data(),
                  mesh.boundaryConditions, mesh.intepartpts, mesh.nsize[27]);
    }

    cpuInitSetup(sol, res, app, master, mesh, tmp, common,
                 filein, fileout, mpiprocs, mpirank, fileoffset, omprank);
}

inline CDiscretization::CDiscretization(exasim::Preprocessed&& pre, std::string fileout, std::string exasimpath,
        Int mpiprocs, Int mpirank, Int fileoffset, Int omprank, Int backend, Int builtinmodelID)
{
    driver_abi = ExasimDriverABI{};            // concrete-M build: no runtime ABI (null fn-pointers)
    common.driver_abi = &driver_abi;
    common.backend = backend;
    common.exasimpath = exasimpath;
    common.builtinmodelID = builtinmodelID;

    if (backend > 1) {
        error("CDiscretization(Preprocessed&&): in-memory GPU construction is not yet implemented "
              "(CPU only for the initial PETSc operator-export path).");
    }

    // CPU: build straight into the member structs (no host-staging needed without a device copy).
    cpuInitInMemory(pre, sol, res, app, master, mesh, tmp, common,
                    /*filein=*/"", fileout, mpiprocs, mpirank, fileoffset, omprank);

    // the struct move overwrote app's model id; re-apply (single concrete model -> index 0).
    app.builtinmodelID = builtinmodelID;
    app.modelnumber = 0;
    common.modelnumber = 0;

    // shared post-init tail (geometry, mass inverse / HDG setup). Solve mode; no vis-count / Paraview
    // overrides (the operator-export path does not write output files).
    finalizeConstruction(backend, ExasimExecutionMode::Solve, 0, 0, 0, 0, 0, 0);
}

#endif // __DISCRETIZATION_INMEMORY_HPP__
