// poisson2d.cc — auto-generated standalone header-only Exasim app (pyt2c app scaffolder).
//
// Drives a steady HDG solve on the concrete text2code-generated model `PdeModel`
// (generated/my_model.hpp) via a genuine PETSc SNES + GMRES on Exasim's exported HDG
// operators (exasim::petsc::solve_steady). There is NO runtime-loaded model ABI (.so)
// and NO hand-rolled PETSc solver code in this app: the whole solver lives in
// <exasim/petsc.hpp>. This is the C++-driven form of a text2code-generated model.
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include <exasim/operators.hpp>   // unity Exasim backend (CSolution<M>/CAssembler<M>/CPreconditioner<M>)
#include <exasim/export.hpp>      // recover_volume
#include <exasim/petsc.hpp>       // exasim::petsc::solve_steady (prepare + SNES+GMRES + recover)

// text2code emits `struct PdeModel : ModelDefaults<PdeModel>` unqualified; with the
// operator-export backend that CRTP base is exasim::ModelDefaults, so bring it into scope.
using exasim::ModelDefaults;
#include "generated/my_model.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    PETSC_COMM_WORLD = MPI_COMM_WORLD;
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    if (!Kokkos::is_initialized()) Kokkos::initialize(argc, argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string filein  = (argc > 1) ? argv[1] : "datain/";
    const std::string fileout = (argc > 2) ? argv[2] : "dataout/out";
    const int backend = 0;  // 0/1 = host CPU (set 2 for CUDA / 3 for HIP builds)

    {
        // No-ABI concrete-model CSolution, built straight from preprocessed datain/.
        CSolution<PdeModel> model(filein, fileout, "", (Int)size, (Int)rank,
                                  (Int)0 /*fileoffset*/, (Int)0 /*gpuid*/,
                                  (Int)backend, (Int)8 /*builtinmodelID*/);
        model.disc.common.nomodels = 1;
        std::vector<Int>     ncarr  = { model.disc.common.components.nc };
        std::vector<dstype*> udgarr = { &model.disc.sol.udg[0] };
        model.disc.common.ncarray = ncarr.data();
        model.disc.sol.udgarray   = udgarr.data();

        // The entire solver: prepare (InitSolution + odg->Gauss) + PETSc SNES+GMRES + recover.
        const int reason = exasim::petsc::solve_steady<PdeModel>(model, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "[poisson2d] steady solve SNESConvergedReason=" << reason << "\n";

        model.writer.SaveSolutions(backend);              // dataout/outudg_np*.bin + outuhat
        if (model.vis.savemode > 0)
            model.writer.SaveParaview(backend, "", true); // dataout/*.vtu when vis is enabled
    }

    Kokkos::finalize();
    PetscFinalize();
    MPI_Finalize();
    return 0;
}
