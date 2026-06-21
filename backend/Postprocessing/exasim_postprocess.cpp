/*
Standalone Exasim postprocessing executable using the ExasimSolver facade.

This entry point intentionally avoids directly including backend implementation
.cpp files. It uses ExasimSolverSetup.hpp to initialize the solver, configure
the active model provider, initialize models, and then run the solver's
runmode-based postprocessing path.

Example:

  mpirun -np 4 exasim_postprocess 1 datain/ dataout/out 0 0

The command-line interface matches the legacy postprocess.cpp executable:

  exasim_postprocess nummodels InputFile(s) OutputFile(s)
                    [restart] [postmode] [nsca] [nvec] [nten] [nsurf] [nvqoi]
*/

#include "ExasimSolverSetup.hpp"

int main(int argc, char** argv)
{
#if defined(HAVE_MPI) || defined(_MPI)
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    MPI_Comm comm = MPI_COMM_NULL;
#endif

    ExasimSolver solver;

    int err = InitializeExasimPostprocessor(solver, argc, argv, comm);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = solver.Postprocess();
    if (err) {
        solver.Finalize();
        return err;
    }

    return solver.Finalize();
}
