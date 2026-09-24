#include "ExasimSolverSetup.hpp"

// External built-in model (ID 107, registered via
// exasim_add_external_builtin_model): getBuiltInLibraryExasimDriverABI() comes
// from the generated provider -- do NOT include my_model.hpp / modelprovider.hpp here.
// (my_model.hpp is the checked-in text2code output for reference; the build
// regenerates it from pdeapp107.txt.)

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    MPI_Comm comm = MPI_COMM_NULL;
#endif

    ExasimSolver solver;
    return RunExasimSolver(solver, argc, argv, comm);
}
