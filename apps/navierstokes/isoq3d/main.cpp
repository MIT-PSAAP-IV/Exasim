// External built-in model (ID 108, registered via
// exasim_add_external_builtin_model): getBuiltInLibraryExasimDriverABI() comes
// from the generated provider.
#include <exasim/ExasimSolverSetup.hpp>

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
