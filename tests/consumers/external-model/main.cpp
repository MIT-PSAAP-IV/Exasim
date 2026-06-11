// External built-in model consumer: runs a model registered via
// exasim_add_external_builtin_model() (model ID 100, Poisson 2D).
// Built ONLY via find_package(Exasim) -- no source tree references.
//
// getBuiltInLibraryExasimDriverABI() is provided by the ext_model_100 target
// (ExternalModelProvider.cpp); do NOT include builtinlibprovider.hpp here.
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
