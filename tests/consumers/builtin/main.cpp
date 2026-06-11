// Out-of-tree builtin-library consumer test: run a built-in PDE model selected by
// builtinmodelID in pdeapp.txt. Built ONLY via find_package(Exasim) -- no source tree.
#include <exasim/ExasimSolverSetup.hpp>
#include <exasim/builtinlibprovider.hpp>

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
