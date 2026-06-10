// Out-of-tree GPU+MPI builtin-library consumer: runs a built-in PDE model on CUDA with MPI.
// pdeapp.txt selects the model (builtinmodelID) and platform ("cuda").
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
