// Out-of-tree GPU builtin-library consumer: runs a built-in PDE model on the CUDA backend.
// pdeapp.txt selects the model (builtinmodelID) and platform ("cuda").
#include <exasim/ExasimSolverSetup.hpp>
#include <exasim/builtinlibprovider.hpp>

int main(int argc, char** argv)
{
    MPI_Comm comm = MPI_COMM_NULL;
    ExasimSolver solver;
    return RunExasimSolver(solver, argc, argv, comm);
}
