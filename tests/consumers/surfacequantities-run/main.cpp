// Out-of-tree consumer: run the builtin Poisson model with SurfaceQuantities saved on
// the ibs boundaries (checked by check.sh).
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
