/*
Build this app from apps/builtinlibrary after configuring Exasim:

  cmake -B build -DExasim_DIR=/path/to/Exasim
  cmake --build build

Run with an input file that uses a built-in model ID:

  mpirun -np 4 build/exasimapp /path/to/pdeapp.txt
*/

#include "ExasimSolverSetup.hpp"
#include "../../backend/Model/BuiltIn/builtinlibprovider.cpp"

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
