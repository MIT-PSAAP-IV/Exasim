/*

Below are the intructions to compile and run main.cpp

cd /path/to/Exasim/apps/poisson/poisson2d

cmake -S . -B build -DExasim_DIR=/Path/to/Exasim/build \

cmake --build build

mpirun -np 4 build/exasimapp pdeapp.txt

*/

#include "ExasimSolverSetup.hpp"
#include "my_model.hpp"
#include "modelprovider.hpp"

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
