// Ordinary installed-model application; the coupled launch selects --hdgldg.
#include <exasim/ExasimSolverSetup.hpp>

int main(int argc,char** argv)
{
    ExasimSolver solver;
    return RunExasimSolver(solver,argc,argv,MPI_COMM_WORLD,{100});
}
