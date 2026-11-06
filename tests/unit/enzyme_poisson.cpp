// Test a complete solve with handwritten C++ physics or generated physics.
// Both providers must supply the Enzyme JVP callbacks.
#include "ExasimSolverSetup.hpp"
#ifdef ENZYME_TEST_BUILTIN
#include "builtinlibprovider.hpp"
#else
#include "../../apps/poisson/poisson2d/my_model.hpp"
#include "modelprovider.hpp"
#endif

int main(int argc, char** argv)
{
    ExasimSolver solver;
    return RunExasimSolver(solver, argc, argv, MPI_COMM_NULL);
}
