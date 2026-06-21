// cmake -S . -B build -DExasim_DIR=/path/to/prefix
// cmake --build build
// mpirun -np N build/exasimapp 1 datain/ dataout/out

#include <string>
#include <vector>

#include <ExasimSolverSetup.hpp>
#include "frontendprovider.cpp"

namespace {

int RunExasimPostprocess(ExasimSolver& solver, int argc, char** argv, MPI_Comm comm)
{
    int err = InitializeExasimPostprocessor(solver, argc, argv, comm);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = solver.Postprocess();
    if (err) {
        solver.Finalize();
        return err;
    }

    return solver.Finalize();
}

int RunExasimSolve(ExasimSolver& solver, int argc, char** argv, MPI_Comm comm)
{
    return RunExasimSolver(solver, argc, argv, comm);
}

} // namespace

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    MPI_Comm comm = MPI_COMM_NULL;
#endif
    ExasimSolver solver;

    if (argc > 1 && std::string(argv[1]) == "postprocess") {
        std::vector<char*> shifted;
        shifted.push_back(argv[0]);
        for (int i = 2; i < argc; i++)
            shifted.push_back(argv[i]);
        return RunExasimPostprocess(solver, static_cast<int>(shifted.size()), shifted.data(), comm);
    }

    if (argc > 1 && std::string(argv[1]) == "solve") {
        std::vector<char*> shifted;
        shifted.push_back(argv[0]);
        for (int i = 2; i < argc; i++)
            shifted.push_back(argv[i]);
        return RunExasimSolve(solver, static_cast<int>(shifted.size()), shifted.data(), comm);
    }

    return RunExasimSolve(solver, argc, argv, comm);
}
