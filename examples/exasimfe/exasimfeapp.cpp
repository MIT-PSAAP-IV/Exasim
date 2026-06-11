// cmake -S . -B build
// cmake --build build

#include <iostream>
#include <stdexcept>
#include <vector>

#include "ExasimSolver.hpp"
#include "kernels/frontendprovider.cpp"

namespace {

const ExasimDriverABI& SelectExasimDriverABI()
{
    return getFrontendGeneratedExasimDriverABI();
}

const char* SelectExasimDriverProviderName()
{
    return "FrontendGenerated";
}

void PrintModelProvider(const int modelnumber, const int builtinmodelID)
{
    int rank = 0;
#ifdef HAVE_MPI
    MPI_Comm_rank(EXASIM_COMM_WORLD, &rank);
#endif

    if (rank == 0) {
        std::cout << "Model " << modelnumber
                  << ": provider = " << SelectExasimDriverProviderName()
                  << ", builtinmodelID = " << builtinmodelID << std::endl;
    }
}

int ConfigureModelDefinitions(ExasimSolver& solver)
{
    for (int i = 0; i < solver.NumModelDefinitions(); i++) {
        const int builtinmodelID = solver.BuiltinModelID(i);
        int err = 0;
        try {
            err = solver.SetModelDefinition(i, builtinmodelID, SelectExasimDriverABI());
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return 1;
        }
        if (err) return err;
        PrintModelProvider(i, builtinmodelID);
    }

    return 0;
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

    int err = solver.InitializeEnvironment(argc, argv, comm);
    if (err) return err;

    err = solver.ParseInputs(argc, argv);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = ConfigureModelDefinitions(solver);
    if (err) {
        solver.Finalize();
        return err;
    }

    err = solver.InitializeModels();
    if (err) return err;

    err = solver.Solve();
    if (err) {
        solver.Finalize();
        return err;
    }

    return solver.Finalize();
}
