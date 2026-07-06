// Out-of-tree builtin-library consumer test: run a built-in PDE model selected by
// builtinmodelID in pdeapp.txt. Built ONLY via find_package(Exasim) -- no source tree.
#include <exasim/ExasimSolverSetup.hpp>
#include <exasim/builtinlibprovider.hpp>
#include <cstdlib>

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    MPI_Comm comm = MPI_COMM_NULL;
#endif
    ExasimSolver solver;

    // Demonstration of multiple named QoI instances in one program (C2): when
    // EXASIM_QOI_DEMO is set, replace the default Domain/Boundary instances with
    // independently-named ones, each owning a slice of the model's QoI output. Uses only
    // the public ExasimSolver API (no knowledge of model internals); out-of-range
    // registrations are rejected by AddQoI, so blind registration is safe.
    if (std::getenv("EXASIM_QOI_DEMO")) {
        const auto& abi = SelectExasimDriverABI();
        int err = solver.InitializeEnvironment(argc, argv, comm);
        if (!err) err = solver.ParseInputs(argc, argv, abi);
        if (!err) err = ConfigureModelDefinitions(solver, abi);
        if (!err) err = solver.InitializeModels();
        if (!err) {
            solver.ClearQoI(0);
            solver.AddQoI(0, "KineticEnergy", 0, /*boundary*/ 0, /*offset*/ 0, /*ncomp*/ 1);
            solver.AddQoI(0, "Dissipation",   0, /*boundary*/ 0, /*offset*/ 1, /*ncomp*/ 1);
            solver.AddQoI(0, "WallFlux",      1, /*boundary*/ 0, /*offset*/ 0, /*ncomp*/ 1);
            err = solver.Solve();
        }
        solver.Finalize();
        return err;
    }

    return RunExasimSolver(solver, argc, argv, comm);
}
