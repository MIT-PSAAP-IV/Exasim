// The native implementation is included only to access its already-owned
// model for this test's output cadence. No replacement integrator is used.
#include "backend/Main/ExasimSolver.cpp"
#define HDGLDG_PUBLIC_NO_MAIN
#include "public.cpp"

int main(int argc,char** argv)
{
    ExasimSolver solver;
    int status=InitializeExasimSolver(solver,argc,argv,MPI_COMM_WORLD);
    if(!status) {
        // Native saveRestart is otherwise fixed to a much longer interval.
        // This changes only checkpoint frequency, not equations or controls.
        solver.Model(0)->disc.common.outputparams.saveRestart=1;
        status=solver.Solve();
    }
    solver.Finalize();
    return status;
}
