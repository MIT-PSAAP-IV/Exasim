// Case app with the generated residual stages plugged in: the normal Exasim solve (exasimapp's command line), with
// SetResidualStages(mkstages::table()) registered first, so Residual<M> -- every Newton residual and every
// finite-difference GMRES matvec -- runs the generated kernels (residualstages.hpp). MK_STAGES=0 runs production.
// Like the harness, the Exasim unity TU is compiled here with the library's flags.
// usage: [flux run -n P] mk_app_<schedule> 1 <datain dir>/ <dataout prefix>
#include <string>
#include <vector>
#include "backend/Main/ExasimSolver.cpp"
#include <exasim/ExasimSolverSetup.hpp>
#ifdef MK_FRONTEND_PROVIDER      // MATLAB-frontend case: the generated kernels are compiled into this TU, model ID from app.bin
#include "frontendprovider.cpp"
#define MK_INIT_SOLVER(s, c, v, m) InitializeExasimSolver(s, c, v, m)
#else                            // language-frontend case: model library (frontend_model), ID 100
#define MK_INIT_SOLVER(s, c, v, m) InitializeExasimSolver(s, c, v, m, {100})
#endif
#include <cstdlib>
#include "mk_common.hpp"
#include "schedule.hpp"
#include "mk_stages.hpp"
#include "mk_pcstages.hpp"
#include "mk_prof.hpp"

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    MPI_Comm comm = MPI_COMM_NULL;
#endif
    const char* e = std::getenv("MK_STAGES");
    const bool on = !(e && e[0] == '0');
    if (on) SetResidualStages(mkprof::wrap(mkstages::table()));
    const char* ep = std::getenv("MK_PC_STAGES");      // preconditioner-build stages (precondstages.hpp); default: off
    const bool pc = ep && ep[0] == '1';
    if (pc) SetPrecondStages(mkprof::wrap(mkpc::table()));
    printf("mk_app: residual stages %s, preconditioner stages %s (%s)\n", on ? "GENERATED" : "production", pc ? "GENERATED" : "production", MK_SCHED);
    ExasimSolver solver;
    if (MK_INIT_SOLVER(solver, argc, argv, comm)) return 1;   // RunExasimSolver, with Solve() timed on its own
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    Kokkos::fence();
    Kokkos::Timer tm;
    const int err = solver.Solve();
    Kokkos::fence();
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    const double ts = tm.seconds();
    int rank = 0;
#ifdef HAVE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if (rank == 0) printf("mk_app: Solve() %.4f s (stages %s, pc %s)\n", ts, on ? "GENERATED" : "production", pc ? "GENERATED" : "production");
    if (rank == 0 && on) printf("mk_app: deferred GetW checks %ld, re-runs %ld (rank 0)\n", mkstages::st().w_calls, mkstages::st().w_redo);
    // MK_SAVE_FINAL=1: write the final udg/wdg of the owned elements, [npe, nc, ne1] header + data, per rank:
    //   <dataout prefix>final_udg_np<rank>.bin, ..._wdg_...   (fluxlab/solcompare.py reads them)
    if (const char* sf = std::getenv("MK_SAVE_FINAL"); sf && sf[0] == '1' && argc > 3) {
        auto* m = solver.Model(0);
        auto& c = m->disc.common;
        const int npe = (int)c.grid.npe, nc = (int)c.components.nc, ncw = (int)c.components.ncw, ne1 = (int)c.meshsizes.ne1;
        auto dump = [&](const char* tag, const double* dptr, int ncomp) {
            std::vector<double> h((size_t)npe * ncomp * ne1);
            hipMemcpy(h.data(), dptr, h.size() * sizeof(double), hipMemcpyDeviceToHost);
            const std::string fn = std::string(argv[3]) + "final_" + tag + "_np" + std::to_string(rank) + ".bin";
            FILE* f = std::fopen(fn.c_str(), "wb"); const double hd[3] = {(double)npe, (double)ncomp, (double)ne1};
            if (f) { std::fwrite(hd, sizeof(double), 3, f); std::fwrite(h.data(), sizeof(double), h.size(), f); std::fclose(f); } };
        dump("udg", m->disc.sol.udg, nc); if (ncw > 0) dump("wdg", m->disc.sol.wdg, ncw);
        if (rank == 0) printf("mk_app: final udg/wdg written (%d x %d x %d per rank 0)\n", npe, nc, ne1);
    }
    if (err) { solver.Finalize(); return err; }
    return solver.Finalize();
}
