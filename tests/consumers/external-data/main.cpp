// Tests the HDG fext interface and recovers tractions via Fint.
// IntializeMeshInterface/setInterfaceFluxes supply uext on the boundary
// (ncuext=2); getInterfaceFluxes recovers sigma.n after the solve
// (ncuint=2).  Manufactured solution ux=x+y, uy=x-y on [0,1]^2
// (zero body force, constant stress sigma = 2*mu*[[1,1],[1,-1]]).

#include <exasim/ExasimSolverSetup.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {
    constexpr double mu = 0.5;    // shear modulus (mu[0] in physicsparam)
    constexpr double tol = 1e-12;  // error threshold
}

int main(int argc, char** argv)
{
    ExasimSolver solver;
    int err = 0;

#ifdef HAVE_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    MPI_Comm comm = MPI_COMM_NULL;
#endif

    err = solver.InitializeEnvironment(argc, argv, comm);
    if (err != 0) { std::fprintf(stderr, "FAIL: InitializeEnvironment\n"); return err; }

    err = solver.ParseInputs(argc, argv);
    if (err != 0) { std::fprintf(stderr, "FAIL: ParseInputs\n"); solver.Finalize(); return err; }

    err = solver.SetModelDefinition(0, solver.BuiltinModelID(0), SelectExasimDriverABI());
    if (err != 0) { std::fprintf(stderr, "FAIL: SetModelDefinition\n"); solver.Finalize(); return err; }

    err = solver.InitializeModels();
    if (err != 0) { std::fprintf(stderr, "FAIL: InitializeModels\n"); return err; }

    const int ncuext = 2;   // ux, uy supplied on boundary
    const int ncuint = 2;   // sigma.n recovered from Fint
    const int ibc     = 0;  // FextCall=1 for BC type 1
    const int comperm = 0;  // no communication permutation
    const int offset  = 0;  // no offset

    err = solver.IntializeMeshInterface(0, ncuext, ncuint, ibc, comperm, offset, comm);
    if (err != 0) { std::fprintf(stderr, "FAIL: IntializeMeshInterface\n"); return err; }

    const auto points = solver.getInterfacePoints();
    const int npoints = static_cast<int>(points.size());

    // Build normals from point position.  The Fint convention on boundary
    // faces uses the inward normal (pointing into the domain).
    std::vector<double> nx(npoints), ny(npoints);
    for (int i = 0; i < npoints; ++i) {
        const double x = points[i].x, y = points[i].y;
        if      (std::fabs(x) < tol)            { nx[i] =  1; ny[i] =  0; }
        else if (std::fabs(x - 1.0) < tol)      { nx[i] = -1; ny[i] =  0; }
        else if (std::fabs(y) < tol)            { nx[i] =  0; ny[i] =  1; }
        else if (std::fabs(y - 1.0) < tol)      { nx[i] =  0; ny[i] = -1; }
        else {
            std::fprintf(stderr, "FAIL: point not on boundary: (%.15g, %.15g)\n", x, y);
            return 1;
        }
    }

    // Analytical traction: t = sigma.n with sigma = 2*mu*[[1,1],[1,-1]]
    // (constant stress for u=x+y, u=x-y with tr(eps)=0, independent of lambda).
    // t_0 = 2*mu*(nx+ny),  t_1 = 2*mu*(nx-ny).
    std::vector<double> trac_exact(2 * npoints);
    for (int i = 0; i < npoints; ++i) {
        trac_exact[0 * npoints + i] = 2.0 * mu * (nx[i] + ny[i]);
        trac_exact[1 * npoints + i] = 2.0 * mu * (nx[i] - ny[i]);
    }

    std::vector<dstype> recv_flux(2 * npoints);
    for (int i = 0; i < npoints; ++i) {
        const dstype x = points[i].x, y = points[i].y;
        recv_flux[0 * npoints + i] = (x + y);
        recv_flux[1 * npoints + i] = (x - y);
    }

    solver.setInterfaceFluxes(recv_flux);
    std::printf("  fext: %d interface points\n", npoints);

    err = solver.Solve();
    if (err != 0) { std::fprintf(stderr, "FAIL: Solve\n"); return err; }

    // Retrieve the computed traction from the Fint function.
    std::vector<dstype> trac_comp;
    solver.getInterfaceFluxes(trac_comp);

    double max_err = 0.0;
    for (int i = 0; i < npoints; ++i) {
        const double e0 = std::fabs(trac_comp[0 * npoints + i] - trac_exact[0 * npoints + i]);
        const double e1 = std::fabs(trac_comp[1 * npoints + i] - trac_exact[1 * npoints + i]);
        max_err = std::max({max_err, e0, e1});
    }
    std::printf("  fint: max traction error = %.2e%s\n", max_err, max_err > tol ? "  FAIL" : "  PASS");
    if (max_err > tol) return 1;

    err = solver.Finalize();
    if (err != 0) { std::fprintf(stderr, "FAIL: Finalize\n"); return err; }

    return 0;
}
