#include <iostream>
#include <cmath>
#include <vector>

#include "ExasimSolver.hpp"
#include "builtinlibprovider.cpp"

void update_boundary_coordinates(ExasimSolver& solver)
{
    // Inclined boundary (ibc=2): the plate surface from (-0.1100, 0.0001) to (0.3492, 0.2471)
    const double x0 = -0.1100, y0 = 0.0001;
    const double x1 =  0.3492, y1 = 0.2471;
    double dx = x1 - x0, dy = y1 - y0;
    double L = std::sqrt(dx*dx + dy*dy);

    // Unit normal pointing into the flow (upward, toward exterior of domain)
    double nx = -dy / L;
    double ny =  dx / L;

    solver.IntializeMeshInterface(
        /*modelnumber=*/0, /*ncuext=*/2, /*ncuint=*/2,
        /*ibc=*/2, /*comperm=*/0, /*offset=*/0, MPI_COMM_NULL);

    auto pts = solver.getInterfacePoints();
    int npts = solver.ngf * solver.nfaces;
    std::vector<dstype> uext(npts * 2, 0.0);

    double amplitude = 0.001;
    for (int i = 0; i < npts; i++) {
        double x = pts[i].x, y = pts[i].y;
        // Only apply bump to points geometrically on the inclined boundary
        if (!(std::abs(y - (1e-4 + 0.538*(x + 0.1101))) < 2e-3))
            continue;
        double s = std::sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0));
        double xi = s / L;
        double umag = amplitude * std::sin(M_PI * xi);
        uext[i + 0*npts] = umag * nx;
        uext[i + 1*npts] = umag * ny;
    }
    solver.setInterfaceFluxes(uext);
}

int main(int argc, char** argv)
{
    ExasimSolver solver;

    int err = solver.InitializeEnvironment(argc, argv, MPI_COMM_NULL);
    if (err) return err;

    err = solver.ParseInputs(argc, argv);
    if (err) return solver.Finalize();

    const ExasimDriverABI& abi = getBuiltInLibraryExasimDriverABI();
    for (int i = 0; i < solver.NumModelDefinitions(); i++)
        if (solver.SetModelDefinition(i, solver.BuiltinModelID(i), abi))
            return solver.Finalize();

    err = solver.InitializeModels();
    if (err) return solver.Finalize();

    update_boundary_coordinates(solver);

    err = solver.Solve();
    return solver.Finalize();
}
