// One-time CPU setup entry shared by native frontend and Text2Code cases.
// This program never generates model kernels or starts a PDE solve.
#include <exasim/operators.hpp>
#include "backend/Preprocessing/hdgldgregions.hpp"

namespace {
struct ExpressionCleanup {
    Mesh& mesh;
    ~ExpressionCleanup() {
        if (mesh.nbndexpr > 0 && mesh.boundaryExprs)
            freeCharArray(mesh.boundaryExprs, mesh.nbndexpr);
        if (mesh.nbndexpr > 0 && mesh.curvedBoundaryExprs)
            freeCharArray(mesh.curvedBoundaryExprs, mesh.nbndexpr);
        if (mesh.nprdexpr > 0 && mesh.periodicExprs1)
            freeCharArray(mesh.periodicExprs1, mesh.nprdexpr*mesh.nprdcom);
        if (mesh.nprdexpr > 0 && mesh.periodicExprs2)
            freeCharArray(mesh.periodicExprs2, mesh.nprdexpr*mesh.nprdcom);
    }
};
}

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: exasim_hdgldg_preprocess pdeapp.txt selection.txt NEW_ABSOLUTE_OUTDIR [--native-metadata]\n";
        return 2;
    }
    int status = 0;
    Kokkos::initialize(argc, argv);
    try {
        namespace rs = exasim::hdgldg;
        const bool nativeMetadata = argc == 5;
        if (nativeMetadata && std::string(argv[4]) != "--native-metadata")
            throw std::invalid_argument("unknown rank-separated preprocessing option");
        const std::filesystem::path out(argv[3]);
        if (!out.is_absolute() || std::filesystem::exists(out))
            throw std::invalid_argument("rank-separated output must be a new absolute directory");
        std::ifstream manifest(argv[2]);
        if (!manifest) throw std::invalid_argument("cannot open rank-separated selection manifest");
        const auto input = rs::readSpec(manifest);
        // Native frontends already know model dimensions and generate their
        // own kernels. They need only the common config/mesh reader, not a
        // Text2Code model export or kernel-generation pass.
        std::unique_ptr<CPreprocessing> preprocessing;
        if (nativeMetadata) {
            InputParams parameters = parseInputFile(argv[1], 0);
            for (const char* name : {"ncu", "ncv", "ncw"})
                if (!parameters.intParams.count(name))
                    throw std::invalid_argument(std::string("native rank-separated metadata must specify ")+name);
            PDE model = initializePDE(parameters, 0);
            preprocessing = std::make_unique<CPreprocessing>(model, parameters, ParsedSpec{}, 0, 1);
        } else preprocessing = std::make_unique<CPreprocessing>(argv[1], 0, 1);
        CPreprocessing& prep = *preprocessing;
        prep.mesh = initializeMesh(prep.params, prep.pde);
        ExpressionCleanup cleanup{prep.mesh};
        prep.pde.nd = prep.pde.ncx = prep.mesh.dim;
        if (nativeMetadata || useBuiltInAppMetadata(prep.pde)) validateBuiltInAppMetadata(prep.pde);
        else applyParsedSpecMetadata(prep.pde, prep.spec);
        finalizePDEModelSizes(prep.pde);
        pdeFinalizeDerived(prep.pde);
        prep.master = initializeMaster(prep.pde, prep.mesh);
        uniformRefineMesh(prep.mesh, prep.pde, prep.master, 0);
        buildMesh(prep.mesh, prep.pde, prep.master);
        const auto topology = rs::topologyFromPreparedMesh(prep.mesh);
        const auto plan = rs::selectAndBuildPlan(topology, input);
        // Extract both regions before writing either: no half-valid setup on
        // unsupported fields. Original expressions remain owned by prep.mesh.
        auto h = rs::extractPreparedRegion(prep.mesh, prep.master, prep.pde, plan, 1);
        auto l = rs::extractPreparedRegion(prep.mesh, prep.master, prep.pde, plan, 0);
        std::filesystem::create_directories(out);
        rs::writePreparedRegion(h, prep.master, plan, (out/"H").string());
        rs::writePreparedRegion(l, prep.master, plan, (out/"L").string());
        std::ofstream accepted(out/"selection.txt");
        rs::writeSpec(accepted, input);
        if (!accepted) throw std::runtime_error("cannot write accepted selection manifest");
        std::cout << "rank-separated preprocessing complete: " << plan.interfaces.size()
                  << " HL faces, H=" << h.mesh.ne << ", L=" << l.mesh.ne << '\n';
    } catch (const std::exception& e) {
        std::cerr << "rank-separated preprocessing: " << e.what() << '\n';
        status = 1;
    }
    Kokkos::finalize();
    return status;
}
