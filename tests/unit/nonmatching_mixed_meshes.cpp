// [nonmatching] Phase 10: mixed triangle/quadrilateral setup, search and
// owner-test transport. Reuse the established synthetic mesh fixture.
#define NONMATCHING_SETUP_NO_MAIN
#include "nonmatching_setup.cpp"
#undef NONMATCHING_SETUP_NO_MAIN

namespace {

struct MixedOutcome {
    double maxOwnerTestError = 0;
    double maxOldDonorTestError = 0;
    double maxTracePolynomialError = 0;
    long long points = 0;
    long long outsidePoints = 0;
};

MixedOutcome run_mixed_case(int porder, bool model0Triangle, int model0Ranks,
        bool curved, double gap,
        bool extension = false)
{
    check(model0Ranks > 0 && model0Ranks < worldSize, "mixed: valid model rank split");
    const int model = worldRank < model0Ranks ? 0 : 1;
    const int modelBegin = model == 0 ? 0 : model0Ranks;
    const int modelRanks = model == 0 ? model0Ranks : worldSize - model0Ranks;
    const int part = worldRank - modelBegin;
    const bool localTriangle = model == 0 ? model0Triangle : !model0Triangle;

    MasterData triangle = make_master(0, porder);
    MasterData quad = make_master(1, porder);
    MasterData& localMaster = localTriangle ? triangle : quad;
    const MasterData& ownerMaster = localTriangle ? quad : triangle;

    ModelMesh left, right;
    left.tri = model0Triangle;
    right.tri = !model0Triangle;
    left.xs = uniform(0.0, 0.5, 4);
    right.xs = uniform(0.5 + gap, 1.0 + gap, 3);
    left.ys = uniform(0.0, extension ? 1.125 : 1.0, 10);
    right.ys = {0.0, 0.09, 0.23, 0.37, 0.52, 0.68, 0.81, 1.0};
    left.xiface = 0.5; left.xfar = 0.0;
    right.xiface = 0.5 + gap; right.xfar = 1.0 + gap;
    if (curved) { left.amp = 0.007; right.amp = 0.011; }

    RankStructs rs;
    build_rank_structs(rs, model == 0 ? left : right, localMaster,
                       part, modelRanks, 1);
    Int interfaceMap = 0;
    rs.app.interfacefluxmap = &interfaceMap;
    std::vector<Int> intfaces;
    for (Int e = 0; e < rs.mesh.ndims[1]; ++e)
        for (Int lf = 0; lf < localMaster.nfe; ++lf)
            if (rs.mesh.bf[lf + localMaster.nfe * e] == 2)
                intfaces.push_back(lf + localMaster.nfe * e);
    rs.mesh.intfaces = intfaces.data();
    rs.mesh.szintfaces = static_cast<Int>(intfaces.size());

    commonstruct common;
    common.components.nc = 1;
    common.components.ncu = 1;
    common.components.ncq = 0;
    common.components.ncx = 2;
    common.components.nco = 0;
    common.components.ncw = 0;
    common.meshsizes.ne = rs.mesh.ndims[1];
    common.meshsizes.ne1 = rs.mesh.ndims[1];
    common.meshsizes.nfe = localMaster.nfe;

    nonmatchingdata nm;
    std::unique_ptr<NonmatchingRuntime<dstype, Int>> runtime;
    std::unique_ptr<OwnerTestData<dstype, Int>> sidecar;
    DonorSearchParams prm;
    DonorSearchResult<dstype, Int> result;
    check(rebuild_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol, common,
              static_cast<Int>(worldSize), static_cast<Int>(worldRank),
              static_cast<Int>(modelBegin), nm, runtime, Int(0), &result, prm,
              true, &sidecar), "mixed: setup ran");
    check(runtime != nullptr && sidecar != nullptr, "mixed: transactional sidecars exist");

    const nonmatchinginterface& owner = nm.dir[model];
    const nonmatchinginterface& donor = nm.dir[1 - model];
    check(owner.npe == localMaster.npe && owner.nfe == localMaster.nfe,
          "mixed: owner dimensions are role-local");
    check(donor.npe == localMaster.npe && donor.nfe == localMaster.nfe,
          "mixed: donor dimensions are role-local");
    check(owner.npf == localMaster.npf && donor.npf == localMaster.npf,
          "mixed: shared trace dimensions");
    check(sidecar->dir[1 - model].enabled == 1,
          "mixed: donor direction requests owner tests");
    check(sidecar->dir[1 - model].values.size() ==
              static_cast<std::size_t>(donor.npf) * donor.npts,
          "mixed: one owner-test payload per donor point");
    check(runtime->dir[1 - model].ptest.size ==
              static_cast<Int>(sidecar->dir[1 - model].values.size()),
          "mixed: runtime mirrors owner tests");

    int localQuadFace3 = 0;
    if (model0Triangle && model == 1 && donor.isdonor)
        for (Int p = 0; p < donor.npts; ++p)
            localQuadFace3 = std::max(localQuadFace3, donor.pa[p] == 3 ? 1 : 0);
    int globalQuadFace3 = 0;
    MPI_Allreduce(&localQuadFace3, &globalQuadFace3, 1, MPI_INT, MPI_MAX,
                  EXASIM_COMM_WORLD);
    if (model0Triangle)
        check(globalQuadFace3 == 1,
              "mixed: triangle owners receive quadrilateral donor local face 3");

    MixedOutcome out;
    out.points = donor.npts;
    for (Int p = 0; p < donor.npts; ++p) {
        const Int g = donor.pg[p];
        double trace = 0;
        for (Int j = 0; j < donor.npf; ++j) {
            const double transported = sidecar->dir[1 - model].values[
                    static_cast<std::size_t>(j) + static_cast<std::size_t>(donor.npf) * p];
            const double expected = ownerMaster.shapfgt[g + ownerMaster.ngf * j];
            const double oldWrong = localMaster.shapfgt[g + localMaster.ngf * j];
            out.maxOwnerTestError = std::max(out.maxOwnerTestError,
                                             std::abs(transported - expected));
            out.maxOldDonorTestError = std::max(out.maxOldDonorTestError,
                                                std::abs(transported - oldWrong));
            check(runtime->dir[1 - model].ptest.data[j + donor.npf * p] == transported,
                  "mixed: runtime owner-test copy is bitwise");
            trace += donor.ppsi[j + donor.npf * p] *
                     std::pow(localMaster.xpf[j], porder);
        }
        out.maxTracePolynomialError = std::max(out.maxTracePolynomialError,
                std::abs(trace - std::pow(donor.pxi[p], porder)));
        bool outside = false;
        for (Int d = 0; d < donor.nd; ++d) {
            const dstype zeta = donor.pzeta[p + static_cast<std::size_t>(donor.npts) * d];
            outside = outside || zeta < -1.0e-10 || zeta > 1.0 + 1.0e-10;
        }
        out.outsidePoints += outside;
    }
    check(out.maxOwnerTestError <= 2.0e-14,
          "mixed: D6 uses the owner face nodal basis");
    check(out.maxTracePolynomialError <= 1.0e-12,
          "mixed: donor trace basis reproduces degree-p data");
    if (porder >= 3 && donor.npts > 0)
        check(out.maxOldDonorTestError >= 1.0e-3,
              "mixed: old donor-master test basis is measurably wrong");

    long long outsideGlobal = 0;
    MPI_Allreduce(&out.outsidePoints, &outsideGlobal, 1, MPI_LONG_LONG, MPI_SUM,
                  EXASIM_COMM_WORLD);
    if (extension)
        check(outsideGlobal > 0, "mixed: extension exercises OUTSIDE extrapolation");

    free_nonmatching_data(nm);
    runtime.reset();
    sidecar.reset();
    CPUFREE(common.requests);
    CPUFREE(common.statuses);
    return out;
}

void run_incompatible_case(const std::string& mode)
{
    const int model = worldRank < worldSize / 2 ? 0 : 1;
    MasterData master = make_master(model == 0 ? 0 : 1, 3);
    if (model == 1 && mode == "porder") master.ndims[3]++;
    if (model == 1 && mode == "pgauss") master.ndims[4]++;
    if (model == 1 && mode == "npf") master.ndims[6]++;
    if (model == 1 && mode == "ngf") master.ndims[8]++;
    if (model == 1 && mode == "quadrature") master.gwf[0] *= 1.01;
    if (model == 1 && mode == "quadrature_local" &&
        worldRank != worldSize / 2) master.gwf[0] *= 1.01;

    ModelMesh mesh;
    mesh.tri = model == 0;
    mesh.xs = model == 0 ? uniform(0.0, 0.5, 3) : uniform(0.5, 1.0, 3);
    mesh.ys = uniform(0.0, 1.0, 5);
    mesh.xiface = 0.5;
    mesh.xfar = model == 0 ? 0.0 : 1.0;
    RankStructs rs;
    build_rank_structs(rs, mesh, master, worldRank - model * (worldSize / 2),
                       worldSize / 2, 1);
    nonmatchingdata nm;
    OwnerTestData<dstype, Int> sidecar;
    setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol,
            Int(worldSize), Int(worldRank), Int(model * (worldSize / 2)), nm,
            static_cast<DonorSearchResult<dstype, Int>*>(nullptr),
            DonorSearchParams(), false, nullptr,
            mode == "sidecar" ? nullptr : &sidecar);
    check(false, "mixed incompatibility must fail");
}

void run_cost_case(bool model0Triangle)
{
    const int model0Ranks = worldSize / 2;
    const int model = worldRank < model0Ranks ? 0 : 1;
    const int modelBegin = model == 0 ? 0 : model0Ranks;
    const int modelRanks = model == 0 ? model0Ranks : worldSize - model0Ranks;
    const bool localTriangle = model == 0 ? model0Triangle : !model0Triangle;
    MasterData triangle = make_master(0, 3), quad = make_master(1, 3);
    MasterData& master = localTriangle ? triangle : quad;
    ModelMesh left, right;
    left.tri = model0Triangle; right.tri = !model0Triangle;
    left.xs = uniform(0.0, 0.5, 4); right.xs = uniform(0.5, 1.0, 3);
    left.ys = uniform(0.0, 1.0, 10);
    right.ys = {0.0, 0.09, 0.23, 0.37, 0.52, 0.68, 0.81, 1.0};
    left.xiface = right.xiface = 0.5; left.xfar = 0; right.xfar = 1;
    RankStructs rs;
    build_rank_structs(rs, model == 0 ? left : right, master,
                       worldRank - modelBegin, modelRanks, 1);
    nonmatchingdata nm;
    OwnerTestData<dstype, Int> sidecar;
    SearchStatistics stats;
    MPI_Barrier(EXASIM_COMM_WORLD);
    const double start = MPI_Wtime();
    check(setup_nonmatching_interface(rs.app, rs.master, rs.mesh, rs.sol,
              Int(worldSize), Int(worldRank), Int(modelBegin), nm,
              static_cast<DonorSearchResult<dstype, Int>*>(nullptr),
              DonorSearchParams(), true, &stats, &sidecar), "mixed cost: setup ran");
    const double elapsed = MPI_Wtime() - start;
    long long local[6] = {stats.ownerPoints, stats.facesPerPointSum, 0, 0, 0, 0};
    for (int d = 0; d < 2; ++d) {
        local[2] += nm.dir[d].npts;
        local[3] += nm.dir[d].nslots;
        local[4] += nm.dir[d].npairs;
        local[5] += static_cast<long long>(sidecar.dir[d].values.size()) * sizeof(dstype);
    }
    long long total[6] = {};
    double setupMax = 0;
    MPI_Reduce(local, total, 6, MPI_LONG_LONG, MPI_SUM, 0, EXASIM_COMM_WORLD);
    MPI_Reduce(&elapsed, &setupMax, 1, MPI_DOUBLE, MPI_MAX, 0, EXASIM_COMM_WORLD);
    if (worldRank == 0)
        std::printf("MIXED COST %s (%d+%d): candidate faces/owner point %.3f, points %lld, slots %lld, "
                    "pairs %lld, owner-test host bytes %lld, setup max %.3f ms\n",
                    model0Triangle ? "tri-quad" : "quad-tri", model0Ranks,
                    worldSize - model0Ranks,
                    total[0] ? static_cast<double>(total[1]) / total[0] : 0.0,
                    total[2], total[3], total[4], total[5], 1.0e3 * setupMax);
    free_nonmatching_data(nm);
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (worldSize < 2 || (worldSize & 1)) {
        if (worldRank == 0) std::printf("nonmatching_mixed_meshes needs an even rank count\n");
        MPI_Finalize();
        return 1;
    }
    if (argc == 3 && std::string(argv[1]) == "--incompatible") {
        run_incompatible_case(argv[2]);
        MPI_Finalize();
        return 1;
    }

    for (int p = 1; p <= 4; ++p)
        for (int orientation = 0; orientation < 2; ++orientation)
            run_mixed_case(p, orientation == 0, worldSize / 2,
                           false, 0.0);

    // Curvature, gap/overlap and OUTSIDE coverage.
    run_mixed_case(2, true, worldSize / 2, true, 0.0);
    run_mixed_case(2, false, worldSize / 2, false, 0.01);
    run_mixed_case(2, true, worldSize / 2, false, -0.005);
    run_mixed_case(2, true, worldSize / 2, false, 0.0, true);

    if (worldSize >= 4) {
        run_mixed_case(3, true, 1, false, 0.0);
        run_mixed_case(3, false, worldSize - 1, false, 0.0);
    }
    run_cost_case(true);
    run_cost_case(false);

    int total = 0;
    MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (worldRank == 0)
        std::printf("%s nonmatching_mixed_meshes (%d ranks)\n",
                    total == 0 ? "PASS" : "FAIL", worldSize);
    MPI_Finalize();
    return total == 0 ? 0 : 1;
}
