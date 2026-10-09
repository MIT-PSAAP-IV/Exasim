// Preconditioner-build harness: the LDG block-Jacobian (ComputeLDGPreconditioner -> [mpi]BlockJacobianLDG) on the real case,
// production vs a staged composition of the same functions with replaceable stages (mk_pcstages.hpp).
// The staged build runs BlockJacobianLDG's / mpiBlockJacobianLDG's exact sequence without the per-stage benchmark fences.
// Production K is not bitwise reproducible (fp64 atomics in the face assembly), so every comparison is reported next to
// production's own build-to-build difference.
// usage: [flux run -n P] mk_pc_<schedule> 1 <datain dir>/ <dataout prefix>
// output (rank 0): MKPCREF / MKPCNOISE / MKPC lines (times, max|dK|/max|K|), MKPCSTAGE per-stage times (slowest rank)
#include "backend/Main/ExasimSolver.cpp"
#include <exasim/ExasimSolverSetup.hpp>
#ifdef MK_FRONTEND_PROVIDER
#include "frontendprovider.cpp"
#define MK_INIT_SOLVER(s, c, v, m) InitializeExasimSolver(s, c, v, m)
#else
#define MK_INIT_SOLVER(s, c, v, m) InitializeExasimSolver(s, c, v, m, {100})
#endif
#include <algorithm>
#include <vector>
#include <cmath>
#include <functional>
#include <array>
#include <string>
#include <cstdlib>
#include "mk_common.hpp"
#include "schedule.hpp"
#include "mk_stages.hpp"
#include "mk_pcstages.hpp"
#include "mk_prof.hpp"

using M = exasim::detail::AbiAdapter;
using Stage = std::pair<std::string, std::function<void()>>;

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    MPI_Comm comm = MPI_COMM_NULL;
#endif
    ExasimSolver solver;
    if (MK_INIT_SOLVER(solver, argc, argv, comm)) { printf("MKERR init\n"); return 1; }
    int rc = 0;
    {
    const Int backend = solver.Backend();
    solver.InitializeSolution(0);
    auto* m = solver.Model(0);
    auto& d = m->disc; auto& sol = d.sol; auto& res = d.res; auto& app = d.app; auto& master = d.master; auto& mesh = d.mesh;
    auto& tmp = d.tmp; auto& common = d.common; auto handle = common.cublasHandle; auto& abi = d.driver_abi;
    const int rank = common.mpiRank, nprocs = common.mpiProcs; const bool root = rank == 0;
    const Int nc = common.components.nc, ncu = common.components.ncu, nco = common.components.nco, ncx = common.components.ncx,
              ncw = common.components.ncw, ncq = common.components.ncq, nd = common.grid.nd, npe = common.grid.npe,
              npf = common.grid.npf, nge = common.grid.nge, ne = common.meshsizes.ne, ne1 = common.meshsizes.ne1,
              nbe = common.meshsizes.nbe, nbf = common.meshsizes.nbf;
    const Int nbe0 = common.meshsizes.nbe0, nbe1 = common.meshsizes.nbe1, nbe2 = common.meshsizes.nbe2;
    (void)ncx; (void)ncq; (void)nge;
    auto gmax = [&](double v) {
#ifdef HAVE_MPI
        if (nprocs > 1) MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_MAX, comm);
#endif
        return v; };
    auto barrier = [&]() {
#ifdef HAVE_MPI
        if (nprocs > 1) MPI_Barrier(comm);
#endif
    };
    auto fail = [&](const char* msg) { printf("MKERR rank %d: %s\n", rank, msg); return 1; };
    std::vector<void*> frees;
    rc = [&]() -> int {
    if (root) printf("MKINFO nprocs=%d ne1=%d nbe=%d nbe0=%d nbe1=%d nbe2=%d preconditioner=%d curvedMesh=%d ncAV=%d\n", nprocs, (int)ne1, (int)nbe,
                     (int)nbe0, (int)nbe1, (int)nbe2, (int)common.solverparams.preconditioner, (int)common.grid.curvedMesh, (int)common.physicsparams.ncAV);
    if (common.spatialScheme != 0 || common.solverparams.preconditioner != 1) return fail("the case does not use the LDG block-Jacobian preconditioner");
    {
#include "mk_setup.inc"
    auto sync = []() { Kokkos::fence(); hipDeviceSynchronize(); };
    const Int n = npe*ncu; const size_t NK = (size_t)n*n*ne1;
    dstype* u = m->solv.sys.u;
    ArrayExtract(u, sol.udg, npe, nc, ne1, 0, npe, 0, ncu, 0, ne1);
    dstype* K = res.K;
    dstype *Kref = nullptr, *Kpre = nullptr;
    hipMalloc(&Kref, NK * sizeof(dstype)); hipMalloc(&Kpre, NK * sizeof(dstype)); frees.push_back(Kref); frees.push_back(Kpre);
    auto maxdiff = [&](const dstype* a, const dstype* b, double& md, double& mx) {
        double r1 = 0, r2 = 0;
        Kokkos::parallel_reduce("mkpc_maxdiff", Kokkos::RangePolicy<>(0, NK), KOKKOS_LAMBDA(const size_t i, double& x1, double& x2) {
            const double dd = fabs(a[i] - b[i]), v = fabs(b[i]);
            if (!(dd <= x1)) x1 = dd;   // NaN-propagating
            if (v > x2) x2 = v; }, Kokkos::Max<double>(r1), Kokkos::Max<double>(r2));
        md = gmax(r1); mx = gmax(r2); };

    // ---- staged build: BlockJacobianLDG / mpiBlockJacobianLDG sequence, stages swappable, no benchmark fences ----
    struct PC { bool gen_bminv = false, gen_fg = false, gen_inv = false, gen_gj = false, gen_elem = false, gen_face = false, gen_fused = false, gen_fused2 = false, gen_sl = false, gen_cross = false; };
    PrecondStageContext pctx{{sol, res, app, master, mesh, tmp, common, handle, backend}, abi};
    int req_count = 0;
    auto build_seq = [&](PC pc, bool with_inverse) {
        std::vector<Stage> s;
        if (nprocs > 1) {
#ifdef HAVE_MPI
            s.push_back({"state", [&]() {
                const Int bsz = npe*ncu;
                ArrayInsert(sol.udg, u, npe, nc, ne, 0, npe, 0, ncu, 0, ne1);
                GetArrayAtIndex(tmp.buffsend, sol.udg, mesh.elemsendind, bsz*common.nelemsend);
                hipDeviceSynchronize();
                Int psend = 0, precv = 0; req_count = 0;
                for (Int nn = 0; nn < common.nnbsd; nn++) { const Int ns = common.elemsendpts[nn]*bsz;
                    if (ns > 0) { MPI_Isend(&tmp.buffsend[psend], ns, mpi_type<dstype>(), common.nbsd[nn], 0, EXASIM_COMM_LOCAL, &common.requests[req_count]); psend += ns; req_count++; } }
                for (Int nn = 0; nn < common.nnbsd; nn++) { const Int nr = common.elemrecvpts[nn]*bsz;
                    if (nr > 0) { MPI_Irecv(&tmp.buffrecv[precv], nr, mpi_type<dstype>(), common.nbsd[nn], 0, EXASIM_COMM_LOCAL, &common.requests[req_count]); precv += nr; req_count++; } }
                GetUhat<M>(sol, res, app, master, mesh, tmp, common, handle, 0, nbf, backend);
                if (ncq > 0) GetQ(sol, res, app, master, mesh, tmp, common, handle, 0, nbe0, 0, nbf, backend);
                if (ncw > 0) GetW<M>(sol, res, app, master, mesh, tmp, common, handle, 0, nbe0, 0, nbf, backend);
                MPI_Waitall(req_count, common.requests, common.statuses);
                PutArrayAtIndex(sol.udg, tmp.buffrecv, mesh.elemrecvind, bsz*common.nelemrecv);
                GetUhat<M>(sol, res, app, master, mesh, tmp, common, handle, 0, nbf, backend);
                if (ncq > 0) GetQ(sol, res, app, master, mesh, tmp, common, handle, nbe0, nbe2, 0, nbf, backend);
                if (ncw > 0) GetW<M>(sol, res, app, master, mesh, tmp, common, handle, nbe0, nbe2, 0, nbf, backend);
                if (common.physicsparams.ncAV > 0 && common.physicsparams.frozenAVflag == 0) GetAv<M>(sol, res, app, master, mesh, tmp, common, handle, backend); }});
#endif
        } else {
            s.push_back({"state", [&]() {
                ArrayInsert(sol.udg, u, npe, nc, ne, 0, npe, 0, ncu, 0, ne);
                GetUhat<M>(sol, res, app, master, mesh, tmp, common, handle, 0, nbf, backend);
                if (ncq > 0) GetQ(sol, res, app, master, mesh, tmp, common, handle, 0, nbe, 0, nbf, backend);
                if (ncw > 0) GetW<M>(sol, res, app, master, mesh, tmp, common, handle, 0, nbe, 0, nbf, backend);
                if (common.physicsparams.ncAV > 0 && common.physicsparams.frozenAVflag == 0) GetAv<M>(sol, res, app, master, mesh, tmp, common, handle, backend); }});
        }
        const Int nb = nprocs > 1 ? nbe1 : nbe;
        // per block: element, element-face, trace, Schur, copy -- one stage each, looping over all blocks
        s.push_back({"blocks", [&, pc, nb]() {
            for (Int j = 0; j < nb; j++) {
                const Int e1 = common.eblks[3*j]-1, e2 = common.eblks[3*j+1], neb = e2 - e1;
                if (pc.gen_fused2) mkjac::elem_fused2(pctx, j, pc.gen_sl);
                else if (pc.gen_fused) mkjac::elem_fused(pctx, j);
                else if (pc.gen_elem) mkjac::elem(pctx, j);
                else uEquationElemBlock<M>(sol, res, app, master, mesh, tmp, common, handle, j, backend);
                if (pc.gen_fused || pc.gen_fused2) {}
                else if (pc.gen_face) mkjac::elemface(pctx, j);
                else uEquationElemFaceBlockLDG(sol, res, app, abi, master, mesh, tmp, common, handle, j, backend);
                uhatEquationElemFaceBlockLDG(sol, res, app, abi, master, mesh, tmp, common, handle, j, backend);
                if (pc.gen_bminv || pc.gen_fg) mkpc::schur(res, common, j, pc.gen_bminv, pc.gen_fg, handle, backend, pc.gen_sl);
                else uEquationSchurBlockLDG(sol, res, app, abi, master, mesh, tmp, common, handle, j, backend, nullptr);
                ArrayCopy(&K[n*n*e1], res.D, n*n*neb);
            } }});
        s.push_back({"cross", [&, pc]() { if (pc.gen_cross) mkjac::cross(pctx, K); else RuFaceCrossDerivOptimized(K, sol, res, app, abi, master, mesh, tmp, common); }});
        if (with_inverse)
            s.push_back({"inverse", [&, pc, nb]() {
                for (Int j = 0; j < nb; j++) {
                    const Int e1 = common.eblks[3*j]-1, e2 = common.eblks[3*j+1], neb = e2 - e1;
                    if (pc.gen_gj) mkpc::gjinverse(&K[n*n*e1], res.H, n, neb);
                    else if (pc.gen_inv) mkpc::inverse(handle, &K[n*n*e1], res.H, n, neb);
                    else Inverse(handle, &K[n*n*e1], res.H, res.ipiv, n, neb, backend);
                } }});
        return s; };
    auto run_seq = [](const std::vector<Stage>& s) { for (auto& st : s) st.second(); };
    auto timeit = [&](auto&& f, int reps) { f(); sync(); barrier(); std::vector<double> t;
        for (int r = 0; r < reps; r++) { barrier(); Kokkos::Timer tm; f(); sync(); barrier(); t.push_back(tm.seconds() * 1e3); }
        std::sort(t.begin(), t.end()); return gmax(t[t.size() / 2]); };
    auto seqtime = [&](const std::vector<Stage>& st, const char* tag, int reps) {
        std::vector<std::vector<double>> t(st.size());
        for (int r = -1; r < reps; r++) { sync(); barrier();
            for (size_t k = 0; k < st.size(); k++) { Kokkos::Timer tm; st[k].second(); sync(); if (r >= 0) t[k].push_back(tm.seconds() * 1e3); } }
        for (size_t k = 0; k < st.size(); k++) { std::sort(t[k].begin(), t[k].end());
            const double ms = gmax(t[k][t[k].size() / 2]); if (root) printf("MKPCSTAGE %s %s %s %.3f\n", MK_SCHED, tag, st[k].first.c_str(), ms); } };
    auto report = [&](const char* tag, double t, const dstype* ref) { double md, mx; maxdiff(K, ref, md, mx);
        if (root) printf("%s %s %.3f ms  max|dK|/max|K| = %.3e (max|K| %.3e)\n", tag, MK_SCHED, t, md / (mx + 1e-300), mx); };

    // ---- production: the real ComputeLDGPreconditioner (first call = warm-up, timed separately) ----
    { barrier(); Kokkos::Timer tm; m->prec.ComputeLDGPreconditioner(d, K, u, backend); sync(); barrier();
      const double tw = gmax(tm.seconds() * 1e3); if (root) printf("MKPCWARM %s first production build %.3f ms\n", MK_SCHED, tw); }
    if (const char* e = std::getenv("MK_PC_PROFONLY"); e && e[0] == '2') {   // stage microbenchmark for counter passes
        // the generated configuration with every stage in a roctx range (MK_PROF_RANGES=1): 2 builds, 3 residuals,
        // 30 Arnoldi steps (FD matvec + apply + CGS) -- ~10k kernels instead of a whole solve's ~110k
        SetResidualStages(mkprof::wrap(mkstages::table())); SetPrecondStages(mkprof::wrap(mkpc::table()));
        for (int r = 0; r < 2; r++) m->prec.ComputeLDGPreconditioner(d, K, u, backend);
        sync();
        const Int N = npe*ncu*ne1; const int nst = 30, n1 = 61;
        dstype *b = nullptr, *V = nullptr; hipMalloc(&b, N*sizeof(dstype)); hipMalloc(&V, (size_t)(nst + 1)*N*sizeof(dstype));
        ArrayInsert(sol.udg, u, npe, nc, ne, 0, npe, 0, ncu, 0, ne1);
        for (int r = 0; r < 3; r++) Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend);
        sync(); ArrayCopy(b, res.Ru, N);
        std::vector<double> H((size_t)n1*n1, 0.0), y(n1 + 1, 0.0);
        const double nrm = PNORM(handle, N, b, backend); ArrayAXPB(V, b, one/nrm, zero, N);
        for (int i = 0; i < nst; i++) { const Int mm = i + 1;
            MatVec<M>(&V[mm*N], sol, res, app, master, mesh, tmp, common, handle, &V[i*N], u, b, backend);
            m->prec.ApplyPreconditioner(&V[mm*N], m->solv.sys, d, backend);
            CGS(handle, V, &H[(size_t)n1*i], y.data(), N, mm, backend); }
        sync(); barrier(); hipFree(b); hipFree(V);
        SetResidualStages(nullptr); SetPrecondStages(nullptr);
        if (root) printf("MKPCPROF stageprof done (%d ranges)\n", mkprof::seq()); return 0; }
    if (const char* e = std::getenv("MK_PC_PROFONLY"); e && e[0] == '1') {   // profiling: production builds only
        for (int r = 0; r < 3; r++) m->prec.ComputeLDGPreconditioner(d, K, u, backend);
        sync(); barrier(); if (root) printf("MKPCPROF done\n"); return 0; }
    const double t_prod = timeit([&]() { m->prec.ComputeLDGPreconditioner(d, K, u, backend); }, 3);
    m->prec.ComputeLDGPreconditioner(d, K, u, backend); sync();
    hipMemcpy(Kref, K, NK * sizeof(dstype), hipMemcpyDeviceToDevice);
    m->prec.ComputeLDGPreconditioner(d, K, u, backend); sync();
    report("MKPCNOISE", t_prod, Kref);    // production vs production (final K)
    // ---- pre-inverse K: production-stage composition vs itself (noise) and vs generated stages ----
    PC P; PC G; G.gen_bminv = true; G.gen_fg = true; G.gen_inv = true;
    { auto sp = build_seq(P, false); run_seq(sp); sync(); hipMemcpy(Kpre, K, NK * sizeof(dstype), hipMemcpyDeviceToDevice);
      run_seq(sp); sync(); report("MKPCPRE_NOISE", 0.0, Kpre);
      auto sg = build_seq(G, false); run_seq(sg); sync(); report("MKPCPRE_GEN", 0.0, Kpre);
      if (root) printf("MKINFO F.G column overflow count: %d\n", mkpc::overflow_count()); }
    // ---- full builds: staged production stages (no fences) and generated stages, vs the real ComputeLDGPreconditioner ----
    { auto sp = build_seq(P, true); const double t = timeit([&]() { run_seq(sp); }, 3); run_seq(sp); sync(); report("MKPC_STAGEDPROD", t, Kref); }
    { auto sg = build_seq(G, true); const double t = timeit([&]() { run_seq(sg); }, 3); run_seq(sg); sync(); report("MKPC_GEN", t, Kref); }
    // ---- the Gauss-Jordan inverse vs rocSOLVER on the same pre-inverse K (generated stages) ----
    PC GJ = G; GJ.gen_gj = true;
    {   dstype* Kx = nullptr; hipMalloc(&Kx, NK * sizeof(dstype)); frees.push_back(Kx);
        auto sg = build_seq(G, false); run_seq(sg); sync(); hipMemcpy(Kpre, K, NK * sizeof(dstype), hipMemcpyDeviceToDevice);
        const Int nb = nprocs > 1 ? nbe1 : nbe;
        auto inv_all = [&](bool gj) { for (Int j = 0; j < nb; j++) { const Int e1 = common.eblks[3*j]-1, neb = common.eblks[3*j+1]-e1;
            if (gj) mkpc::gjinverse(&K[n*n*e1], res.H, n, neb); else Inverse(handle, &K[n*n*e1], res.H, res.ipiv, n, neb, backend); } };
        inv_all(false); sync(); hipMemcpy(Kx, K, NK * sizeof(dstype), hipMemcpyDeviceToDevice);          // rocSOLVER inverse
        const int cnt = std::min<int>(64, ne1);
        const double r_ref = gmax(mkpc::invresid(Kx, Kpre, cnt));
        hipMemcpy(K, Kpre, NK * sizeof(dstype), hipMemcpyDeviceToDevice); inv_all(true); sync();
        const double r_gj = gmax(mkpc::invresid(K, Kpre, cnt));
        double md, mx; maxdiff(K, Kx, md, mx);
        if (root) printf("MKPCGJ %s max|X_gj - X_rocsolver|/max|X| = %.3e (max|X| %.3e)  max|XA-I| first %d/rank: rocsolver %.3e gj %.3e\n",
                         MK_SCHED, md / (mx + 1e-300), mx, cnt, r_ref, r_gj);
        hipMemcpy(Kx, K, NK * sizeof(dstype), hipMemcpyDeviceToDevice);
        for (int rep = 0; rep < 4; rep++) {
            hipMemcpy(K, Kpre, NK * sizeof(dstype), hipMemcpyDeviceToDevice); inv_all(true); sync();
            maxdiff(K, Kx, md, mx); if (root) printf("MKPCGJREP %s gj run-to-run max|dX| = %.3e (%s)\n", MK_SCHED, md, md == 0.0 ? "bitwise" : "NOT bitwise"); }
        const double t_roc = timeit([&]() { hipMemcpy(K, Kpre, NK * sizeof(dstype), hipMemcpyDeviceToDevice); inv_all(false); }, 3);
        const double t_gj = timeit([&]() { hipMemcpy(K, Kpre, NK * sizeof(dstype), hipMemcpyDeviceToDevice); inv_all(true); }, 3);
        const double t_cp = timeit([&]() { hipMemcpy(K, Kpre, NK * sizeof(dstype), hipMemcpyDeviceToDevice); }, 3);
        if (root) printf("MKPCGJT %s inverse (all blocks, minus the %.3f ms restore copy): rocsolver %.3f ms gj %.3f ms (%.1fx)\n",
                         MK_SCHED, t_cp, t_roc - t_cp, t_gj - t_cp, (t_roc - t_cp) / (t_gj - t_cp));
    }
    { auto s2 = build_seq(GJ, true); const double t = timeit([&]() { run_seq(s2); }, 3); run_seq(s2); sync(); report("MKPC_GENGJ", t, Kref); }
    seqtime(build_seq(GJ, true), "gengj", 3);
    // ---- element stage alone: production vs generated on block 0 (D and B right after the stage), then timing ----
    {   const Int j = 0, e1 = common.eblks[0]-1, neb = common.eblks[1]-e1;
        const size_t nD = (size_t)npe*npe*ncu*ncu*neb, nB = (size_t)npe*npe*ncu*ncq*neb;
        dstype *Dp = nullptr, *Bp = nullptr; hipMalloc(&Dp, nD*sizeof(dstype)); hipMalloc(&Bp, nB*sizeof(dstype)); frees.push_back(Dp); frees.push_back(Bp);
        run_seq(build_seq(P, false));                  // state as the build leaves it (w, q)
        uEquationElemBlock<M>(sol, res, app, master, mesh, tmp, common, handle, j, backend); sync();
        hipMemcpy(Dp, res.D, nD*sizeof(dstype), hipMemcpyDeviceToDevice); hipMemcpy(Bp, res.B, nB*sizeof(dstype), hipMemcpyDeviceToDevice);
        mkjac::elem(pctx, j); sync();
        auto cmp = [&](const dstype* a, const dstype* b, size_t nn, double& md, double& mx, long long& ndiff) {
            double r1 = 0, r2 = 0; long long r3 = 0;
            Kokkos::parallel_reduce("mkjac_cmp", Kokkos::RangePolicy<>(0, nn), KOKKOS_LAMBDA(const size_t i, double& x1, double& x2, long long& x3) {
                const double dd = fabs(a[i] - b[i]); if (!(dd <= x1)) x1 = dd; if (fabs(b[i]) > x2) x2 = fabs(b[i]);
                if (a[i] != b[i]) x3++; }, Kokkos::Max<double>(r1), Kokkos::Max<double>(r2), r3);
            md = gmax(r1); mx = gmax(r2); ndiff = (long long)gmax((double)r3); };
        double md, mx; long long nd_;
        cmp(res.D, Dp, nD, md, mx, nd_); if (root) printf("MKJACELEM %s D max|d|/max = %.3e, entries differing %lld / %zu (%s)\n", MK_SCHED, md/(mx+1e-300), nd_, nD, nd_ == 0 ? "bitwise" : "NOT bitwise");
        cmp(res.B, Bp, nB, md, mx, nd_); if (root) printf("MKJACELEM %s B max|d|/max = %.3e, entries differing %lld / %zu (%s)\n", MK_SCHED, md/(mx+1e-300), nd_, nB, nd_ == 0 ? "bitwise" : "NOT bitwise");
        const Int nb = nprocs > 1 ? nbe1 : nbe;
        const double tp = timeit([&]() { for (Int jj = 0; jj < nb; jj++) uEquationElemBlock<M>(sol, res, app, master, mesh, tmp, common, handle, jj, backend); }, 3);
        const double tg = timeit([&]() { for (Int jj = 0; jj < nb; jj++) mkjac::elem(pctx, jj); }, 3);
        if (root) printf("MKJACELEMT %s element stage (all blocks): production %.3f ms generated %.3f ms (%.2fx)\n", MK_SCHED, tp, tg, tp/tg);
    }
    // ---- element-face stage alone on block 0: same element-stage D/B in, production vs generated (and production twice) ----
    {   const Int j = 0, e1 = common.eblks[0]-1, neb = common.eblks[1]-e1;
        const size_t nD = (size_t)npe*npe*ncu*ncu*neb, nB = (size_t)npe*npe*ncu*ncq*neb, nF = (size_t)npe*npf*common.meshsizes.nfe*ncu*ncu*neb;
        std::vector<dstype*> bufs(5); for (auto& b : bufs) { hipMalloc(&b, std::max(nB, nF)*sizeof(dstype)); frees.push_back(b); }
        dstype *D0 = bufs[0], *B0 = bufs[1], *Dp = bufs[2], *Bp = bufs[3], *Fp = bufs[4];
        auto cp = [&](dstype* dst, const dstype* src, size_t nn) { hipMemcpy(dst, src, nn*sizeof(dstype), hipMemcpyDeviceToDevice); };
        run_seq(build_seq(P, false));
        uEquationElemBlock<M>(sol, res, app, master, mesh, tmp, common, handle, j, backend); sync(); cp(D0, res.D, nD); cp(B0, res.B, nB);
        uEquationElemFaceBlockLDG(sol, res, app, abi, master, mesh, tmp, common, handle, j, backend); sync(); cp(Dp, res.D, nD); cp(Bp, res.B, nB); cp(Fp, res.F, nF);
        auto cmp = [&](const char* tag, const dstype* a, const dstype* b, size_t nn) {
            double r1 = 0, r2 = 0; long long r3 = 0;
            Kokkos::parallel_reduce("mkjac_cmpf", Kokkos::RangePolicy<>(0, nn), KOKKOS_LAMBDA(const size_t i, double& x1, double& x2, long long& x3) {
                const double dd = fabs(a[i] - b[i]); if (!(dd <= x1)) x1 = dd; if (fabs(b[i]) > x2) x2 = fabs(b[i]);
                if (a[i] != b[i]) x3++; }, Kokkos::Max<double>(r1), Kokkos::Max<double>(r2), r3);
            const double md = gmax(r1), mx = gmax(r2); const long long nd_ = (long long)gmax((double)r3);
            if (root) printf("MKJACFACE %s %s max|d|/max = %.3e, differing %lld / %zu (%.3f%%)\n", MK_SCHED, tag, md/(mx+1e-300), nd_, nn, 100.0*nd_/nn); };
        cp(res.D, D0, nD); cp(res.B, B0, nB);
        uEquationElemFaceBlockLDG(sol, res, app, abi, master, mesh, tmp, common, handle, j, backend); sync();
        cmp("prod-vs-prod D", res.D, Dp, nD); cmp("prod-vs-prod B", res.B, Bp, nB); cmp("prod-vs-prod F", res.F, Fp, nF);
        cp(res.D, D0, nD); cp(res.B, B0, nB);
        mkjac::elemface(pctx, j); sync();
        cmp("gen-vs-prod D", res.D, Dp, nD); cmp("gen-vs-prod B", res.B, Bp, nB); cmp("gen-vs-prod F", res.F, Fp, nF);
        cp(Dp, res.D, nD); cp(res.D, D0, nD); cp(res.B, B0, nB); mkjac::elemface(pctx, j); sync(); cmp("gen-vs-gen D", res.D, Dp, nD);
        // fused element + element-face (face terms in the element GEMM epilogue) vs production elem then face (Dp/Bp/Fp)
        mkjac::elem_fused(pctx, j); sync();
        cmp("fused-vs-prod D", res.D, Dp, nD); cmp("fused-vs-prod B", res.B, Bp, nB); cmp("fused-vs-prod F", res.F, Fp, nF);
        mkjac::elem_fused2(pctx, j); sync();
        cmp("fused2-vs-prod D", res.D, Dp, nD); cmp("fused2-vs-prod B", res.B, Bp, nB); cmp("fused2-vs-prod F", res.F, Fp, nF);
        const Int nb = nprocs > 1 ? nbe1 : nbe;
        {   const double t1 = timeit([&]() { for (Int jj = 0; jj < nb; jj++) { uEquationElemBlock<M>(sol, res, app, master, mesh, tmp, common, handle, jj, backend);
                uEquationElemFaceBlockLDG(sol, res, app, abi, master, mesh, tmp, common, handle, jj, backend); } }, 3);
            const double t2 = timeit([&]() { for (Int jj = 0; jj < nb; jj++) mkjac::elem_fused(pctx, jj); }, 3);
            const double t3 = timeit([&]() { for (Int jj = 0; jj < nb; jj++) mkjac::elem_fused2(pctx, jj); }, 3);
            if (root) printf("MKJACFUSEDT %s element + element-face (all blocks): production %.3f ms fused %.3f ms (%.2fx) fused2 %.3f ms (%.2fx)\n", MK_SCHED, t1, t2, t1/t2, t3, t1/t3); }
        const double tp = timeit([&]() { for (Int jj = 0; jj < nb; jj++) uEquationElemFaceBlockLDG(sol, res, app, abi, master, mesh, tmp, common, handle, jj, backend); }, 3);
        const double tg = timeit([&]() { for (Int jj = 0; jj < nb; jj++) mkjac::elemface(pctx, jj); }, 3);
        if (mkjac::jt_on()) {
            const char* nm[16] = {"f_gather","f_w_int","f_flux_gemm","-","f_bc_other","f_bc_w","f_asm","-",
                                  "e_gather_interp_w","e_source","e_flux_wchain","-","e_xx_negmm_D","e_xx_negmm_B","-","-"};
            auto run = [&](const char* tag, auto&& f) { for (int k = 0; k < 16; k++) mkjac::jt()[k] = 0;
                for (int r = 0; r < 3; r++) for (Int jj = 0; jj < nb; jj++) f(jj);
                if (root) { printf("MKJACSEC %s %s per build (ms):", MK_SCHED, tag);
                    for (int k = 0; k < 16; k++) if (mkjac::jt()[k] > 0) printf(" %s=%.2f", nm[k == 11 ? 12 : k == 12 ? 13 : k], mkjac::jt()[k] / 3); printf("\n"); } };
            run("elemface", [&](Int jj) { mkjac::elemface(pctx, jj); });
            run("elem", [&](Int jj) { mkjac::elem(pctx, jj); });
            run("fused", [&](Int jj) { mkjac::elem_fused(pctx, jj); });
            run("fused2", [&](Int jj) { mkjac::elem_fused2(pctx, jj); }); }
        if (root) printf("MKJACFACET %s element-face stage (all blocks): production %.3f ms generated %.3f ms (%.2fx)\n", MK_SCHED, tp, tg, tp/tg);
    }
    PC GE = GJ; GE.gen_elem = true;
    { auto s3 = build_seq(GE, true); const double t = timeit([&]() { run_seq(s3); }, 3); run_seq(s3); sync(); report("MKPC_GENELEM", t, Kref); }
    seqtime(build_seq(GE, true), "genelem", 3);
    PC GF = GE; GF.gen_face = true;
    { auto s4 = build_seq(GF, true); const double t = timeit([&]() { run_seq(s4); }, 3); run_seq(s4); sync(); report("MKPC_GENFACE", t, Kref);
      hipMemcpy(Kpre, K, NK * sizeof(dstype), hipMemcpyDeviceToDevice); run_seq(s4); sync(); report("MKPC_GENFACE_REP", 0.0, Kpre); }
    seqtime(build_seq(GF, true), "genface", 3);
    PC GU = GJ; GU.gen_fused = true;
    { auto s5 = build_seq(GU, true); const double t = timeit([&]() { run_seq(s5); }, 3); run_seq(s5); sync(); report("MKPC_GENFUSED", t, Kref); }
    seqtime(build_seq(GU, true), "genfused", 3);
    PC G2 = GJ; G2.gen_fused2 = true;
    { auto s6 = build_seq(G2, true); const double t = timeit([&]() { run_seq(s6); }, 3); run_seq(s6); sync(); report("MKPC_GENFUSED2", t, Kref); }
    seqtime(build_seq(G2, true), "genfused2", 3);
    PC G3 = G2; G3.gen_sl = true;       // + D/F written in Schur layout, Schur relayout skipped
    { auto s2 = build_seq(G2, false); run_seq(s2); sync(); hipMemcpy(Kpre, K, NK * sizeof(dstype), hipMemcpyDeviceToDevice);
      auto s3 = build_seq(G3, false); run_seq(s3); sync(); report("MKPCPRE_SL_vs_FUSED2", 0.0, Kpre); }
    { auto s7 = build_seq(G3, true); const double t = timeit([&]() { run_seq(s7); }, 3); run_seq(s7); sync(); report("MKPC_GENSL", t, Kref); }
    seqtime(build_seq(G3, true), "gensl", 3);
    // ---- cross-face stage: generated vs production on the same pre-cross K ----
    PC G4 = G3; G4.gen_cross = true;
    { auto s3 = build_seq(G3, false); run_seq(s3); sync(); hipMemcpy(Kpre, K, NK * sizeof(dstype), hipMemcpyDeviceToDevice);
      run_seq(s3); sync(); report("MKPCPRE_CROSS_prod_noise", 0.0, Kpre);
      auto s4 = build_seq(G4, false); run_seq(s4); sync(); report("MKPCPRE_CROSS_gen", 0.0, Kpre);
      const double tp = timeit([&]() { RuFaceCrossDerivOptimized(K, sol, res, app, abi, master, mesh, tmp, common); }, 3);
      const double tg = timeit([&]() { mkjac::cross(pctx, K); }, 3);
      if (root) printf("MKJACCROSST %s cross stage: production %.3f ms generated %.3f ms (%.2fx)\n", MK_SCHED, tp, tg, tp/tg);
      if (mkjac::jt_on()) { for (int k = 0; k < 24; k++) mkjac::jt()[k] = 0; for (int r = 0; r < 3; r++) mkjac::cross(pctx, K); sync();
          if (root) printf("MKJACCROSSSEC %s per build (ms): gather=%.2f flux=%.2f dot_gemm_pack=%.2f map_Ef=%.2f gemm_scatter=%.2f\n", MK_SCHED,
                           mkjac::jt()[16]/3, mkjac::jt()[17]/3, mkjac::jt()[18]/3, mkjac::jt()[19]/3, mkjac::jt()[20]/3); } }
    { auto s8 = build_seq(G4, true); const double t = timeit([&]() { run_seq(s8); }, 3); run_seq(s8); sync(); report("MKPC_GENCROSS", t, Kref); }
    // ---- plugged in: production ComputeLDGPreconditioner with the stage table registered (precondstages.hpp) ----
    SetPrecondStages(mkpc::table());
    { const double t = timeit([&]() { m->prec.ComputeLDGPreconditioner(d, K, u, backend); }, 3);
      m->prec.ComputeLDGPreconditioner(d, K, u, backend); sync(); report("MKPCPLUG", t, Kref); }
    SetPrecondStages(nullptr);
    // ---- MFMA GEMM v1 vs v2 (Exasim mfma_gemm.hpp): residual bitwise + timing, production and generated stages ----
    {   const Int N = npe*ncu*ne1; dstype* R1 = nullptr; hipMalloc(&R1, N*sizeof(dstype)); frees.push_back(R1);
        ArrayInsert(sol.udg, u, npe, nc, ne, 0, npe, 0, ncu, 0, ne1);
        for (int gen = 0; gen < 2; gen++) {
            SetResidualStages(gen ? mkstages::table() : nullptr);
            double t[3] = {0, 0, 0};
            for (int v = 1; v <= 2; v++) { exasim_mfma::version() = v;
                Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend); sync();
                if (v == 1) hipMemcpy(R1, res.Ru, N*sizeof(dstype), hipMemcpyDeviceToDevice);
                t[v] = timeit([&]() { Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend); }, 5); }
            Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend); sync();
            double r1 = 0; long long nd_ = 0;
            Kokkos::parallel_reduce("mkgemm_cmp", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const size_t q, double& x1, long long& x3) {
                const double dd = fabs(res.Ru[q] - R1[q]); if (!(dd <= x1)) x1 = dd; if (res.Ru[q] != R1[q]) x3++; }, Kokkos::Max<double>(r1), nd_);
            const double md = gmax(r1); nd_ = (long long)gmax((double)nd_);
            if (root) printf("MKGEMMV %s %s residual: v1 %.3f ms v2 %.3f ms (%.2fx), entries differing %lld (max|d| %.3e)\n", MK_SCHED,
                             gen ? "generated" : "production", t[1], t[2], t[1]/t[2], nd_, md); }
        // generated GetW: device Newton loop vs the host-checked loop (same iterations -> bitwise)
        {   SetResidualStages(mkstages::table());
            double tw[2] = {0, 0};
            for (int dv = 0; dv < 2; dv++) { mkstages::w_device() = dv;
                Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend); sync();
                if (dv == 0) hipMemcpy(R1, res.Ru, N*sizeof(dstype), hipMemcpyDeviceToDevice);
                tw[dv] = timeit([&]() { Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend); }, 5); }
            Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend); sync();
            long long nd_ = 0;
            Kokkos::parallel_reduce("mkw_cmp", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const size_t q, long long& x3) { if (res.Ru[q] != R1[q]) x3++; }, nd_);
            nd_ = (long long)gmax((double)nd_);
            if (root) printf("MKWDEV %s generated residual: host-checked GetW %.3f ms, device GetW %.3f ms (%.2fx), entries differing %lld\n",
                             MK_SCHED, tw[0], tw[1], tw[0]/tw[1], nd_); }
        // block concurrency: the per-block chains on 1 / 2 / 4 streams (same kernels, same arithmetic -> bitwise)
        {   SetResidualStages(mkstages::table());
            const int ps[3] = {1, 2, 4}; double tp[3] = {0, 0, 0};
            for (int pi = 0; pi < 3; pi++) { if (ps[pi] > mkstages::nslot()) continue; mkstages::mstreams() = ps[pi];
                Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend); sync();
                if (pi == 0) hipMemcpy(R1, res.Ru, N*sizeof(dstype), hipMemcpyDeviceToDevice);
                tp[pi] = timeit([&]() { Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend); }, 5);
                long long nd_ = 0;
                Kokkos::parallel_reduce("mks_cmp", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const size_t q, long long& x3) { if (res.Ru[q] != R1[q]) x3++; }, nd_);
                nd_ = (long long)gmax((double)nd_);
                if (root) printf("MKMSTREAM %s generated residual on %d stream(s): %.3f ms (%.2fx vs 1), entries differing %lld\n",
                                 MK_SCHED, ps[pi], tp[pi], tp[0]/tp[pi], nd_); }
            mkstages::mstreams() = std::min(4, mkstages::nslot()); }
        SetResidualStages(nullptr);
        double tb[3] = {0, 0, 0};
        for (int v = 1; v <= 2; v++) { exasim_mfma::version() = v; SetPrecondStages(mkpc::table());
            tb[v] = timeit([&]() { m->prec.ComputeLDGPreconditioner(d, K, u, backend); }, 3); SetPrecondStages(nullptr);
            const double tp = timeit([&]() { m->prec.ComputeLDGPreconditioner(d, K, u, backend); }, 3);
            if (root) printf("MKGEMMV %s build v%d: generated %.3f ms, production %.3f ms\n", MK_SCHED, v, tb[v], tp); }
        exasim_mfma::version() = 2; }
    // ---- GMRES iteration anatomy: nrest real Arnoldi steps (matvec, block-Jacobian apply, CGS), plain timers ----
    {   SetResidualStages(mkstages::table());                 // the generated residual in both runs: isolate the GMRES side
        m->prec.ComputeLDGPreconditioner(d, K, u, backend); sync();
        auto& sys = m->solv.sys;
        const Int N = npe*ncu*ne1; const int nrest = std::min<int>(60, (int)common.solverparams.gmresRestart), n1 = nrest + 1;
        dstype *b = nullptr, *V = nullptr, *Vp = nullptr;
        hipMalloc(&b, N*sizeof(dstype)); hipMalloc(&V, (size_t)n1*N*sizeof(dstype)); hipMalloc(&Vp, (size_t)n1*N*sizeof(dstype));
        frees.push_back(b); frees.push_back(V); frees.push_back(Vp);
        ArrayInsert(sol.udg, u, npe, nc, ne, 0, npe, 0, ncu, 0, ne1);
        Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend); sync();
        ArrayCopy(b, res.Ru, N);
        std::vector<double> H((size_t)n1*n1, 0.0), Hp((size_t)n1*n1, 0.0), y(n1 + 1, 0.0);
        auto arnoldi = [&](bool timed_phases, double* ph) {
            const double nrm = PNORM(handle, N, b, backend);
            ArrayAXPB(V, b, one/nrm, zero, N);
            for (int i = 0; i < nrest; i++) { const Int mm = i + 1; Kokkos::Timer tm;
                MatVec<M>(&V[mm*N], sol, res, app, master, mesh, tmp, common, handle, &V[i*N], u, b, backend);
                if (timed_phases) { sync(); ph[0] += tm.seconds()*1e3; tm.reset(); }
                m->prec.ApplyPreconditioner(&V[mm*N], sys, d, backend);
                if (timed_phases) { sync(); ph[1] += tm.seconds()*1e3; tm.reset(); }
                CGS(handle, V, &H[(size_t)n1*i], y.data(), N, mm, backend);
                if (timed_phases) { sync(); ph[2] += tm.seconds()*1e3; tm.reset(); } } };
        auto run = [&](const char* tag, const PrecondStageTable* tbl) {
            SetPrecondStages(tbl);
            double ph[3] = {0, 0, 0}, w[3] = {0, 0, 0};
            arnoldi(false, w);                                   // warm-up
            for (int r = 0; r < 3; r++) arnoldi(true, ph);
            std::vector<double> tot;
            for (int r = 0; r < 3; r++) { sync(); barrier(); Kokkos::Timer tm; arnoldi(false, w); sync(); barrier(); tot.push_back(tm.seconds()*1e3); }
            std::sort(tot.begin(), tot.end());
            const double t_tot = gmax(tot[1]);
            for (int k = 0; k < 3; k++) ph[k] = gmax(ph[k] / 3);
            if (root) printf("MKGMRES %s %s per %d iterations (ms): matvec %.2f apply %.2f cgs %.2f | phases sum %.2f | loop (no extra syncs) %.2f -> %.3f ms/iteration\n",
                             MK_SCHED, tag, nrest, ph[0], ph[1], ph[2], ph[0]+ph[1]+ph[2], t_tot, t_tot / nrest);
            SetPrecondStages(nullptr); };
        // apply alone: one vector, generated variants vs production (accuracy + bandwidth)
        {   dstype *x0 = nullptr, *xp = nullptr, *xg = nullptr; hipMalloc(&x0, N*sizeof(dstype)); hipMalloc(&xp, N*sizeof(dstype)); hipMalloc(&xg, N*sizeof(dstype));
            frees.push_back(x0); frees.push_back(xp); frees.push_back(xg);
            const double nb = PNORM(handle, N, b, backend); ArrayAXPB(x0, b, one/nb, zero, N);
            hipMemcpy(xp, x0, N*sizeof(dstype), hipMemcpyDeviceToDevice); m->prec.ApplyPreconditioner(xp, sys, d, backend); sync();
            const double tp = timeit([&]() { hipMemcpy(xg, x0, N*sizeof(dstype), hipMemcpyDeviceToDevice); m->prec.ApplyPreconditioner(xg, sys, d, backend); }, 5);
            const double tc = timeit([&]() { hipMemcpy(xg, x0, N*sizeof(dstype), hipMemcpyDeviceToDevice); }, 5);
            if (root) printf("MKGMV %s production apply %.3f ms (%.2f TB/s on K)\n", MK_SCHED, tp - tc, (double)NK*8/1e9/(tp - tc));
            for (int v = 0; v <= 6; v++) {
                hipMemcpy(xg, x0, N*sizeof(dstype), hipMemcpyDeviceToDevice); mkgm::apply_v(v, K, xg, (int)ne1); sync();
                double r1 = 0, r2 = 0;
                Kokkos::parallel_reduce("mkgm_cmpx", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const size_t q, double& x1, double& x2) {
                    const double dd = fabs(xg[q] - xp[q]); if (!(dd <= x1)) x1 = dd; if (fabs(xp[q]) > x2) x2 = fabs(xp[q]); }, Kokkos::Max<double>(r1), Kokkos::Max<double>(r2));
                const double md = gmax(r1), mx = gmax(r2);
                const double tg = timeit([&]() { hipMemcpy(xg, x0, N*sizeof(dstype), hipMemcpyDeviceToDevice); mkgm::apply_v(v, K, xg, (int)ne1); }, 5);
                if (root) printf("MKGMV %s variant %d apply %.3f ms (%.2f TB/s) max|dy|/max|y| = %.3e\n", MK_SCHED, v, tg - tc, (double)NK*8/1e9/(tg - tc), md/(mx+1e-300)); } }
        // Arnoldi sensitivity baseline: production with the GEMMs on rocBLAS instead of MFMA is not switchable here, so measure
        // how far two Arnoldi runs drift when only the starting vector is perturbed at the rounding level
        run("production", nullptr);
        { double w3[3] = {0, 0, 0}; arnoldi(false, w3); } sync();          // production reference V, H
        hipMemcpy(Vp, V, (size_t)n1*N*sizeof(dstype), hipMemcpyDeviceToDevice); Hp = H;
        run("generated", mkpc::table());
        SetPrecondStages(mkpc::table()); { double w3[3] = {0,0,0}; arnoldi(false, w3); } sync(); SetPrecondStages(nullptr);
        double hd = 0, hm = 0; for (int i = 0; i < nrest; i++) for (int k = 0; k <= i + 1; k++) {
            hd = std::max(hd, std::fabs(H[(size_t)n1*i + k] - Hp[(size_t)n1*i + k])); hm = std::max(hm, std::fabs(Hp[(size_t)n1*i + k])); }
        double md, mx; { const size_t NV = (size_t)n1*N; double r1 = 0, r2 = 0;
            Kokkos::parallel_reduce("mkgm_cmp", Kokkos::RangePolicy<>(0, NV), KOKKOS_LAMBDA(const size_t q, double& x1, double& x2) {
                const double dd = fabs(V[q] - Vp[q]); if (!(dd <= x1)) x1 = dd; if (fabs(Vp[q]) > x2) x2 = fabs(Vp[q]); }, Kokkos::Max<double>(r1), Kokkos::Max<double>(r2));
            md = gmax(r1); mx = gmax(r2); }
        {   // baseline: production again with b scaled by (1 + 2^-52): a pure rounding-level perturbation
            std::vector<double> Hg = H; dstype* Vg = nullptr; hipMalloc(&Vg, (size_t)n1*N*sizeof(dstype)); frees.push_back(Vg);
            hipMemcpy(Vg, V, (size_t)n1*N*sizeof(dstype), hipMemcpyDeviceToDevice);
            ArrayMultiplyScalar(b, 1.0 + 2.220446049250313e-16, N);
            { double w3[3] = {0, 0, 0}; arnoldi(false, w3); } sync();
            ArrayMultiplyScalar(b, 1.0 / (1.0 + 2.220446049250313e-16), N);
            double bd = 0; for (int i = 0; i < nrest; i++) for (int k = 0; k <= i + 1; k++) bd = std::max(bd, std::fabs(H[(size_t)n1*i + k] - Hp[(size_t)n1*i + k]));
            double hm2 = 0; for (auto v : Hp) hm2 = std::max(hm2, std::fabs(v));
            if (root) { printf("MKGMRESCMP %s production vs production(b*(1+eps)) after %d steps: max|dH|/max|H| = %.3e\n", MK_SCHED, nrest, bd/(hm2+1e-300));
                for (int i : {0, 4, 9, 19, 39, nrest - 1}) { double a = 0, bb = 0; for (int k = 0; k <= i + 1; k++) { a = std::max(a, std::fabs(Hg[(size_t)n1*i + k] - Hp[(size_t)n1*i + k])); bb = std::max(bb, std::fabs(H[(size_t)n1*i + k] - Hp[(size_t)n1*i + k])); }
                    printf("MKGMRESSTEP %s step %d: gen-vs-prod max|dH| %.3e   perturbed-vs-prod %.3e\n", MK_SCHED, i + 1, a, bb); } }
            H = Hg; hipMemcpy(V, Vg, (size_t)n1*N*sizeof(dstype), hipMemcpyDeviceToDevice); }
        if (root) printf("MKGMRESCMP %s generated vs production after %d Arnoldi steps: max|dH|/max|H| = %.3e, max|dV|/max|V| = %.3e\n", MK_SCHED, nrest, hd/(hm+1e-300), md/(mx+1e-300));
        SetResidualStages(nullptr);
    }
    if (root) printf("MKPCREF %s production ComputeLDGPreconditioner %.3f ms\n", MK_SCHED, t_prod);
    // ---- per-stage attribution (in sequence) ----
    seqtime(build_seq(P, true), "prod", 3);
    seqtime(build_seq(G, true), "gen", 3);
    // Schur sub-steps for one block, production vs generated
    { std::vector<Stage> sp{{"schur_prod(all blocks)", [&]() { for (Int j = 0; j < (nprocs > 1 ? nbe1 : nbe); j++) uEquationSchurBlockLDG(sol, res, app, abi, master, mesh, tmp, common, handle, j, backend, nullptr); }},
                            {"schur_gen(all blocks)", [&]() { for (Int j = 0; j < (nprocs > 1 ? nbe1 : nbe); j++) mkpc::schur(res, common, j, true, true, handle, backend); }},
                            {"inverse_prod", [&]() { for (Int j = 0; j < (nprocs > 1 ? nbe1 : nbe); j++) { const Int e1 = common.eblks[3*j]-1, neb = common.eblks[3*j+1]-e1; Inverse(handle, &K[n*n*e1], res.H, res.ipiv, n, neb, backend); } }},
                            {"inverse_gen", [&]() { for (Int j = 0; j < (nprocs > 1 ? nbe1 : nbe); j++) { const Int e1 = common.eblks[3*j]-1, neb = common.eblks[3*j+1]-e1; mkpc::inverse(handle, &K[n*n*e1], res.H, n, neb); } }}};
      seqtime(sp, "sub", 3); }
    }
    return 0;
    }();
    for (void* p : frees) hipFree(p);
    }
    solver.Finalize();
    return rc;
}
