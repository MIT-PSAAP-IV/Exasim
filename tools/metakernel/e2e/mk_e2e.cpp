// Meta-kernel END-TO-END harness: the real case loaded by Exasim itself, production Residual as the reference,
// and a schedule whose stages are either Exasim's own functions or generated kernels, on the same device arrays.
//
// Built in the case's own app project (same generated model kernels as exasimapp). The Exasim solver library is a
// single unity TU (ExasimSolver.cpp); it is compiled HERE with the library's flags, so every internal template
// (GetUhat, GetQ, GetW, RuElem, RuFace, PutFaceNodes, UpdateSource, ...) is callable without patching Exasim.
// One rank: the RuResidual sequence. Several ranks (gpumpi build): the RuResidualMPI sequence (halo exchange of u
// overlapped with the interior blocks), each stage swappable. Times are the slowest rank's (barriers around each rep).
// usage: [flux run -n P] mk_e2e_<schedule> 1 <datain dir>/ <dataout prefix>        (the exasimapp command line)
// output (rank 0): MKREF <sched> <stage> <ms> | MKRES <sched> <total> <ref> <speedup> <rel diff> <bitwise> ... | MKKER ...
#include "backend/Main/ExasimSolver.cpp"
#include <exasim/ExasimSolverSetup.hpp>
#ifdef MK_FRONTEND_PROVIDER      // MATLAB-frontend case: the generated kernels are compiled into this TU, model ID from app.bin
#include "frontendprovider.cpp"
#define MK_INIT_SOLVER(s, c, v, m) InitializeExasimSolver(s, c, v, m)
#else                            // language-frontend case: model library (frontend_model), ID 100
#define MK_INIT_SOLVER(s, c, v, m) InitializeExasimSolver(s, c, v, m, {100})
#endif
#include <algorithm>
#include <vector>
#include <cmath>
#include <functional>
#include <array>
#include <string>
#include <cstring>
#include <cstdint>
#include "mk_common.hpp"
#include "schedule.hpp"          // generated: element-residual kernels (per block) + MK_SCHED / MK_NK / mk_run(a) + optional stages
#include "mk_stages.hpp"         // the generated stages as an Exasim ResidualStageTable

using M = exasim::detail::AbiAdapter;
// -ffast-math (the library's flags) lets the compiler assume no NaN/Inf, so std::isfinite is unreliable here: test the bits
static inline bool mk_finite(double x) { uint64_t u; std::memcpy(&u, &x, 8); return (u & 0x7ff0000000000000ull) != 0x7ff0000000000000ull; }
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
    auto& tmp = d.tmp; auto& common = d.common; auto handle = common.cublasHandle;
    const int rank = common.mpiRank, nprocs = common.mpiProcs; const bool root = rank == 0;
    const Int nc = common.components.nc, ncu = common.components.ncu, nco = common.components.nco, ncx = common.components.ncx,
              ncw = common.components.ncw, ncs = common.components.ncs, ncq = common.components.ncq, nd = common.grid.nd,
              npe = common.grid.npe, npf = common.grid.npf, nge = common.grid.nge, ne = common.meshsizes.ne,
              nbe = common.meshsizes.nbe, nbf = common.meshsizes.nbf;
    const Int nbe0 = common.meshsizes.nbe0, nbe1 = common.meshsizes.nbe1, nbe2 = common.meshsizes.nbe2;
    auto gmax = [&](double v) {
#ifdef HAVE_MPI
        if (nprocs > 1) MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_MAX, comm);
#endif
        return v; };
    auto gsum = [&](double v) {
#ifdef HAVE_MPI
        if (nprocs > 1) MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, comm);
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
    if (root) printf("MKINFO nprocs=%d nc=%d ncu=%d nco=%d ncx=%d ncw=%d ncs=%d ncq=%d nd=%d npe=%d npf=%d nge=%d ngf=%d "
           "tdep=%d tdfunc=%d source=%d ncAV=%d frozenAV=%d wave=%d subproblem=%d dae_alpha=%g dae_beta=%g curvedMesh=%d preconditioner=%d matvecOrder=%d matvecTol=%g spatialScheme=%d\n",
           nprocs, (int)nc, (int)ncu, (int)nco, (int)ncx, (int)ncw, (int)ncs, (int)ncq, (int)nd, (int)npe, (int)npf, (int)nge, (int)common.grid.ngf,
           (int)common.timeparams.tdep, (int)common.timeparams.tdfunc, (int)common.physicsparams.source, (int)common.physicsparams.ncAV,
           (int)common.physicsparams.frozenAVflag, (int)common.timeparams.wave, (int)common.timeparams.subproblem,
           common.timeparams.dae_alpha, common.timeparams.dae_beta, (int)common.grid.curvedMesh,
           (int)common.solverparams.preconditioner, (int)common.solverparams.matvecOrder, (double)common.solverparams.matvecTol, (int)common.spatialScheme);
    printf("MKRANK %d ne=%d ne1=%d nf=%d nbe=%d nbe0=%d nbe1=%d nbe2=%d nbf=%d ndof1=%d nelemsend=%d nelemrecv=%d nnbsd=%d\n", rank, (int)ne,
           (int)common.meshsizes.ne1, (int)common.meshsizes.nf, (int)nbe, (int)nbe0, (int)nbe1, (int)nbe2, (int)nbf, (int)common.sizes.ndof1,
           (int)common.nelemsend, (int)common.nelemrecv, (int)common.nnbsd);
    if (nc != MK_NC || ncu != MK_NCU || nco != MK_NCO || ncw != MK_NCW || nd != MK_ND || npe != MK_NPE || nge != MK_NGE || ncx != MK_NCX) return fail("case sizes do not match mk_common.hpp");
    if (common.physicsparams.ncAV > 0 && common.physicsparams.frozenAVflag == 0) return fail("unfrozen AV (GetAv inside the residual) not replicated");
    {
#include "mk_setup.inc"

    auto sync = []() { Kokkos::fence(); hipDeviceSynchronize(); };
    auto timeit = [&](auto&& f, int reps) { f(); sync(); barrier(); std::vector<double> t;
        for (int r = 0; r < reps; r++) { barrier(); Kokkos::Timer tm; f(); sync(); barrier(); t.push_back(tm.seconds() * 1e3); }
        std::sort(t.begin(), t.end()); return gmax(t[t.size() / 2]); };
    const Int ndof1 = common.sizes.ndof1, f1all = common.firstFace(), f2all = common.lastFace();

    // ---- production stages (the calls RuResidual / RuResidualMPI make) ----
    // fsel (pass selection) for uhat / RqFace: 0 all faces, 1 faces not touching a ghost element, 2 faces touching one
    struct Impl { std::string tag; std::function<void(int)> uhat; std::function<void(Int, Int, int)> q; std::function<void(Int, Int)> w, elem;
                  std::function<void()> face, tail; };
    Impl P;
    P.tag = "exasim";
    P.uhat = [&](int) { GetUhat<M>(sol, res, app, master, mesh, tmp, common, handle, 0, nbf, backend); };
    P.q = [&](Int b1, Int b2, int) { if (ncq > 0) GetQ(sol, res, app, master, mesh, tmp, common, handle, b1, b2, 0, nbf, backend); };
    P.w = [&](Int b1, Int b2) { if (ncw > 0) GetW<M>(sol, res, app, master, mesh, tmp, common, handle, b1, b2, 0, nbf, backend); };
    P.elem = [&](Int b1, Int b2) { RuElem<M>(sol, res, app, master, mesh, tmp, common, handle, b1, b2, backend); };
    P.face = [&]() { RuFace<M>(sol, res, app, master, mesh, tmp, common, handle, 0, nbf, backend); };
    auto p_scatter = [&]() {
        if (nprocs > 1) PutFaceNodes(res.Ru, res.Rh, mesh.facecon, npf, ncu, npe, ncu, f1all, f2all);                 // RuResidualMPI
        else PutFaceNodes(res.Ru, &res.Rh[npf*ncu*f1all], mesh.facecon, npf, ncu, npe, ncu, f1all, f2all); };        // RuResidual
    auto p_finalize = [&]() {
        ArrayMultiplyScalar(res.Ru, minusone, ndof1);
        if (common.timeparams.tdep == 1) ArrayMultiplyScalar(res.Ru, one / common.timestate.dtfactor, ndof1); };
    P.tail = [&]() { p_scatter(); p_finalize(); };

    // ---- halo exchange of u (RuResidualMPI, verbatim) ----
    int req_count = 0;
    auto post = [&]() {
#ifdef HAVE_MPI
        const Int bsz = npe * ncu;
        GetArrayAtIndex(tmp.buffsend, sol.udg, mesh.elemsendind, bsz*common.nelemsend);
        hipDeviceSynchronize();
        Int psend = 0, precv = 0; req_count = 0;
        for (Int n = 0; n < common.nnbsd; n++) { const Int nsend = common.elemsendpts[n]*bsz;
            if (nsend > 0) { MPI_Isend(&tmp.buffsend[psend], nsend, mpi_type<dstype>(), common.nbsd[n], 0, EXASIM_COMM_LOCAL, &common.requests[req_count]);
                             psend += nsend; req_count++; } }
        for (Int n = 0; n < common.nnbsd; n++) { const Int nrecv = common.elemrecvpts[n]*bsz;
            if (nrecv > 0) { MPI_Irecv(&tmp.buffrecv[precv], nrecv, mpi_type<dstype>(), common.nbsd[n], 0, EXASIM_COMM_LOCAL, &common.requests[req_count]);
                             precv += nrecv; req_count++; } }
#endif
    };
    auto wait_unpack = [&]() {
#ifdef HAVE_MPI
        MPI_Waitall(req_count, common.requests, common.statuses);
        PutArrayAtIndex(sol.udg, tmp.buffrecv, mesh.elemrecvind, npe*ncu*common.nelemrecv);
#endif
    };
    // the residual as an ordered stage list (same order and ranges as production)
    const bool frozen = common.physicsparams.frozenAVflag == 1;
    int FA = 0, FB = 0;   // pass selection used by the generated uhat / RqFace (set below when enabled)
    auto build_seq = [&](Impl& I) {
        std::vector<Stage> s;
        if (nprocs > 1) {
            s = {{"post", post}, {"uhat.a", [&]() { I.uhat(FA); }}, {"q.a", [&]() { I.q(0, nbe0, FA); }}, {"w.a", [&]() { I.w(0, nbe0); }},
                 {"elem.a", [&]() { if (frozen) I.elem(0, nbe0); }}, {"wait+unpack", wait_unpack}, {"uhat.b", [&]() { I.uhat(FB); }},
                 {"q.b", [&]() { I.q(nbe0, nbe2, FB); }}, {"w.b", [&]() { I.w(nbe0, nbe2); }},
                 {"elem.b", [&]() { if (frozen) I.elem(nbe0, nbe1); else I.elem(0, nbe1); }}, {"face", I.face}, {"tail", I.tail}};
        } else {
            s = {{"uhat", [&]() { I.uhat(0); }}, {"q", [&]() { I.q(0, nbe, 0); }}, {"w", [&]() { I.w(0, nbe); }}, {"elem", [&]() { I.elem(0, nbe); }},
                 {"face", I.face}, {"tail", I.tail}};
        }
        return s; };
    auto run_seq = [](const std::vector<Stage>& s) { for (auto& st : s) st.second(); };
    auto seqtime = [&](const std::vector<Stage>& st, const char* tag, int reps) {
        std::vector<std::vector<double>> t(st.size());
        for (int r = -1; r < reps; r++) { sync(); barrier();
            for (size_t k = 0; k < st.size(); k++) { Kokkos::Timer tm; st[k].second(); sync(); if (r >= 0) t[k].push_back(tm.seconds() * 1e3); } }
        double sum = 0;
        for (size_t k = 0; k < st.size(); k++) { std::sort(t[k].begin(), t[k].end()); const double ms = gmax(t[k][t[k].size() / 2]); sum += ms;
            if (root) printf("%s %s %s %.4f\n", tag, MK_SCHED, st[k].first.c_str(), ms); }
        if (root) printf("%s %s sum %.4f\n", tag, MK_SCHED, sum); };

    // ---- correctness mode: every residual starts from the same perturbed w, so GetW's Newton updates run ----
    const char* pw_env = std::getenv("MK_PERTURB_W"); const double pw = pw_env ? std::atof(pw_env) : 0.0;
    dstype* w0 = nullptr; const size_t nw = (size_t)npe * ncw * ne;
    if (pw != 0.0) { std::vector<dstype> h(nw); hipMemcpy(h.data(), sol.wdg, nw * sizeof(dstype), hipMemcpyDeviceToHost);
        for (size_t i = 0; i < nw; i++) h[i] *= (1.0 + pw * (((i * 2654435761u) % 1000) / 1000.0 - 0.5));
        hipMalloc(&w0, nw * sizeof(dstype)); hipMemcpy(w0, h.data(), nw * sizeof(dstype), hipMemcpyHostToDevice); frees.push_back(w0);
        if (root) printf("MKINFO perturbing w by up to +-%.2e before every residual (correctness mode; timings include the reset copy)\n", pw / 2); }
    auto reset_w = [&]() { if (w0) hipMemcpyAsync(sol.wdg, w0, nw * sizeof(dstype), hipMemcpyDeviceToDevice, mk_stream()); };

    // ---- reference: production Residual ----
    auto reference = [&]() { reset_w(); Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend); };
    const double t_ref = timeit(reference, 20);
    std::vector<dstype> Rref(ndof1), Rmk(ndof1);
    reference(); sync();
    hipMemcpy(Rref.data(), res.Ru, ndof1 * sizeof(dstype), hipMemcpyDeviceToHost);
    auto compare = [&](const char* tag, double t) {
#pragma float_control(precise, on)   // the error statistic must not inherit -ffast-math (NaN/Inf assumptions, reassociation)
        hipMemcpy(Rmk.data(), res.Ru, ndof1 * sizeof(dstype), hipMemcpyDeviceToHost);
        double md = 0, mx = 0, ndiff = 0, nf = 0, nfr = 0; Int first = -1;
        for (Int i = 0; i < ndof1; i++) { if (Rmk[i] != Rref[i]) ndiff++;
            if (!mk_finite(Rmk[i])) { nf++; if (first < 0) first = i; }
            if (!mk_finite(Rref[i])) nfr++;
            else if (mk_finite(Rmk[i])) { mx = std::max(mx, std::fabs(Rref[i])); md = std::max(md, std::fabs(Rmk[i] - Rref[i])); } }
        if (first >= 0) { const Int e = first / (npe*ncu), j = (first / npe) % ncu; Int blk = -1;
            for (Int b = 0; b < nbe; b++) if (e >= common.eblks[3*b] - 1 && e < common.eblks[3*b+1]) blk = b;
            printf("MKNAN %s rank %d: %.0f non-finite; first at elem %d comp %d node %d, block %d (%s)\n", tag, rank, nf, (int)e, (int)j, (int)(first % npe),
                   (int)blk, blk < nbe0 ? "interior" : (blk < nbe1 ? "interface" : "ghost")); }
        { double lmd = 0; Int imd = -1; for (Int i = 0; i < ndof1; i++) { const double dd = std::fabs(Rmk[i] - Rref[i]); if (dd > lmd || !(dd == dd)) { lmd = dd; imd = i; if (!(dd == dd)) break; } }
          if (std::getenv("MK_DIAG")) printf("MKDIAG %s rank %d: md=%.6e mx=%.6e ndiff=%.0f maxdiff at %d (elem %d comp %d): mk=%.17e ref=%.17e\n", tag, rank, md, mx, ndiff,
              (int)imd, imd >= 0 ? (int)(imd / (npe*ncu)) : -1, imd >= 0 ? (int)((imd / npe) % ncu) : -1, imd >= 0 ? Rmk[imd] : 0.0, imd >= 0 ? Rref[imd] : 0.0); }
        md = gmax(md); mx = gmax(mx); ndiff = gsum(ndiff); nf = gsum(nf); nfr = gsum(nfr);
        if (root && nfr > 0) printf("MKNAN %s reference has %.0f non-finite entries\n", tag, nfr);
        if (root) printf("%s %s %.4f %.4f %.3f %.2e %d nonfinite=%.0f backend=%s nprocs=%d\n", tag, MK_SCHED, t, t_ref, t_ref / t,
                         md / (mx + 1e-300), ndiff == 0, nf, MK_BACKEND, nprocs); };
    // harness self-check: production stages composed by this harness must reproduce Residual bitwise
    { auto sp = build_seq(P); const double tp = timeit([&]() { reset_w(); run_seq(sp); }, 5); reset_w(); run_seq(sp); sync(); compare("MKSELF", tp); }
    seqtime(build_seq(P), "MKREF", 20);
    if (root) printf("MKREF %s total %.4f\n", MK_SCHED, t_ref);

    // ---- generated stages: the case-side stage table (mk_stages.hpp), i.e. exactly what the solver runs ----
    ResidualStageContext ctx{sol, res, app, master, mesh, tmp, common, handle, backend};
    if (!mkstages::build(ctx)) return fail(mkstages::st().why.c_str());
    Impl G = P; G.tag = "gen";
    G.uhat = [&](int pass) { mkstages::t_uhat(ctx, pass); };
    G.q = [&](Int b1, Int b2, int pass) { mkstages::t_q(ctx, b1, b2, pass); };
    G.w = [&](Int b1, Int b2) { mkstages::t_w(ctx, b1, b2); };
    G.elem = [&](Int b1, Int b2) { mkstages::t_elem(ctx, b1, b2); };
    G.face = [&]() { mkstages::t_face(ctx); };
    G.tail = [&]() { mkstages::t_tail(ctx); };
    std::string elabel = MK_NK > 0 ? std::string("gen:elem{") + MK_KLABEL[0] + (MK_NK > 1 ? ",..." : "") + "}" : "exasim:elem";
    std::string flabel = "exasim:face", tlabel = "exasim:scatter+finalize", ulabel = "exasim:uhat", qlabel = "exasim:q", wlabel = "exasim:w";
#ifdef MK_HAS_FACE
    flabel = std::string("gen:") + MK_FACE_LABEL;
#endif
#ifdef MK_GEN_SCATTER
    tlabel = "gen:scatter+finalize";
#endif
#ifdef MK_HAS_UHAT
    ulabel = "gen:uhat";
#endif
#ifdef MK_HAS_Q
#ifdef MK_HAS_RQFACE
    qlabel = std::string("gen:RqFace+") + MK_Q_LABEL;
#else
    qlabel = std::string("gen:exasimRqFace+") + MK_Q_LABEL;
#endif
#endif
#ifdef MK_HAS_W
    wlabel = "gen:w";
#endif
    // the plugged path: Residual<M> itself with the stage table registered (residualstages.hpp) -- what the solver executes
    { SetResidualStages(mkstages::table());
      const double tpl = timeit([&]() { reset_w(); Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend); }, 20);
      reset_w(); Residual<M>(sol, res, app, master, mesh, tmp, common, handle, backend); sync(); compare("MKPLUG", tpl);
      SetResidualStages(nullptr); }
    {
    if (nprocs > 1) { FA = 1; FB = 2; }   // pass numbers, as RuResidualStaged passes them (production stages ignore them)
    auto sg = build_seq(G);
    const double t = timeit([&]() { reset_w(); run_seq(sg); }, 20);
    reset_w(); run_seq(sg); sync();
    compare("MKRES", t);
    { const char* rep = std::getenv("MK_REPEAT_CHECK"); const int nrep = rep ? std::atoi(rep) : 0;   // re-run + check N times
      for (int r = 0; r < nrep; r++) { reset_w(); run_seq(sg); sync(); compare("MKREP", t); } }
    // per-stage attribution in sequence, with the implementation behind each stage in the label
    auto lab = [&](const std::string& st) {
        const std::string b = st.substr(0, st.find('.'));
        if (b == "uhat") return ulabel; if (b == "q") return qlabel; if (b == "w") return wlabel; if (b == "elem") return elabel;
        if (b == "face") return flabel; if (b == "tail") return tlabel; return std::string("exasim:") + b; };
    std::vector<Stage> named; for (auto& s : sg) named.push_back({s.first + "=" + lab(s.first), s.second});
    std::vector<std::vector<double>> tt(named.size());
    for (int r = -1; r < 20; r++) { sync(); barrier(); for (size_t k = 0; k < named.size(); k++) { Kokkos::Timer tm; named[k].second(); sync(); if (r >= 0) tt[k].push_back(tm.seconds() * 1e3); } }
    for (size_t k = 0; k < named.size(); k++) { std::sort(tt[k].begin(), tt[k].end()); const double ms = gmax(tt[k][tt[k].size() / 2]);
        if (root) printf("MKKER %s x%d %.4f %s\n", MK_SCHED, (int)k, ms, named[k].first.c_str()); }
    double wi = gmax((double)mkstages::st().w_iters_max);
    if (root) printf("MKINFO %s GetW Newton updates: max %.0f per call over all calls\n", MK_SCHED, wi);
    if (root) printf("MKINFO %s deferred GetW checks: %ld calls, %ld residuals re-ran elem+face (rank 0)\n", MK_SCHED, mkstages::st().w_calls, mkstages::st().w_redo);
    }
    }
    return 0;
    }();
    for (void* p : frees) hipFree(p);
    }
    solver.Finalize();
    return rc;
}
