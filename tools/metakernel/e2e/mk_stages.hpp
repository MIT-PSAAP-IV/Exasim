#include <chrono>
// mk_stages.hpp -- the generated residual stages as an Exasim ResidualStageTable (backend/Discretization/residualstages.hpp).
//
// Case-side code: include after the Exasim unity TU (backend/Main/ExasimSolver.cpp), mk_common.hpp and the generated
// schedule.hpp; then SetResidualStages(mkstages::table()) makes Residual<M> -- and so every Newton residual and every
// finite-difference GMRES matvec -- run the generated kernels. Stages the schedule does not generate stay null in the
// table, i.e. production. Mesh-only data (block tables, ordered face-contribution lists, face-node geometry, ghost-face
// flags) is built on first use and keyed on the mesh; time and dtfactor are refreshed on every call.
// If any precondition of the generated stages fails, the table is not offered (table() returns nullptr) and the
// reason is printed once, so the solver runs production.
#pragma once
#include <vector>
#include <array>
#include <algorithm>
#include <cstdio>
#include <string>
#include <cmath>

namespace mkstages {
using M = exasim::detail::AbiAdapter;
using Ctx = ResidualStageContext;

// ---- block concurrency: MK_MSTREAM = streams in the pool (default 4; 0 or 1 = the blocks in sequence on the Kokkos stream)
// (settable at run time up to nslot(), the scratch sets allocated at setup: MK_MSTREAM_SLOTS, default 4)
inline int& mstreams() { static int v = [] { const char* e = getenv("MK_MSTREAM"); return e ? atoi(e) : 1; }(); return v; }   // default 1: overlap measured 0.93-0.95x (kernels contend for memory)
inline int nslot() { static int v = [] { const char* e = getenv("MK_MSTREAM_SLOTS"); return std::max(1, e ? atoi(e) : 4); }(); return v; }
inline int& cur_slot() { static int v = 0; return v; }   // scratch slot of the block being issued (set by fanout)
struct State {
    bool built = false, ok = false; const void* key = nullptr; std::string why;
    std::vector<MkIn> blocks; std::vector<MkFaceIn> fblocks, rqblocks; std::vector<MkQIn> qblocks;
    std::vector<MkIn> eslot; std::vector<MkFaceIn> fslot;
    MkFaceIn* rq_dblocks = nullptr; int* rq_wgoff = nullptr; int rq_nwg = 0;   // one-launch rqface (MK_RQFACE_ALL)
    int *qfs = nullptr, *qcode = nullptr, *fnga = nullptr; size_t* fofs = nullptr;  // q v7 tables (MK_Q_FACES)
    dstype** pbase = nullptr; int *pnga = nullptr, *pnode = nullptr, *pinv = nullptr;   // producer face values (MK_FACE_PRODUCER)
    struct PFix { dstype* dst; int nga, f, side; }; PFix* pfix = nullptr; int npfix = 0;   // ghost sides, gathered in face()
    std::vector<MkFaceIn> fint, fbou;                                            // interior / boundary face blocks (MK_FACE_ALL)   // per-stream-slot stage scratch (only the scratch pointers are used)
#ifdef MK_HAS_UHAT
    std::vector<MkUhIn> uhblocks;
#endif
    int *nodeptr = nullptr, *contrib = nullptr, *ghost = nullptr, *efaces = nullptr, *lcontrib = nullptr;
    dstype *wF = nullptr, *wdw = nullptr;
    int w_iters_max = 0; int w_guess = 3;   // device GetW: iterations the previous call needed
    struct WPend { const EosBlockBatch* bb; MkWIn wa; Int b1, b2; int* fd; int* fh; int launched; };
    std::vector<WPend> wpend; std::vector<std::pair<Int, Int>> elem_calls; long w_redo = 0, w_calls = 0;
};
inline State& st() { static State s; return s; }

inline dstype* dmalloc(size_t n) { dstype* p = nullptr; if (hipMalloc(&p, std::max<size_t>(n, 1) * sizeof(dstype)) != hipSuccess) p = nullptr; return p; }

// ---- build the mesh-only tables (once per mesh) ----
inline bool build(Ctx& c) {
    State& S = st();
    if (S.built && S.key == (const void*)c.mesh.facecon) return S.ok;
    S = State(); S.built = true; S.key = c.mesh.facecon;
    struct BuildTimer { std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now(); ~BuildTimer() { hipDeviceSynchronize();
        fprintf(stderr, "MKINFO stage build %.3f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()); } } mk_bt;
    auto& sol = c.sol; auto& res = c.res; auto& app = c.app; auto& master = c.master; auto& mesh = c.mesh; auto& tmp = c.tmp;
    auto& common = c.common; auto handle = c.handle; const Int backend = c.backend;
    const Int nc = common.components.nc, ncu = common.components.ncu, nco = common.components.nco, ncx = common.components.ncx,
              ncw = common.components.ncw, ncs = common.components.ncs, nd = common.grid.nd, npe = common.grid.npe,
              npf = common.grid.npf, nge = common.grid.nge, ngf = common.grid.ngf, ne = common.meshsizes.ne,
              nbe = common.meshsizes.nbe, nbf = common.meshsizes.nbf, nf = common.meshsizes.nf;
    const int nprocs = common.mpiProcs;
    auto fail = [&](const char* w) { S.why = w; S.ok = false; if (common.mpiRank == 0) printf("mkstages: generated residual stages disabled: %s\n", w); return false; };
    if (nc != MK_NC || ncu != MK_NCU || nco != MK_NCO || ncw != MK_NCW || nd != MK_ND || npe != MK_NPE || nge != MK_NGE || ncx != MK_NCX)
        return fail("case sizes do not match the generated kernels");
    if (npf != MK_NPF || ngf != MK_NGF) return fail("face sizes do not match the generated kernels");
    const bool fullrange = common.firstFace() == 0 && common.lastFace() == nf;
    // element blocks
    Int nebmax = 0; for (Int j = 0; j < nbe; j++) nebmax = std::max(nebmax, common.eblks[3*j+1] - common.eblks[3*j] + 1);
    // one set of stage buffers per stream slot (fanout runs block k of a call on stream k % nslot; same slot = same stream)
    const size_t ngm = (size_t)nge * nebmax;
    for (int sl = 0; sl < nslot(); sl++) {
        MkIn z{}; z.g_u = dmalloc(ngm * MK_NC); z.g_w = dmalloc(ngm * MK_NCW); z.g_s = dmalloc(ngm * MK_NCU); z.g_sg = dmalloc(ngm * MK_NCU);
        z.g_pr = dmalloc(ngm * MK_NPR); z.g_f = dmalloc(ngm * MK_NF); z.g_rg = dmalloc(ngm * MK_NRG);
        z.un = dmalloc((size_t)npe * MK_NC * nebmax); z.wn = dmalloc((size_t)npe * MK_NCW * nebmax); S.eslot.push_back(z); }
    dstype *g_u = nullptr, *g_w = nullptr, *g_s = nullptr, *g_sg = nullptr, *g_pr = nullptr, *g_f = nullptr, *g_rg = nullptr, *un = nullptr, *wn = nullptr;
    for (Int j = 0; j < nbe; j++) {
        const Int e1 = common.eblks[3*j] - 1, e2 = common.eblks[3*j+1], neb = e2 - e1, nga = nge * neb, nm = nge * e1 * (ncx + nd*nd + 1);
        S.blocks.push_back(MkIn{&sol.elemg[nm], &sol.odgg[nge*nco*e1], &sol.udg[npe*nc*e1], &sol.wdg[npe*ncw*e1], &sol.sdgg[nge*ncs*e1],
            &sol.elemg[nm + nga*(ncx + nd*nd)], &sol.elemg[nm + nga*ncx], master.shapegt, master.shapegw, app.uinf, app.physicsparam,
            g_u, g_w, g_s, g_sg, g_pr, g_f, g_rg, &res.Rue[npe*ncu*e1], un, wn, &mesh.eindudg1[npe*nc*e1], sol.udg,
            common.timestate.time, common.timestate.dtfactor, (int)nga, (int)neb});
    }
    // face blocks
    Int nfbmax = 0; for (Int j = 0; j < nbf; j++) nfbmax = std::max(nfbmax, common.fblks[3*j+1] - common.fblks[3*j] + 1);
    for (int sl = 0; sl < nslot(); sl++) {
        MkFaceIn z{}; z.gf1 = dmalloc((size_t)ngf * nfbmax * MK_NF); z.gug = dmalloc((size_t)ngf * nfbmax * (2*MK_NC + 2*MK_NCW + MK_NCU));
        z.gat = dmalloc((size_t)npf * nfbmax * (2*MK_NC + 2*MK_NCW + MK_NCU)); S.fslot.push_back(z); }
#ifndef MK_FACE_ALL
    dstype *g_f1 = nullptr, *g_ug = nullptr, *g_at = nullptr;
#endif
    for (Int j = 0; j < nbf; j++) {
        const Int f1 = common.fblks[3*j] - 1, f2 = common.fblks[3*j+1], ib = common.fblks[3*j+2], nfb = f2 - f1, nga = ngf * nfb, nm = ngf * f1 * (ncx + nd + 1);
#ifdef MK_FACE_ALL
        // interior blocks run concurrently in one launch: each gets its own scratch region
        dstype *g_f1 = nullptr, *g_ug = nullptr, *g_at = nullptr;
#ifdef MK_FACE_BOU_ALL
        if (true) {   // boundary blocks run concurrently too
#else
        if (ib == 0) {
#endif
          g_f1 = dmalloc((size_t)ngf * nfb * MK_NF); g_ug = dmalloc((size_t)ngf * nfb * (2*MK_NC + 2*MK_NCW + MK_NCU)); }
#endif
        S.fblocks.push_back(MkFaceIn{&sol.faceg[nm], &sol.faceg[nm + nga*ncx], &sol.faceg[nm + nga*(ncx + nd)], &sol.og1[ngf*nco*f1], &sol.og2[ngf*nco*f1],
            sol.udg, sol.wdg, sol.uh, &mesh.findudg1[npf*nc*f1], ib == 0 ? &mesh.findudg2[npf*nc*f1] : nullptr, mesh.facecon,
            master.shapfgt, master.shapfgw, app.uinf, app.physicsparam, app.tau, &res.Rh[npf*ncu*f1], g_f1, g_ug, g_at,
            common.timestate.time, (int)f1, (int)nfb, (int)ib});
    }
#ifdef MK_FACE_CONN
    { // the connectivity interpolation reads udg through facecon: require findudg1/2 == the facecon formula, exactly
      // (checked on the device; a host copy of findudg is ~0.5 GB on the larger meshes)
      if (const char* e = getenv("MK_CONN_CHECK"); !(e && e[0] == '0')) {
      unsigned long long* bad_d; hipMalloc(&bad_d, sizeof(unsigned long long)); hipMemset(bad_d, 0, sizeof(unsigned long long));
      for (const MkFaceIn& fb : S.fblocks) { if (fb.ib != 0) continue;
        const int* fc = mesh.facecon; const int* g1 = fb.find1; const int* g2 = fb.find2; const int nfb = fb.nf, f1 = fb.f1;
        const int NPF = npf, NPE = npe, NC = nc; const long N = (long)nfb * NPF * 2 * NC;
        Kokkos::parallel_for("mk_conn_check", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const long t) {
          const int cc = (int)(t % NC); long r = t / NC; const int side = (int)(r % 2); r /= 2; const int q = (int)(r % NPF), fl = (int)(r / NPF);
          const int k = fc[2*(NPF*(f1 + fl) + q) + side], m = k % NPE, n = k / NPE;
          const int v = (side ? g2 : g1)[q + NPF*fl + (size_t)NPF*nfb*cc];
          if (v != m + NPE*cc + NPE*NC*n) atomicAdd(bad_d, 1ull); }); }
      unsigned long long bad = 0; hipMemcpy(&bad, bad_d, sizeof(bad), hipMemcpyDeviceToHost); hipFree(bad_d);
      if (bad) { printf("MKERR facecon/findudg mismatch at %llu entries: interp=conn is not valid for this mesh\n", bad); std::abort(); } } }
#endif
#ifdef MK_FACE_PRODUCER
    { // each element writes its own side of its interior faces; sides owned by elements the element stage never runs
      // (ghosts) are gathered in face() instead
      Int eend = ne;
      if (nprocs > 1) eend = common.eblks[3*(common.meshsizes.nbe1 - 1) + 1];
      std::vector<int> fc((size_t)2 * npf * nf); hipMemcpy(fc.data(), mesh.facecon, fc.size() * sizeof(int), hipMemcpyDeviceToHost);
      std::vector<dstype*> pb((size_t)6 * ne, nullptr); std::vector<int> pn((size_t)6 * ne, 0), pd((size_t)6 * npf * ne, 0), used(ne, 0);
      std::vector<int> pinv((size_t)6 * npe * ne, -1);
      std::vector<State::PFix> fix;
      for (const MkFaceIn& fb : S.fblocks) { if (fb.ib != 0) continue; const int nga = ngf * fb.nf;
        for (int fl = 0; fl < fb.nf; fl++) { const int f = fb.f1 + fl;
          for (int side = 0; side < 2; side++) {
            dstype* dst = fb.gug + (size_t)ngf * fl + (size_t)nga * side * (MK_NC + MK_NCW);
            const int E = fc[2*(npf*f) + side] / npe;
            if (E >= eend) { fix.push_back({dst, nga, f, side}); continue; }
            if (used[E] == 6) return fail("producer face values: an element has more than 6 interior faces");
            const int sl = used[E]++; pb[6*E + sl] = dst; pn[6*E + sl] = nga;
            for (int q = 0; q < npf; q++) { const int k = fc[2*(npf*f + q) + side];
              if (k / npe != E) return fail("producer face values: a face's nodes span two elements");
              pd[(size_t)npf * (6*E + sl) + q] = k % npe; pinv[(size_t)npe * (6*E + sl) + k % npe] = q; } } } }
      hipMalloc(&S.pbase, pb.size() * sizeof(dstype*)); hipMalloc(&S.pnga, pn.size() * sizeof(int)); hipMalloc(&S.pnode, pd.size() * sizeof(int));
      hipMemcpy(S.pbase, pb.data(), pb.size() * sizeof(dstype*), hipMemcpyHostToDevice); hipMemcpy(S.pnga, pn.data(), pn.size() * sizeof(int), hipMemcpyHostToDevice);
      hipMemcpy(S.pnode, pd.data(), pd.size() * sizeof(int), hipMemcpyHostToDevice);
      hipMalloc(&S.pinv, pinv.size() * sizeof(int)); hipMemcpy(S.pinv, pinv.data(), pinv.size() * sizeof(int), hipMemcpyHostToDevice);
      S.npfix = (int)fix.size(); hipMalloc(&S.pfix, std::max<size_t>(fix.size(), 1) * sizeof(State::PFix));
      if (!fix.empty()) hipMemcpy(S.pfix, fix.data(), fix.size() * sizeof(State::PFix), hipMemcpyHostToDevice);
      for (Int j = 0; j < (Int)S.blocks.size(); j++) { MkIn& a = S.blocks[j]; a.fgt = master.shapfgt; a.pbase = S.pbase; a.pnga = S.pnga; a.pnode = S.pnode; a.pinv = S.pinv;
        a.pe1 = (int)(common.eblks[3*j] - 1); } }
#endif
    for (MkFaceIn fb : S.fblocks) { fb.Rh = res.Rh; S.rqblocks.push_back(fb); }
#ifdef MK_FACE_ALL
    for (const MkFaceIn& fb : S.fblocks) { if (fb.ib == 0) S.fint.push_back(fb); else S.fbou.push_back(fb); }
#endif
    { // one-launch rqface: the blocks on the device + each block's first workgroup (64 / ngf faces per workgroup)
      std::vector<int> wo(S.rqblocks.size() + 1, 0); const int fpw = 64 / ngf;
      for (size_t j = 0; j < S.rqblocks.size(); j++) wo[j + 1] = wo[j] + (S.rqblocks[j].nf + fpw - 1) / fpw;
      S.rq_nwg = wo.back();
      hipMalloc(&S.rq_dblocks, std::max<size_t>(S.rqblocks.size(), 1) * sizeof(MkFaceIn)); hipMalloc(&S.rq_wgoff, wo.size() * sizeof(int));
      hipMemcpy(S.rq_dblocks, S.rqblocks.data(), S.rqblocks.size() * sizeof(MkFaceIn), hipMemcpyHostToDevice);
      hipMemcpy(S.rq_wgoff, wo.data(), wo.size() * sizeof(int), hipMemcpyHostToDevice); }
    // ordered face-contribution lists per element node (production's PutFaceNodesGather order) + ghost-face flags
    { std::vector<int> fc((size_t)2 * npf * nf);
      hipMemcpy(fc.data(), mesh.facecon, fc.size() * sizeof(int), hipMemcpyDeviceToHost);
      std::vector<std::array<int, 3>> tup; tup.reserve(fc.size());
      for (Int i = 0; i < npf * nf; i++) { const int k1 = fc[2*i], k2 = fc[2*i+1]; tup.push_back({k1, (int)i, 0}); if (k1 != k2) tup.push_back({k2, (int)i, 1}); }
      std::sort(tup.begin(), tup.end());
      std::vector<int> ptr((size_t)npe * ne + 1, 0), con(tup.size());
      for (size_t t = 0; t < tup.size(); t++) { ptr[tup[t][0] + 1]++; con[t] = (tup[t][1] << 1) | tup[t][2]; }
      for (size_t k = 0; k < (size_t)npe * ne; k++) ptr[k + 1] += ptr[k];
      hipMalloc(&S.nodeptr, ptr.size() * sizeof(int)); hipMalloc(&S.contrib, std::max<size_t>(con.size(), 1) * sizeof(int));
      hipMemcpy(S.nodeptr, ptr.data(), ptr.size() * sizeof(int), hipMemcpyHostToDevice);
      hipMemcpy(S.contrib, con.data(), con.size() * sizeof(int), hipMemcpyHostToDevice);
      { // per element: its faces in order of first appearance (<= 6 for hexes) and the contributions re-coded against them
        std::vector<int> ef((size_t)6 * ne, -1), lc(con.size(), 0);
        for (Int e = 0; e < ne; e++) {
          int nsl = 0;
          for (int p = 0; p < npe; p++) { const size_t node = p + (size_t)npe * e;
            for (int r = ptr[node]; r < ptr[node + 1]; r++) { const int ii = con[r] >> 1, f = ii / npf, q = ii % npf; int sl = -1;
              for (int k = 0; k < nsl; k++) if (ef[6*e + k] == f) sl = k;
              if (sl < 0) { if (nsl == 6) return fail("an element touches more than 6 faces"); sl = nsl; ef[6*e + nsl++] = f; }
              lc[r] = ((sl * npf + q) << 1) | (con[r] & 1); } } }
#if defined(MK_Q_FACES) || defined(MK_Q_HYBRID)
        { // q v7: faces of each element in ascending id + node -> (face node, side) per slot; requires each node's
          // contributions to come from strictly ascending (hence distinct) faces, so face-by-face accumulation = gather order
          std::vector<int> qfs((size_t)6 * ne, -1), qcode((size_t)6 * npe * ne, -1);
          for (Int e = 0; e < ne; e++) {
            std::vector<int> fl; for (int k = 0; k < 6; k++) if (ef[6*e + k] >= 0) fl.push_back(ef[6*e + k]);
            std::sort(fl.begin(), fl.end());
            for (size_t k = 0; k < fl.size(); k++) qfs[6*e + k] = fl[k];
            for (int p = 0; p < npe; p++) { const size_t node = p + (size_t)npe * e; int last = -1;
              for (int r = ptr[node]; r < ptr[node + 1]; r++) { const int ii = con[r] >> 1, f = ii / npf, q = ii % npf;
                if (f <= last) return fail("q v7: a node's face contributions are not in strictly ascending face order");
                last = f; const int sl = (int)(std::lower_bound(fl.begin(), fl.end(), f) - fl.begin());
                qcode[6 * npe * e + npe * sl + p] = (q << 1) | (con[r] & 1); } } }
          std::vector<size_t> fo(nf, 0); std::vector<int> fn(nf, 0);
          for (Int j = 0; j < nbf; j++) { const Int f1 = common.fblks[3*j] - 1, f2 = common.fblks[3*j+1], nfb = f2 - f1, nm = ngf * f1 * (ncx + nd + 1);
            for (Int f = f1; f < f2; f++) { fo[f] = (size_t)nm + (size_t)ngf * (f - f1); fn[f] = (int)(ngf * nfb); } }
          hipMalloc(&S.qfs, qfs.size() * sizeof(int)); hipMalloc(&S.qcode, qcode.size() * sizeof(int));
          hipMalloc(&S.fofs, std::max<size_t>(fo.size(), 1) * sizeof(size_t)); hipMalloc(&S.fnga, std::max<size_t>(fn.size(), 1) * sizeof(int));
          hipMemcpy(S.qfs, qfs.data(), qfs.size() * sizeof(int), hipMemcpyHostToDevice); hipMemcpy(S.qcode, qcode.data(), qcode.size() * sizeof(int), hipMemcpyHostToDevice);
          hipMemcpy(S.fofs, fo.data(), fo.size() * sizeof(size_t), hipMemcpyHostToDevice); hipMemcpy(S.fnga, fn.data(), fn.size() * sizeof(int), hipMemcpyHostToDevice); }
#endif
        hipMalloc(&S.efaces, ef.size() * sizeof(int)); hipMalloc(&S.lcontrib, std::max<size_t>(lc.size(), 1) * sizeof(int));
        hipMemcpy(S.efaces, ef.data(), ef.size() * sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(S.lcontrib, lc.data(), lc.size() * sizeof(int), hipMemcpyHostToDevice); }
      const Int nbe1 = common.meshsizes.nbe1, nbe2 = common.meshsizes.nbe2;
      const Int eg0 = (nprocs > 1 && nbe1 < nbe2) ? common.eblks[3*nbe1] - 1 : ne;
      std::vector<int> gh(nf, 0);
      for (Int i = 0; i < npf * nf; i++) { if (fc[2*i] / npe >= eg0 || fc[2*i+1] / npe >= eg0) gh[i / npf] = 1; }
#ifdef HAVE_MPI
      if (nprocs > 1) {   // every received element must be a ghost, or skipping faces in pass 1 would be wrong
          std::vector<int> ri((size_t)npe * ncu * common.nelemrecv);
          if (!ri.empty()) hipMemcpy(ri.data(), mesh.elemrecvind, ri.size() * sizeof(int), hipMemcpyDeviceToHost);
          for (int v : ri) if (v / (npe * nc) < eg0) return fail("received elements outside the ghost block range");
      }
#endif
      hipMalloc(&S.ghost, std::max<size_t>(gh.size(), 1) * sizeof(int));
      hipMemcpy(S.ghost, gh.data(), gh.size() * sizeof(int), hipMemcpyHostToDevice); }
    for (Int j = 0; j < nbe; j++) { const Int e1 = common.eblks[3*j] - 1, e2 = common.eblks[3*j+1], neb = e2 - e1, nga = nge * neb, nm = nge * e1 * (ncx + nd*nd + 1);
        S.qblocks.push_back(MkQIn{sol.udg, &sol.elemg[nm + nga*ncx], master.shapegt, master.shapegw, res.Rh, res.Minv, S.nodeptr, S.contrib, S.efaces, S.lcontrib, (int)e1, (int)neb,
                                  (int)common.grid.curvedMesh});
        MkQIn& qb = S.qblocks.back(); qb.uh = sol.uh; qb.fgt = master.shapfgt; qb.fgw = master.shapfgw; qb.faceg = sol.faceg;
        qb.qfs = S.qfs; qb.qcode = S.qcode; qb.fofs = S.fofs; qb.fnga = S.fnga; }
#if defined(MK_GEN_SCATTER) || defined(MK_HAS_Q)
    if (!fullrange) return fail("the generated q path / scatter need the face blocks to cover [0, nf)");
#endif
#ifdef MK_HAS_UHAT
    for (Int j = 0; j < nbf; j++) if (common.fblks[3*j+2] > 0 && isin<M>(common.fblks[3*j+2], common.stgparams.stgib, common.stgparams.nstgib))
        return fail("the generated uhat does not implement STG inflow blocks");
    for (Int j = 0; j < nbf; j++) {
        const Int f1 = common.fblks[3*j] - 1, f2 = common.fblks[3*j+1], ib = common.fblks[3*j+2], nfb = f2 - f1, nn = npf * nfb;
        if (ib == 0) continue;
        const Int n1 = nn*ncx, n2 = nn*(ncx + nd), n3 = nn*(ncx + nd + 1);
        dstype* geo = dmalloc((size_t)n3 + 2*(size_t)nn*nd);
        GetArrayAtIndex(tmp.tempn, sol.xdg, &mesh.findxdg1[npf*ncx*f1], nn*ncx);
        Node2Gauss(handle, &geo[0], tmp.tempn, master.shapfnt, npf, npf, nfb*ncx, backend);
        Node2Gauss(handle, &geo[n3], tmp.tempn, &master.shapfnt[npf*npf], npf, npf, nfb*nd, backend);
        Node2Gauss(handle, &geo[n3 + nn*nd], tmp.tempn, &master.shapfnt[2*npf*npf], npf, npf, nfb*nd, backend);
        FaceGeom3D(&geo[n2], &geo[n1], &geo[n3], nn);
        S.uhblocks.push_back(MkUhIn{sol.uh, sol.udg, geo, &mesh.findudg1[npf*nc*f1], mesh.facecon, app.physicsparam, app.uinf, app.tau,
                                    common.timestate.time, (int)f1, (int)nfb, (int)ib, S.ghost, 0});
    }
#endif
#ifdef MK_HAS_Q
    if (common.grid.curvedMesh != 0 && common.grid.curvedMesh != 1) return fail("unknown curvedMesh flag");
#endif
#ifdef MK_HAS_W
    if (ncw != 1) return fail("the generated GetW needs ncw == 1");
    if (!(common.timeparams.subproblem == 0 && common.timeparams.wave == 0 && std::fabs(common.timeparams.dae_alpha) < 1e-10 &&
          std::fabs(common.timeparams.dae_beta) < 1e-10 && EosBatchEnabled(common)))
        return fail("the generated GetW implements the batched EoS Newton branch only");
    S.wF = dmalloc((size_t)npe * ne); S.wdw = dmalloc((size_t)npe * ne);
#endif
    Kokkos::fence(); hipDeviceSynchronize();
    S.ok = true;
    if (common.mpiRank == 0) printf("mkstages: generated residual stages enabled (%s)\n", MK_SCHED);
    return true;
}

// ---- stages (signatures of ResidualStageTable) ----
inline int fsel(int pass) {
#ifdef MK_FSEL
    return pass;          // 0 all faces, 1 faces untouched by the halo, 2 faces adjacent to a ghost element
#else
    (void)pass; return 0;
#endif
}
// ---- block concurrency: each block's kernel chain on its own stream (blocks touch disjoint outputs and own scratch) ----
// MK_MSTREAM = streams in the pool (default 4; 0 or 1 = the blocks in sequence on the Kokkos stream, as before)
// device tables for one-launch-per-range dispatch (MK_BATCHED): the blocks [b1, b2) and each block's first workgroup
template <class T, class W> struct RangeTab { T* blocks = nullptr; int* wgoff = nullptr; int nb = 0, nwg = 0; };
template <class T, class W> inline RangeTab<T, W>& range_tab(const std::vector<T>& all, Int b1, Int b2, W wg) {
    static std::map<std::pair<Int, Int>, RangeTab<T, W>> m;
    auto key = std::make_pair(b1, b2); auto it = m.find(key); if (it != m.end()) return it->second;
    RangeTab<T, W> r; r.nb = (int)(b2 - b1); std::vector<int> wo(r.nb + 1, 0);
    for (int j = 0; j < r.nb; j++) wo[j + 1] = wo[j] + wg(all[b1 + j]);
    r.nwg = wo.back();
    hipMalloc(&r.blocks, std::max(1, r.nb) * sizeof(T)); hipMalloc(&r.wgoff, wo.size() * sizeof(int));
    hipMemcpy(r.blocks, &all[b1], r.nb * sizeof(T), hipMemcpyHostToDevice); hipMemcpy(r.wgoff, wo.data(), wo.size() * sizeof(int), hipMemcpyHostToDevice);
    return m.emplace(key, r).first->second;
}
inline void useslot(MkIn& a, const MkIn& z) { a.g_u = z.g_u; a.g_w = z.g_w; a.g_s = z.g_s; a.g_sg = z.g_sg; a.g_pr = z.g_pr; a.g_f = z.g_f; a.g_rg = z.g_rg; a.un = z.un; a.wn = z.wn; }
inline void useslot(MkFaceIn& a, const MkFaceIn& z) { a.gf1 = z.gf1; a.gug = z.gug; a.gat = z.gat; }
template <class F> inline void fanout(int nblocks, F&& f) {
    const int P = std::min(std::min(mstreams(), nslot()), nblocks);
    if (P <= 1) { cur_slot() = 0; for (int j = 0; j < nblocks; j++) f(j); return; }
    static std::vector<hipStream_t> pool; static std::vector<hipEvent_t> done; static hipEvent_t fork = nullptr;
    while ((int)pool.size() < P) { hipStream_t q; hipStreamCreateWithFlags(&q, hipStreamNonBlocking); pool.push_back(q);
                                   hipEvent_t ev; hipEventCreateWithFlags(&ev, hipEventDisableTiming); done.push_back(ev); }
    if (!fork) hipEventCreateWithFlags(&fork, hipEventDisableTiming);
    const hipStream_t main = Kokkos::HIP().hip_stream();
    hipEventRecord(fork, main);
    for (int p = 0; p < P; p++) hipStreamWaitEvent(pool[p], fork, 0);
    for (int j = 0; j < nblocks; j++) { mk_cur() = pool[j % P]; cur_slot() = j % P; f(j); }   // slot p only ever runs on stream p
    mk_cur() = nullptr; cur_slot() = 0;
    for (int p = 0; p < P; p++) { hipEventRecord(done[p], pool[p]); hipStreamWaitEvent(main, done[p], 0); }
}
inline void elem(Ctx& c, Int b1, Int b2) {
    State& S = st();
    if (!S.wpend.empty()) S.elem_calls.emplace_back(b1, b2);   // re-run if a deferred w check finds an unconverged block
    const dstype tm = c.common.timestate.time, dtf = c.common.timestate.dtfactor;
#ifdef MK_ELEM_ALL
    { auto& r = range_tab(S.blocks, b1, b2, [](const MkIn& a) { return mk_elem_wg(a); });
      mk_elem_all(r.blocks, r.wgoff, r.nb, r.nwg, tm, dtf); return; }
#endif
    fanout((int)(b2 - b1), [&](int k) { MkIn a = S.blocks[b1 + k]; a.time = tm; a.dtf = dtf; useslot(a, S.eslot[cur_slot()]); mk_run(a); }); }
#ifdef MK_HAS_FACE
#ifdef MK_FACE_PRODUCER
// the ghost sides of interior faces: the face interpolation kernel's sums for just those sides
__global__ void __launch_bounds__(256) mk_pfix(const State::PFix* fx, const int n, const dstype* udg, const dstype* wdg, const int* facecon, const dstype* gt) {
    const long t = (long)blockIdx.x * 256 + threadIdx.x; const int NCOL = MK_NC + MK_NCW;
    if (t >= (long)n * MK_NGF * NCOL) return;
    const int g = (int)(t % MK_NGF); const long r = t / MK_NGF; const int col = (int)(r % NCOL), it = (int)(r / NCOL);
    const State::PFix x = fx[it]; dstype mk_s = 0;
#pragma unroll
    for (int q = 0; q < MK_NPF; q++) { const int k = facecon[2*(MK_NPF*x.f + q) + x.side], m = k % MK_NPE, nn = k / MK_NPE;
        const dstype v = col < MK_NC ? udg[m + MK_NPE*col + (size_t)MK_NPE*MK_NC*nn] : wdg[m + MK_NPE*(col - MK_NC) + (size_t)MK_NPE*MK_NCW*nn];
        mk_s += gt[g + MK_NGF*q] * v; }
    x.dst[g + (size_t)x.nga * col] = mk_s; }
#endif
inline void face(Ctx& c) {
#ifdef MK_FACE_PRODUCER
    { State& S = st(); if (S.npfix > 0) { const long N = (long)S.npfix * MK_NGF * (MK_NC + MK_NCW);
        hipLaunchKernelGGL(mk_pfix, dim3((unsigned)((N + 255) / 256)), dim3(256), 0, mk_stream(), S.pfix, S.npfix, c.sol.udg, c.sol.wdg, c.mesh.facecon, c.master.shapfgt); } }
#endif
#ifdef MK_FACE_ALL
    { State& S = st(); const dstype tm = c.common.timestate.time; const Int ni = (Int)S.fint.size();
      if (ni > 0) {
        auto& ri = range_tab(S.fint, 0, ni, [](const MkFaceIn& a) { return mk_fi_wg(a); });
        auto& r1 = range_tab(S.fint, 0, ni, [](const MkFaceIn& a) { return mk_f1_wg(a); });
        auto& r2 = range_tab(S.fint, 0, ni, [](const MkFaceIn& a) { return mk_f2_wg(a); });
        mk_face_interior_all(ri.blocks, ri.wgoff, r1.wgoff, r2.wgoff, ri.nb, ri.nwg, r1.nwg, r2.nwg, tm); }
#ifdef MK_FACE_BOU_ALL
      if (!S.fbou.empty()) { const Int nb = (Int)S.fbou.size();
        auto& ri = range_tab(S.fbou, 0, nb, [](const MkFaceIn& a) { return mk_fib_wg(a); });
        auto& rb = range_tab(S.fbou, 0, nb, [](const MkFaceIn& a) { return mk_fb_wg(a); });
        mk_face_bou_all(ri.blocks, ri.wgoff, rb.wgoff, ri.nb, ri.nwg, rb.nwg, tm); }
#else
      for (MkFaceIn fb : S.fbou) { fb.time = tm; useslot(fb, S.fslot[0]); mk_face_block(fb); }
#endif
      return; }
#endif
    auto& fbs = st().fblocks; const dstype tm = c.common.timestate.time;
    fanout((int)fbs.size(), [&](int k) { MkFaceIn fb = fbs[k]; fb.time = tm; useslot(fb, st().fslot[cur_slot()]); mk_face_block(fb); }); }
#endif
#ifdef MK_HAS_UHAT
inline void uhat(Ctx& c, int pass) {
    const int fs = fsel(pass);
    mk_uhat_int(c.sol.uh, c.sol.udg, c.mesh.facecon, (int)c.common.meshsizes.nf, st().ghost, fs);
#ifdef MK_UHAT_BOU_ALL
    { auto& ubs = st().uhblocks; if (!ubs.empty()) {
        auto& r = range_tab(ubs, 0, (Int)ubs.size(), [](const MkUhIn& a) { return mk_ub_wg(a); });
        mk_uhat_bou_all(r.blocks, r.wgoff, r.nb, r.nwg, fs, c.common.timestate.time); } }
#else
    for (MkUhIn ub : st().uhblocks) { ub.fsel = fs; ub.time = c.common.timestate.time; mk_uhat_bou(ub); }
#endif
    }
#endif
#ifdef MK_HAS_Q
inline void q(Ctx& c, Int b1, Int b2, int pass) {
    if (b2 <= b1) return;
#if defined(MK_Q_HYBRID)
    if (pass == 2) {   // the ghost-side elements: face integrals inside the kernel, no RqFace launch
  #ifdef MK_Q_ALL
        { auto& r = range_tab(st().qblocks, b1, b2, [](const MkQIn& a) { return mk_q_wg(a); }); mk_qf_all(r.blocks, r.wgoff, r.nb, r.nwg); return; }
  #else
        fanout((int)(b2 - b1), [&](int k) { mk_qf_block(st().qblocks[b1 + k]); }); return;
  #endif
    }
#endif
#if defined(MK_Q_FACES)
    (void)pass;   // the face integrals are computed inside the q kernel
#elif defined(MK_RQFACE_ALL)
    mk_rqface_all(st().rq_dblocks, st().rq_wgoff, (int)st().rqblocks.size(), st().rq_nwg, st().ghost, fsel(pass));
#elif defined(MK_HAS_RQFACE)
    { auto& rbs = st().rqblocks; const int fs = fsel(pass); fanout((int)rbs.size(), [&](int k) { mk_rqface_block(rbs[k], st().ghost, fs); }); }
#else
    (void)pass; RqFace(c.sol, c.res, c.app, c.master, c.mesh, c.tmp, c.common, c.handle, 0, c.common.meshsizes.nbf, c.backend);
#endif
#ifdef MK_Q_ALL
    { auto& r = range_tab(st().qblocks, b1, b2, [](const MkQIn& a) { return mk_q_wg(a); }); mk_q_all(r.blocks, r.wgoff, r.nb, r.nwg); return; }
#endif
    fanout((int)(b2 - b1), [&](int k) { mk_q_block(st().qblocks[b1 + k]); }); }
#endif
#ifdef MK_HAS_W
// per-block EoS residual norms: production's EosBlockNorms2 kernel verbatim (same team policy -> same summation order,
// so the same convergence decisions), without its device->host copy
inline void w_norms_dev(const EosBlockBatch& b, const dstype* f) {
    using Policy = Kokkos::TeamPolicy<>;
    const int* off = b.off; double* nrm2 = b.nrm2;
    Kokkos::parallel_for("EosBlockNorms2", Policy(b.nb, Kokkos::AUTO), KOKKOS_LAMBDA(const Policy::member_type& team) {
        const int j = team.league_rank();
        double s = 0;
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, off[j], off[j+1]), [&](const int i, double& acc) {
            acc += (double)f[i]*(double)f[i]; }, s);
        Kokkos::single(Kokkos::PerTeam(team), [&]() { nrm2[j] = s; });
    }); }
// production's host test on the device: a block stops when sqrt(n2) < 1e-6; flag[0] = done (no block active),
// flag[1] counts the iterations that went on to an update (production's `it`)
inline void w_decide_dev(const EosBlockBatch& b, int* flag) {
    int* act = b.active; const double* nrm2 = b.nrm2; const int nb = b.nb;
    Kokkos::parallel_for("mk_w_decide", Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const int) {
        if (flag[0]) return;
        bool any = false;
        for (int j = 0; j < nb; j++) if (act[j]) { if (sqrt(nrm2[j]) < 1e-6) act[j] = 0; else any = true; }
        if (!any) flag[0] = 1; else flag[1] += 1; }); }
// fast block norms: chunks of <= 4096 points inside one EoS block, one 256-thread workgroup per chunk with a fixed-order
// tree reduction, then the decide kernel adds each block's chunk partials in order (deterministic; the norm differs from
// production's DOT only by summation order). MK_W_FASTNORM=0 keeps the team-per-block reduction.
inline int& w_fastnorm() { static int v = [] { const char* e = getenv("MK_W_FASTNORM"); return (e && e[0] == '0') ? 0 : 1; }(); return v; }
struct WChunks { int n = 0; int *start = nullptr, *end = nullptr, *cptr = nullptr; double* part = nullptr; };
inline WChunks& w_chunks(const EosBlockBatch& b) {
    static std::map<const EosBlockBatch*, WChunks> m;
    auto it = m.find(&b); if (it != m.end()) return it->second;
    std::vector<int> off(b.nb + 1); hipMemcpy(off.data(), b.off, off.size() * sizeof(int), hipMemcpyDeviceToHost);
    std::vector<int> st_, en_, cp(b.nb + 1, 0);
    for (int j = 0; j < b.nb; j++) { for (int a = off[j]; a < off[j+1]; a += 4096) { st_.push_back(a); en_.push_back(std::min(off[j+1], a + 4096)); } cp[j+1] = (int)st_.size(); }
    WChunks w; w.n = (int)st_.size();
    hipMalloc(&w.start, std::max(1, w.n) * sizeof(int)); hipMalloc(&w.end, std::max(1, w.n) * sizeof(int)); hipMalloc(&w.cptr, cp.size() * sizeof(int));
    hipMalloc(&w.part, std::max(1, w.n) * sizeof(double));
    hipMemcpy(w.start, st_.data(), st_.size() * sizeof(int), hipMemcpyHostToDevice); hipMemcpy(w.end, en_.data(), en_.size() * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(w.cptr, cp.data(), cp.size() * sizeof(int), hipMemcpyHostToDevice);
    return m.emplace(&b, w).first->second;
}
__global__ void __launch_bounds__(256) mk_w_partials(const dstype* f, const int* start, const int* end, double* part) {
    __shared__ double sm[256];
    const int c = blockIdx.x, t = threadIdx.x; double s = 0;
    for (int i = start[c] + t; i < end[c]; i += 256) s += (double)f[i] * (double)f[i];
    sm[t] = s; __syncthreads();
    for (int h = 128; h > 0; h >>= 1) { if (t < h) sm[t] += sm[t + h]; __syncthreads(); }
    if (t == 0) part[c] = sm[0];
}
__global__ void mk_w_decide_fast(int* act, double* nrm2, const double* part, const int* cptr, const int nb, int* flag) {
    if (flag[0]) return;
    bool any = false;
    for (int j = 0; j < nb; j++) { double s = 0; for (int c = cptr[j]; c < cptr[j+1]; c++) s += part[c]; nrm2[j] = s;
        if (act[j]) { if (sqrt(s) < 1e-6) act[j] = 0; else any = true; } }
    if (!any) flag[0] = 1; else flag[1] += 1;
}
// partials + decide in ONE launch: the last workgroup to finish (atomic ticket) runs mk_w_decide_fast's loop verbatim,
// so the sums and decisions are identical; it resets the ticket for the next launch
__global__ void __launch_bounds__(256) mk_w_partials_decide(const dstype* f, const int* start, const int* end, double* part,
        int* act, double* nrm2, const int* cptr, const int nb, int* flag, unsigned* ticket, const int nch, const int first, int* fh) {
    __shared__ double sm[256]; __shared__ bool last;
    const int c = blockIdx.x, t = threadIdx.x; double s = 0;
    for (int i = start[c] + t; i < end[c]; i += 256) s += (double)f[i] * (double)f[i];
    sm[t] = s; __syncthreads();
    for (int h = 128; h > 0; h >>= 1) { if (t < h) sm[t] += sm[t + h]; __syncthreads(); }
    if (t == 0) { part[c] = sm[0]; __threadfence(); last = atomicAdd(ticket, 1u) == (unsigned)(nch - 1); }
    __syncthreads();
    if (!last || t != 0) return;
    __threadfence(); *ticket = 0;
    // first: the call's EosBatchSetActive(all 1) + flag reset, folded in
    int f0 = first ? 0 : flag[0], f1 = first ? 0 : flag[1];
    if (f0) return;
    const volatile double* vp = part;
    bool any = false;
    for (int j = 0; j < nb; j++) { double s2 = 0; for (int k = cptr[j]; k < cptr[j+1]; k++) s2 += vp[k]; nrm2[j] = s2;
        int aj = first ? 1 : act[j]; if (aj) { if (sqrt(s2) < 1e-6) aj = 0; else any = true; } act[j] = aj; }
    if (!any) f0 = 1; else f1 += 1;
    flag[0] = f0; flag[1] = f1;
    if (fh) { fh[0] = f0; fh[1] = f1; __threadfence_system(); }   // straight to pinned host memory: no copy launch
}
__global__ void mk_w_reset(int* act, const int nb, int* flag) {   // EosBatchSetActive(all 1) + flag memset, on the device
    for (int j = threadIdx.x; j < nb; j += blockDim.x) act[j] = 1;
    if (threadIdx.x == 0) { flag[0] = 0; flag[1] = 0; } }
inline unsigned* w_ticket() { static unsigned* t = [] { unsigned* p; hipMalloc(&p, sizeof(unsigned)); hipMemset(p, 0, sizeof(unsigned)); return p; }(); return t; }
inline int& w_defer() { static int v = [] { const char* e = getenv("MK_W_DEFER"); return (e && e[0] == '0') ? 0 : 1; }(); return v; }
inline void w_norms_decide(const EosBlockBatch& b, const dstype* f, int* flag, int first = 0, int* fh = nullptr) {
    if (!w_fastnorm()) { w_norms_dev(b, f); w_decide_dev(b, flag); return; }
    if (w_defer()) { WChunks& w = w_chunks(b);
        hipLaunchKernelGGL(mk_w_partials_decide, dim3(w.n), dim3(256), 0, mk_stream(), f, w.start, w.end, w.part,
                           b.active, b.nrm2, w.cptr, b.nb, flag, w_ticket(), w.n, first, fh); return; }
    WChunks& w = w_chunks(b);
    hipLaunchKernelGGL(mk_w_partials, dim3(w.n), dim3(256), 0, mk_stream(), f, w.start, w.end, w.part);
    hipLaunchKernelGGL(mk_w_decide_fast, dim3(1), dim3(1), 0, mk_stream(), b.active, b.nrm2, w.part, w.cptr, b.nb, flag);
}
inline int& w_device() { static int v = [] { const char* e = getenv("MK_W_DEVICE"); return (e && e[0] == '0') ? 0 : 1; }(); return v; }   // settable for A/B
inline void w(Ctx& c, Int b1, Int b2) {
    if (b2 <= b1) return;
    State& S = st(); auto& sol = c.sol; auto& common = c.common; const Int npe = common.grid.npe;
    auto& bb = EosBlockBatchGet(common.eblks, b1, b2, npe);
    if (bb.nb <= 0) return;
    const MkWIn wa{&sol.xdg[npe*common.components.ncx*bb.E1], &sol.udg[npe*common.components.nc*bb.E1], &sol.odg[npe*common.components.nco*bb.E1],
                   &sol.wdg[npe*common.components.ncw*bb.E1], S.wF, S.wdw, c.app.physicsparam, c.app.uinf, c.app.tau, common.timestate.time,
                   (int)(npe * (bb.E2 - bb.E1))};
    const bool unfrozenAV = common.physicsparams.ncAV > 0 && common.physicsparams.frozenAVflag == 0;
#ifndef MK_GEN_SCATTER
    const bool can_defer = false;    // the deferred check is resolved in the generated tail
#else
    const bool can_defer = true;
#endif
    if (can_defer && w_device() && w_fastnorm() && w_defer() && !unfrozenAV) {
        // Same iterations, no host synchronisation here: launch the iterations the previous residual needed (+1 to see
        // convergence; iterations after convergence are masked no-ops) and copy the flags to pinned memory. tail()
        // checks them once per residual; an unconverged range continues there and its consumers (elem, face) re-run.
        struct Fl { int* d; int* h; }; static std::map<const EosBlockBatch*, Fl> fl;
        auto it = fl.find(&bb); if (it == fl.end()) { Fl f; hipMalloc(&f.d, 2*sizeof(int)); hipHostMalloc(&f.h, 2*sizeof(int)); it = fl.emplace(&bb, f).first; }
        int* flag = it->second.d;
        static const int cap = [] { const char* e = getenv("MK_W_FIRST"); return e ? std::max(1, atoi(e)) : 10; }();   // test knob: force the fallback
        const int first = std::min(cap, S.w_guess + 1);
        for (int k = 0; k < first; k++) { mk_w_fused(wa); w_norms_decide(bb, S.wF, flag, k == 0, it->second.h);
            EosMaskedUpdate(bb, &sol.wdg[npe*common.components.ncw*bb.E1], S.wdw, npe); }
        S.wpend.push_back({&bb, wa, b1, b2, flag, it->second.h, first}); S.w_calls++;
        return;
    }
    std::vector<int> act(bb.nb, 1); EosBatchSetActive(bb, act);
    if (w_device()) {
        // Same iterations as below with no host round trip per iteration: the convergence test runs on the device and an
        // iteration after convergence changes nothing (every block is masked), so the host checks once, after the
        // iteration count the previous call needed (+1 to see it converge), and continues one at a time if not done.
        static int* flag = nullptr; if (!flag) hipMalloc(&flag, 2*sizeof(int));
        hipMemsetAsync(flag, 0, 2*sizeof(int), mk_stream());
        auto iter = [&]() { mk_w_fused(wa); w_norms_decide(bb, S.wF, flag);
                            EosMaskedUpdate(bb, &sol.wdg[npe*common.components.ncw*bb.E1], S.wdw, npe); };
        int h[2] = {0, 0}, launched = 0;
        const int first = std::min(10, S.w_guess + 1);
        for (; launched < first; launched++) iter();
        auto peek = [&]() { hipMemcpyAsync(h, flag, 2*sizeof(int), hipMemcpyDeviceToHost, mk_stream()); hipStreamSynchronize(mk_stream()); };
        peek();
        while (!h[0] && launched < 10) { iter(); launched++; peek(); }
        S.w_guess = h[1]; S.w_iters_max = std::max(S.w_iters_max, h[1]);
        return;
    }
    int it = 0;
    for (int iter = 0; iter < 10; iter++) {
        mk_w_fused(wa);
        const std::vector<double> n2 = EosBlockNorms2(bb, S.wF);
        bool any = false;
        for (int j = 0; j < bb.nb; j++) if (act[j]) { if (std::sqrt(n2[j]) < 1e-6) act[j] = 0; else any = true; }
        if (!any) break;
        EosBatchSetActive(bb, act);
        EosMaskedUpdate(bb, &sol.wdg[npe*common.components.ncw*bb.E1], S.wdw, npe); it++;
    }
    S.w_iters_max = std::max(S.w_iters_max, it); }
#endif
#ifdef MK_HAS_W
inline void elem(Ctx& c, Int b1, Int b2);
#ifdef MK_HAS_FACE
inline void face(Ctx& c);
#endif
// the deferred w checks of this residual: one synchronisation; continue any unconverged range exactly as the synchronous
// path would (one iteration at a time until converged or 10), then re-run what consumed w (elem calls, face)
inline void w_resolve(Ctx& c) {
    State& S = st(); if (S.wpend.empty()) return;
    hipStreamSynchronize(mk_stream());
    bool redo = false; int need = 0; const Int npe = c.common.grid.npe;
    for (auto& p : S.wpend) {
        int launched = p.launched;
        while (!p.fh[0] && launched < 10) {
            mk_w_fused(p.wa); w_norms_decide(*p.bb, S.wF, p.fd, 0, p.fh);
            EosMaskedUpdate(*p.bb, &c.sol.wdg[npe*c.common.components.ncw*p.bb->E1], S.wdw, npe); launched++;
            hipStreamSynchronize(mk_stream()); redo = true; }
        need = std::max(need, p.fh[1]); S.w_iters_max = std::max(S.w_iters_max, p.fh[1]); }
    S.w_guess = need;
    auto calls = S.elem_calls; S.wpend.clear(); S.elem_calls.clear();
    if (!redo) return;
    S.w_redo++;
    for (auto& e : calls) elem(c, e.first, e.second);
#ifdef MK_HAS_FACE
    face(c);
#else
    ProductionResidualStages<M>::face(c);
#endif
}
#endif
#ifdef MK_GEN_SCATTER
inline void tail(Ctx& c) {
#ifdef MK_HAS_W
    w_resolve(c);
#endif
    const dstype* Rh = c.res.Rh; dstype* Ru = c.res.Ru; const int* np_ = st().nodeptr; const int* cn = st().contrib;
    const dstype sc = one / c.common.timestate.dtfactor; const int nrow = (int)(c.common.grid.npe * c.common.meshsizes.ne);
    const int tdep = (int)c.common.timeparams.tdep, N = nrow * MK_NCU;
    Kokkos::parallel_for("mk_scatter_finalize", Kokkos::RangePolicy<>(0, N), KOKKOS_LAMBDA(const int t) {
        const int node = t % nrow, j = t / nrow, mm = node % MK_NPE, e = node / MK_NPE;
        const size_t ix = mm + (size_t)MK_NPE * j + (size_t)MK_NPE * MK_NCU * e;
        dstype s = Ru[ix];
        for (int r = np_[node]; r < np_[node + 1]; r++) { const int ci = cn[r], ii = ci >> 1;
            const dstype v = Rh[(ii % MK_NPF) + MK_NPF * (j + MK_NCU * (ii / MK_NPF))]; if (ci & 1) s += v; else s -= v; }
        s = (dstype)(-1.0) * s; if (tdep == 1) s = sc * s;       // ArrayMultiplyScalar(Ru, minusone), then (1/dtfactor)
        Ru[ix] = s; }); }
#endif

// the table offered to SetResidualStages; building happens on the first residual (the stage entries call build)
inline bool ready(Ctx& c) { return build(c); }
#define MKS_WRAP(ret_call, prod_call) do { if (ready(c)) { ret_call; } else { prod_call; } } while (0)
using P = ProductionResidualStages<M>;
inline void t_uhat(Ctx& c, int pass) {
#ifdef MK_HAS_UHAT
    MKS_WRAP(uhat(c, pass), P::uhat(c, pass));
#else
    P::uhat(c, pass);
#endif
}
inline void t_q(Ctx& c, Int b1, Int b2, int pass) {
#ifdef MK_HAS_Q
    MKS_WRAP(q(c, b1, b2, pass), P::q(c, b1, b2, pass));
#else
    P::q(c, b1, b2, pass);
#endif
}
inline void t_w(Ctx& c, Int b1, Int b2) {
#ifdef MK_HAS_W
    MKS_WRAP(w(c, b1, b2), P::w(c, b1, b2));
#else
    P::w(c, b1, b2);
#endif
}
inline void t_elem(Ctx& c, Int b1, Int b2) { if (MK_NK > 0) { MKS_WRAP(elem(c, b1, b2), P::elem(c, b1, b2)); } else P::elem(c, b1, b2); }
inline void t_face(Ctx& c) {
#ifdef MK_HAS_FACE
    MKS_WRAP(face(c), P::face(c));
#else
    P::face(c);
#endif
}
inline void t_tail(Ctx& c) {
#ifdef MK_GEN_SCATTER
    MKS_WRAP(tail(c), P::tail(c));
#else
    P::tail(c);
#endif
}
inline const ResidualStageTable* table() {
    static ResidualStageTable t = [] { ResidualStageTable v; v.name = MK_SCHED;
        v.uhat = t_uhat; v.q = t_q; v.w = t_w; v.elem = t_elem; v.face = t_face; v.tail = t_tail; return v; }();
    return &t;
}
}  // namespace mkstages
