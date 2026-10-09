// Meta-kernel driver: one generated schedule (schedule.hpp) against the production staged element residual.
// Real state from datain (sol1.bin, master.bin, app.bin); geometry (Xx, jac) and sdg are synthetic but shared.
// Output lines:  MKRES <sched> <total ms> <ref ms> <speedup> <rel diff> <bitwise 0/1>
//                MKKER <sched> <j> <ms> <label>
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
typedef double dstype;
#include "mfma_gemm.hpp"
#include "KokkosFlux.cpp"
#include "KokkosSource.cpp"
#include "mk_common.hpp"
#include "schedule.hpp"

static std::vector<double> readall(const std::string& f) {
    std::ifstream in(f, std::ios::binary | std::ios::ate); size_t n = in.tellg() / 8; in.seekg(0);
    std::vector<double> v(n); in.read((char*)v.data(), n * 8); return v;
}
static std::vector<std::vector<double>> sections(const std::vector<double>& a) {
    int L = (int)a[0]; std::vector<int> ns(L); for (int i = 0; i < L; i++) ns[i] = (int)a[1 + i];
    std::vector<std::vector<double>> out; size_t o = 1 + L;
    for (int i = 0; i < L; i++) { out.emplace_back(a.begin() + o, a.begin() + o + ns[i]); o += ns[i]; }
    return out;
}
using V = Kokkos::View<double*>;
static V up(const std::vector<double>& h) { V d("d", h.size()); auto hv = Kokkos::create_mirror_view(d); for (size_t i = 0; i < h.size(); i++) hv(i) = h[i]; Kokkos::deep_copy(d, hv); return d; }

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
    const std::string D = argv[1];
    auto M = sections(readall(D + "/master.bin")), A = sections(readall(D + "/app.bin")), S = sections(readall(D + "/sol1.bin"));
    const int nd = (int)M[0][0], npe = (int)M[0][5], nge = (int)M[0][7];
    const int nc = (int)A[0][5], ncu = (int)A[0][6], nco = (int)A[0][9], ncx = (int)A[0][11], ncw = (int)A[0][13];
    const std::vector<double>& shapegt = M[1], &shapegw = M[2], &uinfh = A[3], &param = A[6];
    const std::vector<double>& xdg = S[1], &udg = S[2], &odg = S[3], &wdg = S[4];
    const int ne = (int)(udg.size() / ((size_t)npe * nc)), nga = nge * ne, ndp1 = nd + 1;
    if (nd != MK_ND || npe != MK_NPE || nge != MK_NGE || nc != MK_NC || ncu != MK_NCU || nco != MK_NCO || ncx != MK_NCX || ncw != MK_NCW) {
        printf("case sizes do not match mk_common.hpp\n"); return 1; }
    auto interp = [&](const std::vector<double>& nod, int ncomp) {
        std::vector<double> g((size_t)nga * ncomp, 0.0);
        for (int e = 0; e < ne; e++) for (int k = 0; k < ncomp; k++) for (int q = 0; q < nge; q++) {
            double s = 0; for (int p = 0; p < npe; p++) s += shapegt[q + nge * p] * nod[p + (size_t)npe * (k + (size_t)ncomp * e)];
            g[(q + (size_t)nge * e) + (size_t)nga * k] = s; }
        return g; };
    std::vector<double> odgg = interp(odg, nco), xg = interp(xdg, ncx);
    std::vector<double> Xx((size_t)nga * nd * nd, 0.0), jac((size_t)nga, 1.0), sdgg((size_t)nga * ncu, 0.0);
    for (size_t i = 0; i < (size_t)nga; i++) { for (int d = 0; d < nd; d++) Xx[i + (size_t)nga * (d + nd * d)] = 2.0 + 0.1 * d;
        for (int d = 0; d < nd; d++) for (int c = 0; c < nd; c++) if (c != d) Xx[i + (size_t)nga * (c + nd * d)] = 0.01 * (1 + c + 2 * d);
        jac[i] = 0.125 + 1e-3 * (i % 7); for (int k = 0; k < ncu; k++) sdgg[i + (size_t)nga * k] = 1e-3 * ((i + k) % 5); }
    std::vector<int> eind((size_t)npe * nc * ne);
    for (int e = 0; e < ne; e++) for (int k = 0; k < nc; k++) for (int p = 0; p < npe; p++) eind[p + (size_t)npe * (e + (size_t)ne * k)] = p + npe * k + npe * nc * e;
    const double dtfactor = 1.0 / 3.0e-5, time = 1e-4;

    V d_udg = up(udg), d_wdg = up(wdg), d_odgg = up(odgg), d_xg = up(xg), d_Xx = up(Xx), d_jac = up(jac), d_sdgg = up(sdgg);
    V d_gt = up(shapegt), d_gw = up(shapegw), d_uinf = up(uinfh), d_par = up(param);
    Kokkos::View<int*> d_eind("eind", eind.size()); { auto h = Kokkos::create_mirror_view(d_eind); for (size_t i = 0; i < eind.size(); i++) h(i) = eind[i]; Kokkos::deep_copy(d_eind, h); }
    V un("un", (size_t)npe * nc * ne), wn("wn", (size_t)npe * ncw * ne);
    V gu("gu", (size_t)nga * nc), gw_("gw", (size_t)nga * ncw), gs("gs", (size_t)nga * ncu), gsg("gsg", (size_t)nga * ncu), gpr("gpr", (size_t)nga * MK_NPR),
      gf("gf", (size_t)nga * MK_NF), grg("grg", (size_t)nga * MK_NRG), R("R", (size_t)npe * ncu * ne), Rref("Rref", (size_t)npe * ncu * ne);
    // ---- reference: the production staged element residual ----
    // ---- reference: the production staged element residual, as separately timeable steps ----
    dstype *pu = d_udg.data(), *pw = d_wdg.data(), *pun = un.data(), *pwn = wn.data(), *pug = gu.data(), *pwg = gw_.data(), *pfg = gf.data(), *psg = gsg.data(), *prg = grg.data(), *ps = gs.data();
    const dstype *pgt = d_gt.data(), *pgw = d_gw.data(), *po = d_odgg.data(), *px = d_xg.data(), *pXx = d_Xx.data(), *pj = d_jac.data(), *psd = d_sdgg.data(), *puinf = d_uinf.data(), *ppar = d_par.data();
    const int* pe = d_eind.data(); dstype* pRref = Rref.data();
    auto r_interp = [=]() {
        Kokkos::parallel_for("gather", (size_t)npe * nc * ne, KOKKOS_LAMBDA(const size_t i) { pun[i] = pu[pe[i]]; });
        exasim_mfma::gemm_nn(pug, pgt, pun, nge, npe, ne * nc, nge);
        Kokkos::parallel_for("gatherw", (size_t)npe * ncw * ne, KOKKOS_LAMBDA(const size_t idx) {
            const int p = idx % npe; const size_t r = idx / npe; const int e = r % ne, k = r / ne; pwn[idx] = pw[p + npe * k + (size_t)npe * ncw * e]; });
        exasim_mfma::gemm_nn(pwg, pgt, pwn, nge, npe, ne * ncw, nge); };
    auto r_source = [=]() { KokkosSource(ps, px, pug, po, pwg, puinf, ppar, time, 0, nga, nc, ncu, nd, ncx, nco, ncw); };
    auto r_time = [=]() { Kokkos::parallel_for("timeterm", (size_t)nga * ncu, KOKKOS_LAMBDA(const size_t i) { double t = 1.0 * psd[i] + (-dtfactor) * pug[i]; psg[i] = 1.0 * t + 1.0 * ps[i]; }); };
    auto r_flux = [=]() { KokkosFlux(pfg, px, pug, po, pwg, puinf, ppar, time, 0, nga, nc, ncu, nd, ncx, nco, ncw); };
    auto r_scale = [=]() {
        const int Mm = nga, N = Mm * ncu, P = Mm * nd, I = nge * ndp1, J = I * ncu;
        Kokkos::parallel_for("ApplyXx4", (size_t)N, KOKKOS_LAMBDA(const size_t idx) {
            const int i = idx % Mm, k = idx / Mm, g = i % nge, e = i / nge, ge = g + nge * e, ke = I * k + J * e;
            prg[g + ke] = psg[idx] * pj[i];
            for (int m = 0; m < nd; m++) { const int gem = ge + P * m; double s = pfg[idx] * pXx[gem];
                for (int j = 1; j < nd; j++) s += pfg[idx + (size_t)N * j] * pXx[gem + Mm * j];
                prg[g + nge * (m + 1) + ke] = s; }
        }); };
    auto r_integrate = [=]() { exasim_mfma::gemm_nn(pRref, pgw, prg, npe, nge * ndp1, ncu * ne, npe); };
    auto reference = [&]() { r_interp(); r_source(); r_time(); r_flux(); r_scale(); r_integrate(); };
    auto timeit = [&](auto&& f, int reps) { f(); Kokkos::fence(); std::vector<double> t;
        for (int r = 0; r < reps; r++) { Kokkos::Timer tm; f(); Kokkos::fence(); t.push_back(tm.seconds() * 1e3); }
        std::sort(t.begin(), t.end()); return t[t.size() / 2]; };
    const double t_ref = timeit(reference, 30);
    auto href = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Rref);
    { const char* nm[] = {"interp", "source", "time", "flux", "scale", "integrate"}; double tt[6];
      tt[0] = timeit(r_interp, 30); tt[1] = timeit(r_source, 30); tt[2] = timeit(r_time, 30); tt[3] = timeit(r_flux, 30); tt[4] = timeit(r_scale, 30); tt[5] = timeit(r_integrate, 30);
      for (int j = 0; j < 6; j++) printf("MKREF %s %d %.4f %s\n", MK_SCHED, j, tt[j], nm[j]); }

    MkIn a{d_xg.data(), d_odgg.data(), d_udg.data(), d_wdg.data(), d_sdgg.data(), d_jac.data(), d_Xx.data(), d_gt.data(), d_gw.data(), d_uinf.data(), d_par.data(),
           gu.data(), gw_.data(), gs.data(), gsg.data(), gpr.data(), gf.data(), grg.data(), R.data(), un.data(), wn.data(), d_eind.data(), d_udg.data(), time, dtfactor, nga, ne};
    Kokkos::deep_copy(gu, 0.0); Kokkos::deep_copy(grg, 0.0); Kokkos::deep_copy(gf, 0.0);   // no stale values from the reference
    const double t = timeit([&]() { mk_run(a); }, 30);
    auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), R);
    double md = 0, mx = 0; size_t ndiff = 0, nf = 0;
    for (size_t i = 0; i < h.extent(0); i++) { if (h(i) != href(i)) ndiff++; if (!std::isfinite(h(i))) nf++;
        if (std::isfinite(href(i))) { mx = std::max(mx, std::fabs(href(i))); md = std::max(md, std::fabs(h(i) - href(i))); } }
    printf("MKRES %s %.4f %.4f %.3f %.2e %d nonfinite=%zu backend=%s\n", MK_SCHED, t, t_ref, t_ref / t, md / (mx + 1e-300), ndiff == 0, nf, MK_BACKEND);
    for (int j = 0; j < MK_NK; j++) printf("MKKER %s %d %.4f %s\n", MK_SCHED, j, timeit([&]() { mk_run_kernel(a, j); }, 30), MK_KLABEL[j]);
    }
    Kokkos::finalize();
    return 0;
}
