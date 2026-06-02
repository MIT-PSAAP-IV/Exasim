template <class Model>
static void KokkosUbouJacTemplate(dstype* f, dstype* f_udg,
                                  dstype* f_wdg, dstype* f_uhg,
                                  const dstype* xdg, const dstype* udg,
                                  const dstype* odg, const dstype* wdg,
                                  const dstype* uhg, const dstype* nlg,
                                  const dstype* tau, const dstype* uinf,
                                  const dstype* param, const dstype time,
                                  const int modelnumber, const int ib,
                                  const int ng, const int nc_runtime,
                                  const int ncu_runtime,
                                  const int nd_runtime, const int ncx,
                                  const int nco_runtime,
                                  const int ncw_runtime)
{
    constexpr int nd = Model::nd;
    constexpr int ncu = Model::ncu;
    constexpr int nc = ncu * (1 + nd);
    constexpr int nco = Model::nco;
    constexpr int ncw = Model::ncw;

    (void)modelnumber;
    (void)nc_runtime;
    (void)ncu_runtime;
    (void)nd_runtime;
    (void)ncx;
    (void)nco_runtime;
    (void)ncw_runtime;

    Kokkos::parallel_for("UbouJac", ng, KOKKOS_LAMBDA(const size_t i) {
        constexpr int nd = Model::nd;
        constexpr int ncu = Model::ncu;
        constexpr int nc = ncu * (1 + nd);
        constexpr int nco = Model::nco;
        constexpr int ncw = Model::ncw;
        constexpr int ntau = Model::ntau;
        dstype x[nd];
        dstype uq[nc];
        dstype v[(nco > 0) ? nco : 1];
        dstype w[(ncw > 0) ? ncw : 1];
        dstype uh[ncu];
        dstype n[nd];
        dstype tau_local[ntau];
        dstype ub_local[ncu];
        dstype ub_uq_local[ncu * nc];
        dstype ub_w_local[(ncw > 0) ? ncu * ncw : 1];
        dstype ub_uh_local[ncu * ncu];

        for (int k = 0; k < nd; ++k) x[k] = xdg[k * ng + i];
        for (int k = 0; k < nc; ++k) uq[k] = udg[k * ng + i];
        for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
        for (int k = 0; k < ncw; ++k) w[k] = wdg[k * ng + i];
        for (int k = 0; k < ncu; ++k) uh[k] = uhg[k * ng + i];
        for (int k = 0; k < nd; ++k) n[k] = nlg[k * ng + i];
        for (int k = 0; k < ntau; ++k) tau_local[k] = tau[k];

        Model::ubou(ub_local, ib, x, uq, v, w, uh, n, tau_local, param,
                    uinf, time);
        Model::ubou_jac_uq(ub_uq_local, ib, x, uq, v, w, uh, n, tau_local,
                           param, uinf, time);
        Model::ubou_jac_w(ub_w_local, ib, x, uq, v, w, uh, n, tau_local,
                          param, uinf, time);
        Model::ubou_jac_uh(ub_uh_local, ib, x, uq, v, w, uh, n, tau_local,
                           param, uinf, time);

        for (int k = 0; k < ncu; ++k) f[k * ng + i] = ub_local[k];
        for (int k = 0; k < ncu * nc; ++k) f_udg[k * ng + i] = ub_uq_local[k];
        for (int k = 0; k < ncu * ncw; ++k) f_wdg[k * ng + i] = ub_w_local[k];
        for (int k = 0; k < ncu * ncu; ++k) f_uhg[k * ng + i] = ub_uh_local[k];
    });
}

void KokkosUbouJac(dstype* f, dstype* f_udg, dstype* f_wdg,
                   dstype* f_uhg, const dstype* xdg, const dstype* udg,
                   const dstype* odg, const dstype* wdg, const dstype* uhg,
                   const dstype* nlg, const dstype* tau,
                   const dstype* uinf, const dstype* param,
                   const dstype time, const int modelnumber, const int ib,
                   const int ng, const int nc, const int ncu, const int nd,
                   const int ncx, const int nco, const int ncw)
{
    KokkosUbouJacTemplate<PdeModel>(f, f_udg, f_wdg, f_uhg, xdg, udg, odg,
                                    wdg, uhg, nlg, tau, uinf, param, time,
                                    modelnumber, ib, ng, nc, ncu, nd, ncx,
                                    nco, ncw);
}
