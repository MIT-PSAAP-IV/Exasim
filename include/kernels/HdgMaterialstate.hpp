template <class Model>
static void HdgMaterialstateTemplate(dstype* f, dstype* f_udg, dstype* f_wdg,
                                     const dstype* xdg, const dstype* udg,
                                     const dstype* odg, const dstype* wdg,
                                     const dstype* uinf, const dstype* param,
                                     const dstype time, const int modelnumber,
                                     const int ng, const int nc_runtime,
                                     const int ncu_runtime,
                                     const int nd_runtime, const int ncx,
                                     const int nco_runtime,
                                     const int ncw_runtime,
                                     const int nmaterialstate_runtime)
{
    constexpr int nd = Model::nd;
    constexpr int ncu = Model::ncu;
    constexpr int nc = ncu * (1 + nd);
    constexpr int nco = Model::nco;
    constexpr int ncw = Model::ncw;
    constexpr int nmaterialstate = Model::nmaterialstate;

    (void)modelnumber;
    (void)nc_runtime;
    (void)ncu_runtime;
    (void)nd_runtime;
    (void)ncx;
    (void)nco_runtime;
    (void)ncw_runtime;
    (void)nmaterialstate_runtime;

    Kokkos::parallel_for("HdgMaterialstate", ng, KOKKOS_LAMBDA(const size_t i) {
        dstype x[nd];
        dstype uq[nc];
        dstype v[(nco > 0) ? nco : 1];
        dstype w[(ncw > 0) ? ncw : 1];
        dstype state[(nmaterialstate > 0) ? nmaterialstate : 1];
        dstype state_uq[(nmaterialstate * nc > 0) ? nmaterialstate * nc : 1];
        dstype state_w[(nmaterialstate * ncw > 0) ? nmaterialstate * ncw : 1];

        for (int k = 0; k < nd; ++k) x[k] = xdg[k * ng + i];
        for (int k = 0; k < nc; ++k) uq[k] = udg[k * ng + i];
        for (int k = 0; k < nco; ++k) v[k] = odg[k * ng + i];
        for (int k = 0; k < ncw; ++k) w[k] = wdg[k * ng + i];

        Model::materialstate(state, x, uq, v, w, param, uinf, time);

        for (int k = 0; k < nmaterialstate; ++k) f[k * ng + i] = state[k];
        Model::materialstate_jac_uq(state_uq, x, uq, v, w, param, uinf, time);
        for (int k = 0; k < nmaterialstate * nc; ++k) f_udg[k * ng + i] = state_uq[k];
        Model::materialstate_jac_w(state_w, x, uq, v, w, param, uinf, time);
        for (int k = 0; k < nmaterialstate * ncw; ++k) f_wdg[k * ng + i] = state_w[k];
    });
}

void HdgMaterialstate(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xdg,
                      const dstype* udg, const dstype* odg, const dstype* wdg,
                      const dstype* uinf, const dstype* param, const dstype time,
                      const int modelnumber, const int ng, const int nc,
                      const int ncu, const int nd, const int ncx,
                      const int nco, const int ncw,
                      const int nmaterialstate)
{
    HdgMaterialstateTemplate<PdeModel>(f, f_udg, f_wdg, xdg, udg, odg, wdg,
                                       uinf, param, time, modelnumber, ng, nc,
                                       ncu, nd, ncx, nco, ncw,
                                       nmaterialstate);
}
