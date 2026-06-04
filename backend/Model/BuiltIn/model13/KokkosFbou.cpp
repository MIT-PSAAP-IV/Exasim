void KokkosFbou1(dstype* f, const dstype* x, const dstype* uq, const dstype* v, const dstype* w, const dstype* uhat, const dstype* n, const dstype* tau, const dstype* eta, const dstype* mu, const dstype t, const int modelnumber, const int N, const int szx, const int szuq, const int szv, const int szw, const int szuhat, const int szn, const int sztau, const int szeta, const int szmu)
{

  Kokkos::parallel_for("Fbou", N, KOKKOS_LAMBDA(const size_t i) {
    dstype x1 = x[1*N+i];
    dstype uq0 = uq[0*N+i];
    dstype uq1 = uq[1*N+i];
    dstype uq2 = uq[2*N+i];
    dstype uhat0 = uhat[0*N+i];
    dstype n0 = n[0*N+i];
    dstype n1 = n[1*N+i];
    dstype tau0 = tau[0];
    dstype mu0 = mu[0];

    dstype x0 = x1*mu0;

    f[0 * N + i] = tau0*(-uhat0 + uq0) + n0*x0*uq1 + n1*x0*uq2;
  });
}

void KokkosFbouJac1(dstype* f, dstype* J1, dstype* J2, dstype* J3, const dstype* x, const dstype* uq, const dstype* v, const dstype* w, const dstype* uhat, const dstype* n, const dstype* tau, const dstype* eta, const dstype* mu, const dstype t, const int modelnumber, const int N, const int szx, const int szuq, const int szv, const int szw, const int szuhat, const int szn, const int sztau, const int szeta, const int szmu)
{

  Kokkos::parallel_for("Fbou", N, KOKKOS_LAMBDA(const size_t i) {
    dstype x1 = x[1*N+i];
    dstype uq0 = uq[0*N+i];
    dstype uq1 = uq[1*N+i];
    dstype uq2 = uq[2*N+i];
    dstype uhat0 = uhat[0*N+i];
    dstype n0 = n[0*N+i];
    dstype n1 = n[1*N+i];
    dstype tau0 = tau[0];
    dstype mu0 = mu[0];
    dstype x0 = x1*mu0;
    dstype x2 = n1*x0;
    dstype x3 = n0*x0;

    f[0 * N + i] = tau0*(-uhat0 + uq0) + x2*uq2 + x3*uq1;
    J1[0 * N + i] = tau0;
    J1[1 * N + i] = x3;
    J1[2 * N + i] = x2;
    J3[0 * N + i] = -tau0;
  });
}

void KokkosFbou(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg,
           const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time,
           const int modelnumber, const int ib, const int ng, const int nc, const int ncu, const int nd,
           const int ncx, const int nco, const int ncw) {
    if (ib == 1 )
        KokkosFbou1(f, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, modelnumber,
                        ng, nc, ncu, nd, ncx, nco, ncw, nc, ncu, nd);
}
void KokkosFbouJac(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg,
           const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time,
           const int modelnumber, const int ib, const int ng, const int nc, const int ncu, const int nd,
           const int ncx, const int nco, const int ncw) {
    if (ib == 1 )
        KokkosFbouJac1(f, f_udg, f_wdg, f_uhg, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, modelnumber,
                        ng, nc, ncu, nd, ncx, nco, ncw, nc, ncu, nd);
}
