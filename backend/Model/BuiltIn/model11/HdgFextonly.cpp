void HdgFextonly1(dstype* f, const dstype* x, const dstype* uq, const dstype* v, const dstype* w, const dstype* uhat, const dstype* n, const dstype* uext, const dstype* tau, const dstype* eta, const dstype* mu, const dstype t, const int modelnumber, const int N, const int szx, const int szuq, const int szv, const int szw, const int szuhat, const int szn, const int szuext, const int sztau, const int szeta, const int szmu)
{

  Kokkos::parallel_for("Fext", N, KOKKOS_LAMBDA(const size_t i) {
    dstype uq0 = uq[0*N+i];
    dstype uhat0 = uhat[0*N+i];
    dstype uhat1 = uhat[1*N+i];
    dstype uhat2 = uhat[2*N+i];
    dstype uhat3 = uhat[3*N+i];
    dstype uext0 = uext[0*N+i];
    dstype mu8 = mu[8];
    dstype mu9 = mu[9];


    f[0 * N + i] = -uhat0 + uq0;
    f[1 * N + i] = -uhat1;
    f[2 * N + i] = -uhat2;
    f[3 * N + i] = -uhat3 + mu8*uhat0*uext0/mu9;
  });
}

void HdgFextonly(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg,
           const dstype* nlg, const dstype* uext, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time,
           const int modelnumber, const int ib, const int ng, const int nc, const int ncu, const int nd,
           const int ncx, const int nco, const int ncw) {
    if (ib == 1 )
        HdgFextonly1(f, xdg, udg, odg, wdg, uhg, nlg, uext, tau, uinf, param, time, modelnumber,
                        ng, nc, ncu, nd, ncx, nco, ncw, nc, ncu, nd, ncx);
}
