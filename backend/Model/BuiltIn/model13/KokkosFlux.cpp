void KokkosFlux(dstype* f, const dstype* x, const dstype* uq, const dstype* v, const dstype* w, const dstype* eta, const dstype* mu, const dstype t, const int modelnumber, const int N, const int szx, const int szuq, const int szv, const int szw, const int szeta, const int szmu)
{

  Kokkos::parallel_for("Flux", N, KOKKOS_LAMBDA(const size_t i) {
    dstype x1 = x[1*N+i];
    dstype uq1 = uq[1*N+i];
    dstype uq2 = uq[2*N+i];
    dstype mu0 = mu[0];

    dstype x0 = x1*mu0;

    f[0 * N + i] = x0*uq1;
    f[1 * N + i] = x0*uq2;
  });
}

