void KokkosFlux(dstype* f, const dstype* x, const dstype* uq, const dstype* v, const dstype* w, const dstype* eta, const dstype* mu, const dstype t, const int modelnumber, const int N, const int szx, const int szuq, const int szv, const int szw, const int szeta, const int szmu)
{

  Kokkos::parallel_for("Flux", N, KOKKOS_LAMBDA(const size_t i) {
    dstype uq2 = uq[2*N+i];
    dstype uq3 = uq[3*N+i];
    dstype uq4 = uq[4*N+i];
    dstype uq5 = uq[5*N+i];
    dstype mu0 = mu[0];
    dstype mu1 = mu[1];


    f[0 * N + i] = uq2*mu0;
    f[1 * N + i] = uq3*mu0;
    f[2 * N + i] = uq4*mu1;
    f[3 * N + i] = uq5*mu1;
  });
}

