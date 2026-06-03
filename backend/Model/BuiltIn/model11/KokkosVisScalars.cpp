void KokkosVisScalars(dstype* f, const dstype* x, const dstype* uq, const dstype* v, const dstype* w, const dstype* eta, const dstype* mu, const dstype t, const int modelnumber, const int N, const int szx, const int szuq, const int szv, const int szw, const int szeta, const int szmu)
{

  Kokkos::parallel_for("VisScalars", N, KOKKOS_LAMBDA(const size_t i) {
    dstype uq0 = uq[0*N+i];
    dstype uq1 = uq[1*N+i];
    dstype uq2 = uq[2*N+i];
    dstype uq3 = uq[3*N+i];
    dstype mu0 = mu[0];
    dstype mu1 = mu[1];
    dstype mu3 = mu[3];
    dstype mu9 = mu[9];

    dstype x0 = pow(uq0, -1);
    dstype x1 = pow(uq0, -2);
    dstype x2 = uq3 - 0.5*(x1*pow(uq1, 2) + x1*pow(uq2, 2))*uq0;
    dstype x3 = x2*(-1.0 + mu0);
    dstype x4 = x3*pow(mu3, 2)*mu9*mu0;

    f[0 * N + i] = uq0;
    f[1 * N + i] = x0*uq1;
    f[2 * N + i] = x0*uq2;
    f[3 * N + i] = x3;
    f[4 * N + i] = x0*x2;
    f[5 * N + i] = 1.0*x0*x4;
    f[6 * N + i] = 4.14087039670808e-07*x4*mu1*sqrt(pow(mu9, 3))/(110.4 + mu9);
  });
}

