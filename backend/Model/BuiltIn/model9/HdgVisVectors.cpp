void HdgVisVectors(dstype* f, dstype* J1, dstype* J2, const dstype* x, const dstype* uq, const dstype* v, const dstype* w, const dstype* eta, const dstype* mu, const dstype t, const int modelnumber, const int N, const int szx, const int szuq, const int szv, const int szw, const int szeta, const int szmu)
{

  Kokkos::parallel_for("VisVectors", N, KOKKOS_LAMBDA(const size_t i) {
    dstype uq2 = uq[2*N+i];
    dstype uq3 = uq[3*N+i];
    dstype uq4 = uq[4*N+i];
    dstype uq5 = uq[5*N+i];

    f[0 * N + i] = uq2;
    f[1 * N + i] = uq3;
    f[2 * N + i] = uq4;
    f[3 * N + i] = uq5;
    J1[0 * N + i] = 0;
    J1[1 * N + i] = 0;
    J1[2 * N + i] = 0;
    J1[3 * N + i] = 0;
    J1[4 * N + i] = 0;
    J1[5 * N + i] = 0;
    J1[6 * N + i] = 0;
    J1[7 * N + i] = 0;
    J1[8 * N + i] = 1;
    J1[9 * N + i] = 0;
    J1[10 * N + i] = 0;
    J1[11 * N + i] = 0;
    J1[12 * N + i] = 0;
    J1[13 * N + i] = 1;
    J1[14 * N + i] = 0;
    J1[15 * N + i] = 0;
    J1[16 * N + i] = 0;
    J1[17 * N + i] = 0;
    J1[18 * N + i] = 1;
    J1[19 * N + i] = 0;
    J1[20 * N + i] = 0;
    J1[21 * N + i] = 0;
    J1[22 * N + i] = 0;
    J1[23 * N + i] = 1;
  });
}

