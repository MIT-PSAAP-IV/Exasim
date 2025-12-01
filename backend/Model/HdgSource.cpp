void HdgSource(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{

  Kokkos::parallel_for("Source", N, KOKKOS_LAMBDA(const size_t i) {
    dstype x0 = x[0*N+i];
    dstype x1 = x[1*N+i];

    f[0 * N + i] = 2*pow(acos(-1), 2)*sin(acos(-1)*x1)*sin(acos(-1)*x0);
    J1[0 * N + i] = 0;
    J1[1 * N + i] = 0;
    J1[2 * N + i] = 0;
  });
}

