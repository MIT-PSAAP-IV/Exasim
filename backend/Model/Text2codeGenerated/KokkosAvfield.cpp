void KokkosAvfield(dstype* f, const dstype* x, const dstype* uq, const dstype* v, const dstype* w, const dstype* eta, const dstype* mu, const dstype t, const int modelnumber, const int N, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw, const int nce, const int npe, const int ne)
{

  Kokkos::parallel_for("Avfield", N, KOKKOS_LAMBDA(const size_t i) {
    int p = i%npe;
    int e = i/npe;
    dstype uq0 = uq[p+npe*0+npe*nc*e];
    dstype uq1 = uq[p+npe*1+npe*nc*e];
    dstype uq2 = uq[p+npe*2+npe*nc*e];
    dstype uq4 = uq[p+npe*4+npe*nc*e];
    dstype uq5 = uq[p+npe*5+npe*nc*e];
    dstype uq8 = uq[p+npe*8+npe*nc*e];
    dstype uq10 = uq[p+npe*10+npe*nc*e];
    dstype v0 = v[p+npe*0+npe*nco*e];
    dstype mu11 = mu[11];
    dstype mu12 = mu[12];

    dstype x0 = pow(uq0, -1);
    dstype x1 = tanh(v0*mu12)*(x0*(uq10 - x0*uq2*uq8) + x0*(uq5 - x0*uq4*uq1));
    dstype x2 = x1*(0.5 + 0.318309886183791*atan(1000.0*x1));

    f[p+npe*0+npe*nce*e] = x2 - (0.000318309780080517 - mu11 + x2)*(0.5 + 0.318309886183791*atan(0.318309780080517 - 1000.0*mu11 + 1000.0*x2));
  });
}
