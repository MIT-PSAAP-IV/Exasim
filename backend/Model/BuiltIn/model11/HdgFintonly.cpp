void HdgFintonly1(dstype* f, const dstype* x, const dstype* uq, const dstype* v, const dstype* w, const dstype* uhat, const dstype* n, const dstype* tau, const dstype* eta, const dstype* mu, const dstype t, const int modelnumber, const int N, const int szx, const int szuq, const int szv, const int szw, const int szuhat, const int szn, const int sztau, const int szeta, const int szmu)
{

  Kokkos::parallel_for("Fint", N, KOKKOS_LAMBDA(const size_t i) {
    dstype x1 = x[1*N+i];
    dstype uq0 = uq[0*N+i];
    dstype uq1 = uq[1*N+i];
    dstype uq2 = uq[2*N+i];
    dstype uq3 = uq[3*N+i];
    dstype uq4 = uq[4*N+i];
    dstype uq5 = uq[5*N+i];
    dstype uq6 = uq[6*N+i];
    dstype uq7 = uq[7*N+i];
    dstype uq8 = uq[8*N+i];
    dstype uq9 = uq[9*N+i];
    dstype uq10 = uq[10*N+i];
    dstype uq11 = uq[11*N+i];
    dstype uhat0 = uhat[0*N+i];
    dstype uhat1 = uhat[1*N+i];
    dstype uhat2 = uhat[2*N+i];
    dstype uhat3 = uhat[3*N+i];
    dstype n0 = n[0*N+i];
    dstype n1 = n[1*N+i];
    dstype tau0 = tau[0];
    dstype mu0 = mu[0];
    dstype mu1 = mu[1];
    dstype mu2 = mu[2];
    dstype mu3 = mu[3];
    dstype mu9 = mu[9];

    dstype x0 = -1.0 + mu0;
    dstype x2 = pow(uhat0, -2);
    dstype x3 = 0.5*(x2*pow(uhat1, 2) + x2*pow(uhat2, 2));
    dstype x4 = uhat3 - x3*uhat0;
    dstype x5 = x0*x4;
    dstype x6 = pow(uhat0, -1);
    dstype x7 = x6*uq8;
    dstype x8 = uq10 - x7*uhat2;
    dstype x9 = x2*uhat2;
    dstype x10 = uq9 - x7*uhat1;
    dstype x11 = x2*uhat1;
    dstype x12 = x0*uhat0;
    dstype x13 = 110.4 + mu9;
    dstype x14 = 1.0*mu0;
    dstype x15 = pow(mu3, 2)*mu9;
    dstype x16 = sqrt(pow(x0, 3)*pow(x4, 3)*pow(mu3, 6)*pow(mu0, 3)/pow(uhat0, 3))/(110.4 + x6*x5*x15*x14);
    dstype x17 = x2*x14*x13*x16/(x0*mu2*mu1);
    dstype x18 = x6*uq4;
    dstype x19 = uq6 - x18*uhat2;
    dstype x20 = uq5 - x18*uhat1;
    dstype x21 = sqrt(pow(mu9, 3));
    dstype x22 = x21*x15*mu0*mu1/x13;
    dstype x23 = pow(uq0, -2);
    dstype x24 = x6*uhat2/x1;
    dstype x25 = x6*x8;
    dstype x26 = x6*x20;
    dstype x27 = x21*x16;
    dstype x28 = 0.000982141192491313*x27;

    f[0 * N + i] = 0.000418426671846558*x22*(tau0*(-uhat3 + uq3) + n0*x17*(x12*(uq7 - uhat0*(x20*x11 + x9*x19) - x3*uq4) - x5*uq4) + n1*x17*(x12*(uq11 - uhat0*(x11*x10 + x8*x9) - x3*uq8) - x5*uq8));
    f[1 * N + i] = 4.14087039670808e-07*x0*x22*(uq3 - 0.5*(pow(uq1, 2)*x23 + pow(uq2, 2)*x23)*uq0);
    f[2 * N + i] = x28*(x24 - x25 + 2*x26);
    f[3 * N + i] = 0.00147321178873697*(x6*x10 + x6*x19)*x27;
    f[4 * N + i] = x28*(x24 + 2*x25 - x26);
  });
}

void HdgFintonly(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg,
           const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time,
           const int modelnumber, const int ib, const int ng, const int nc, const int ncu, const int nd,
           const int ncx, const int nco, const int ncw) {
    if (ib == 1 )
        HdgFintonly1(f, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, modelnumber,
                        ng, nc, ncu, nd, ncx, nco, ncw, nc, ncu, nd);
}
