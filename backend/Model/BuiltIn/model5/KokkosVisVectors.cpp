void KokkosVisVectors(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("VisVectors", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype xdg1 = xdg[0*ng+i];
		dstype xdg2 = xdg[1*ng+i];
		dstype t2 = sqrt(5.0);
		dstype t3 = (xdg1*3.141592653589793)/1.0E+1;
		dstype t4 = xdg2*3.141592653589793*(2.0/1.5E+1);
		dstype t5 = sin(t3);
		dstype t6 = sin(t4);
		f[0*ng+i] = t5*t6*sin(t2*time*3.141592653589793*(2.0/2.5E+1))*2.0;
		f[1*ng+i] = t5*t6*sin(t2*time*3.141592653589793*(4.0/2.5E+1))*(3.0/2.0);
	});
}

