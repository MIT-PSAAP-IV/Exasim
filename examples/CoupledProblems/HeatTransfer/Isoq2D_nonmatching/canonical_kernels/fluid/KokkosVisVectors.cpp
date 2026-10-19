void KokkosVisVectors(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("VisVectors", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype param4 = param[3];
		dstype param10 = param[9];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype udg3 = udg[2*ng+i];
		dstype t2 = param1*param10;
		dstype t3 = 1.0/udg1;
		dstype t4 = sqrt(2.87E+2);
		dstype t5 = sqrt(t2);
		f[0*ng+i] = param4*t3*t4*t5*udg2;
		f[1*ng+i] = param4*t3*t4*t5*udg3;
	});
}

