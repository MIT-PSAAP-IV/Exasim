void KokkosFlux(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("Flux", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype param7 = param[6];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype udg3 = udg[2*ng+i];
		dstype udg4 = udg[3*ng+i];
		dstype odg1 = odg[0*ng+i];
		dstype t2 = 1.0/3.141592653589793;
		dstype t3 = udg1*1.0E+6;
		dstype t4 = atan(t3);
		dstype t5 = t2*t4;
		dstype t6 = t5+1.0/2.0;
		dstype t7 = t6*udg1;
		f[0*ng+i] = udg2+odg1*udg3*3.0E+1;
		f[1*ng+i] = (udg2*udg2)/(param7+t7+3.183098861714306E-7)+odg1*udg4*3.0E+1+(param1*pow(t7+3.183098861714306E-7,2.0))/2.0;
	});
}

