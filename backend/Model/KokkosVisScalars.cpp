void KokkosVisScalars(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("VisScalars", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype udg3 = udg[2*ng+i];
		dstype udg4 = udg[3*ng+i];
		dstype t2 = udg2*udg2;
		dstype t3 = udg3*udg3;
		dstype t4 = -udg4;
		dstype t5 = 1.0/udg1;
		dstype t6 = t5*t5;
		dstype t7 = (t2*t5)/2.0;
		dstype t8 = (t3*t5)/2.0;
		dstype t9 = t4+t7+t8;
		f[0*ng+i] = -t5*t9;
		f[1*ng+i] = sqrt(t2*t6+t3*t6)*1.0/sqrt(param1*t5*fabs(t9*(param1-1.0)));
	});
}

