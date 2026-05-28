void KokkosVisTensors(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("VisTensors", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype udg3 = udg[2*ng+i];
		dstype udg4 = udg[3*ng+i];
		dstype udg5 = udg[4*ng+i];
		dstype udg6 = udg[5*ng+i];
		f[0*ng+i] = udg3;
		f[1*ng+i] = udg4;
		f[2*ng+i] = udg5;
		f[3*ng+i] = udg6;
	});
}

