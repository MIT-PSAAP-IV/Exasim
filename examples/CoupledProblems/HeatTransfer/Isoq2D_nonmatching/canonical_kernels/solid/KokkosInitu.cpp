void KokkosInitu(dstype* f, const dstype* xdg, const dstype* uinf, const dstype* param, const int modelnumber, const int ng, const int ncx, const int nce, const int npe, const int ne)
{
	Kokkos::parallel_for("Initu", ng, KOKKOS_LAMBDA(const size_t i) {
		int j = i%npe;
		int k = i/npe;
		dstype param2 = param[1];
		f[j+npe*0+npe*nce*k] = param2;
	});
}

