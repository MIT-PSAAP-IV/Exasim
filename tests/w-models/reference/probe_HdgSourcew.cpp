// ABI reference kernels for the HDG w-equation source (Sourcew) equivalence
// test — hand-authored in the exact style text2code emits for the libpdemodel
// `HdgSourcew` / `HdgSourcewonly` symbols (see backend/Model/BuiltIn/model4/
// HdgSourcew.cpp). They are the ground-truth reference that
// tests/w-models/compare_hdg_sourcew.cpp compares the templated
// exasim::hdg_sourcew_kernel<M> / hdg_sourcewonly_kernel<M> against, the same
// relationship compare_model4.cpp has with backend/Model/BuiltIn/model4/.
//
// Probe model: nd=2, ncu=2, ncw=2, nparam=1, nco=0  =>  Nq = nc = ncu*(1+nd) = 6.
//   sw[0] = param0*udg0 + 3*udg4 + wdg0*wdg1
//   sw[1] = udg1*udg2 + 5*udg5 + 2*wdg0 + wdg1^2
// Non-trivial in BOTH udg and wdg so the input-index-outer SoA layout of
// f_udg (size ncw*Nq) and f_wdg (size ncw*ncw) is fully exercised:
//   f_udg[(j*ncw + o)*ng + i] = d sw[o] / d udg[j]
//   f_wdg[(j*ncw + o)*ng + i] = d sw[o] / d wdg[j]

void HdgSourcew(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("Sourcew", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype udg3 = udg[2*ng+i];
		dstype udg5 = udg[4*ng+i];
		dstype udg6 = udg[5*ng+i];
		dstype wdg1 = wdg[0*ng+i];
		dstype wdg2 = wdg[1*ng+i];
		// value (ncw = 2)
		f[0*ng+i] = param1*udg1 + 3.0*udg5 + wdg1*wdg2;
		f[1*ng+i] = udg2*udg3 + 5.0*udg6 + 2.0*wdg1 + wdg2*wdg2;
		// d sw / d udg  (input-index-outer: [(j*ncw+o)*ng+i])
		f_udg[0*ng+i] = param1;   // d sw0/d udg0
		f_udg[1*ng+i] = 0.0;      // d sw1/d udg0
		f_udg[2*ng+i] = 0.0;      // d sw0/d udg1
		f_udg[3*ng+i] = udg3;     // d sw1/d udg1
		f_udg[4*ng+i] = 0.0;      // d sw0/d udg2
		f_udg[5*ng+i] = udg2;     // d sw1/d udg2
		f_udg[6*ng+i] = 0.0;      // d sw0/d udg3
		f_udg[7*ng+i] = 0.0;      // d sw1/d udg3
		f_udg[8*ng+i] = 3.0;      // d sw0/d udg4
		f_udg[9*ng+i] = 0.0;      // d sw1/d udg4
		f_udg[10*ng+i] = 0.0;     // d sw0/d udg5
		f_udg[11*ng+i] = 5.0;     // d sw1/d udg5
		// d sw / d wdg  (input-index-outer: [(j*ncw+o)*ng+i])
		f_wdg[0*ng+i] = wdg2;     // d sw0/d wdg0
		f_wdg[1*ng+i] = 2.0;      // d sw1/d wdg0
		f_wdg[2*ng+i] = wdg1;     // d sw0/d wdg1
		f_wdg[3*ng+i] = 2.0*wdg2; // d sw1/d wdg1
	});
}

void HdgSourcewonly(dstype* f, dstype* f_wdg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("Sourcewonly", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype udg3 = udg[2*ng+i];
		dstype udg5 = udg[4*ng+i];
		dstype udg6 = udg[5*ng+i];
		dstype wdg1 = wdg[0*ng+i];
		dstype wdg2 = wdg[1*ng+i];
		// value (ncw = 2)
		f[0*ng+i] = param1*udg1 + 3.0*udg5 + wdg1*wdg2;
		f[1*ng+i] = udg2*udg3 + 5.0*udg6 + 2.0*wdg1 + wdg2*wdg2;
		// d sw / d wdg  (input-index-outer)
		f_wdg[0*ng+i] = wdg2;
		f_wdg[1*ng+i] = 2.0;
		f_wdg[2*ng+i] = wdg1;
		f_wdg[3*ng+i] = 2.0*wdg2;
	});
}
