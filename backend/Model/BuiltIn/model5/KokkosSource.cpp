void KokkosSource(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("Source", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype xdg1 = xdg[0*ng+i];
		dstype xdg2 = xdg[1*ng+i];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype udg3 = udg[2*ng+i];
		dstype udg4 = udg[3*ng+i];
		dstype t2 = 3.141592653589793*3.141592653589793;
		dstype t3 = 3.141592653589793*3.141592653589793*3.141592653589793;
		dstype t4 = sqrt(5.0);
		dstype t5 = (xdg1*3.141592653589793)/1.0E+1;
		dstype t6 = xdg2*3.141592653589793*(2.0/1.5E+1);
		dstype t7 = cos(t5);
		dstype t8 = cos(t6);
		dstype t9 = sin(t5);
		dstype t10 = sin(t6);
		dstype t11 = t4*time*3.141592653589793*(2.0/2.5E+1);
		dstype t12 = t4*time*3.141592653589793*(4.0/2.5E+1);
		dstype t13 = cos(t11);
		dstype t14 = cos(t12);
		dstype t15 = sin(t11);
		dstype t16 = sin(t12);
		dstype t17 = (t7*t10*t15*3.141592653589793)/5.0;
		dstype t18 = (t8*t9*t16*3.141592653589793)/5.0;
		dstype t21 = t3*t4*t7*t8*t9*t10*t13*t16*(2.0/6.25E+2);
		dstype t22 = t3*t4*t7*t8*t9*t10*t14*t15*(4.0/6.25E+2);
		dstype t19 = t17+1.0;
		dstype t20 = t18+1.0;
		dstype t25 = t2*t4*t7*t10*t13*t20*(2.0/1.25E+2);
		dstype t26 = t2*t4*t8*t9*t14*t19*(4.0/1.25E+2);
		f[0*ng+i] = udg1*(t21+t22-t25-t26);
		f[1*ng+i] = udg2*(t21+t22-t25-t26);
		f[2*ng+i] = udg3*(t21+t22-t25-t26);
		f[3*ng+i] = udg4*(t21+t22-t25-t26);
	});
}

