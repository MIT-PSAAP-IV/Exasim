void KokkosTdfunc(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("Tdfunc", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype xdg1 = xdg[0*ng+i];
		dstype xdg2 = xdg[1*ng+i];
		dstype t2 = 3.141592653589793*3.141592653589793;
		dstype t3 = sqrt(5.0);
		dstype t4 = (xdg1*3.141592653589793)/1.0E+1;
		dstype t5 = xdg2*3.141592653589793*(2.0/1.5E+1);
		dstype t6 = cos(t4);
		dstype t7 = cos(t5);
		dstype t8 = sin(t4);
		dstype t9 = sin(t5);
		dstype t10 = t3*time*3.141592653589793*(2.0/2.5E+1);
		dstype t11 = t3*time*3.141592653589793*(4.0/2.5E+1);
		dstype t12 = sin(t10);
		dstype t13 = sin(t11);
		dstype t14 = (t6*t9*t12*3.141592653589793)/5.0;
		dstype t15 = (t7*t8*t13*3.141592653589793)/5.0;
		dstype t18 = (t2*t6*t7*t8*t9*t12*t13)/2.5E+1;
		dstype t16 = t14+1.0;
		dstype t17 = t15+1.0;
		dstype t19 = -t18;
		dstype t20 = t16*t17;
		dstype t21 = t19+t20;
		f[0*ng+i] = t21;
		f[1*ng+i] = t21;
		f[2*ng+i] = t21;
		f[3*ng+i] = t21;
	});
}

