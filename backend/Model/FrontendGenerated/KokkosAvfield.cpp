void KokkosAvfield(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw, const int nce, const int npe, const int ne)
{
	Kokkos::parallel_for("Avfield", ng, KOKKOS_LAMBDA(const size_t i) {
		int j = i%npe;
		int k = i/npe;
		dstype param2 = param[1];
		dstype param7 = param[6];
		dstype xdg1 = xdg[j+npe*0+npe*ncx*k];
		dstype udg1 = udg[j+npe*0+npe*nc*k];
		dstype udg2 = udg[j+npe*1+npe*nc*k];
		dstype udg3 = udg[j+npe*2+npe*nc*k];
		dstype udg4 = udg[j+npe*3+npe*nc*k];
		dstype t2 = udg1*udg1;
		dstype t3 = 1.0/3.141592653589793;
		dstype t4 = udg1*1.0E+3;
		dstype t5 = atan(t4);
		dstype t6 = t2*1.0E+6;
		dstype t7 = t3*t5;
		dstype t8 = t6+1.0;
		dstype t9 = t7+1.0/2.0;
		dstype t10 = 1.0/t8;
		dstype t11 = t9*udg1;
		dstype t12 = t3*t4*t10;
		dstype t13 = t9+t12;
		dstype t16 = param7+t11+3.183097800805168E-4;
		dstype t14 = t13*udg2*udg3;
		dstype t17 = t16*udg4;
		dstype t18 = 1.0/(t16*t16);
		dstype t20 = t18*(t14-t17)*-1.0E+6;
		dstype t21 = atan(t20);
		dstype t22 = t3*t21;
		dstype t23 = t22+1.0/2.0;
		dstype t25 = t20*t23;
		dstype t27 = t25-8.996816902199195E+2;
		dstype t28 = atan(t27);
		dstype t29 = t3*t28;
		dstype t30 = t29+1.0/2.0;
		f[j+npe*0+npe*nce*k] = tanh(((param2-xdg1)*1.0E+1)/param2)*((t3*atan(t30*(t18*t23*(t14-t17)*1.0E+3+8.996816902199195E-1)*(-1.111111111111111E+3)+t18*t23*(t14-t17)*1.111111111111111E+6+1.0E+2)-1.0/2.0)*(t30*(t18*t23*(t14-t17)*1.0E+3+8.996816902199195E-1)*(-1.0E+1/9.0)+t18*t23*(t14-t17)*1.111111111111111E+3+1.0/1.0E+1)+3.183097800805168E-4);
	});
}

