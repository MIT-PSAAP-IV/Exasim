void HdgSource(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("Source", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype param2 = param[1];
		dstype param3 = param[2];
		dstype param4 = param[3];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		{
		dstype t2 = 1.0/3.141592653589793;
		dstype t3 = udg1*1.0E+6;
		dstype t4 = atan(t3);
		dstype t5 = t2*t4;
		dstype t6 = t5+1.0/2.0;
		f[0*ng+i] = 0.0;
		f[1*ng+i] = udg2*(tanh(t6*udg1*1.0E+3-9.996816901138286E-1)/2.0E+1-1.0/2.0E+1)+(param1*(param3-param4)*(t6*udg1+3.183098861714306E-7))/param2;
		}
		{
		dstype t2 = udg1*udg1;
		dstype t3 = 1.0/3.141592653589793;
		dstype t4 = udg1*1.0E+6;
		dstype t5 = atan(t4);
		dstype t7 = t2*1.0E+12;
		dstype t6 = t3*t5;
		dstype t8 = t7+1.0;
		dstype t9 = t6+1.0/2.0;
		dstype t10 = 1.0/t8;
		dstype t11 = t9*udg1*1.0E+3;
		dstype t12 = t11-9.996816901138286E-1;
		dstype t13 = tanh(t12);
		f_udg[0*ng+i] = 0.0;
		f_udg[1*ng+i] = udg2*(t13*t13-1.0)*(t6*1.0E+3+t3*t10*udg1*1.0E+9+5.0E+2)*(-1.0/2.0E+1)+(param1*(t9+t3*t4*t10)*(param3-param4))/param2;
		f_udg[2*ng+i] = 0.0;
		f_udg[3*ng+i] = t13/2.0E+1-1.0/2.0E+1;
		f_udg[4*ng+i] = 0.0;
		f_udg[5*ng+i] = 0.0;
		f_udg[6*ng+i] = 0.0;
		f_udg[7*ng+i] = 0.0;
		}
	});
}

