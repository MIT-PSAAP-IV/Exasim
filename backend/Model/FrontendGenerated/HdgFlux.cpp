void HdgFlux(dstype* f, dstype* f_udg, dstype* f_wdg, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("Flux", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype param7 = param[6];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype udg3 = udg[2*ng+i];
		dstype udg4 = udg[3*ng+i];
		dstype odg1 = odg[0*ng+i];
		{
		dstype t2 = 1.0/3.141592653589793;
		dstype t3 = udg1*1.0E+6;
		dstype t4 = atan(t3);
		dstype t5 = t2*t4;
		dstype t6 = t5+1.0/2.0;
		dstype t7 = t6*udg1;
		f[0*ng+i] = udg2+odg1*udg3*3.0E+1;
		f[1*ng+i] = (udg2*udg2)/(param7+t7+3.183098861714306E-7)+odg1*udg4*3.0E+1+(param1*pow(t7+3.183098861714306E-7,2.0))/2.0;
		}
		{
		dstype t2 = udg1*udg1;
		dstype t3 = 1.0/3.141592653589793;
		dstype t4 = odg1*3.0E+1;
		dstype t5 = udg1*1.0E+6;
		dstype t6 = atan(t5);
		dstype t8 = t2*1.0E+12;
		dstype t7 = t3*t6;
		dstype t9 = t8+1.0;
		dstype t10 = t7+1.0/2.0;
		dstype t12 = 1.0/t9;
		dstype t11 = t10*udg1;
		dstype t13 = t3*t5*t12;
		dstype t14 = param7+t11+3.183098861714306E-7;
		dstype t15 = t10+t13;
		f_udg[0*ng+i] = 0.0;
		f_udg[1*ng+i] = param1*t15*(t11+3.183098861714306E-7)-1.0/(t14*t14)*t15*(udg2*udg2);
		f_udg[2*ng+i] = 1.0;
		f_udg[3*ng+i] = (udg2*2.0)/t14;
		f_udg[4*ng+i] = t4;
		f_udg[5*ng+i] = 0.0;
		f_udg[6*ng+i] = 0.0;
		f_udg[7*ng+i] = t4;
		}
	});
}

