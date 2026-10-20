void KokkosVisScalars(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("VisScalars", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype param2 = param[1];
		dstype param4 = param[3];
		dstype param10 = param[9];
		dstype param13 = param[12];
		dstype param14 = param[13];
		dstype param15 = param[14];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype udg3 = udg[2*ng+i];
		dstype udg4 = udg[3*ng+i];
		dstype odg1 = odg[0*ng+i];
		dstype odg2 = odg[1*ng+i];
		dstype t2 = param1*param10;
		dstype t3 = param4*param4;
		dstype t4 = param10*param10*param10;
		dstype t5 = udg2*udg2;
		dstype t6 = udg3*udg3;
		dstype t7 = param1-1.0;
		dstype t8 = -udg4;
		dstype t9 = 1.0/udg1;
		dstype t11 = sqrt(2.87E+2);
		dstype t13 = param10+5.52E+2/5.0;
		dstype t10 = t9*t9;
		dstype t12 = 1.0/sqrt(t2);
		dstype t14 = (t5*t9)/2.0;
		dstype t15 = (t6*t9)/2.0;
		dstype t16 = 1.0/t13;
		dstype t17 = t4*4.906777729315601E-8;
		dstype t18 = sqrt(t17);
		dstype t19 = t8+t14+t15;
		f[0*ng+i] = (param2*t11*t12*t16*t18*udg1*2.293281533101045E-5)/param4;
		f[1*ng+i] = (param2*param4*t7*t11*t16*t18*t19*(-6.581717999999999E-3))/t12;
		f[2*ng+i] = -t2*t3*t7*t9*t19;
		f[3*ng+i] = (param4*sqrt(t5*t10+t6*t10)*1.0/sqrt(-param1*t2*t3*t7*t9*t19))/t12;
		f[4*ng+i] = tanh(odg1*param13)*(param14+odg2*param15);
	});
}

