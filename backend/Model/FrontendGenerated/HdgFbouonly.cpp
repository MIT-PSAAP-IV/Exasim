void HdgFbouonly1(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("Fbouonly1", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype uhg1 = uhg[0*ng+i];
		dstype uhg2 = uhg[1*ng+i];
		dstype nlg1 = nlg[0*ng+i];
		dstype t2 = 3.141592653589793*3.141592653589793;
		dstype t3 = 3.141592653589793*3.141592653589793*3.141592653589793;
		dstype t4 = uhg1*uhg1;
		dstype t5 = uhg1*uhg1*uhg1;
		dstype t6 = 1.0/3.141592653589793;
		dstype t7 = uhg1*1.0E+6;
		dstype t8 = atan(t7);
		dstype t9 = t8*t8;
		dstype t10 = t6*t8;
		dstype t11 = t10+1.0/2.0;
		dstype t12 = t11*uhg1;
		dstype t13 = t12+3.183098861714306E-7;
		dstype t14 = param1*t13;
		dstype t15 = 1.0/t13;
		dstype t16 = t15*uhg2;
		dstype t17 = sqrt(t14);
		dstype t18 = 1.0/t17;
		dstype t20 = t16+t17;
		dstype t24 = nlg1*(t16-t17)*-1.0E+2;
		dstype t22 = nlg1*t20*1.0E+2;
		dstype t25 = tanh(t24);
		dstype t23 = tanh(t22);
		dstype t26 = t23+t25;
		f[0*ng+i] = udg1/2.0-uhg1-(udg1*((t18*t20*t25)/2.0+(t18*t23*(t16-t17))/2.0))/2.0+(t18*t26*udg2)/4.0;
		f[1*ng+i] = udg2/2.0-uhg2+(udg2*((t18*t20*t23)/2.0+(t18*t25*(t16-t17))/2.0))/2.0+(t6*t18*t26*udg1*1.0/pow(3.141592653589793*5.734161139E+9+uhg1*3.141592653589793*9.007199254740992E+15+t8*uhg1*1.801439850948198E+16,2.0)*(param1*t3*1.885426815002567E+29-t3*(uhg2*uhg2)*5.846006549323612E+48+param1*t5*(t8*t8*t8)*5.846006549323612E+48+param1*t3*t4*1.395631259454478E+42+param1*t3*t5*7.307508186654515E+47+param1*t3*uhg1*8.884864546684903E+35+param1*t4*t9*3.141592653589793*5.58252503781791E+42+param1*t5*t9*3.141592653589793*8.769009823985418E+48+param1*t2*t4*t8*5.58252503781791E+42+param1*t2*t5*t8*4.384504911992709E+48+param1*t2*t8*uhg1*1.776972909336981E+36))/7.205759403792794E+16;
	});
}

void HdgFbouonly2(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("Fbouonly2", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype param2 = param[1];
		dstype param3 = param[2];
		dstype param4 = param[3];
		dstype param5 = param[4];
		dstype param6 = param[5];
		dstype xdg1 = xdg[0*ng+i];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype uhg1 = uhg[0*ng+i];
		dstype uhg2 = uhg[1*ng+i];
		dstype nlg1 = nlg[0*ng+i];
		dstype t2 = param6*time;
		dstype t3 = 3.141592653589793*3.141592653589793;
		dstype t4 = 3.141592653589793*3.141592653589793*3.141592653589793;
		dstype t5 = uhg1*uhg1;
		dstype t6 = uhg1*uhg1*uhg1;
		dstype t7 = 1.0/3.141592653589793;
		dstype t9 = -param3;
		dstype t10 = -param4;
		dstype t11 = 1.0/param2;
		dstype t12 = 1.0/udg1;
		dstype t16 = uhg1*1.0E+6;
		dstype t8 = sin(t2);
		dstype t14 = param3+t10;
		dstype t17 = atan(t16);
		dstype t13 = param5*t8;
		dstype t18 = t11*t14*xdg1;
		dstype t19 = t17*t17;
		dstype t21 = t7*t17;
		dstype t15 = -t13;
		dstype t20 = -t18;
		dstype t22 = t21+1.0/2.0;
		dstype t24 = t9+t13+t18;
		dstype t23 = t22*uhg1;
		dstype t25 = param3+t15+t20+udg1;
		dstype t26 = t12*t24*udg2;
		dstype t29 = t23+3.183098861714306E-7;
		dstype t30 = param1*t29;
		dstype t31 = 1.0/t29;
		dstype t32 = t31*uhg2;
		dstype t33 = sqrt(t30);
		dstype t34 = 1.0/t33;
		dstype t36 = t32+t33;
		dstype t40 = nlg1*(t32-t33)*-1.0E+2;
		dstype t38 = nlg1*t36*1.0E+2;
		dstype t41 = tanh(t40);
		dstype t39 = tanh(t38);
		dstype t42 = t39+t41;
		f[0*ng+i] = param3*(-1.0/2.0)+t13/2.0+t18/2.0+udg1/2.0-uhg1-(t25*((t34*t36*t41)/2.0+(t34*t39*(t32-t33))/2.0))/2.0-(t34*t42*(t26-udg2))/4.0;
		f[1*ng+i] = t26/2.0+udg2/2.0-uhg2-(((t34*t36*t39)/2.0+(t34*t41*(t32-t33))/2.0)*(t26-udg2))/2.0+(t7*t25*t34*t42*1.0/pow(3.141592653589793*5.734161139E+9+uhg1*3.141592653589793*9.007199254740992E+15+t17*uhg1*1.801439850948198E+16,2.0)*(param1*t4*1.885426815002567E+29-t4*(uhg2*uhg2)*5.846006549323612E+48+param1*t6*(t17*t17*t17)*5.846006549323612E+48+param1*t4*t5*1.395631259454478E+42+param1*t4*t6*7.307508186654515E+47+param1*t4*uhg1*8.884864546684903E+35+param1*t5*t19*3.141592653589793*5.58252503781791E+42+param1*t6*t19*3.141592653589793*8.769009823985418E+48+param1*t3*t5*t17*5.58252503781791E+42+param1*t3*t6*t17*4.384504911992709E+48+param1*t3*t17*uhg1*1.776972909336981E+36))/7.205759403792794E+16;
	});
}

void HdgFbouonly(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ib, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	if (ib == 1)
		HdgFbouonly1(f, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, modelnumber, ng, nc, ncu, nd, ncx, nco, ncw);
	else if (ib == 2)
		HdgFbouonly2(f, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, modelnumber, ng, nc, ncu, nd, ncx, nco, ncw);
}

