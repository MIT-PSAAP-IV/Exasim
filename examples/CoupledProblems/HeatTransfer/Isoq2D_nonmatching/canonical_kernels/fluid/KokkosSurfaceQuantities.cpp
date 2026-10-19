void KokkosSurfaceQuantities1(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("SurfaceQuantities1", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype param2 = param[1];
		dstype param3 = param[2];
		dstype param5 = param[4];
		dstype param6 = param[5];
		dstype param7 = param[6];
		dstype param8 = param[7];
		dstype param9 = param[8];
		dstype param10 = param[9];
		dstype tau1 = tau[0];
		dstype xdg2 = xdg[1*ng+i];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype udg3 = udg[2*ng+i];
		dstype udg4 = udg[3*ng+i];
		dstype udg5 = udg[4*ng+i];
		dstype udg6 = udg[5*ng+i];
		dstype udg7 = udg[6*ng+i];
		dstype udg8 = udg[7*ng+i];
		dstype udg9 = udg[8*ng+i];
		dstype udg10 = udg[9*ng+i];
		dstype udg11 = udg[10*ng+i];
		dstype udg12 = udg[11*ng+i];
		dstype uhg1 = uhg[0*ng+i];
		dstype uhg2 = uhg[1*ng+i];
		dstype uhg3 = uhg[2*ng+i];
		dstype uhg4 = uhg[3*ng+i];
		dstype nlg1 = nlg[0*ng+i];
		dstype nlg2 = nlg[1*ng+i];
		dstype t2 = param6*param6;
		dstype t3 = param7*param7;
		dstype t4 = uhg2*uhg2;
		dstype t5 = uhg3*uhg3;
		dstype t6 = param1-1.0;
		dstype t7 = 1.0/param2;
		dstype t8 = 1.0/param5;
		dstype t10 = 1.0/param9;
		dstype t11 = 1.0/uhg1;
		dstype t13 = 1.0/xdg2;
		dstype t19 = param10+5.52E+2/5.0;
		dstype t9 = t8*t8;
		dstype t12 = t11*t11;
		dstype t14 = t11*udg5*uhg2;
		dstype t15 = t11*udg5*uhg3;
		dstype t16 = t11*udg9*uhg2;
		dstype t17 = t11*udg9*uhg3;
		dstype t18 = 1.0/t6;
		dstype t26 = t11*t13*uhg3;
		dstype t20 = t2*t9;
		dstype t21 = t3*t9;
		dstype t27 = (t4*t12)/2.0;
		dstype t28 = (t5*t12)/2.0;
		dstype t35 = t20+t21;
		dstype t37 = t27+t28;
		dstype t36 = 1.0/t35;
		dstype t38 = t37*uhg1;
		dstype t42 = -t10*t11*(t38-uhg4);
		dstype t46 = -1.0/(param10*t10*t11*(t38-uhg4)-5.52E+2/5.0);
		dstype t44 = pow(t42,3.0/2.0);
		f[0*ng+i] = t8*t36*(t6*(t38-uhg4)+t6*(param8-t8*(t2/2.0+t3/2.0)))*-2.0;
		f[1*ng+i] = t8*t36*(nlg2*(tau1*(udg2-uhg2)-(nlg1*t7*t19*t44*(t26-t11*(t14-udg6)*2.0+t11*(t17-udg11))*(2.0/3.0))/(param10*t10*t11*(t38-uhg4)-5.52E+2/5.0)+(nlg2*t7*t19*t44*(t11*(t15-udg7)+t11*(t16-udg10)))/(param10*t10*t11*(t38-uhg4)-5.52E+2/5.0))-nlg1*(tau1*(udg3-uhg3)-(nlg2*t7*t19*t44*(t26+t11*(t14-udg6)-t11*(t17-udg11)*2.0)*(2.0/3.0))/(param10*t10*t11*(t38-uhg4)-5.52E+2/5.0)+(nlg1*t7*t19*t44*(t11*(t15-udg7)+t11*(t16-udg10)))/(param10*t10*t11*(t38-uhg4)-5.52E+2/5.0)))*-2.0;
		f[2*ng+i] = t8*1.0/pow(t35,3.0/2.0)*(tau1*(udg4-uhg4)+(param1*t7*t19*t44*t46*(nlg1*t12*t18*(t6*uhg1*(udg8+uhg1*(t12*uhg2*(t14-udg6)+t12*uhg3*(t15-udg7))-t37*udg5)+t6*udg5*(t38-uhg4))+nlg2*t12*t18*(t6*uhg1*(udg12+uhg1*(t12*uhg2*(t16-udg10)+t12*uhg3*(t17-udg11))-t37*udg9)+t6*udg9*(t38-uhg4))))/param3);
	});
}

void KokkosSurfaceQuantities(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ib, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	KokkosSurfaceQuantities1(f, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, modelnumber, ng, nc, ncu, nd, ncx, nco, ncw);
}

