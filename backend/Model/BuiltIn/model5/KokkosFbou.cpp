void KokkosFbou1(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("Fbou1", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype tau1 = tau[0];
		dstype xdg1 = xdg[0*ng+i];
		dstype xdg2 = xdg[1*ng+i];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype udg3 = udg[2*ng+i];
		dstype udg4 = udg[3*ng+i];
		dstype uhg1 = uhg[0*ng+i];
		dstype uhg2 = uhg[1*ng+i];
		dstype uhg3 = uhg[2*ng+i];
		dstype uhg4 = uhg[3*ng+i];
		dstype nlg1 = nlg[0*ng+i];
		dstype nlg2 = nlg[1*ng+i];
		dstype t2 = udg2*udg2;
		dstype t3 = udg3*udg3;
		dstype t4 = param1-1.0;
		dstype t5 = 1.0/udg1;
		dstype t7 = sqrt(5.0);
		dstype t9 = (xdg1*3.141592653589793)/1.0E+1;
		dstype t10 = xdg2*3.141592653589793*(2.0/1.5E+1);
		dstype t6 = t5*t5;
		dstype t8 = t5*udg2*udg3;
		dstype t11 = t2*t5;
		dstype t12 = t3*t5;
		dstype t13 = cos(t9);
		dstype t14 = cos(t10);
		dstype t15 = sin(t9);
		dstype t16 = sin(t10);
		dstype t19 = t7*time*3.141592653589793*(2.0/2.5E+1);
		dstype t20 = t7*time*3.141592653589793*(4.0/2.5E+1);
		dstype t17 = (t2*t6)/2.0;
		dstype t18 = (t3*t6)/2.0;
		dstype t21 = cos(t19);
		dstype t22 = cos(t20);
		dstype t23 = sin(t19);
		dstype t24 = sin(t20);
		dstype t25 = t17+t18;
		dstype t30 = (t13*t16*t23*3.141592653589793)/5.0;
		dstype t31 = (t14*t15*t24*3.141592653589793)/5.0;
		dstype t35 = t7*t15*t16*t21*udg1*3.141592653589793*(4.0/2.5E+1);
		dstype t36 = t7*t15*t16*t21*udg2*3.141592653589793*(4.0/2.5E+1);
		dstype t37 = t7*t15*t16*t21*udg3*3.141592653589793*(4.0/2.5E+1);
		dstype t38 = t7*t15*t16*t21*udg4*3.141592653589793*(4.0/2.5E+1);
		dstype t39 = t7*t15*t16*t22*udg1*3.141592653589793*(6.0/2.5E+1);
		dstype t40 = t7*t15*t16*t22*udg2*3.141592653589793*(6.0/2.5E+1);
		dstype t41 = t7*t15*t16*t22*udg3*3.141592653589793*(6.0/2.5E+1);
		dstype t42 = t7*t15*t16*t22*udg4*3.141592653589793*(6.0/2.5E+1);
		dstype t26 = t25*udg1;
		dstype t33 = t30+1.0;
		dstype t34 = t31+1.0;
		dstype t45 = -t37;
		dstype t48 = -t40;
		dstype t29 = -t4*(t26-udg4);
		dstype t55 = t8+t45;
		dstype t56 = t8+t48;
		dstype t32 = t29+udg4;
		dstype t51 = t5*t32*udg2;
		dstype t52 = t5*t32*udg3;
		f[0*ng+i] = -nlg1*(t34*(t35-udg2)-t14*t15*t23*3.141592653589793*(t39-udg3)*(4.0/1.5E+1))-nlg2*(t33*(t39-udg3)-t13*t16*t24*3.141592653589793*(t35-udg2)*(3.0/2.0E+1))+tau1*(udg1-uhg1);
		f[1*ng+i] = tau1*(udg2-uhg2)-nlg1*(t34*(-t11+t36+t4*(t26-udg4))+t14*t15*t23*t56*3.141592653589793*(4.0/1.5E+1))+nlg2*(t33*t56+t13*t16*t24*3.141592653589793*(-t11+t36+t4*(t26-udg4))*(3.0/2.0E+1));
		f[2*ng+i] = tau1*(udg3-uhg3)+nlg1*(t34*t55+t14*t15*t23*3.141592653589793*(-t12+t41+t4*(t26-udg4))*(4.0/1.5E+1))-nlg2*(t33*(-t12+t41+t4*(t26-udg4))+t13*t16*t24*t55*3.141592653589793*(3.0/2.0E+1));
		f[3*ng+i] = -nlg1*(t34*(t38-t51)-t14*t15*t23*3.141592653589793*(t42-t52)*(4.0/1.5E+1))-nlg2*(t33*(t42-t52)-t13*t16*t24*3.141592653589793*(t38-t51)*(3.0/2.0E+1))+tau1*(udg4-uhg4);
	});
}

void KokkosFbou(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ib, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	if (ib == 1)
		KokkosFbou1(f, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, modelnumber, ng, nc, ncu, nd, ncx, nco, ncw);
}

