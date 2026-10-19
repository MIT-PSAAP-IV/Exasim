void HdgFintonly1(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("Fintonly1", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype param2 = param[1];
		dstype param3 = param[2];
		dstype param4 = param[3];
		dstype param10 = param[9];
		dstype param13 = param[12];
		dstype param14 = param[13];
		dstype param15 = param[14];
		dstype tau1 = tau[0];
		dstype xdg2 = xdg[1*ng+i];
		dstype udg1 = udg[0*ng+i];
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
		dstype odg1 = odg[0*ng+i];
		dstype odg2 = odg[1*ng+i];
		dstype nlg1 = nlg[0*ng+i];
		dstype nlg2 = nlg[1*ng+i];
		dstype t2 = odg1*param13;
		dstype t3 = odg2*param15;
		dstype t4 = param4*param4;
		dstype t5 = uhg2*uhg2;
		dstype t6 = uhg3*uhg3;
		dstype t7 = 1.0/3.141592653589793;
		dstype t9 = param1-1.0;
		dstype t10 = 1.0/param2;
		dstype t11 = 1.0/param3;
		dstype t12 = 1.0/xdg2;
		dstype t14 = uhg1*4.0E+3;
		dstype t16 = uhg1-1.0/2.0E+1;
		dstype t17 = param10+5.52E+2/5.0;
		dstype t8 = tanh(t2);
		dstype t13 = param14+t3;
		dstype t15 = 1.0/t9;
		dstype t18 = t14-2.0E+2;
		dstype t19 = atan(t18);
		dstype t20 = t7*t19;
		dstype t21 = t20+1.0/2.0;
		dstype t22 = t16*t21;
		dstype t23 = t22*4.0E+3;
		dstype t24 = t22+7.957746988807024E-5;
		dstype t26 = t22+5.007957746988807E-2;
		dstype t25 = t24*t24;
		dstype t27 = t23+3.18309879552281E-1;
		dstype t29 = 1.0/t26;
		dstype t28 = atan(t27);
		dstype t30 = t29*t29;
		dstype t31 = t29*uhg4;
		dstype t33 = t25*1.6E+7;
		dstype t35 = t12*t29*uhg3;
		dstype t32 = t7*t28;
		dstype t34 = t33+1.0;
		dstype t36 = (t5*t30)/2.0;
		dstype t37 = (t6*t30)/2.0;
		dstype t38 = 1.0/t34;
		dstype t40 = t36+t37;
		dstype t39 = t7*t27*t38;
		dstype t41 = t26*t40;
		dstype t44 = t32+t39+1.0/2.0;
		dstype t46 = t9*(t41-uhg4)*-4.0E+3;
		dstype t48 = t46-8.0;
		dstype t52 = t29*t44*udg5*uhg2;
		dstype t53 = t29*t44*udg5*uhg3;
		dstype t54 = t29*t44*udg9*uhg2;
		dstype t55 = t29*t44*udg9*uhg3;
		dstype t49 = atan(t48);
		dstype t50 = t7*t49;
		dstype t51 = t50+1.0/2.0;
		dstype t67 = t51*(t9*(t41-uhg4)+1.0/5.0E+2)*-4.0E+3;
		dstype t69 = pow(t51*(t9*(t41-uhg4)+1.0/5.0E+2)-7.957746988807024E-5,2.0);
		dstype t77 = -t29*(t51*(t9*(t41-uhg4)+1.0/5.0E+2)-2.07957746988807E-3);
		dstype t82 = -1.0/(param1*param10*t4*t29*(t51*(t9*(t41-uhg4)+1.0/5.0E+2)-2.07957746988807E-3)-5.52E+2/5.0);
		dstype t70 = t67+3.18309879552281E-1;
		dstype t73 = t69*1.6E+7;
		dstype t78 = param1*t4*t77;
		dstype t83 = t31+t77;
		dstype t72 = atan(t70);
		dstype t75 = t73+1.0;
		dstype t80 = pow(t78,3.0/2.0);
		dstype t74 = t7*t72;
		dstype t76 = 1.0/t75;
		dstype t85 = -t7*t76*(t51*(t9*(t41-uhg4)+1.0/5.0E+2)*4.0E+3-3.18309879552281E-1);
		dstype t86 = t74+t85+1.0/2.0;
		f[0*ng+i] = tau1*(udg4-uhg4)+nlg1*(t83*uhg2+t8*t13*udg8+(t10*t17*t29*t80*uhg3*(t29*(t53-udg7)+t29*(t54-udg10)))/(param1*param10*t4*t29*(t51*(t9*(t41-uhg4)+1.0/5.0E+2)-2.07957746988807E-3)-5.52E+2/5.0)-(t10*t17*t29*t80*uhg2*(t35-t29*(t52-udg6)*2.0+t29*(t55-udg11))*(2.0/3.0))/(param1*param10*t4*t29*(t51*(t9*(t41-uhg4)+1.0/5.0E+2)-2.07957746988807E-3)-5.52E+2/5.0)+param1*t10*t11*t15*t17*t30*t80*t82*(t44*udg5*(t51*(t9*(t41-uhg4)+1.0/5.0E+2)-2.07957746988807E-3)+t9*t26*t86*(udg8+t26*(t30*uhg2*(t52-udg6)+t30*uhg3*(t53-udg7))-t40*t44*udg5)))+nlg2*(t83*uhg3+t8*t13*udg12+(t10*t17*t29*t80*uhg2*(t29*(t53-udg7)+t29*(t54-udg10)))/(param1*param10*t4*t29*(t51*(t9*(t41-uhg4)+1.0/5.0E+2)-2.07957746988807E-3)-5.52E+2/5.0)-(t10*t17*t29*t80*uhg3*(t35+t29*(t52-udg6)-t29*(t55-udg11)*2.0)*(2.0/3.0))/(param1*param10*t4*t29*(t51*(t9*(t41-uhg4)+1.0/5.0E+2)-2.07957746988807E-3)-5.52E+2/5.0)+param1*t10*t11*t15*t17*t30*t80*t82*(t44*udg9*(t51*(t9*(t41-uhg4)+1.0/5.0E+2)-2.07957746988807E-3)+t9*t26*t86*(udg12+t26*(t30*uhg2*(t54-udg10)+t30*uhg3*(t55-udg11))-t40*t44*udg9)));
	});
}

void HdgFintonly(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uhg, const dstype* nlg, const dstype* tau, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ib, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	if (ib == 1)
		HdgFintonly1(f, xdg, udg, odg, wdg, uhg, nlg, tau, uinf, param, time, modelnumber, ng, nc, ncu, nd, ncx, nco, ncw);
}

