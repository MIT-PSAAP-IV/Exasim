void KokkosInitu(dstype* f, const dstype* xdg, const dstype* uinf, const dstype* param, const int modelnumber, const int ng, const int ncx, const int nce, const int npe, const int ne)
{
	Kokkos::parallel_for("Initu", ng, KOKKOS_LAMBDA(const size_t i) {
		int j = i%npe;
		int k = i/npe;
		dstype param1 = param[0];
		dstype param2 = param[1];
		dstype param3 = param[2];
		dstype param4 = param[3];
		dstype param5 = param[4];
		dstype param6 = param[5];
		dstype xdg1 = xdg[j+npe*0+npe*ncx*k];
		dstype xdg2 = xdg[j+npe*1+npe*ncx*k];
		dstype t2 = param3*param3;
		dstype t3 = 1.0/3.141592653589793;
		dstype t5 = param1-1.0;
		dstype t6 = 1.0/param4;
		dstype t8 = -xdg1;
		dstype t9 = -xdg2;
		dstype t10 = sqrt(5.0);
		dstype t4 = t3*t3;
		dstype t7 = t6*t6;
		dstype t11 = param5+t8;
		dstype t12 = param6+t9;
		dstype t13 = 1.0/t5;
		dstype t16 = (param2*t10)/5.0;
		dstype t17 = param2*t10*(2.0/5.0);
		dstype t14 = t11*t11;
		dstype t15 = t12*t12;
		dstype t18 = t14+t15;
		dstype t19 = t7*t18;
		dstype t20 = -t19;
		dstype t21 = t19/2.0;
		dstype t22 = -t21;
		dstype t23 = t20+1.0;
		dstype t24 = exp(t23);
		dstype t25 = t22+1.0/2.0;
		dstype t26 = exp(t25);
		dstype t27 = (t2*t4*t5*t24)/8.0;
		dstype t28 = -t27;
		dstype t30 = (param3*t3*t6*t11*t26)/2.0;
		dstype t31 = (param3*t3*t6*t12*t26)/2.0;
		dstype t29 = t28+1.0;
		dstype t32 = -t30;
		dstype t34 = t17+t31;
		dstype t33 = pow(t29,t13);
		dstype t35 = t16+t32;
		f[j+npe*0+npe*nce*k] = t33;
		f[j+npe*1+npe*nce*k] = t33*t34;
		f[j+npe*2+npe*nce*k] = t33*t35;
		f[j+npe*3+npe*nce*k] = (t33*(t34*t34+t35*t35))/2.0+(t13*pow(t29,param1*t13))/param1;
	});
}

