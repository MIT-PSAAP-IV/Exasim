void KokkosVisScalars(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
	Kokkos::parallel_for("VisScalars", ng, KOKKOS_LAMBDA(const size_t i) {
		dstype param1 = param[0];
		dstype param2 = param[1];
		dstype param3 = param[2];
		dstype param4 = param[3];
		dstype param5 = param[4];
		dstype param6 = param[5];
		dstype xdg1 = xdg[0*ng+i];
		dstype xdg2 = xdg[1*ng+i];
		dstype udg1 = udg[0*ng+i];
		dstype udg2 = udg[1*ng+i];
		dstype udg3 = udg[2*ng+i];
		dstype udg4 = udg[3*ng+i];
		dstype t2 = param1-1.0;
		dstype t3 = 1.0/(udg1*udg1);
		dstype t4 = sqrt(5.0);
		dstype t5 = (xdg1*3.141592653589793)/1.0E+1;
		dstype t6 = xdg2*3.141592653589793*(2.0/1.5E+1);
		dstype t7 = sin(t5);
		dstype t8 = sin(t6);
		f[0*ng+i] = udg1;
		f[1*ng+i] = t2*(udg4-(udg1*(t3*(udg2*udg2)+t3*(udg3*udg3)))/2.0);
		f[2*ng+i] = udg1-pow((param3*param3)*t2*1.0/(3.141592653589793*3.141592653589793)*exp(-1.0/(param4*param4)*(pow(param5-xdg1-t7*t8*sin(t4*time*3.141592653589793*(2.0/2.5E+1))*2.0+param2*t4*time*(2.0/5.0),2.0)+pow(param6-xdg2-t7*t8*sin(t4*time*3.141592653589793*(4.0/2.5E+1))*(3.0/2.0)+(param2*t4*time)/5.0,2.0))+1.0)*(-1.0/8.0)+1.0,1.0/t2);
	});
}

