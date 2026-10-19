void KokkosAvfield(dstype* f, const dstype* xdg, const dstype* udg, const dstype* odg, const dstype* wdg, const dstype* uinf, const dstype* param, const dstype time, const int modelnumber, const int ng, const int nc, const int ncu, const int nd, const int ncx, const int nco, const int ncw, const int nce, const int npe, const int ne)
{
	Kokkos::parallel_for("Avfield", ng, KOKKOS_LAMBDA(const size_t i) {
		int j = i%npe;
		int k = i/npe;
		dstype param1 = param[0];
		dstype param2 = param[1];
		dstype param4 = param[3];
		dstype param10 = param[9];
		dstype param12 = param[11];
		dstype param13 = param[12];
		dstype xdg2 = xdg[j+npe*1+npe*ncx*k];
		dstype udg1 = udg[j+npe*0+npe*nc*k];
		dstype udg2 = udg[j+npe*1+npe*nc*k];
		dstype udg3 = udg[j+npe*2+npe*nc*k];
		dstype udg4 = udg[j+npe*3+npe*nc*k];
		dstype udg5 = udg[j+npe*4+npe*nc*k];
		dstype udg6 = udg[j+npe*5+npe*nc*k];
		dstype udg9 = udg[j+npe*8+npe*nc*k];
		dstype udg11 = udg[j+npe*10+npe*nc*k];
		dstype odg1 = odg[j+npe*0+npe*nco*k];
		dstype t2 = odg1*param13;
		dstype t3 = 1.0/3.141592653589793;
		dstype t5 = 1.0/udg1;
		dstype t6 = 1.0/xdg2;
		dstype t4 = tanh(t2);
		dstype t7 = t5*udg2*udg5;
		dstype t8 = t5*udg3*udg9;
		dstype t11 = t5*t6*udg3;
		dstype t17 = t4*(-t11+t5*(t7-udg6)+t5*(t8-udg11))*-1.0E+3;
		dstype t18 = atan(t17);
		dstype t19 = t3*t18;
		dstype t20 = t19+1.0/2.0;
		dstype t21 = -t4*t20*(-t11+t5*(t7-udg6)+t5*(t8-udg11));
		f[j+npe*0+npe*nce*k] = t21+(t3*atan(param12*-1.0E+3+t17*t20+3.183097800805168E-1)+1.0/2.0)*(param12+t4*t20*(-t11+t5*(t7-udg6)+t5*(t8-udg11))-3.183097800805168E-4);
		f[j+npe*1+npe*nce*k] = (sqrt(2.87E+2)*param1*param2*param4*param10*sqrt((param10*param10*param10)*4.906777729315601E-8)*1.0/sqrt(param1*param10)*(param1-1.0)*(-udg4+(t5*(udg2*udg2))/2.0+(t5*(udg3*udg3))/2.0)*(-6.581717999999999E-3))/(param10+5.52E+2/5.0);
	});
}

