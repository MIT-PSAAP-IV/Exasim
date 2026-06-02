void cpuInitu(dstype* f, const dstype* xdg, const dstype* uinf, const dstype* param, const int modelnumber, const int ng, const int ncx, const int nce, const int npe, const int ne)
{
	for (int i = 0; i <ng; i++) {
		int j = i%npe;
		int k = i/npe;
		dstype param2 = param[1];
		dstype param3 = param[2];
		dstype param4 = param[3];
		dstype xdg1 = xdg[j+npe*0+npe*ncx*k];
		dstype t2 = -param4;
		dstype t3 = 1.0/param2;
		dstype t4 = param3+t2;
		f[j+npe*0+npe*nce*k] = (param3-t3*t4*xdg1)*(atan(param3*1.0E+6-t3*t4*xdg1*1.0E+6)/3.141592653589793-1.0/2.0)+3.183098861714306E-7;
		f[j+npe*1+npe*nce*k] = 0.0;
	}
}

