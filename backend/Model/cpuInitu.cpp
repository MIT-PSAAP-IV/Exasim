void cpuInitu(dstype* f, const dstype* xdg, const dstype* uinf, const dstype* param, const int modelnumber, const int ng, const int ncx, const int nce, const int npe, const int ne)
{

  for (int i = 0; i < N; ++i) {
    int p = i%npe; 
    int e = i/npe; 


    f[p+npe*0 +npe*nce*e] = 0.0;
  }
}

