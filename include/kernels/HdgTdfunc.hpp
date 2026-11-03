// Value and derivatives of the time-derivative mass coefficient.
template <class Model>
static void HdgTdfuncTemplate(dstype* f, dstype* f_udg, dstype* f_wdg,
        const dstype* xdg, const dstype* udg, const dstype* odg,
        const dstype* wdg, const dstype* uinf, const dstype* param,
        const dstype time, const int modelnumber, const int ng,
        const int nc, const int ncu, const int nd, const int ncx,
        const int nco, const int ncw)
{
    (void)modelnumber; (void)nc; (void)ncu; (void)nd;
    (void)ncx; (void)nco; (void)ncw;
    Kokkos::parallel_for("HdgTdfunc", ng, KOKKOS_LAMBDA(const size_t i) {
        constexpr int nd=Model::nd, ncu=Model::ncu, nc=ncu*(1+nd);
        constexpr int nco=Model::nco, ncw=Model::ncw;
        dstype x[nd], uq[nc], v[nco>0?nco:1], w[ncw>0?ncw:1];
        for(int k=0;k<nd;++k) x[k]=xdg[k*ng+i];
        for(int k=0;k<nc;++k) uq[k]=udg[k*ng+i];
        for(int k=0;k<nco;++k) v[k]=odg[k*ng+i];
        for(int k=0;k<ncw;++k) w[k]=wdg[k*ng+i];
        dstype m[ncu], m_uq[ncu*nc];
        Model::tdfunc(m,x,uq,v,w,param,uinf,time);
        Model::tdfunc_jac_uq(m_uq,x,uq,v,w,param,uinf,time);
        for(int k=0;k<ncu;++k) f[k*ng+i]=m[k];
        for(int k=0;k<ncu*nc;++k) f_udg[k*ng+i]=m_uq[k];
        dstype m_w[(ncu*ncw>0) ? ncu*ncw : 1];
        Model::tdfunc_jac_w(m_w,x,uq,v,w,param,uinf,time);
        for(int k=0;k<ncu*ncw;++k) f_wdg[k*ng+i]=m_w[k];
    });
}

void KokkosTdfunc(dstype* f, dstype* f_udg, dstype* f_wdg,
        const dstype* xdg, const dstype* udg, const dstype* odg,
        const dstype* wdg, const dstype* uinf, const dstype* param,
        const dstype time, const int modelnumber, const int ng,
        const int nc, const int ncu, const int nd, const int ncx,
        const int nco, const int ncw)
{
    HdgTdfuncTemplate<PdeModel>(f,f_udg,f_wdg,xdg,udg,odg,wdg,uinf,param,
        time,modelnumber,ng,nc,ncu,nd,ncx,nco,ncw);
}

