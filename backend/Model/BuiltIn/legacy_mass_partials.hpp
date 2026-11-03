// Included inside a known legacy built-in model namespace by its CMake overlay.
// Models 1--15 have unknown-independent mass coefficients. This adapter is not
// valid for custom/state-dependent models: regenerate those with current codegen.
inline void KokkosTdfunc(dstype* f, dstype* f_udg, dstype* f_wdg,
        const dstype* xdg, const dstype* udg, const dstype* odg,
        const dstype* wdg, const dstype* uinf, const dstype* param,
        const dstype time, const int modelnumber, const int ng, const int nc,
        const int ncu, const int nd, const int ncx, const int nco, const int ncw)
{
    KokkosTdfunc(f,xdg,udg,odg,wdg,uinf,param,time,modelnumber,ng,nc,ncu,nd,ncx,nco,ncw);
    Kokkos::parallel_for("legacy built-in mass partials", ng*ncu*nc,
        KOKKOS_LAMBDA(const size_t i) { f_udg[i]=0; });
    if(ncw>0)
        Kokkos::parallel_for("legacy built-in mass w partials", ng*ncu*ncw,
            KOKKOS_LAMBDA(const size_t i) { f_wdg[i]=0; });
}

