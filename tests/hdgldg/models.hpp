#pragma once
#include <Kokkos_Core.hpp>
#include <limits>

template<int ND,bool Periodic=false>
struct PolynomialDiffusionMath {
    static constexpr int nd=ND, ncu=3, ncw=0, nco=0, nparam=0, Nq=3*(ND+1);
    KOKKOS_INLINE_FUNCTION static double coefficient(int c,int a) {
        const double K[9]={2,.2,.1,.2,1.5,-.1,.1,-.1,1}; return K[3*c+a];
    }
    KOKKOS_INLINE_FUNCTION static double exact(int c,double x,double y,int derivative=0,double z=0) {
        if constexpr(Periodic){x=0;if(derivative==1)return 0;}
        const double a[3]={.7,1.2,-.4}, b[3]={.3,-.5,1.1}, d[3]={-.2,.8,-.7};
        const double e[3]={.4,-.6,.2}, f[3]={.13,.27,-.31}, g[3]={.9,.35,-.45};
        const double h[3]={.2,-.1,.6},i[3]={.11,.17,-.09},j[3]={-.13,.23,.07},k[3]={.3,-.2,.15};
        if(derivative==1) return -(b[c]+2*e[c]*x+f[c]*y+(ND==3?i[c]*z:0));
        if(derivative==2) return -(d[c]+f[c]*x+2*g[c]*y+(ND==3?j[c]*z:0));
        if(derivative==3) return -(h[c]+i[c]*x+j[c]*y+2*k[c]*z);
        const double xy=a[c]+b[c]*x+d[c]*y+e[c]*x*x+f[c]*x*y+g[c]*y*y;
        if constexpr(ND==2)return xy;
        return xy+z*(h[c]+i[c]*x+j[c]*y)+k[c]*z*z;
    }
    KOKKOS_INLINE_FUNCTION static double value(int c,const double x[],int derivative=0)
    { return exact(c,x[0],x[1],derivative,ND==3?x[2]:0); }
    KOKKOS_INLINE_FUNCTION static double gradient(const double uq[],int a,int d)
    {
        double result=uq[3+3*d+a];
        if constexpr(Periodic)if(d<2)result+=.25*uq[3+3*(1-d)+a];
        return result;
    }
#define VOLUME_ARGS const double x[],const double uq[],const double[],const double[],const double[],const double[],double
    KOKKOS_INLINE_FUNCTION static void flux(double out[],VOLUME_ARGS) {
        for(int d=0;d<ND;++d)for(int c=0;c<3;++c){out[c+3*d]=0;
            for(int a=0;a<3;++a)out[c+3*d]+=coefficient(c,a)*gradient(uq,a,d);}
    }
    KOKKOS_INLINE_FUNCTION static void flux_jac_uq(double out[],VOLUME_ARGS) {
        for(int k=0;k<3*ND*Nq;++k)out[k]=0;
        for(int d=0;d<ND;++d)for(int c=0;c<3;++c)for(int a=0;a<3;++a)
            {out[c+3*d+(3*ND)*(3+3*d+a)]=coefficient(c,a);
             if constexpr(Periodic)if(d<2)out[c+3*d+(3*ND)*(3+3*(1-d)+a)]=.25*coefficient(c,a);}
    }
    KOKKOS_INLINE_FUNCTION static void source(double out[],VOLUME_ARGS) {
        const double laplacian[3]={(Periodic?1.8:2.6)+(ND==3?.6:0),(Periodic?.7:-.5)+(ND==3?-.4:0),(Periodic?-.9:-.5)+(ND==3?.3:0)};
        for(int c=0;c<3;++c){out[c]=0;for(int a=0;a<3;++a)out[c]-=coefficient(c,a)*laplacian[a];}
    }
#undef VOLUME_ARGS
#define BOUNDARY_ARGS int ib,const double x[],const double uq[],const double[],const double[],const double uh[],const double n[],const double tau[],const double[],const double[],double
    KOKKOS_INLINE_FUNCTION static void ubou(double out[],BOUNDARY_ARGS) {
        for(int c=0;c<3;++c)out[c]=ib==2*ND+1?std::numeric_limits<double>::quiet_NaN():value(c,x);
    }
    KOKKOS_INLINE_FUNCTION static void fbou(double out[],BOUNDARY_ARGS) {
        for(int c=0;c<3;++c){out[c]=tau[0]*(uq[c]-uh[c]);
            for(int a=0;a<3;++a)for(int d=0;d<ND;++d)out[c]+=coefficient(c,a)*gradient(uq,a,d)*n[d];}
        if(ib==2*ND+1)for(int c=0;c<3;++c)out[c]=std::numeric_limits<double>::quiet_NaN();
    }
    KOKKOS_INLINE_FUNCTION static void fbou_hdg(double out[],BOUNDARY_ARGS) {
        for(int c=0;c<3;++c)out[c]=ib==2*ND+1?std::numeric_limits<double>::quiet_NaN():tau[0]*(value(c,x)-uh[c]);
    }
    KOKKOS_INLINE_FUNCTION static void fbou_hdg_jac_uh(double out[],BOUNDARY_ARGS) {
        for(int c=0;c<3;++c)for(int a=0;a<3;++a)out[3*c+a]=c==a?-tau[0]:0;
    }
#undef BOUNDARY_ARGS
    KOKKOS_INLINE_FUNCTION static void fhat(double out[],const double[],const double u1[],const double u2[],
        const double[],const double[],const double[],const double[],const double[],const double n[],
        const double tau[],const double[],const double[],double) {
        for(int c=0;c<3;++c){out[c]=tau[0]*(u1[c]-u2[c]);
            for(int a=0;a<3;++a)for(int d=0;d<ND;++d)
                out[c]+=.5*coefficient(c,a)*(gradient(u1,a,d)+gradient(u2,a,d))*n[d];}
    }
};
