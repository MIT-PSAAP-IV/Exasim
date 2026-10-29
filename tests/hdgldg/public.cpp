// Exercise the same provider/launcher API as a generated application.
#include "models.hpp"
#include "ExasimSolverSetup.hpp"
#include "modeldefaults.hpp"

struct PdeModel : ModelDefaults<PdeModel> {
    using Math=PolynomialDiffusionMath<2>;
    static constexpr int nd=2,ncu=3,ncw=0,nco=0,nparam=0,Nq=9;
    // Forward the shared point model without duplicating its equations.
#define REUSE_MATH(name) template<class... Args> KOKKOS_INLINE_FUNCTION static void name(Args... args){Math::name(args...);}
    REUSE_MATH(flux) REUSE_MATH(flux_jac_uq) REUSE_MATH(source)
    REUSE_MATH(ubou) REUSE_MATH(fbou) REUSE_MATH(fbou_hdg)
    REUSE_MATH(fbou_hdg_jac_uh) REUSE_MATH(fhat)
#undef REUSE_MATH
    KOKKOS_INLINE_FUNCTION static void initu(double out[],const double x[],const double[],const double[])
    {for(int c=0;c<ncu;++c)out[c]=Math::value(c,x);}
#define BOUNDARY_ARGUMENTS int,const double[],const double[],const double[],const double[],const double[],const double n[],const double tau[],const double[],const double[],double
    KOKKOS_INLINE_FUNCTION static void fbou_jac_uq(double out[],BOUNDARY_ARGUMENTS)
    {
        for(int a=0;a<Nq;++a)for(int c=0;c<ncu;++c){double v=a==c?tau[0]:0;
            if(a>=ncu)v+=Math::coefficient(c,(a-ncu)%ncu)*n[(a-ncu)/ncu];
            out[c+ncu*a]=v;}
    }
    KOKKOS_INLINE_FUNCTION static void fbou_jac_uh(double out[],BOUNDARY_ARGUMENTS)
    {for(int a=0;a<ncu;++a)for(int c=0;c<ncu;++c)out[c+ncu*a]=a==c?-tau[0]:0;}
#undef BOUNDARY_ARGUMENTS
};
#include "modelprovider.hpp"

#ifndef HDGLDG_PUBLIC_NO_MAIN
int main(int argc,char** argv)
{
    ExasimSolver solver;
    return RunExasimSolver(solver,argc,argv,MPI_COMM_WORLD);
}
#endif
