#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
using std::partial_sum;
#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/pblas.h"
#include "../../backend/Discretization/material_properties.hpp"
#include "../../backend/Discretization/wequation.hpp"

// Exercise the real matvec and auxiliary Newton routines. Only the volume
// model and global PDE residual are small analytic fixtures, not copied solver
// implementations. This executable uses CPU BLAS, like auxiliary_eos_trial.
static int residualKind = 0;
static int residualCalls = 0;
static int sourceCalls = 0;

static void sourceOnly(double* f, double* fw, const double*, const double* u,
    const double*, const double* w, const double*, const double*, double,
    int model, int ng, int, int, int, int, int, int)
{
    ++sourceCalls;
    Kokkos::parallel_for("consistency_source", ng, KOKKOS_LAMBDA(int i) {
        if (model == 1) {
            f[i] = log(w[i]) + u[i];
            fw[i] = 1.0/w[i];
        } else if (model == 2) {
            f[i] = u[i] - 2*w[i] + 0.5*w[i+ng];
            f[i+ng] = u[i+ng] + 0.25*w[i] - 3*w[i+ng];
            fw[i] = -2; fw[i+ng] = 0.25;
            fw[i+2*ng] = 0.5; fw[i+3*ng] = -3;
        } else {
            // A deliberately inconsistent local model never converges.
            f[i] = 1.0;
            fw[i] = 1.0;
        }
    });
}

static void sourceWithDerivative(double* f, double* fu, double* fw,
    const double* x, const double* u, const double* o, const double* w,
    const double* ui, const double* p, double t, int model, int ng, int nc,
    int ncu, int nd, int ncx, int nco, int ncw)
{
    sourceOnly(f,fw,x,u,o,w,ui,p,t,model,ng,nc,ncu,nd,ncx,nco,ncw);
    Kokkos::parallel_for("consistency_source_derivative", ng*ncw*nc,
        KOKKOS_LAMBDA(int i) {
            const int q = i/ng;
            fu[i] = q % ncw == q / ncw ? 1.0 : 0.0;
        });
}

template<class M, class T, class I>
void Residual(solstructT<T,I>& sol, resstructT<T,I>& res, appstructT<T,I>& app,
    masterstructT<T,I>&, meshstructT<T,I>&, tempstructT<T,I>& tmp,
    commonstructT<T,I>& common, cublasHandle_t, Int backend)
{
    ++residualCalls;
    const int n = common.sizes.ndof1;
    if (residualKind == 0) {
        ArrayCopy(res.Ru, sol.udg, n);
    } else {
        wEquation<M>(sol.wdg,sol.xdg,sol.udg,sol.odg,sol.wsrc,tmp.tempg,
                     app,common,n,backend,nullptr);
        ArrayAXPBY(res.Ru,sol.udg,sol.wdg,T(1),T(2),n);
    }
}
#include "../../backend/Discretization/matvec.hpp"

static int failures = 0;
static void require(bool okay, const char* message)
{
    if (!okay) {
        std::fprintf(stderr,"FAIL: %s\n",message);
        ++failures;
    }
}

static bool close(double got, double want, double tolerance)
{
    return is_finite_bitwise(got) && std::abs(got-want) <= tolerance;
}

static void checkMatvec()
{
    solstruct sol{}; resstruct res{}; appstruct app{}; masterstruct master{};
    meshstruct mesh{}; tempstruct tmp{}; commonstruct common{};
    Kokkos::View<double*> u("base",2), v("direction",2), jv("jv",2),
        ru("base_residual",2), udg("udg",2), r("residual",2), w("auxiliary",2),
        zeroes("zeroes",2), scratch("scratch",4);
    common.grid.npe=1; common.grid.nd=1;
    common.components.nc=1; common.components.ncu=1; common.components.ncx=1;
    common.meshsizes.ne1=2; common.sizes.ndof1=2;
    common.solverparams.matvecTol=1e-6;
    sol.udg=udg.data();
    sol.xdg=zeroes.data(); sol.odg=zeroes.data(); sol.wsrc=zeroes.data();
    res.Ru=r.data(); tmp.tempg=scratch.data();
    auto set = [](auto view, double a, double b) {
        auto host = Kokkos::create_mirror_view(view);
        host(0)=a; host(1)=b; Kokkos::deep_copy(view,host);
    };
    auto apply = [&](bool alias, double knownNorm=-1) {
        MatVec<exasim::detail::AbiAdapter>(alias ? v.data() : jv.data(),sol,res,
            app,master,mesh,tmp,common,common.cublasHandle,v.data(),u.data(),
            ru.data(),0,knownNorm);
        Kokkos::fence();
        return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),alias?v:jv);
    };
    set(u,1,2); set(ru,1,2);
    for (int order : {1,2}) {
        common.solverparams.matvecOrder=order;
        for (bool alias : {false,true}) {
            for (double scale : {1.0,1e-8,1e-12,1e-16}) {
                set(v,scale,-2*scale);
                auto out=apply(alias);
                require(close(out(0)/scale,1,2e-9),"scaled/aliased identity Jv first component");
                require(close(out(1)/scale,-2,2e-9),"scaled/aliased identity Jv second component");
            }
            set(v,0,0);
            const int before=residualCalls;
            auto out=apply(alias);
            require(out(0)==0 && out(1)==0,"zero direction must give zero Jv");
            require(residualCalls==before,"zero direction must not evaluate residual");
        }
        set(v,1/std::sqrt(5.0),-2/std::sqrt(5.0));
        auto out=apply(false,1);
        require(close(out(0),1/std::sqrt(5.0),2e-9),"known normalized direction");
    }
    auto base=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),ru);
    require(base(0)==1 && base(1)==2,"central difference must preserve base residual");

    ExasimDriverABI abi{};
    abi.hdgjac.HdgSourcewonly=sourceOnly;
    abi.hdgjac.HdgSourcew=sourceWithDerivative;
    common.driver_abi=&abi; common.modelnumber=1; common.components.ncw=1;
    sol.wdg=w.data(); sol.szwdg=2;
    common.timeparams.wave=0; common.timeparams.dae_alpha=0; common.timeparams.dae_beta=0;
    residualKind=1;
    set(u,0,0); set(ru,2,2); set(w,1,1);
    for (int order : {1,2}) {
        common.solverparams.matvecOrder=order;
        // An intentionally small probe checks that the EOS no longer returns
        // the unchanged seed just because the source residual is below 1e-6.
        set(v,1e-6,0);
        auto out=apply(false,1);
        require(close(out(0)/1e-6,-1,5e-4),"EOS sensitivity retained in tiny FD probe");
        auto seed=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),w);
        require(seed(0)==1 && seed(1)==1,"matvec must restore auxiliary seed");
    }
    residualKind=0;
}

static void checkAuxiliaryAgainstHdg()
{
    appstruct app{}; commonstruct common{}; ExasimDriverABI abi{};
    abi.hdgjac.HdgSourcewonly=sourceOnly;
    abi.hdgjac.HdgSourcew=sourceWithDerivative;
    common.driver_abi=&abi; common.grid.nd=1;
    common.components.nc=1; common.components.ncu=1;
    common.components.ncw=1; common.components.ncx=1;
    common.timeparams.wave=0; common.timeparams.dae_alpha=0; common.timeparams.dae_beta=0;
    common.modelnumber=1;
    double w=1, u=1e-12, derivative=0, zeros[4]{}, scratch[20]{};
    wEquation<exasim::detail::AbiAdapter>(&w,zeros,&u,zeros,zeros,scratch,app,common,1,0,nullptr);
    const double ldg=w;
    w=1;
    wEquation<exasim::detail::AbiAdapter>(&w,&derivative,zeros,&u,zeros,zeros,scratch,app,common,1,0,nullptr);
    require(ldg<1 && ldg==w,"LDG and HDG apply the same tiny EOS correction");
    require(close(derivative,-std::exp(-u),2e-15),"HDG implicit EOS derivative preserved");

    // Two coupled ordinary DAE variables catch an erroneous scalar broadcast
    // onto off-diagonal Jacobian entries. A = I - S_w = [3,-.5;-.25,4].
    common.modelnumber=2; common.components.ncw=2;
    common.components.nc=2; common.components.ncu=2;
    common.timeparams.dae_beta=1;
    double state[2]{}, input[2]={2,-1}, deriv[4]{};
    const double det=12-0.125;
    const double expected[2]={(4*input[0]+0.5*input[1])/det,
                              (0.25*input[0]+3*input[1])/det};
    wEquation<exasim::detail::AbiAdapter>(state,zeros,input,zeros,zeros,scratch,app,common,1,0,nullptr);
    require(close(state[0],expected[0],1e-14) && close(state[1],expected[1],1e-14),
            "coupled DAE LDG state");
    state[0]=state[1]=0;
    wEquation<exasim::detail::AbiAdapter>(state,deriv,zeros,input,zeros,zeros,scratch,app,common,1,0,nullptr);
    require(close(state[0],expected[0],1e-14) && close(state[1],expected[1],1e-14),
            "coupled DAE HDG state");
    require(close(deriv[0],4/det,1e-14) && close(deriv[1],0.25/det,1e-14) &&
            close(deriv[2],0.5/det,1e-14) && close(deriv[3],3/det,1e-14),
            "coupled DAE HDG implicit derivative");

    common.timeparams.wave=1; common.timestate.dtfactor=2;
    common.components.nc=3; common.components.ncu=3;
    double waveInput[3]={2,-1,7}, waveDerivative[6];
    for (int nw : {1,2}) {
        common.components.ncw=nw;
        wEquation<exasim::detail::AbiAdapter>(state,waveDerivative,zeros,waveInput,
            zeros,zeros,scratch,app,common,1,0,nullptr);
        for (int col=0; col<3; ++col)
            for (int row=0; row<nw; ++row)
                require(waveDerivative[row+nw*col]==(row==col ? 0.5 : 0),
                        "HDG wave derivative is a rectangular identity");
        require(state[0]==1 && (nw==1 || state[1]==-0.5),"wave state unchanged");
    }
    common.timeparams.wave=0;
    common.modelnumber=3; common.components.ncw=1;
    common.components.nc=1; common.components.ncu=1;
    common.timeparams.dae_beta=0;
    w=1; u=0; sourceCalls=0;
    wEquation<exasim::detail::AbiAdapter>(&w,zeros,&u,zeros,zeros,scratch,app,common,1,0,nullptr);
    require(sourceCalls==20 && !is_finite_bitwise(w),"LDG EOS iteration exhaustion must propagate failure");
    w=1; derivative=123; sourceCalls=0;
    wEquation<exasim::detail::AbiAdapter>(&w,&derivative,zeros,&u,zeros,zeros,scratch,app,common,1,0,nullptr);
    require(sourceCalls==20 && !is_finite_bitwise(w) && !is_finite_bitwise(derivative),
            "HDG EOS exhaustion must not leave a stale derivative");

    common.modelnumber=1; w=1; u=2; derivative=123; sourceCalls=0;
    wEquation<exasim::detail::AbiAdapter>(&w,&derivative,zeros,&u,zeros,zeros,scratch,app,common,1,0,nullptr);
    require(sourceCalls==2 && !is_finite_bitwise(w) && !is_finite_bitwise(derivative),
            "HDG invalid EOS trial must propagate failure promptly");
}

int main(int argc,char** argv)
{
    Kokkos::initialize(argc,argv);
    {
        checkMatvec();
        checkAuxiliaryAgainstHdg();
    }
    Kokkos::finalize();
    std::printf("LDG matvec / HDG auxiliary consistency: %d failures\n",failures);
    return failures ? 1 : 0;
}
