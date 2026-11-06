// Use Enzyme to differentiate the pointwise physics in the LDG residual.
// Use two P1 elements, one interior face, and two boundary faces.
// Check gradient recovery, implicit auxiliary variables, mass, and smoothed AV.
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
using std::partial_sum;
#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/pblas.h"
#include <modeldefaults.hpp>

struct PdeModel : ModelDefaults<PdeModel> {
    static constexpr int nd=1, ncu=1, ncw=1, nco=2, nparam=1, nmaterialstate=1;
    KOKKOS_INLINE_FUNCTION static void initu(dstype* u, const dstype*, const dstype*, const dstype*) { u[0]=0; }
    KOKKOS_INLINE_FUNCTION static void flux(dstype* f, const dstype*, const dstype* u,
        const dstype* o, const dstype* w, const dstype*, const dstype*, dstype) {
        f[0] = (1+0.1*u[0]*u[0])*u[1] + 0.2*w[0] + o[0]*u[0] + 0.3*o[1];
    }
    KOKKOS_INLINE_FUNCTION static void source(dstype* f, const dstype*, const dstype* u,
        const dstype* o, const dstype* w, const dstype*, const dstype*, dstype) {
        f[0] = u[0]*u[0] + 0.2*u[1]*w[0] + o[0]*u[0];
    }
    KOKKOS_INLINE_FUNCTION static void tdfunc(dstype* f, const dstype*, const dstype* u,
        const dstype* o, const dstype* w, const dstype*, const dstype*, dstype) {
        f[0] = 1 + 0.1*u[0]*u[0] + 0.2*w[0] + 0.1*o[0];
    }
    KOKKOS_INLINE_FUNCTION static void sourcew(dstype* f, const dstype*, const dstype* u,
        const dstype* o, const dstype* w, const dstype*, const dstype*, dstype) {
        f[0] = 0.4*u[0]*u[0] + 0.1*u[1] - 2*w[0] + 0.2*o[0];
    }
    KOKKOS_INLINE_FUNCTION static void avfield(dstype* f, const dstype*, const dstype* u,
        const dstype* o, const dstype* w, const dstype*, const dstype*, dstype) {
        f[0] = 0.02*u[0]*u[0] + 0.01*u[1]*u[1] + 0.03*w[0] + 0.1*o[0]*o[0] + 0.2*o[1];
    }
    KOKKOS_INLINE_FUNCTION static void materialstate(dstype* f, const dstype*, const dstype* u,
        const dstype* o, const dstype*, const dstype*, const dstype*, dstype) {
        f[0] = 1 + 0.2*u[0]*u[0] + 0.1*u[1] + 0.05*o[0];
    }
    KOKKOS_INLINE_FUNCTION static void ubou(dstype* f, int, const dstype* x,
        const dstype* u, const dstype* o, const dstype* w, const dstype* uh,
        const dstype*, const dstype*, const dstype*, const dstype*, dstype) {
        f[0] = x[0] + 0.1*u[0]*u[0] + 0.01*uh[0]*uh[0] + 0.03*w[0] + 0.02*o[0];
    }
    KOKKOS_INLINE_FUNCTION static void fbou(dstype* f, int, const dstype* x,
        const dstype* u, const dstype* o, const dstype* w, const dstype* uh,
        const dstype* n, const dstype* tau, const dstype* p, const dstype* ui, dstype t) {
        flux(f,x,u,o,w,p,ui,t);
        f[0] = f[0]*n[0] + tau[0]*(u[0]-uh[0]);
    }
};
#include <modelprovider.hpp>
#include "../../backend/Model/ModelDispatch/model_drivers_abi.cpp"
#include "../../backend/Discretization/ioutilities.hpp"
#include "../../backend/Discretization/residual.hpp"
#include "../../backend/Discretization/matvec.hpp"

struct Fixture {
    solstruct sol; resstruct res; appstruct app; masterstruct master;
    meshstruct mesh; tempstruct tmp; commonstruct c;
    std::array<dstype,8> udg{}, dudg{}, odg{}, dodg{}, sdg{};
    std::array<dstype,4> x{0,0.5,0.5,1}, w{}, dw{}, history{}, Ru{}, dRu{};
    std::array<dstype,3> uh{}, duh{}, Rh{}, dRh{};
    std::array<dstype,8> Rq{}, dRq{}, sdgg{}, odgg{}, dodgg{};
    std::array<dstype,6> og1{},og2{},dog1{},dog2{};
    std::array<dstype,12> elemg{};
    std::array<dstype,9> faceg{0.5,1,1, 0,1,-1,1,1,1};
    std::array<dstype,8> minv{8,-4,-4,8,8,-4,-4,8}, shapegw{};
    std::array<dstype,4> shapegt{};
    std::array<dstype,1> shapeFace{1}, tau{1.3}, params{0};
    std::array<dstype,2048> tempn{},tempg{};
    std::array<Int,3> eblks{1,2,0};
    std::array<Int,6> fblks{1,1,0,2,3,1}, facecon{1,2,0,0,3,3};
    std::array<Int,8> eind{0,1,4,5,2,3,6,7};
    std::array<Int,6> find1{1,3,0,5,2,7},find2{4,6,0,5,2,7};
    std::array<Int,3> findx{1,0,3};
    std::array<Int,4> cgelcon{0,1,1,2},row{0,1,3,4},cols{0,0,1,1};
    std::array<dstype,2> materialState{-10,10}, materialValues{-28,32};
    std::array<Int,1> materialCounts{1};
    std::array<Int,2> materialOffsets{0,2};
    std::array<Int,64> tempi{};
    Fixture() {
        const dstype a=(1+1/std::sqrt(3.0))/2, b=1-a;
        shapegt={a,b,b,a}; shapegw={a,b,b,a,-0.5,0.5,-0.5,0.5};
        elemg={0.5*b,0.5*a,0.5+0.5*b,0.5+0.5*a,1,1,1,1,0.25,0.25,0.25,0.25};
        sol.udg=udg.data(); sol.dudg=dudg.data(); sol.xdg=x.data();
        sol.wdg=w.data(); sol.dwdg=dw.data(); sol.odg=odg.data(); sol.dodg=dodg.data();
        sol.uh=uh.data(); sol.duh=duh.data(); sol.sdg=sdg.data(); sol.sdgg=sdgg.data();
        sol.wsrc=history.data(); sol.elemg=elemg.data(); sol.faceg=faceg.data();
        sol.odgg=odgg.data(); sol.dodgg=dodgg.data(); sol.og1=og1.data(); sol.og2=og2.data();
        sol.dog1=dog1.data(); sol.dog2=dog2.data(); sol.szwdg=4; sol.szudg=8;
        res.Ru=res.Rue=Ru.data(); res.dRu=res.dRue=dRu.data();
        res.Rq=res.Rqe=Rq.data(); res.dRq=res.dRqe=dRq.data();
        res.Rh=Rh.data(); res.dRh=dRh.data(); res.Minv=minv.data();
        master.shapegt=shapegt.data(); master.shapegw=shapegw.data();
        master.shapfgt=master.shapfgw=master.shapfnt=shapeFace.data();
        mesh.eindudg1=eind.data(); mesh.findudg1=find1.data(); mesh.findudg2=find2.data();
        mesh.findxdg1=findx.data(); mesh.facecon=facecon.data();
        mesh.cgelcon=cgelcon.data(); mesh.rowent2elem=row.data(); mesh.colent2elem=cols.data();
        tmp.tempn=tempn.data(); tmp.tempg=tempg.data(); tmp.tempi=tempi.data();
        app.tau=tau.data(); app.physicsparam=app.uinf=params.data();
        c.components.nc=2; c.components.ncu=c.components.ncq=c.components.ncw=c.components.ncx=1;
        c.components.nco=c.components.ncs=2; c.components.ntau=1;
        c.grid.nd=c.grid.npf=c.grid.ngf=1; c.grid.npe=c.grid.nge=2; c.grid.curvedMesh=1;
        c.meshsizes.ne=c.meshsizes.ne1=c.meshsizes.ne2=2; c.meshsizes.nf=3;
        c.meshsizes.nbe=1; c.meshsizes.nbf=2; c.eblks=eblks.data(); c.fblks=fblks.data();
        c.sizes.ndof1=4; c.sizes.ndofucg=3; c.mpiProcs=1;
        c.physicsparams.ncAV=1; c.physicsparams.frozenAVflag=0; c.physicsparams.source=1;
        c.physicsparams.AVsmoothingIter=1;
        c.timeparams.tdfunc=1; c.timeparams.dae_alpha=1; c.timeparams.dae_beta=2;
        c.timestate.dtfactor=2;
        c.driver_abi=const_cast<ExasimDriverABI*>(&getKokkosKernelExasimDriverABI());
        app.materialdb_nstate=1; app.materialdb_ne=1; app.materialdb_npe=2; app.materialdb_porder=1;
        app.materialdb_statecoords=app.materialdb_elemcoords=materialState.data();
        app.materialdb_propvalues=materialValues.data(); app.materialdb_elementcounts=materialCounts.data();
        app.materialdb_elemoffset=materialOffsets.data();
        reset();
    }
    void reset() {
        udg={0.2,0.35,0.4,0.5,0.7,0.8,0.6,0.9};
        odg={0.3,0.4,0.7,0.8,0.5,0.6,0.9,1.0};
        w={0.1,0.2,0.3,0.4}; uh={0.2,0.3,0.4}; history={0.5,0.6,0.7,0.8};
        sdg={0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1}; sdgg.fill(0.6);
        // Fixed auxiliary values at quadrature points are overwritten only with unfrozen AV.
        odgg.fill(0.4); og1.fill(0.4); og2.fill(0.4);
    }
};

static void require(bool ok,const char* text) { if (!ok) throw std::runtime_error(text); }
template<class Model> static void verify(Fixture& f, int time, int wave, bool frozen) {
    f.c.timeparams.tdep=time; f.c.timeparams.wave=wave;
    f.c.physicsparams.frozenAVflag=frozen ? 1:0;
    const std::array<dstype,4> u{0.2,0.35,0.7,0.8};
    std::array<dstype,4> v{0.4,-0.3,0.2,0.5}, result{}, repeated{}, plus{},minus{},probe{};
    auto evaluate=[&](const std::array<dstype,4>& state,std::array<dstype,4>& out) {
        f.reset();
        ArrayInsert(f.sol.udg,state.data(),2,2,2,0,2,0,1,0,2);
        Residual<Model>(f.sol,f.res,f.app,f.master,f.mesh,f.tmp,f.c,f.c.cublasHandle,0);
        Kokkos::fence(); out=f.Ru;
    };
    f.reset(); const auto originalU=f.udg, originalO=f.odg; const auto originalW=f.w; const auto originalH=f.uh;
    auto jvp=[&](std::array<dstype,4>& direction,std::array<dstype,4>& out) {
        MatVec<Model>(out.data(),f.sol,f.res,f.app,f.master,f.mesh,f.tmp,f.c,
            f.c.cublasHandle,direction.data(),const_cast<dstype*>(u.data()),f.Ru.data(),0);
        Kokkos::fence();
    };
    jvp(v,result);
    require(f.udg==originalU && f.odg==originalO && f.w==originalW && f.uh==originalH,"MatVec changed its input state");
    auto other=v; other[0]*=-2; jvp(other,repeated); jvp(v,repeated);
    require(result==repeated,"Repeated JVP changed with fixed input state");
    const dstype eps=1e-6;
    for(int i=0;i<4;++i)probe[i]=u[i]+eps*v[i]; evaluate(probe,plus);
    for(int i=0;i<4;++i)probe[i]=u[i]-eps*v[i]; evaluate(probe,minus);
    dstype error=0;
    for(int i=0;i<4;++i) {
        require(std::isfinite(result[i]) && std::isfinite(plus[i]) && std::isfinite(minus[i]),
                "Residual JVP or finite-difference input is not finite");
        error=std::max(error,std::abs(result[i]-(plus[i]-minus[i])/(2*eps)));
    }
    std::printf("residual tdep=%d wave=%d frozenAV=%d material=%d max JVP error %.3e\n",
        time,wave,frozen,f.app.materialdb_nprop,(double)error);
    require(error<2e-7,"Enzyme residual JVP disagrees with central differences");
    f.reset(); v.fill(0); jvp(v,result);
    for(auto value:result)require(value==0,"Zero direction has nonzero JVP");
}
#ifndef EXASIM_ENZYME_RESIDUAL_FIXTURE_ONLY
int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv); int code=0;
    try {
        Fixture f;
        for(int typed=0;typed<2;++typed)for(int time=0;time<2;++time)for(int wave=0;wave<2;++wave)for(bool frozen:{false,true}) {
            if(typed)verify<PdeModel>(f,time,wave,frozen);
            else verify<exasim::detail::AbiAdapter>(f,time,wave,frozen);
        }
        f.app.materialdb_nprop=1;
        verify<PdeModel>(f,1,0,false);
        verify<exasim::detail::AbiAdapter>(f,1,0,false);
    } catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what());code=1; }
    Kokkos::finalize();return code;
}
#endif
