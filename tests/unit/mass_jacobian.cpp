// Real Q1 element assembly: nonlinear mass, source/flux partials and local w.
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <backend/Common/common.h>
#include <backend/Common/cpuimpl.h>
#include <backend/Common/kokkosimpl.h>
#include <backend/Common/pblas.h>
#include <backend/Discretization/ioutilities.hpp>
#include <backend/Discretization/residual.hpp>

namespace {
constexpr int np=2, ng=2, ncu=2, nc=4;

void require(bool condition, const char* message)
{
    if(!condition) throw std::runtime_error(message);
}

// Each mass variant shares nonzero sources and fluxes. w is eliminated from
// sourcew = u0 + .25*u1 + .1*q0 - w; its total derivative is known exactly.
template<int Mode> struct Model : exasim::ModelDefaults<Model<Mode>> {
    static constexpr int nd=1, ncu=2, ncw=1, nparam=1;
    static constexpr auto disc=exasim::Discretization::HDG;
    KOKKOS_INLINE_FUNCTION static void sourcew(dstype* s, const dstype*,
            const dstype* u, const dstype*, const dstype* w,
            const dstype*, const dstype*, dstype) {
        s[0]=u[0]+.25*u[1]+.1*u[2]-w[0];
    }
    KOKKOS_INLINE_FUNCTION static void sourcew_jac_uq(dstype* s, const dstype*,
            const dstype*, const dstype*, const dstype*,
            const dstype*, const dstype*, dstype) {
        s[0]=1; s[1]=.25; s[2]=.1; s[3]=0;
    }
    KOKKOS_INLINE_FUNCTION static void sourcew_jac_w(dstype* s, const dstype*,
            const dstype*, const dstype*, const dstype*,
            const dstype*, const dstype*, dstype) { s[0]=-1; }

    KOKKOS_INLINE_FUNCTION static void source(dstype* s, const dstype*,
            const dstype* u, const dstype*, const dstype* w,
            const dstype*, const dstype*, dstype) {
        s[0]=.3*u[0]+.2*u[3]+.4*w[0];
        s[1]=-.2*u[1]+.1*u[2]-.1*w[0];
    }
    KOKKOS_INLINE_FUNCTION static void source_jac_uq(dstype* s, const dstype*,
            const dstype*, const dstype*, const dstype*,
            const dstype*, const dstype*, dstype) {
        for(int k=0;k<8;++k) s[k]=0;
        s[0]=.3; s[3]=-.2; s[5]=.1; s[6]=.2;
    }
    KOKKOS_INLINE_FUNCTION static void source_jac_w(dstype* s, const dstype*,
            const dstype*, const dstype*, const dstype*,
            const dstype*, const dstype*, dstype) { s[0]=.4; s[1]=-.1; }

    KOKKOS_INLINE_FUNCTION static void flux(dstype* f, const dstype*,
            const dstype* u, const dstype*, const dstype* w,
            const dstype*, const dstype*, dstype) {
        f[0]=.2*u[1]+.5*u[2]+.1*w[0];
        f[1]=-.1*u[0]+.4*u[3]-.2*w[0];
    }
    KOKKOS_INLINE_FUNCTION static void flux_jac_uq(dstype* f, const dstype*,
            const dstype*, const dstype*, const dstype*,
            const dstype*, const dstype*, dstype) {
        for(int k=0;k<8;++k) f[k]=0;
        f[1]=-.1; f[2]=.2; f[4]=.5; f[7]=.4;
    }
    KOKKOS_INLINE_FUNCTION static void flux_jac_w(dstype* f, const dstype*,
            const dstype*, const dstype*, const dstype*,
            const dstype*, const dstype*, dstype) { f[0]=.1; f[1]=-.2; }

    KOKKOS_INLINE_FUNCTION static void tdfunc(dstype* m, const dstype*,
            const dstype* u, const dstype*, const dstype* w,
            const dstype*, const dstype*, dstype) {
        if constexpr(Mode==0) { m[0]=2; m[1]=3; }
        else if constexpr(Mode==5) { m[0]=1; m[1]=1; }
        else {
            const dstype a=Mode==4?w[0]:u[Mode==1?0:Mode==2?1:2];
            const dstype b=Mode==4?w[0]:u[Mode==1?1:Mode==2?0:3];
            m[0]=1+a*a; m[1]=2+.5*b*b;
        }
    }
    KOKKOS_INLINE_FUNCTION static void tdfunc_jac_uq(dstype* m, const dstype*,
            const dstype* u, const dstype*, const dstype*,
            const dstype*, const dstype*, dstype) {
        for(int k=0;k<8;++k) m[k]=0;
        if constexpr(Mode>=1 && Mode<=3) {
            constexpr int a=Mode==1?0:Mode==2?1:2;
            constexpr int b=Mode==1?1:Mode==2?0:3;
            m[2*a]=2*u[a]; m[1+2*b]=u[b];
        }
    }
    KOKKOS_INLINE_FUNCTION static void tdfunc_jac_w(dstype* m, const dstype*,
            const dstype*, const dstype*, const dstype* w,
            const dstype*, const dstype*, dstype) {
        m[0]=Mode==4?2*w[0]:0; m[1]=Mode==4?w[0]:0;
    }
};

struct Fixture {
    solstruct sol{}; resstruct res{}; appstruct app{}; masterstruct master{};
    meshstruct mesh{}; tempstruct tmp{}; commonstruct common{};
    // Exact live scratch spans for this element; canaries bound each buffer.
    std::vector<dstype> nodes=std::vector<dstype>(32+2), gauss=std::vector<dstype>(66+2);
    std::vector<dstype> u{.7,.9,.4,.3,.2,.15,-.1,.05}, w{0,0}, oldW{0,0};
    std::vector<dstype> stage{.6,.55,-.2,-.15}, geom=std::vector<dstype>(3*ng);
    std::vector<dstype> shape=std::vector<dstype>(np*ng);
    std::vector<dstype> test=std::vector<dstype>(np*ng*2);
    std::vector<dstype> product=std::vector<dstype>(np*np*ng*2);
    std::vector<dstype> ru=std::vector<dstype>(np*ncu);
    std::vector<dstype> D=std::vector<dstype>(np*np*ncu*ncu);
    std::vector<dstype> B=std::vector<dstype>(np*np*ncu*(nc-ncu));
    std::vector<Int> blocks{1,1,0}, indices{0,1,2,3,4,5,6,7};
    dstype dummy=0;

    Fixture() {
        common.components.nc=nc; common.components.ncu=ncu;
        common.components.ncq=nc-ncu; common.components.ncw=1;
        common.components.ncx=1; common.components.ncs=ncu;
        common.grid.nd=1; common.grid.npe=np; common.grid.nge=ng;
        common.meshsizes.ne=1; common.meshsizes.ne2=1;
        common.eblks=blocks.data(); common.mpiProcs=1;
        common.timeparams.tdep=1; common.timeparams.tdfunc=1;
        common.timeparams.dae_alpha=0; common.timeparams.dae_beta=0;
        common.timestate.dtfactor=2; common.physicsparams.source=1;
        sol.udg=u.data(); sol.wdg=w.data(); sol.wsrc=oldW.data();
        sol.sdgg=stage.data(); sol.elemg=geom.data(); sol.odgg=&dummy;
        app.physicsparam=&dummy; app.uinf=&dummy;
        tmp.tempn=nodes.data()+1; tmp.sztempn=32;
        tmp.tempg=gauss.data()+1; tmp.sztempg=66;
        nodes.front()=gauss.front()=-7123; nodes.back()=gauss.back()=-7124;
        mesh.eindudg1=indices.data();
        master.shapegt=shape.data(); master.shapegw=test.data();
        master.shapegwdotshapeg=product.data();
        res.Ru=ru.data(); res.D=D.data(); res.B=B.data();
        const dstype z=1/std::sqrt(dstype(3));
        for(int q=0;q<ng;++q) {
            const dstype x=q?z:-z;
            geom[q]=x; geom[q+ng]=geom[q+2*ng]=1;
            for(int a=0;a<np;++a) {
                shape[q+ng*a]=.5*(1+(a?1:-1)*x);
                test[a+np*q]=shape[q+ng*a];
                test[a+np*(q+ng)]=a?.5:-.5;
            }
            for(int d=0;d<2;++d) for(int b=0;b<np;++b) for(int a=0;a<np;++a)
                product[a+np*b+np*np*(q+ng*d)]=test[a+np*(q+ng*d)]*shape[q+ng*b];
        }
    }

    template<int Mode> void assemble(bool jacobian) {
        if(jacobian) uEquationElemBlock<Model<Mode>>(sol,res,app,master,mesh,tmp,common,0,0,0);
        else RuEquationElemBlock<Model<Mode>>(sol,res,app,master,mesh,tmp,common,0,0,0);
        Kokkos::fence();
        require(nodes.front()==-7123 && nodes.back()==-7124 &&
                gauss.front()==-7123 && gauss.back()==-7124,"assembly scratch overwrite");
    }
};

template<int Mode> void checkJacobian(const char* label)
{
    Fixture f;
    f.assemble<Mode>(true);
    const auto D=f.D, B=f.B, reference=f.ru;
    f.assemble<Mode>(false);
    for(int row=0;row<np*ncu;++row)
        require(std::abs(reference[row]-f.ru[row])<2e-14,"value/Jacobian residual disagreement");
    dstype worst=0;
    for(dstype h:{dstype(1e-5),dstype(1e-6)}) for(int col=0;col<np*nc;++col) {
        const dstype saved=f.u[col];
        f.u[col]=saved+h; f.assemble<Mode>(false); const auto plus=f.ru;
        f.u[col]=saved-h; f.assemble<Mode>(false); const auto minus=f.ru;
        f.u[col]=saved;
        const int input=col/np, node=col%np;
        for(int out=0;out<ncu;++out) for(int row=0;row<np;++row) {
            const dstype fd=(plus[row+np*out]-minus[row+np*out])/(2*h);
            const dstype stored=input<ncu?D[row+np*node+np*np*(out+ncu*input)]:
                B[row+np*node+np*np*(out+ncu*(input-ncu))];
            // D and B store the negative residual derivative.
            worst=std::max(worst,std::abs(fd+stored)/(1+std::abs(fd)));
        }
    }
    std::cout<<label<<" scaled assembled Jacobian error "<<worst<<'\n';
    require(worst<1e-8,"mass assembled Jacobian finite-difference failure");
}

void checkIdentityPreservation()
{
    Fixture enabled, disabled;
    disabled.common.timeparams.tdfunc=0;
    enabled.assemble<5>(true); disabled.assemble<5>(true);
    require(enabled.ru==disabled.ru && enabled.D==disabled.D && enabled.B==disabled.B,
            "identity mass changed the no-Tdfunc assembly");
}

void checkConstantArithmetic()
{
    // Compare the new fused update bitwise with the original constant-mass
    // arithmetic, independently of the element finite-difference comparison.
    std::vector<dstype> source{.2,-.1,.4,.3}, mass{2,2,3,3};
    std::vector<dstype> stage{.6,.55,-.2,-.15}, u{.7,.9,.4,.3};
    std::vector<dstype> old=source, actual=source, rate(4);
    std::vector<dstype> oldJac(16,.125), actualJac=oldJac, partial(16,0);
    std::vector<dstype> oldW(4,.25), actualW=oldW, partialW(4,0);
    ArrayAXPBY(rate.data(),stage.data(),u.data(),one,dstype(-2),4);
    ArrayAXY(rate.data(),rate.data(),mass.data(),one,4);
    ArrayAXPBY(old.data(),old.data(),rate.data(),one,one,4);
    ApplyDtcoef(oldJac.data(),mass.data(),dstype(-2),ng,ncu);
    ApplyMassTime(actual.data(),actualJac.data(),actualW.data(),mass.data(),
        partial.data(),partialW.data(),stage.data(),u.data(),dstype(2),ng,ncu,nc,1);
    Kokkos::fence();
    require(std::memcmp(old.data(),actual.data(),4*sizeof(dstype))==0 &&
            std::memcmp(oldJac.data(),actualJac.data(),16*sizeof(dstype))==0 &&
            std::memcmp(oldW.data(),actualW.data(),4*sizeof(dstype))==0,
            "constant mass changed existing floating-point arithmetic");
}
} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc,argv);
    int result=0;
    try {
        checkJacobian<0>("constant"); checkJacobian<1>("own u");
        checkJacobian<2>("cross u"); checkJacobian<3>("q");
        checkJacobian<4>("eliminated w");
        checkIdentityPreservation(); checkConstantArithmetic();
        std::cout<<"mass Jacobian regression PASS\n";
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n'; result=1;
    }
    Kokkos::finalize();
    return result;
}

