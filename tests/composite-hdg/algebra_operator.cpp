#include <exasim/operators.hpp>
#include <backend/Discretization/CompositeHDG/operator_check.hpp>
#include <cassert>
#ifndef COMPOSITE_TEST_DIM
#define COMPOSITE_TEST_DIM 3
#endif
using Shape=exasim::composite::Shape<COMPOSITE_TEST_DIM>;
constexpr int ND=COMPOSITE_TEST_DIM,P=Shape::P,PF=Shape::PF,Q=Shape::Q,N=Shape::N,M=Shape::M,NC=Shape::NC,FLOW=Shape::flow,AF=Shape::AF,BF=Shape::BF,HS=Shape::HS,FS=Shape::FS,NF=4*ND-1;

const ExasimDriverABI& GetHelmholtzModelABI(int){throw runtime_error("unused");}
const ExasimDriverABI& GetLinearElasticityModelABI(int){throw runtime_error("unused");}

static void testSourcew(double*,double*,double*,const double*,const double*,const double*,const double*,const double*,const double*,double,int,int,int,int,int,int,int,int){}
// Algebra-only mock: no physical model is evaluated by this fixture.
static ExasimDriverABI testAbi=[] {ExasimDriverABI a{};a.abi_version=kExasimDriverABIVersion;a.struct_size=sizeof(ExasimDriverABI);a.ncu=NC;a.ncw=1;a.hdgjac.HdgFlux=testSourcew;a.hdgjac.HdgSource=testSourcew;a.hdgjac.HdgSourcew=testSourcew;return a;}();
extern "C" const exasim::composite::ModelV1* GetExasimCompositeHdgModelV1(const ExasimDriverABI* a,int modelnumber){static exasim::composite::ModelV1 m{1,sizeof(exasim::composite::ModelV1),ND==3?1u:2u,100,testSourcew};return modelnumber==100 && a->hdgjac.HdgSourcew==testSourcew?&m:nullptr;}

int main(int argc,char** argv){
 setenv("EXASIM_COMPOSITE_VALIDATE","1",1);

 Kokkos::initialize(argc,argv);int code=0;
 try{
  InputParams parameters;parameters.stringParams["exasimpath"]=".";
  parameters.intParams["compositehdg"]=1;parameters.doubleParams["compositebatchsize"]=2048.;
  auto pde=initializePDE(parameters,1);assert(pde.compositehdg==1&&pde.compositebatchsize==2048);
  auto wire=exasim::buildAppStruct(pde);assert(exasim::composite::decode(wire).batch==2048);wire.freememory(0);
  parameters.doubleParams["compositebatchsize"]=1024.5;bool rejected=false;try{initializePDE(parameters,1);}catch(const std::exception&){rejected=true;}assert(rejected);
  printf("CPP_TEXT_AND_INMEMORY_CONFIG_PASS\n");
  commonstruct c{};c.grid.nd=ND;c.grid.npe=P;c.grid.npf=PF;c.meshsizes.nfe=2*ND;c.meshsizes.ne=2;c.meshsizes.ne1=2;c.meshsizes.neb=1;c.meshsizes.nf=NF;
  c.components.ncu=NC;c.components.nc=Shape::state;c.components.nco=Shape::other;c.components.ncw=1;c.solverparams.preconditioner=1;c.backend=0;c.compositeConfig.enabled=true;c.compositeConfig.batch=1;c.driver_abi=&testAbi;c.modelnumber=100;c.mpiRank=0;
  fieldsplit::Context ctx;ctx.init(c);assert(ctx.bytes()==fieldsplit::Context::plannedBytes(c));c.composite=&ctx;
  auto c2=c;c2.meshsizes.neb=2;fieldsplit::Context other;other.init(c2);assert(ctx.hf.data()!=other.hf.data());assert(ctx.batch==1&&other.batch==2);other.clear();assert(ctx.hf.extent(0)==M*M);
  Kokkos::View<int*> ec("unit elemcon",2*Q),f2e("unit f2e",4*NF);auto eh=Kokkos::create_mirror_view(ec),fh=Kokkos::create_mirror_view(f2e);
  for(int e=0;e<2;e++)for(int lf=0;lf<2*ND;lf++)for(int node=0;node<PF;node++)eh[e*Q+lf*PF+node]=lf==0?(e?PF-1-node:node):PF*(e?lf+2*ND-1:lf)+node;
  for(int f=0;f<NF;f++){fh[4*f]=f<2*ND?0:1;fh[4*f+1]=f<2*ND?f:f-(2*ND-1);fh[4*f+2]=-1;fh[4*f+3]=-1;}
  fh[2]=1;fh[3]=0;Kokkos::deep_copy(ec,eh);Kokkos::deep_copy(f2e,fh);
  meshstruct mesh{};mesh.elemcon=ec.data();mesh.f2e=f2e.data();
  auto v=ctx.vy,t=ctx.ty;
  for(int trace=0;trace<2;trace++){auto basis=trace?t:v;int points=basis.extent(0)/9;
   Kokkos::parallel_for("unit nonuniform basis",points,KOKKOS_LAMBDA(int p){double sum=0.;for(int s=0;s<5;s++){basis[9*p+s]=.01+(s+1)*(.1+.001*(p%7));sum+=basis[9*p+s];}for(int s=0;s<5;s++)basis[9*p+s]/=sum;for(int s=0;s<4;s++)basis[9*p+5+s]=.13*(s+1)+.002*(p%3);});}
  solstruct sol{};appstruct app{};fieldsplit::selfCheck(ctx,sol,app,mesh,c,0,false);
  using V=Kokkos::View<double*>;V D("unit D",N*N),B("unit B",N*M),C("unit C",M*N),H("unit H",M*M),Ru("unit Ru",N),Rh("unit Rh",M),temp("unit temp",M*M),ch("unit compact H",2*HS),cf("unit compact F",2*FS),be("unit ASM",2*HS);
  std::vector<int> pivots(M);resstruct res{};res.ipiv=pivots.data();res.H=ch.data();res.F=cf.data();tempstruct tmp{};tmp.tempn=temp.data();
  for(int e=0;e<2;e++){
   Kokkos::parallel_for("unit D fill",N*N,KOKKOS_LAMBDA(int k){D[k]=double(k%N==k/N)*10.+.0001*sin(k*.02+e);});
   Kokkos::parallel_for("unit BC fill",N*M,KOKKOS_LAMBDA(int k){B[k]=.001*sin(.013*k+e);C[k]=.002*cos(.017*k+e);});
   Kokkos::parallel_for("unit H fill",M*M,KOKKOS_LAMBDA(int k){H[k]=double(k%M==k/M)*5.+.0003*sin(.019*k+e);});
   Kokkos::parallel_for("unit RHS fill",M,KOKKOS_LAMBDA(int k){Rh[k]=cos(.1*k);if(k<N)Ru[k]=sin(.1*k);});
   fieldsplit::condense(ctx,D.data(),B.data(),C.data(),H.data(),Ru.data(),Rh.data(),res,mesh,tmp,c,e,1,0);
  }
  for(int fp=0;fp<2;fp++){ctx.asm_fp32=fp;fieldsplit::schwarz(ctx,be.data(),ch.data(),res,mesh,tmp,c,0);
   // Independently assemble host face blocks; both elements share reversed nodes.
   auto hc=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),ch);std::vector<double> inv(2*HS);
   if(fp){std::vector<float> packed(2*HS);TemplateCopytoHost(packed.data(),reinterpret_cast<float*>(be.data()),packed.size(),0);for(size_t k=0;k<inv.size();k++)inv[k]=packed[k];}
   else TemplateCopytoHost(inv.data(),be.data(),inv.size(),0);
   double worst=0.;for(int e=0;e<2;e++)for(int field=0;field<5;field++){
    int m=field?Q:BF,comps=field?1:FLOW,off=field?BF*BF+(field-1)*Q*Q:0;std::vector<double> a(m*m),expected(m*m,0.);std::vector<int> piv(m);
    for(int col=0;col<m;col++)for(int row=0;row<m;row++){double value=hc[e*HS+off+row+m*col];if(row<PF*comps&&col<PF*comps){int rr=(PF-1-row/comps)*comps+row%comps,cc=(PF-1-col/comps)*comps+col%comps;value+=hc[(1-e)*HS+off+rr+m*cc];}a[row+m*col]=value;expected[row+m*col]=double(row==col);}
    int status=0;dgetrf_(&m,&m,a.data(),&m,piv.data(),&status);if(status)throw runtime_error("unit host ASM factor failed");char op='N';fieldsplit::dgetrs_(&op,&m,&m,a.data(),&m,piv.data(),expected.data(),&m,&status);if(status)throw runtime_error("unit host ASM solve failed");
    double num=0.,den=0.;for(int k=0;k<m*m;k++){double ref=fp?double(float(expected[k])):expected[k],err=inv[e*HS+off+k]-ref;num+=err*err;den+=ref*ref;}worst=std::max(worst,sqrt(num/den));
   }
   printf("COMPOSITE_ASM_REFERENCE fp32=%d relative=%.17g\n",fp,worst);if(!std::isfinite(worst)||worst>(fp?1.e-6:1.e-11))throw runtime_error("Composite ASM reference mismatch");
  }
printf("COMPOSITE_OPERATOR_UNIT_PASS\n");
 }catch(const std::exception& e){fprintf(stderr,"COMPOSITE_OPERATOR_UNIT_FAIL %s\n",e.what());code=1;}
 Kokkos::finalize();return code;
}
