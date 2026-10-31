#include <exasim/operators.hpp>
#include <backend/Discretization/CompositeHDG/operator_check.hpp>
#include <cassert>

const ExasimDriverABI& GetHelmholtzModelABI(int){throw runtime_error("unused");}
const ExasimDriverABI& GetLinearElasticityModelABI(int){throw runtime_error("unused");}

static void testSourcew(double*,double*,double*,const double*,const double*,const double*,const double*,const double*,const double*,double,int,int,int,int,int,int,int,int){}
// Algebra-only mock: no physical model is evaluated by this fixture.
static ExasimDriverABI testAbi=[] {ExasimDriverABI a{};a.abi_version=kExasimDriverABIVersion;a.struct_size=sizeof(ExasimDriverABI);a.ncu=9;a.ncw=1;a.hdgjac.HdgFlux=testSourcew;a.hdgjac.HdgSource=testSourcew;a.hdgjac.HdgSourcew=testSourcew;return a;}();
extern "C" const exasim::composite::ModelV1* GetExasimCompositeHdgModelV1(const ExasimDriverABI* a,int modelnumber){static exasim::composite::ModelV1 m{1,sizeof(exasim::composite::ModelV1),1,100,testSourcew};return modelnumber==100 && a->hdgjac.HdgSourcew==testSourcew?&m:nullptr;}

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
  commonstruct c{};c.grid.nd=3;c.grid.npe=27;c.grid.npf=9;c.meshsizes.nfe=6;c.meshsizes.ne=2;c.meshsizes.ne1=2;c.meshsizes.neb=1;c.meshsizes.nf=11;
  c.components.ncu=9;c.components.nc=36;c.components.nco=14;c.components.ncw=1;c.solverparams.preconditioner=1;c.backend=0;c.compositeConfig.enabled=true;c.compositeConfig.batch=1;c.driver_abi=&testAbi;c.modelnumber=100;c.mpiRank=0;
  fieldsplit::Context ctx;ctx.init(c);assert(ctx.bytes()==fieldsplit::Context::plannedBytes(c));c.composite=&ctx;
  auto c2=c;c2.meshsizes.neb=2;fieldsplit::Context other;other.init(c2);assert(ctx.hf.data()!=other.hf.data());assert(ctx.batch==1&&other.batch==2);other.clear();assert(ctx.hf.extent(0)==486*486);
  Kokkos::View<int*> ec("unit elemcon",108),f2e("unit f2e",44);auto eh=Kokkos::create_mirror_view(ec),fh=Kokkos::create_mirror_view(f2e);
  for(int e=0;e<2;e++)for(int lf=0;lf<6;lf++)for(int node=0;node<9;node++)eh[e*54+lf*9+node]=lf==0?(e?8-node:node):9*(e?lf+5:lf)+node;
  for(int f=0;f<11;f++){fh[4*f]=f<6?0:1;fh[4*f+1]=f<6?f:f-5;fh[4*f+2]=-1;fh[4*f+3]=-1;}
  fh[2]=1;fh[3]=0;Kokkos::deep_copy(ec,eh);Kokkos::deep_copy(f2e,fh);
  meshstruct mesh{};mesh.elemcon=ec.data();mesh.f2e=f2e.data();
  auto v=ctx.vy,t=ctx.ty;
  for(int trace=0;trace<2;trace++){auto basis=trace?t:v;int points=basis.extent(0)/9;
   Kokkos::parallel_for("unit nonuniform basis",points,KOKKOS_LAMBDA(int p){double sum=0.;for(int s=0;s<5;s++){basis[9*p+s]=.01+(s+1)*(.1+.001*(p%7));sum+=basis[9*p+s];}for(int s=0;s<5;s++)basis[9*p+s]/=sum;for(int s=0;s<4;s++)basis[9*p+5+s]=.13*(s+1)+.002*(p%3);});}
  solstruct sol{};appstruct app{};fieldsplit::selfCheck(ctx,sol,app,mesh,c,0,false);
  using V=Kokkos::View<double*>;V D("unit D",243*243),B("unit B",243*486),C("unit C",486*243),H("unit H",486*486),Ru("unit Ru",243),Rh("unit Rh",486),temp("unit temp",486*486),ch("unit compact H",2*fieldsplit::HS),cf("unit compact F",2*fieldsplit::FS),be("unit ASM",2*fieldsplit::HS);
  std::vector<int> pivots(486);resstruct res{};res.ipiv=pivots.data();res.H=ch.data();res.F=cf.data();tempstruct tmp{};tmp.tempn=temp.data();
  for(int e=0;e<2;e++){
   Kokkos::parallel_for("unit D fill",243*243,KOKKOS_LAMBDA(int k){D[k]=double(k%243==k/243)*10.+.0001*sin(k*.02+e);});
   Kokkos::parallel_for("unit BC fill",243*486,KOKKOS_LAMBDA(int k){B[k]=.001*sin(.013*k+e);C[k]=.002*cos(.017*k+e);});
   Kokkos::parallel_for("unit H fill",486*486,KOKKOS_LAMBDA(int k){H[k]=double(k%486==k/486)*5.+.0003*sin(.019*k+e);});
   Kokkos::parallel_for("unit RHS fill",486,KOKKOS_LAMBDA(int k){Rh[k]=cos(.1*k);if(k<243)Ru[k]=sin(.1*k);});
   fieldsplit::condense(ctx,D.data(),B.data(),C.data(),H.data(),Ru.data(),Rh.data(),res,mesh,tmp,c,e,1,0);
  }
  for(int fp=0;fp<2;fp++){ctx.asm_fp32=fp;fieldsplit::schwarz(ctx,be.data(),ch.data(),res,mesh,tmp,c,0);
   // Independently assemble host face blocks; both elements share reversed nodes.
   auto hc=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),ch);std::vector<double> inv(2*fieldsplit::HS);
   if(fp){std::vector<float> packed(2*fieldsplit::HS);TemplateCopytoHost(packed.data(),reinterpret_cast<float*>(be.data()),packed.size(),0);for(size_t k=0;k<inv.size();k++)inv[k]=packed[k];}
   else TemplateCopytoHost(inv.data(),be.data(),inv.size(),0);
   double worst=0.;for(int e=0;e<2;e++)for(int field=0;field<5;field++){
    int m=field?54:270,comps=field?1:5,off=field?72900+(field-1)*2916:0;std::vector<double> a(m*m),expected(m*m,0.);std::vector<int> piv(m);
    for(int col=0;col<m;col++)for(int row=0;row<m;row++){double value=hc[e*fieldsplit::HS+off+row+m*col];if(row<9*comps&&col<9*comps){int rr=(8-row/comps)*comps+row%comps,cc=(8-col/comps)*comps+col%comps;value+=hc[(1-e)*fieldsplit::HS+off+rr+m*cc];}a[row+m*col]=value;expected[row+m*col]=double(row==col);}
    int status=0;dgetrf_(&m,&m,a.data(),&m,piv.data(),&status);if(status)throw runtime_error("unit host ASM factor failed");char op='N';fieldsplit::dgetrs_(&op,&m,&m,a.data(),&m,piv.data(),expected.data(),&m,&status);if(status)throw runtime_error("unit host ASM solve failed");
    double num=0.,den=0.;for(int k=0;k<m*m;k++){double ref=fp?double(float(expected[k])):expected[k],err=inv[e*fieldsplit::HS+off+k]-ref;num+=err*err;den+=ref*ref;}worst=std::max(worst,sqrt(num/den));
   }
   printf("FS3_ASM_REFERENCE fp32=%d relative=%.17g\n",fp,worst);if(!std::isfinite(worst)||worst>(fp?1.e-6:1.e-11))throw runtime_error("3D ASM reference mismatch");
  }
printf("FS3_OPERATOR_UNIT_PASS\n");
 }catch(const std::exception& e){fprintf(stderr,"FS3_OPERATOR_UNIT_FAIL %s\n",e.what());code=1;}
 Kokkos::finalize();return code;
}
