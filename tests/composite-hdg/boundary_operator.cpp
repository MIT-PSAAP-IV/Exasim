#include <exasim/operators.hpp>
#include <backend/Discretization/CompositeHDG/operator_check.hpp>
#include <cassert>
#ifndef COMPOSITE_TEST_DIM
#define COMPOSITE_TEST_DIM 3
#endif
using Shape=exasim::composite::Shape<COMPOSITE_TEST_DIM>;
constexpr int ND=COMPOSITE_TEST_DIM,P=Shape::P,PF=Shape::PF,Q=Shape::Q,N=Shape::N,M=Shape::M,NC=Shape::NC,FLOW=Shape::flow,AF=Shape::AF,BF=Shape::BF,HS=Shape::HS,FS=Shape::FS;

const ExasimDriverABI& GetHelmholtzModelABI(int){throw runtime_error("unused");}
const ExasimDriverABI& GetLinearElasticityModelABI(int){throw runtime_error("unused");}

static void testSourcew(double*,double*,double*,const double*,const double*,const double*,const double*,const double*,const double*,double,int,int,int,int,int,int,int,int){}
// Algebra-only mock: no physical model is evaluated by this fixture.
static ExasimDriverABI testAbi=[] {ExasimDriverABI a{};a.abi_version=kExasimDriverABIVersion;a.struct_size=sizeof(ExasimDriverABI);a.ncu=NC;a.ncw=1;a.hdgjac.HdgFlux=testSourcew;a.hdgjac.HdgSource=testSourcew;a.hdgjac.HdgSourcew=testSourcew;return a;}();
extern "C" const exasim::composite::ModelV1* GetExasimCompositeHdgModelV1(const ExasimDriverABI* a,int modelnumber){static exasim::composite::ModelV1 m{1,sizeof(exasim::composite::ModelV1),ND==3?1u:2u,100,testSourcew};return modelnumber==100 && a->hdgjac.HdgSourcew==testSourcew?&m:nullptr;}


using V=Kokkos::View<double*>;
using IV=Kokkos::View<int*>;
static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static double worst=0.;
static void close(double actual,double expected){
 double err=std::abs(actual-expected)/(1.+std::abs(expected));worst=std::max(worst,err);
 if(!std::isfinite(actual)||err>2.e-11)throw std::runtime_error("boundary dense reference mismatch");
}
static std::vector<double> host(V v){auto h=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),v);return {h.data(),h.data()+h.extent(0)};}
static void put(V v,const std::vector<double>& values){check(v.extent(0)==values.size(),"fixture size");auto h=Kokkos::create_mirror_view(v);std::copy(values.begin(),values.end(),h.data());Kokkos::deep_copy(v,h);}
// Build the physical-variable basis independently. Invert this small dense T
// with LAPACK instead of using the backend's analytic L or packing helpers.
static std::vector<double> basis(const double* y,bool inverse=false){
 std::vector<double> t(NC*NC,0.);for(int i=0;i<5;++i)t[i]=y[i];
 for(int c=1;c<FLOW;++c)t[c+4+NC*c]=1.;
 int species[4]={0,1,2,4};for(int s=0;s<4;++s){t[species[s]+NC*(FLOW+s)]=1.;t[3+NC*(FLOW+s)]=-1.;t[NC-1+NC*(FLOW+s)]=y[5+s];}
 if(inverse){int n=NC,info=0;std::vector<int> piv(n);std::vector<double> id(n*n,0.);for(int i=0;i<n;++i)id[i+n*i]=1.;dgetrf_(&n,&n,t.data(),&n,piv.data(),&info);check(info==0,"basis factor");char op='N';fieldsplit::dgetrs_(&op,&n,&n,t.data(),&n,piv.data(),id.data(),&n,&info);check(info==0,"basis inverse");return id;}
 return t;
}
static std::vector<double> pack(const std::vector<double>& x,const std::vector<double>& y,const std::vector<int>& ec,int e){
 std::vector<double> z(M);for(int node=0;node<Q;++node){auto l=basis(y.data()+9*ec[e*Q+node],true);for(int c=0;c<NC;++c){double value=0.;for(int j=0;j<NC;++j)value+=l[c+NC*j]*x[NC*node+j];int i=c<FLOW?FLOW*node+c:BF+4*node+c-FLOW;z[i]=value;}}return z;
}
static std::vector<double> volumeProduct(const std::vector<double>& f,const std::vector<double>& x,const std::vector<double>& y,int e){
 // Expand the persisted block layout, multiply on host, then return physical u.
 std::vector<double> z(N,0.),out(N,0.);
 for(int row=0;row<N;++row)for(int col=0;col<M;++col){double a=0.;if(row<AF&&col<BF)a=f[row+AF*col];
 else if(row>=AF&&col>=BF&&(row-AF)/P==(col-BF)%4)a=f[AF*BF+((row-AF)/P)*P*Q+(row-AF)%P+P*((col-BF)/4)];z[row]+=a*x[col];}
 for(int p=0;p<P;++p){auto t=basis(y.data()+9*(e*P+p));for(int i=0;i<NC;++i)for(int j=0;j<NC;++j){int k=j<FLOW?p+P*j:AF+p+P*(j-FLOW);out[p+P*i]+=t[i+NC*j]*z[k];}}
 return out;
}
#ifdef HAVE_HIP
static constexpr int backend=3;
struct TestBlas {
 hipblasHandle_t handle{};
 TestBlas(){CHECK_HIPBLAS(hipblasCreate(&handle));CHECK_HIPBLAS(hipblasSetStream(handle,exasim_stream::compute()));CHECK_HIPBLAS(hipblasSetPointerMode(handle,HIPBLAS_POINTER_MODE_HOST));}
 ~TestBlas(){hipblasDestroy(handle);}
};
#else
static constexpr int backend=0;
struct TestBlas {int handle=0;};
#endif
static void run(int cap,bool reverse,const std::vector<int>& codes){
 constexpr int ne=3,nfe=2*ND,nf=ne*nfe;
 commonstruct c{};c.grid.nd=ND;c.grid.npe=P;c.grid.npf=PF;c.meshsizes.nfe=nfe;c.meshsizes.ne=ne;c.meshsizes.ne1=ne;c.meshsizes.neb=cap;c.meshsizes.nf=nf;
 c.components.ncu=NC;c.components.nc=Shape::state;c.components.nco=Shape::other;c.components.ncw=1;c.solverparams.preconditioner=1;c.backend=backend;c.compositeConfig.enabled=true;c.compositeConfig.batch=cap;c.driver_abi=&testAbi;c.modelnumber=100;
 TestBlas blas;c.cublasHandle=blas.handle;
 std::vector<int> blocks;for(int first=0;first<ne;first+=cap){blocks.insert(blocks.end(),{first+1,std::min(ne,first+cap),0});}
 c.meshsizes.nbe1=c.meshsizes.nbe=blocks.size()/3;c.eblks=blocks.data();c.meshsizes.maxnbc=1001;
 std::vector<int> prefix(1+1001*c.meshsizes.nbe),bf;
 // BC2 and5 share element0; code1001 is deliberately far from its segment index.
 const int codeByFace[3][6]={{0,5,2,3,0,0},{0,1001,5,3,0,0},{0,2,3,1001,0,0}};
 for(int b=0;b<c.meshsizes.nbe;++b)for(int code=1;code<=1001;++code){for(int e=blocks[3*b]-1;e<blocks[3*b+1];++e)for(int face=0;face<nfe;++face)if(codeByFace[e][face]==code)bf.push_back((e-(blocks[3*b]-1))*nfe+face);prefix[1001*b+code]=bf.size();}
 c.nboufaces=prefix.data();
 IV ecv("boundary test ec",ne*Q),bfv("boundary test face map",bf.size());auto ecHost=Kokkos::create_mirror_view(ecv),bfHost=Kokkos::create_mirror_view(bfv);std::vector<int> ec(ne*Q);
 for(int e=0;e<ne;++e)for(int p=0;p<Q;++p)ec[p+e*Q]=e*Q+(p/PF)*PF+(e%2?PF-1-p%PF:p%PF);
 std::copy(ec.begin(),ec.end(),ecHost.data());std::copy(bf.begin(),bf.end(),bfHost.data());Kokkos::deep_copy(ecv,ecHost);Kokkos::deep_copy(bfv,bfHost);
 meshstruct mesh{};mesh.elemcon=ecv.data();mesh.boufaces=bfv.data();mesh.szboufaces=bf.size();
 fieldsplit::Context base; base.init(c);
 auto cc=c;cc.compositeConfig.boundaryCodes=codes;
 fieldsplit::Context ctx;ctx.init(cc);ctx.boundary.init(cc,mesh);
 check(ctx.bytes()==fieldsplit::Context::plannedBytes(cc),"allocation ledger");
 for(int trace=0;trace<2;++trace){auto dest=trace?ctx.ty:ctx.vy;std::vector<double> values(dest.extent(0));for(size_t p=0;p<values.size()/9;++p){double sum=0.;for(int s=0;s<5;++s){values[9*p+s]=s==0?1.e-15:.1*(s+1)+.001*(p%7);sum+=values[9*p+s];}for(int s=0;s<5;++s)values[9*p+s]/=sum;for(int s=0;s<4;++s)values[9*p+5+s]=.13*(s+1)+.002*(p%3);}put(dest,values);put(trace?base.ty:base.vy,values);}
 auto vy=host(ctx.vy),ty=host(ctx.ty);
 V ch("selected H",ne*HS),cf("selected F",ne*FS),bh("base H",ne*HS),bff("base F",ne*FS),temp("temp",cap*M*M);
 IV piv("boundary test pivots",cap*M);resstruct res{},br{};res.H=ch.data();res.F=cf.data();res.ipiv=piv.data();br=res;br.H=bh.data();br.F=bff.data();tempstruct tmp{};tmp.tempn=temp.data();
 V xin("trace x",ne*M),out("trace out",ne*M),bout("base out",ne*M);std::vector<double> xv(ne*M);for(size_t k=0;k<xv.size();++k)xv[k]=sin(.031*k);put(xin,xv);
 std::vector<double> expected(ne*M),savedBaseR(ne*M),savedSelectedR(ne*M),originalR(ne*M),savedC(ne*M*N),savedH(ne*M*M),savedZ(ne*N);
 for(int bi=0;bi<c.meshsizes.nbe;++bi){int b=reverse?c.meshsizes.nbe-1-bi:bi,e1=blocks[3*b]-1,count=blocks[3*b+1]-e1;
  V D("D",count*N*N),B("B",count*N*M),C("C",count*M*N),H("H",count*M*M),ru("ru",count*N),rh("rh",count*M),bru("bru",count*N),brh("brh",count*M);
  std::vector<double> ds(D.extent(0)),bs(B.extent(0)),cs(C.extent(0)),hs(H.extent(0)),rus(ru.extent(0)),rhs(rh.extent(0));
  for(int e=0;e<count;++e){int ge=e1+e;for(int k=0;k<N*N;++k)ds[e*N*N+k]=10.*(k%N==k/N)+.0001*sin(.02*k+ge);
   for(int k=0;k<N*M;++k){bs[e*N*M+k]=.001*sin(.013*k+ge);cs[e*M*N+k]=.002*cos(.017*k+ge);savedC[ge*M*N+k]=cs[e*M*N+k];}
   for(int k=0;k<M*M;++k){hs[e*M*M+k]=5.*(k%M==k/M)+.0003*sin(.019*k+ge);savedH[ge*M*M+k]=hs[e*M*M+k];}
   for(int j=0;j<NC;++j){for(int p=0;p<P;++p)rus[P*count*j+P*e+p]=sin(.1*(p+P*j)+ge);for(int p=0;p<Q;++p){double v=cos(.1*(NC*p+j)+ge);rhs[Q*count*j+Q*e+p]=v;originalR[ge*M+NC*p+j]=v;}}
  }
  put(D,ds);put(B,bs);put(C,cs);put(H,hs);put(ru,rus);put(rh,rhs);put(bru,rus);put(brh,rhs);
  fieldsplit::condense(base,D.data(),B.data(),C.data(),H.data(),bru.data(),brh.data(),br,mesh,tmp,c,e1,count,backend);
  fieldsplit::condense(ctx,D.data(),B.data(),C.data(),H.data(),ru.data(),rh.data(),res,mesh,tmp,cc,e1,count,backend);
  check(host(D)==ds&&host(B)==bs&&host(C)==cs&&host(H)==hs,"source matrix overwritten");
  auto z=host(bru),r0=host(brh),r1=host(rh);check(host(ru)==z,"volume recovery RHS changed");
  std::copy(z.begin(),z.end(),savedZ.begin()+e1*N);std::copy(r0.begin(),r0.end(),savedBaseR.begin()+e1*M);std::copy(r1.begin(),r1.end(),savedSelectedR.begin()+e1*M);
 }
 check(host(ch)==host(bh)&&host(cf)==host(bff),"compact matrices changed");check(vy==host(ctx.vy)&&ty==host(ctx.ty),"bases changed");
 fieldsplit::multiply(base,bout.data(),bh.data(),xin.data(),0,ne,mesh,c,backend);expected=host(bout);
 const auto fullF=host(cf);
 for(int e=0;e<ne;++e){std::vector<double> x(xv.begin()+e*M,xv.begin()+(e+1)*M),f(fullF.begin()+e*FS,fullF.begin()+(e+1)*FS);auto px=pack(x,ty,ec,e);auto u=volumeProduct(f,px,vy,e);
  for(int face=0;face<nfe;++face)for(int p=0;p<PF;++p)for(int comp=0;comp<NC;++comp){int row=NC*(face*PF+p)+comp;bool selected=std::find(codes.begin(),codes.end(),codeByFace[e][face])!=codes.end();
   double rhs=savedBaseR[e*M+row];if(selected){rhs=originalR[e*M+row];double value=0.;for(int j=0;j<M;++j)value+=savedH[e*M*M+row+M*j]*x[j];for(int j=0;j<N;++j){value+=savedC[e*M*N+row+M*j]*u[j];rhs-=savedC[e*M*N+row+M*j]*savedZ[e*N+j];}expected[e*M+row]=value;}
   close(savedSelectedR[e*M+row],rhs);
  }
 }
 fieldsplit::multiply(ctx,out.data(),ch.data(),xin.data(),0,ne,mesh,cc,backend);auto got=host(out);for(size_t i=0;i<got.size();++i)close(got[i],expected[i]);
 // MPI's split matvec uses partial owned ranges, not necessarily whole batches.
 Kokkos::deep_copy(out,999.);for(int e=ne-1;e>=0;--e)fieldsplit::multiply(ctx,out.data()+e*M,ch.data()+e*HS,xin.data()+e*M,e,1,mesh,cc,backend);
 auto partial=host(out);for(size_t i=0;i<partial.size();++i)close(partial[i],expected[i]);
 ctx.asm_fp32=base.asm_fp32=false;
 fieldsplit::multiply(ctx,out.data(),ch.data(),xin.data(),0,ne,mesh,cc,backend,true);
 fieldsplit::multiply(base,bout.data(),bh.data(),xin.data(),0,ne,mesh,c,backend,true);
 check(host(out)==host(bout),"ASM path changed");check(host(xin)==xv,"input trace overwritten");
 if(!codes.empty()){
  auto bad=cc;std::vector<int> bp=prefix;bad.nboufaces=bp.data();bp[5]=-1;bool rejected=false;try{fieldsplit::BoundaryRows::plannedBytes(bad);}catch(const std::exception&){rejected=true;}if(std::find(codes.begin(),codes.end(),5)!=codes.end())check(rejected,"invalid prefix accepted");
 }
 if(codes==std::vector<int>({5})) {
  auto badMesh=mesh;badMesh.szboufaces=0;fieldsplit::BoundaryRows invalid;bool rejected=false;
  try{invalid.init(cc,badMesh);}catch(const std::exception&){rejected=true;}check(rejected,"short map accepted");
  auto altered=bf;
  // Make both BC5 entries refer to the same face within a multi-element batch.
  if(cap==2) {int start=prefix[4];check(prefix[5]-start==2,"duplicate fixture");altered[start+1]=altered[start];
   std::copy(altered.begin(),altered.end(),bfHost.data());Kokkos::deep_copy(bfv,bfHost);
   rejected=false;try{invalid.init(cc,mesh);}catch(const std::exception&){rejected=true;}check(rejected,"duplicate map accepted");
   std::copy(bf.begin(),bf.end(),bfHost.data());Kokkos::deep_copy(bfv,bfHost);
  }
  rejected=false;try{ctx.boundary.batch(0,ne+1);}catch(const std::exception&){rejected=true;}check(rejected,"bad batch accepted");
 }
 printf("COMPOSITE_BOUNDARY_CASE nd=%d cap=%d reverse=%d codes=%zu faces=%zu bytes=%zu relative=%.17g\n",ND,cap,reverse,codes.size(),ctx.boundary.faces.extent(0),ctx.boundary.bytes(),worst);
}
int main(int argc,char** argv){Kokkos::initialize(argc,argv);int code=0;try{
#ifdef HAVE_HIP
 int runtime=0,driver=0,device=0;hipDeviceProp_t props{};
 CHECK(hipRuntimeGetVersion(&runtime));CHECK(hipDriverGetVersion(&driver));CHECK(hipGetDevice(&device));CHECK(hipGetDeviceProperties(&props,device));
 printf("COMPOSITE_BOUNDARY_GPU runtime=%d driver=%d compile=%d arch=%s device=%s\n",runtime,driver,HIP_VERSION,props.gcnArchName,props.name);
#endif

 for(int cap:{1,2,3})for(bool reverse:{false,true})run(cap,reverse,{5,1001,2,5});
 run(2,false,{5});run(2,false,{});run(2,false,{999});
 printf("COMPOSITE_BOUNDARY_UNIT_PASS\n");
 }catch(const std::exception& e){fprintf(stderr,"COMPOSITE_BOUNDARY_UNIT_FAIL %s\n",e.what());code=1;}Kokkos::finalize();return code;}
