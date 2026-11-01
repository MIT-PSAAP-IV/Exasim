// Actual-state basis checks and deterministic dense operator checks on GPU.
namespace fieldsplit {
template<int ND> inline void selfCheckDimension(Context& ctx,const solstruct& sol,const appstruct& app,const meshstruct& mesh,const commonstruct& c,int backend,bool freezeState=true){

 using K=Kernels<ND>;
 constexpr int N=K::N,M=K::M,NC=K::NC,FLOW=K::FLOW,P=K::P,Q=K::Q,AF=K::AF,BF=K::BF,HS=K::HS,FS=K::FS;
 if(!ctx.enabled||!ctx.validate)return;
 if(freezeState)K::freeze(ctx,sol,app,c);
 for(int trace=0;trace<2;trace++){
  auto basis=trace?ctx.ty:ctx.vy;int points=basis.extent(0)/9;double err=0.;
  Kokkos::parallel_reduce("COMPOSITE inverse basis check",points*NC*NC,KOKKOS_LAMBDA(int k,double& maximum){int p=k/(NC*NC),i=k%NC,j=(k/NC)%NC;const double* y=basis.data()+9*p;double value=0.,scale=1.;for(int l=0;l<NC;l++){double term=K::L(i,l,y)*K::T(l,j,y);value+=term;scale+=fabs(term);}double diff=fabs(value-double(i==j))/scale;if(!exasim::composite::finiteValue(diff))diff=1.e100;if(diff>maximum)maximum=diff;},Kokkos::Max<double>(err));
  printf("COMPOSITE_BASIS rank=%d trace=%d points=%d scaled_inverse_error=%.17g\n",c.mpiRank,trace,points,err);if(!exasim::composite::finiteValue(err)||err>2.e-12)throw std::runtime_error("COMPOSITE basis inverse check failed");
  int n=trace?M:N,count=std::min(3,ctx.owned);View x("check physical",n*count),y("check packed",n*count),z("check roundtrip",n*count);
  Kokkos::parallel_for("check vector",n*count,KOKKOS_LAMBDA(int k){x[k]=sin(.17*k)+.01*cos(.3*k);});K::packResidual(ctx,y.data(),x.data(),trace,0,count,mesh);K::unpack(ctx,z.data(),y.data(),trace,0,count,mesh);
  double numerator=0.,denominator=0.;Kokkos::parallel_reduce("check roundtrip numerator",n*count,KOKKOS_LAMBDA(int k,double& sum){sum+=(x[k]-z[k])*(x[k]-z[k]);},numerator);Kokkos::parallel_reduce("check roundtrip norm",n*count,KOKKOS_LAMBDA(int k,double& sum){sum+=x[k]*x[k];},denominator);
  double relative=sqrt(numerator/denominator);printf("COMPOSITE_ROUNDTRIP rank=%d trace=%d relative=%.17g\n",c.mpiRank,trace,relative);if(!exasim::composite::finiteValue(relative)||relative>1.e-10)throw std::runtime_error("COMPOSITE pack/unpack check failed");
 }
 if(c.mpiRank!=0)return;
 for(int rt=0;rt<2;rt++)for(int ct=0;ct<2;ct++)for(int field=0;field<2;field++){
  int nr=rt?M:N,nc=ct?M:N,comps=field?4:FLOW,a=(rt?Q:P)*comps,b=(ct?Q:P)*comps;
  View full("check full J",nr*nc),sparse("check sparse LJT",a*b),dense("check generic LJT",a*b);
  Kokkos::parallel_for("check full matrix",nr*nc,KOKKOS_LAMBDA(int k){full[k]=sin(.123*k)+.01*cos(.3*k);});
  K::extract(ctx,sparse.data(),full.data(),rt,ct,field,0,1,mesh);K::extractGeneric(ctx,dense.data(),full.data(),rt,ct,field,0,1,mesh);
  double num=0.,den=0.;Kokkos::parallel_reduce("check transform difference",a*b,KOKKOS_LAMBDA(int k,double& v){v+=(sparse[k]-dense[k])*(sparse[k]-dense[k]);},num);Kokkos::parallel_reduce("check transform norm",a*b,KOKKOS_LAMBDA(int k,double& v){v+=dense[k]*dense[k];},den);
  double rel=sqrt(num/den);printf("COMPOSITE_TRANSFORM rowtrace=%d coltrace=%d field=%d relative=%.17g\n",rt,ct,field,rel);if(!exasim::composite::finiteValue(rel)||rel>1.e-12)throw std::runtime_error("COMPOSITE sparse/generic transform check failed");
 }
 // Independent dense reference for both shapes and both output precisions.
 for(int recovery=0;recovery<2;recovery++){
  int rows=recovery?N:M,fr=recovery?AF:BF,pr=recovery?P:Q,stride=recovery?FS:HS;
  View mat("check matrix",stride),x("check x",M),out("check product",rows);Kokkos::View<float*> fm("check float",stride);
  Kokkos::parallel_for("check matrix values",stride,KOKKOS_LAMBDA(int k){mat[k]=sin(.0123*k);fm[k]=float(mat[k]);});Kokkos::parallel_for("check x values",M,KOKKOS_LAMBDA(int k){x[k]=cos(.0345*k);});
  auto ah=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),mat),xh=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),x);
  for(int fp=0;fp<2;fp++){
   if(fp)K::product(out.data(),fm.data(),x.data(),recovery,1);else K::product(out.data(),mat.data(),x.data(),recovery,1);
   auto yh=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),out);double num=0.,den=0.;
   for(int comp=0;comp<NC;comp++)for(int node=0;node<pr;node++){
    int row=K::packedIndex(comp,node,!recovery);double reference=0.;
    if(comp<FLOW){for(int col=0;col<BF;col++){double value=ah[row+fr*col];reference+=(fp?double(float(value)):value)*xh[col];}}
    else for(int col=0;col<Q;col++){double value=ah[fr*BF+(comp-FLOW)*pr*Q+node+pr*col];reference+=(fp?double(float(value)):value)*xh[BF+4*col+comp-FLOW];}
    num+=(yh[row]-reference)*(yh[row]-reference);den+=reference*reference;
   }
   double error=sqrt(num/den);printf("COMPOSITE_PRODUCT recovery=%d fp32=%d relative=%.17g\n",recovery,fp,error);if(!exasim::composite::finiteValue(error)||error>1.e-12)throw std::runtime_error("COMPOSITE compact product check failed");
  }
 }
}
inline void selfCheck(Context& ctx,const solstruct& sol,const appstruct& app,const meshstruct& mesh,const commonstruct& c,int backend,bool freezeState=true){
 if(ctx.dimension==2)selfCheckDimension<2>(ctx,sol,app,mesh,c,backend,freezeState);
 else selfCheckDimension<3>(ctx,sol,app,mesh,c,backend,freezeState);
}
}
