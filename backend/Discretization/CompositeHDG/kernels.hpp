// Compile-time geometry specializations keep the established 3D operation order.
// Included inside fieldsplit after Context and inverse().
template<int ND> struct Kernels {
 using S=exasim::composite::Shape<ND>;
 static constexpr int P=S::P, PF=S::PF, Q=S::Q, NC=S::NC, STATE=S::state, OTHER=S::other;
 static constexpr int FLOW=S::flow, N=S::N, M=S::M, AF=S::AF, BF=S::BF, AS=S::AS, BS=S::BS;
 static constexpr int HS=S::HS, FS=S::FS;
KOKKOS_INLINE_FUNCTION static int species(int s){return s<3?s:4;}
KOKKOS_INLINE_FUNCTION static double T(int old,int c,const double* y){
 if(c==0)return old<5?y[old]:0.;
 if(c<FLOW)return old==c+4?1.:0.;
 if(old==NC-1)return y[5+c-FLOW];
 return old==species(c-FLOW)?1.:(old==3?-1.:0.);
}
KOKKOS_INLINE_FUNCTION static double L(int c,int old,const double* y){
 if(c==0)return old<5?1.:0.;
 if(c==FLOW-1){if(old==NC-1)return 1.;if(old>=5)return 0.;double v=0.;for(int s=0;s<4;s++)v+=y[5+s]*y[species(s)];for(int s=0;s<4;s++)if(old==species(s))v-=y[5+s];return v;}
 if(c<FLOW-1)return old==c+4?1.:0.;
 return (old==species(c-FLOW)?1.:0.)-(old<5?y[species(c-FLOW)]:0.);
}
static void freeze(Context& ctx,const solstruct& sol,const appstruct& app,const commonstruct& c){ if(!ctx.enabled)return;
 auto v=ctx.vy,t=ctx.ty;auto u=sol.udg,h=sol.uh;
 Kokkos::parallel_for("COMPOSITE volume fractions",ctx.owned*P,KOKKOS_LAMBDA(int k){int e=k/P,i=k%P;double rho=0.;for(int s=0;s<5;s++)rho+=u[size_t(e)*STATE*P+s*P+i];for(int s=0;s<5;s++)v[9*k+s]=u[size_t(e)*STATE*P+s*P+i]/rho;for(int s=5;s<9;s++)v[9*k+s]=0.;});
 Kokkos::parallel_for("COMPOSITE trace fractions",c.meshsizes.nf*PF,KOKKOS_LAMBDA(int k){double rho=0.;for(int s=0;s<5;s++)rho+=h[size_t(k)*NC+s];for(int s=0;s<5;s++)t[9*k+s]=h[size_t(k)*NC+s]/rho;for(int s=5;s<9;s++)t[9*k+s]=0.;});
 if(ctx.thermal.extent(0)){
  const int cap=ctx.batch*P;auto xp=ctx.thermal.data(),up=xp+ND*cap,op=up+STATE*cap,wp=op+OTHER*cap,src=wp+cap,fu=src+cap,fw=fu+STATE*cap;
  auto eos=c.driver_abi->hdgjac.HdgSourcew;
  for(int trace=0;trace<2;trace++){
   int points=trace?c.meshsizes.nf*PF:ctx.owned*P;auto dest=trace?t:v;auto ww=sol.wdg;
   for(int first=0;first<points;first+=cap){int ng=std::min(cap,points-first);
    ArraySetValue(xp,0.,ND*ng);ArraySetValue(op,0.,OTHER*ng);ArraySetValue(up,0.,STATE*ng);
    Kokkos::parallel_for("COMPOSITE thermal state",ng,KOKKOS_LAMBDA(int k){int p=first+k;for(int j=0;j<NC;j++)up[k+ng*j]=trace?h[size_t(p)*NC+j]:u[size_t(p/P)*STATE*P+j*P+p%P];wp[k]=trace?1.:ww[p];});
    int iters=trace?8:1;
    for(int it=0;it<iters;it++){eos(src,fu,fw,xp,up,op,wp,app.uinf,app.physicsparam,c.timestate.time,ctx.model.modelnumber,ng,STATE,NC,ND,ND,OTHER,1);
     if(trace&&it+1<iters)Kokkos::parallel_for("COMPOSITE trace EOS estimate",ng,KOKKOS_LAMBDA(int k){double next=wp[k]-src[k]/fw[k];wp[k]=next>0.01?next:0.01;});}
    Kokkos::parallel_for("COMPOSITE thermal directions",ng*4,KOKKOS_LAMBDA(int k){int i=k%ng,s=k/ng;dest[9*(first+i)+5+s]=-(fu[i+ng*species(s)]-fu[i+ng*3])/fu[i+ng*(NC-1)];});
   }
  }
 }
}
static void clearBatch(Context& ctx,int count){if(ctx.enabled){ArraySetValue(ctx.hf.data(),0.,M*M*count);ArraySetValue(ctx.ff.data(),0.,N*M*count);}}
static void extractGeneric(Context& ctx,double* out,const double* in,bool rt,bool ct,int field,int e1,int count,const meshstruct& mesh){ const int comps=field?4:FLOW,base=field?FLOW:0,nr=rt?M:N,nc=ct?M:N,a=(rt?Q:P)*comps,b=(ct?Q:P)*comps;
 auto vy=ctx.vy,ty=ctx.ty;auto ec=mesh.elemcon;
 Kokkos::parallel_for("COMPOSITE transformed field",a*b*count,KOKKOS_LAMBDA(int k){
  int r=k%a,col=(k/a)%b,e=k/(a*b),rp=rt?r/comps:r%P,rc=base+(rt?r%comps:r/P),cp=ct?col/comps:col%P,cc=base+(ct?col%comps:col/P);
  const double* yr=rt?ty.data()+9*ec[(e1+e)*Q+rp]:vy.data()+9*((e1+e)*P+rp);
  const double* yc=ct?ty.data()+9*ec[(e1+e)*Q+cp]:vy.data()+9*((e1+e)*P+cp);
  double sum=0.;for(int i=0;i<NC;i++){double l=L(rc,i,yr);if(l!=0.)for(int j=0;j<NC;j++){double t=T(j,cc,yc);if(t!=0.)sum+=l*in[size_t(e)*nr*nc+(rt?NC*rp+i:rp+P*i)+nr*(ct?NC*cp+j:cp+P*j)]*t;}}
  out[k]=sum;
 });
}
#include "sparse_transform.hpp"

KOKKOS_INLINE_FUNCTION static int packedIndex(int c,int p,bool trace){return c<FLOW?(trace?FLOW*p+c:p+P*c):(trace?BF+4*p+c-FLOW:AF+p+P*(c-FLOW));}
static void packResidual(Context& ctx,double* out,const double* in,bool trace,int e1,int count,const meshstruct& mesh){
 const int n=trace?M:N,points=trace?Q:P;auto vy=ctx.vy,ty=ctx.ty;auto ec=mesh.elemcon;
 Kokkos::parallel_for("COMPOSITE pack",n*count,KOKKOS_LAMBDA(int k){int e=k/n,p=(k%n)/NC,c=k%NC;const double* y=trace?ty.data()+9*ec[(e1+e)*Q+p]:vy.data()+9*((e1+e)*P+p);double sum=0.;for(int j=0;j<NC;j++)sum+=L(c,j,y)*in[size_t(e)*n+(trace?NC*p+j:p+points*j)];out[size_t(e)*n+packedIndex(c,p,trace)]=sum;});
}
static void unpack(Context& ctx,double* out,const double* in,bool trace,int e1,int count,const meshstruct& mesh){
 const int n=trace?M:N,points=trace?Q:P;auto vy=ctx.vy,ty=ctx.ty;auto ec=mesh.elemcon;
 Kokkos::parallel_for("COMPOSITE unpack",n*count,KOKKOS_LAMBDA(int k){int e=k/n,r=k%n,p=trace?r/NC:r%P,c=trace?r%NC:r/P;const double* y=trace?ty.data()+9*ec[(e1+e)*Q+p]:vy.data()+9*((e1+e)*P+p);double sum=0.;for(int j=0;j<NC;j++)sum+=T(c,j,y)*in[size_t(e)*n+packedIndex(j,p,trace)];out[k]=sum;});
}
static void compress(double* out,const double* in,bool recovery,int field,int count){
 const int a=recovery?AF:BF,b=BF,pa=recovery?P:Q,sa=4*pa,sb=BS,stride=recovery?FS:HS;
 if(!field){Kokkos::parallel_for("COMPOSITE persist flow",a*b*count,KOKKOS_LAMBDA(int k){int e=k/(a*b);out[size_t(e)*stride+k%(a*b)]=in[k];});}
 else Kokkos::parallel_for("COMPOSITE persist species diagonals",4*pa*Q*count,KOKKOS_LAMBDA(int k){int r=k%pa,col=(k/pa)%Q,s=(k/(pa*Q))%4,e=k/(4*pa*Q);int ri=recovery?r+P*s:4*r+s;out[size_t(e)*stride+a*b+s*pa*Q+r+pa*col]=in[size_t(e)*sa*sb+ri+sa*(4*col+s)];});
}
#include "local_reference.hpp"
#include "boundary_kernels.hpp"

static void condense(Context& ctx,double* D,double* B,double* C,double* Hfull,double* Ru,double* Rh,resstruct& res,meshstruct& mesh,tempstruct& tmp,commonstruct& common,int e1,int count,int backend){ const bool check=ctx.validate&&!ctx.validated&&common.mpiRank==0;auto pv=ctx.vecin.data(),ph=ctx.vecout.data();
 schurVectorRu(tmp.tempn,Ru,P,NC,count);packResidual(ctx,pv,tmp.tempn,false,e1,count,mesh);
 schurVectorRh(tmp.tempn,Rh,Q,NC,count);packResidual(ctx,ph,tmp.tempn,true,e1,count,mesh);
 boundaryPrepare(ctx,C,Hfull,Rh,mesh,common,e1,count);
 for(int field=0;field<2;field++){
  const int a=field?AS:AF,b=field?BS:BF,vo=field?AF:0,ho=field?BF:0;
  auto A=ctx.work.data(),E=A+size_t(a)*a*count,G=E+size_t(a)*b*count,S=G+size_t(b)*a*count,rv=S+size_t(b)*b*count,rh=rv+a*count,zv=rh+b*count,Fout=zv+a*count;
  extract(ctx,A,D,false,false,field,e1,count,mesh);extract(ctx,E,B,false,true,field,e1,count,mesh);extract(ctx,G,C,true,false,field,e1,count,mesh);extract(ctx,S,Hfull,true,true,field,e1,count,mesh);
  Kokkos::parallel_for("COMPOSITE gather volume RHS",a*count,KOKKOS_LAMBDA(int k){rv[k]=pv[(k/a)*N+vo+k%a];});
  Kokkos::parallel_for("COMPOSITE gather trace RHS",b*count,KOKKOS_LAMBDA(int k){rh[k]=ph[(k/b)*M+ho+k%b];});
  std::unique_ptr<LocalReference> reference;if(check)reference=std::make_unique<LocalReference>(a,b,field,A,E,G,S,rv,rh,backend);
  inverse(ctx,common.cublasHandle,A,tmp.tempn,res.ipiv,a,count,backend);
  PGEMNMStridedBached(common.cublasHandle,a,b,a,-1.,A,a,E,a,0.,Fout,a,count,backend);
  PGEMNMStridedBached(common.cublasHandle,a,1,a,1.,A,a,rv,a,0.,zv,a,count,backend);
  PGEMNMStridedBached(common.cublasHandle,b,b,a,1.,G,b,Fout,a,1.,S,b,count,backend);
  PGEMNMStridedBached(common.cublasHandle,b,1,a,-1.,G,b,zv,a,1.,rh,b,count,backend);
  compress(res.H+size_t(e1)*HS,S,false,field,count);compress(res.F+size_t(e1)*FS,Fout,true,field,count);
  if(reference)reference->check(Fout,zv,S,rh,res.H+size_t(e1)*HS,res.F+size_t(e1)*FS,backend);
  Kokkos::parallel_for("COMPOSITE scatter volume RHS",a*count,KOKKOS_LAMBDA(int k){pv[(k/a)*N+vo+k%a]=zv[k];});
  Kokkos::parallel_for("COMPOSITE scatter trace RHS",b*count,KOKKOS_LAMBDA(int k){ph[(k/b)*M+ho+k%b]=rh[k];});
 }
 boundaryFinish(ctx,res.F+size_t(e1)*FS,res.H+size_t(e1)*HS,pv,ph,common,e1,count);
 unpack(ctx,Ru,pv,false,e1,count,mesh);unpack(ctx,Rh,ph,true,e1,count,mesh);ctx.validated=true;
}
template<class Scalar>static void product(double* out,const Scalar* matrix,const double* in,bool recovery,int count){
 const int rows=recovery?N:M,flowrows=recovery?AF:BF,pa=recovery?P:Q,stride=recovery?FS:HS;
 Kokkos::parallel_for("COMPOSITE compact product",rows*count,KOKKOS_LAMBDA(int k){int e=k/rows,r=k%rows;const Scalar* a=matrix+size_t(e)*stride;const double* x=in+size_t(e)*M;double sum=0.;
  if(r<flowrows){for(int j=0;j<BF;j++)sum+=double(a[r+flowrows*j])*x[j];}
  else{int p=recovery?(r-flowrows)%P:(r-flowrows)/4,s=recovery?(r-flowrows)/P:(r-flowrows)%4;for(int j=0;j<Q;j++)sum+=double(a[flowrows*BF+s*pa*Q+p+pa*j])*x[BF+4*j+s];}out[k]=sum;});
}
static void multiply(Context& ctx,double* out,double* matrix,const double* in,int e1,int count,meshstruct& mesh,commonstruct& c,int backend,bool isASM=false){
 if(!count)return;auto x=ctx.vecin.data(),y=ctx.vecout.data();packResidual(ctx,x,in,true,e1,count,mesh);
 if(isASM&&ctx.asm_fp32)product(y,reinterpret_cast<const float*>(matrix),x,false,count);else product(y,matrix,x,false,count);
 if(!isASM)boundaryProduct(ctx,y,x,e1,count);
 unpack(ctx,out,y,true,e1,count,mesh);
}
static void recover(Context& ctx,double* w,double* f,const double* ve,meshstruct& mesh,commonstruct& c,int backend){
 int count=c.meshsizes.ne1;auto x=ctx.vecin.data(),y=ctx.vecout.data();packResidual(ctx,x,ve,true,0,count,mesh);product(y,f,x,true,count);unpack(ctx,x,y,false,0,count,mesh);ArrayAXPBY(w,w,x,1.,1.,N*count);
}
static void schwarz(Context& ctx,double* BE,const double* AE,resstruct& res,meshstruct& mesh,tempstruct& tmp,commonstruct& c,int backend){ auto w=ctx.hf.data();auto ec=mesh.elemcon,f2e=mesh.f2e;auto bad=ctx.asm_status.data();Kokkos::deep_copy(ctx.asm_status,0);const bool fp32=ctx.asm_fp32;auto packed=reinterpret_cast<float*>(BE);
 for(int first=0;first<ctx.owned;first+=ctx.batch){int count=std::min(ctx.batch,ctx.owned-first);
  for(int startField=0;startField<2;startField++){
   int fields=startField?4:1,m=startField?Q:BF,q=m*m,comps=startField?1:FLOW;
   Kokkos::parallel_for("COMPOSITE ASM gather",q*count*fields,KOKKOS_LAMBDA(int k){int block=k/q,e=first+block%count,field=startField+block/count,r=k%q%m,col=(k%q)/m,off=field?BF*BF+(field-1)*Q*Q:0;double value=AE[size_t(e)*HS+off+r+m*col];int lf=r/(PF*comps);
    if(lf==col/(PF*comps)){int f=ec[Q*e+PF*lf]/PF,a=f2e[4*f],b=f2e[4*f+2],neighbor=e==a?b:a,nlf=e==a?f2e[4*f+3]:f2e[4*f+1];
     if(neighbor>=0){int gr=ec[Q*e+r/comps],gc=ec[Q*e+col/comps],nr=-1,nc=-1;for(int node=0;node<PF;node++){int g=ec[Q*neighbor+PF*nlf+node];if(g==gr)nr=node;if(g==gc)nc=node;}
      if(nr<0||nc<0)Kokkos::atomic_add(bad,1);else{int rr=comps*(PF*nlf+nr)+r%comps,cc=comps*(PF*nlf+nc)+col%comps;value+=AE[size_t(neighbor)*HS+off+rr+m*cc];}}
    }w[k]=value;});
   inverse(ctx,c.cublasHandle,w,tmp.tempn,res.ipiv,m,count*fields,backend);
   Kokkos::parallel_for("COMPOSITE ASM inverse store",q*count*fields,KOKKOS_LAMBDA(int k){int block=k/q,e=first+block%count,field=startField+block/count,off=field?BF*BF+(field-1)*Q*Q:0;double value=w[k];size_t idx=size_t(e)*HS+off+k%q;
    if(exasim::composite::invalidInverseValue(value,fp32))Kokkos::atomic_add(bad,1);
    if(fp32)packed[idx]=float(value);else BE[idx]=value;});
  }
 }
 int status=0;TemplateCopytoHost(&status,bad,1,backend);if(status)throw std::runtime_error("COMPOSITE ASM invalid orientation or nonfinite/out-of-range inverse");
}

};
