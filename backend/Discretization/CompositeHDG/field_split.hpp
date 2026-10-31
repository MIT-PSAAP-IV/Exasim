#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <limits>
#include "composite_finite.hpp"
namespace fieldsplit {
using View=Kokkos::View<double*>;
constexpr int P=27,Q=54,NC=9,N=243,M=486,AF=135,BF=270,AS=108,BS=216;
constexpr int HS=BF*BF+4*Q*Q,FS=AF*BF+4*P*Q;
inline int checkedCount(uint64_t count,const char* name){
 if(count>uint64_t(std::numeric_limits<int>::max()))throw std::runtime_error(std::string("3D HDG array extent exceeds signed 32-bit interface: ")+name+" count="+std::to_string(count));
 return int(count);
}
inline size_t checkedStorage(uint64_t count,const char* name){
 if(count>uint64_t(std::numeric_limits<ptrdiff_t>::max())/sizeof(double))
  throw std::runtime_error(std::string("3D HDG storage byte extent overflows pointer range: ")+name);
 return size_t(count);
}
// Preserve the original kernel for ordinary counts; chunk only oversized arrays.
template<class T> inline void setStorage(T* p,T value,size_t n){
 if(n<=size_t(std::numeric_limits<int>::max())){ArraySetValue(p,value,int(n));return;}
 constexpr size_t chunk=size_t(1)<<28;
 for(size_t off=0;off<n;off+=chunk)ArraySetValue(p+off,value,int(std::min(chunk,n-off)));
}
template<class T> inline void copyStorage(T* dst,const T* src,size_t n){
 if(n<=size_t(std::numeric_limits<int>::max())){ArrayCopy(dst,src,int(n));return;}
 constexpr size_t chunk=size_t(1)<<28;
 for(size_t off=0;off<n;off+=chunk)ArrayCopy(dst+off,src+off,int(std::min(chunk,n-off)));
}
#ifndef USE_FLOAT
struct Context {
 exasim::composite::ModelV1 model{};
 bool validate=false,validated=false;
 bool enabled=false,shared=true,cached_inverse=false,species_batch=true,asm_fp32=false,species_kron=false;
 int species_matrices=4,n=N,m=M,ne=0,owned=0,batch=0;
 View hf,ff,work,vy,ty,vecin,vecout,thermal;
 Kokkos::View<int*> asm_status,pivots,info;
 Kokkos::View<uintptr_t*> aPointers,cPointers;
 static size_t plannedBytes(const commonstruct& c){
  const size_t batch=c.meshsizes.neb,owned=c.meshsizes.ne1,nf=c.meshsizes.nf;
  size_t bytes=8*(size_t(M)*M*batch+size_t(N)*M*batch+batch*(AF*AF+3*AF*BF+BF*BF+2*AF+BF)+owned*P*9+nf*81+2*size_t(M)*owned+batch*P*92)+sizeof(int);
#ifdef HAVE_HIP
  bytes+=sizeof(int)*(size_t(BF)*batch+4*batch)+sizeof(uintptr_t)*8*batch;
#endif
  return bytes;
 }
 void init(const commonstruct& c){
  enabled=c.compositeConfig.enabled;asm_fp32=enabled;
  if(!enabled)return;
  if(c.grid.nd!=3||c.grid.npe!=P||c.grid.npf!=9||c.meshsizes.nfe!=6||c.components.ncu!=NC||c.components.nc!=36||c.components.nco!=14||c.components.ncw!=1||c.solverparams.preconditioner!=1||c.couplingparams.coupledinterface>0||c.outputparams.debugMode)
    throw std::runtime_error("compositehdg requires P2 hex CNS5air, elemental ASM, no coupled interfaces/debug dumps");
  if(c.backend==2)throw std::runtime_error("compositehdg CUDA is not validated; use HIP or CPU");
  model=exasim::composite::requireModel(*c.driver_abi,c.modelnumber);
  validate=std::getenv("EXASIM_COMPOSITE_VALIDATE")!=nullptr;
  ne=c.meshsizes.ne;owned=c.meshsizes.ne1;batch=c.meshsizes.neb;
  checkedCount(uint64_t(HS)*ne,"compact H");checkedCount(uint64_t(FS)*ne,"compact F");
  hf=View("FS batch full H",size_t(M)*M*batch);ff=View("FS batch full F",size_t(N)*M*batch);
  // Sequential field work: A,E,G,S,rv,rh,zv,F. Neither full assembly input
  // is overwritten until both fields have been extracted and condensed.
  work=View("FS sequential field workspace",size_t(batch)*(AF*AF+3*AF*BF+BF*BF+2*AF+BF));
  vy=View("FS volume basis",size_t(owned)*P*9);ty=View("FS trace basis",size_t(c.meshsizes.nf)*9*9);
  vecin=View("FS packed volume or trace input",size_t(M)*owned);vecout=View("FS packed output",size_t(M)*owned);
  thermal=View("FS thermal basis workspace",size_t(batch)*P*92);
  asm_status=Kokkos::View<int*>("FS ASM validity",1);
#ifdef HAVE_HIP
  cached_inverse=c.backend==3;
  pivots=Kokkos::View<int*>("FS LU pivots",size_t(BF)*batch);
  info=Kokkos::View<int*>("FS LU info",4*batch);
  aPointers=Kokkos::View<uintptr_t*>("FS LU inputs",4*batch);cPointers=Kokkos::View<uintptr_t*>("FS LU outputs",4*batch);
#endif
 }
 void clear(){hf=View();ff=View();work=View();vy=View();ty=View();vecin=View();vecout=View();thermal=View();asm_status={};pivots={};info={};aPointers={};cPointers={};}
 size_t bytes()const{return 8*(hf.extent(0)+ff.extent(0)+work.extent(0)+vy.extent(0)+ty.extent(0)+vecin.extent(0)+vecout.extent(0)+thermal.extent(0))+sizeof(int)*(asm_status.extent(0)+pivots.extent(0)+info.extent(0))+sizeof(uintptr_t)*(aPointers.extent(0)+cPointers.extent(0));}
};

#include "cached_inverse.hpp"
inline int Hstride(Context& ctx){return ctx.enabled?HS:M*M;}
inline int Fstride(Context& ctx){return ctx.enabled?FS:N*M;}
inline int ASMstride(Context& ctx){return Hstride(ctx);}
inline int ASMstorageSlots(Context& ctx){return ctx.asm_fp32?HS/2:Hstride(ctx);}
inline bool fullLocalEnabled(){return false;}
template<class S> inline S* H(Context* ctx,S* p,size_t offset){if constexpr(std::is_same_v<S,double>) { if(ctx) return ctx->hf.data(); } return p+offset;}
template<class S> inline S* F(Context* ctx,S* p,size_t offset){if constexpr(std::is_same_v<S,double>) { if(ctx) return ctx->ff.data(); } return p+offset;}
KOKKOS_INLINE_FUNCTION int species(int s){return s<3?s:4;}
KOKKOS_INLINE_FUNCTION double T(int old,int c,const double* y){
 if(c==0)return old<5?y[old]:0.;
 if(c<5)return old==c+4?1.:0.;
 if(old==8)return y[c];
 return old==species(c-5)?1.:(old==3?-1.:0.);
}
KOKKOS_INLINE_FUNCTION double L(int c,int old,const double* y){
 if(c==0)return old<5?1.:0.;
 if(c==4){if(old==8)return 1.;if(old>=5)return 0.;double v=0.;for(int s=0;s<4;s++)v+=y[5+s]*y[species(s)];for(int s=0;s<4;s++)if(old==species(s))v-=y[5+s];return v;}
 if(c<4)return old==c+4?1.:0.;
 return (old==species(c-5)?1.:0.)-(old<5?y[species(c-5)]:0.);
}
inline void freeze(Context& ctx,const solstruct& sol,const appstruct& app,const commonstruct& c){ if(!ctx.enabled)return;
 auto v=ctx.vy,t=ctx.ty;auto u=sol.udg,h=sol.uh;
 Kokkos::parallel_for("FS3 volume fractions",ctx.owned*P,KOKKOS_LAMBDA(int k){int e=k/P,i=k%P;double rho=0.;for(int s=0;s<5;s++)rho+=u[size_t(e)*972+s*P+i];for(int s=0;s<5;s++)v[9*k+s]=u[size_t(e)*972+s*P+i]/rho;for(int s=5;s<9;s++)v[9*k+s]=0.;});
 Kokkos::parallel_for("FS3 trace fractions",c.meshsizes.nf*9,KOKKOS_LAMBDA(int k){double rho=0.;for(int s=0;s<5;s++)rho+=h[size_t(k)*9+s];for(int s=0;s<5;s++)t[9*k+s]=h[size_t(k)*9+s]/rho;for(int s=5;s<9;s++)t[9*k+s]=0.;});
 if(ctx.thermal.extent(0)){
  const int cap=ctx.batch*P;auto xp=ctx.thermal.data(),up=xp+3*cap,op=up+36*cap,wp=op+14*cap,src=wp+cap,fu=src+cap,fw=fu+36*cap;
  auto eos=c.driver_abi->hdgjac.HdgSourcew;
  for(int trace=0;trace<2;trace++){
   int points=trace?c.meshsizes.nf*9:ctx.owned*P;auto dest=trace?t:v;auto ww=sol.wdg;
   for(int first=0;first<points;first+=cap){int ng=std::min(cap,points-first);
    ArraySetValue(xp,0.,3*ng);ArraySetValue(op,0.,14*ng);ArraySetValue(up,0.,36*ng);
    Kokkos::parallel_for("FS3 thermal state",ng,KOKKOS_LAMBDA(int k){int p=first+k;for(int j=0;j<9;j++)up[k+ng*j]=trace?h[size_t(p)*9+j]:u[size_t(p/P)*972+j*P+p%P];wp[k]=trace?1.:ww[p];});
    int iters=trace?8:1;
    for(int it=0;it<iters;it++){eos(src,fu,fw,xp,up,op,wp,app.uinf,app.physicsparam,c.timestate.time,ctx.model.modelnumber,ng,36,9,3,3,14,1);
     if(trace&&it+1<iters)Kokkos::parallel_for("FS3 trace EOS estimate",ng,KOKKOS_LAMBDA(int k){double next=wp[k]-src[k]/fw[k];wp[k]=next>0.01?next:0.01;});}
    Kokkos::parallel_for("FS3 thermal directions",ng*4,KOKKOS_LAMBDA(int k){int i=k%ng,s=k/ng;dest[9*(first+i)+5+s]=-(fu[i+ng*species(s)]-fu[i+ng*3])/fu[i+ng*8];});
   }
  }
 }
}
inline void clearBatch(Context& ctx,int count){if(ctx.enabled){ArraySetValue(ctx.hf.data(),0.,M*M*count);ArraySetValue(ctx.ff.data(),0.,N*M*count);}}
inline void extractGeneric(Context& ctx,double* out,const double* in,bool rt,bool ct,int field,int e1,int count,const meshstruct& mesh){ const int comps=field?4:5,base=field?5:0,nr=rt?M:N,nc=ct?M:N,a=(rt?Q:P)*comps,b=(ct?Q:P)*comps;
 auto vy=ctx.vy,ty=ctx.ty;auto ec=mesh.elemcon;
 Kokkos::parallel_for("FS3 transformed field",a*b*count,KOKKOS_LAMBDA(int k){
  int r=k%a,col=(k/a)%b,e=k/(a*b),rp=rt?r/comps:r%P,rc=base+(rt?r%comps:r/P),cp=ct?col/comps:col%P,cc=base+(ct?col%comps:col/P);
  const double* yr=rt?ty.data()+9*ec[(e1+e)*Q+rp]:vy.data()+9*((e1+e)*P+rp);
  const double* yc=ct?ty.data()+9*ec[(e1+e)*Q+cp]:vy.data()+9*((e1+e)*P+cp);
  double sum=0.;for(int i=0;i<9;i++){double l=L(rc,i,yr);if(l!=0.)for(int j=0;j<9;j++){double t=T(j,cc,yc);if(t!=0.)sum+=l*in[size_t(e)*nr*nc+(rt?9*rp+i:rp+P*i)+nr*(ct?9*cp+j:cp+P*j)]*t;}}
  out[k]=sum;
 });
}
#include "sparse_transform.hpp"
KOKKOS_INLINE_FUNCTION int packedIndex(int c,int p,bool trace){return c<5?(trace?5*p+c:p+P*c):(trace?BF+4*p+c-5:AF+p+P*(c-5));}
inline void packResidual(Context& ctx,double* out,const double* in,bool trace,int e1,int count,const meshstruct& mesh){
 const int n=trace?M:N,points=trace?Q:P;auto vy=ctx.vy,ty=ctx.ty;auto ec=mesh.elemcon;
 Kokkos::parallel_for("FS3 pack",n*count,KOKKOS_LAMBDA(int k){int e=k/n,p=(k%n)/9,c=k%9;const double* y=trace?ty.data()+9*ec[(e1+e)*Q+p]:vy.data()+9*((e1+e)*P+p);double sum=0.;for(int j=0;j<9;j++)sum+=L(c,j,y)*in[size_t(e)*n+(trace?9*p+j:p+points*j)];out[size_t(e)*n+packedIndex(c,p,trace)]=sum;});
}
inline void unpack(Context& ctx,double* out,const double* in,bool trace,int e1,int count,const meshstruct& mesh){
 const int n=trace?M:N,points=trace?Q:P;auto vy=ctx.vy,ty=ctx.ty;auto ec=mesh.elemcon;
 Kokkos::parallel_for("FS3 unpack",n*count,KOKKOS_LAMBDA(int k){int e=k/n,r=k%n,p=trace?r/9:r%P,c=trace?r%9:r/P;const double* y=trace?ty.data()+9*ec[(e1+e)*Q+p]:vy.data()+9*((e1+e)*P+p);double sum=0.;for(int j=0;j<9;j++)sum+=T(c,j,y)*in[size_t(e)*n+packedIndex(j,p,trace)];out[k]=sum;});
}
inline void compress(double* out,const double* in,bool recovery,int field,int count){
 const int a=recovery?AF:BF,b=BF,pa=recovery?P:Q,sa=4*pa,sb=BS,stride=recovery?FS:HS;
 if(!field){Kokkos::parallel_for("FS3 persist flow",a*b*count,KOKKOS_LAMBDA(int k){int e=k/(a*b);out[size_t(e)*stride+k%(a*b)]=in[k];});}
 else Kokkos::parallel_for("FS3 persist species diagonals",4*pa*Q*count,KOKKOS_LAMBDA(int k){int r=k%pa,col=(k/pa)%Q,s=(k/(pa*Q))%4,e=k/(4*pa*Q);int ri=recovery?r+P*s:4*r+s;out[size_t(e)*stride+a*b+s*pa*Q+r+pa*col]=in[size_t(e)*sa*sb+ri+sa*(4*col+s)];});
}
#include "local_reference.hpp"
inline void condense(Context& ctx,double* D,double* B,double* C,double* Hfull,double* Ru,double* Rh,resstruct& res,meshstruct& mesh,tempstruct& tmp,commonstruct& common,int e1,int count,int backend){ const bool check=ctx.validate&&!ctx.validated&&common.mpiRank==0;auto pv=ctx.vecin.data(),ph=ctx.vecout.data();
 schurVectorRu(tmp.tempn,Ru,P,9,count);packResidual(ctx,pv,tmp.tempn,false,e1,count,mesh);
 schurVectorRh(tmp.tempn,Rh,Q,9,count);packResidual(ctx,ph,tmp.tempn,true,e1,count,mesh);
 for(int field=0;field<2;field++){
  const int a=field?AS:AF,b=field?BS:BF,vo=field?AF:0,ho=field?BF:0;
  auto A=ctx.work.data(),E=A+size_t(a)*a*count,G=E+size_t(a)*b*count,S=G+size_t(b)*a*count,rv=S+size_t(b)*b*count,rh=rv+a*count,zv=rh+b*count,Fout=zv+a*count;
  extract(ctx,A,D,false,false,field,e1,count,mesh);extract(ctx,E,B,false,true,field,e1,count,mesh);extract(ctx,G,C,true,false,field,e1,count,mesh);extract(ctx,S,Hfull,true,true,field,e1,count,mesh);
  Kokkos::parallel_for("FS3 gather volume RHS",a*count,KOKKOS_LAMBDA(int k){rv[k]=pv[(k/a)*N+vo+k%a];});
  Kokkos::parallel_for("FS3 gather trace RHS",b*count,KOKKOS_LAMBDA(int k){rh[k]=ph[(k/b)*M+ho+k%b];});
  std::unique_ptr<LocalReference> reference;if(check)reference=std::make_unique<LocalReference>(a,b,field,A,E,G,S,rv,rh,backend);
  inverse(ctx,common.cublasHandle,A,tmp.tempn,res.ipiv,a,count,backend);
  PGEMNMStridedBached(common.cublasHandle,a,b,a,-1.,A,a,E,a,0.,Fout,a,count,backend);
  PGEMNMStridedBached(common.cublasHandle,a,1,a,1.,A,a,rv,a,0.,zv,a,count,backend);
  PGEMNMStridedBached(common.cublasHandle,b,b,a,1.,G,b,Fout,a,1.,S,b,count,backend);
  PGEMNMStridedBached(common.cublasHandle,b,1,a,-1.,G,b,zv,a,1.,rh,b,count,backend);
  compress(res.H+size_t(e1)*HS,S,false,field,count);compress(res.F+size_t(e1)*FS,Fout,true,field,count);
  if(reference)reference->check(Fout,zv,S,rh,res.H+size_t(e1)*HS,res.F+size_t(e1)*FS,backend);
  Kokkos::parallel_for("FS3 scatter volume RHS",a*count,KOKKOS_LAMBDA(int k){pv[(k/a)*N+vo+k%a]=zv[k];});
  Kokkos::parallel_for("FS3 scatter trace RHS",b*count,KOKKOS_LAMBDA(int k){ph[(k/b)*M+ho+k%b]=rh[k];});
 }
 unpack(ctx,Ru,pv,false,e1,count,mesh);unpack(ctx,Rh,ph,true,e1,count,mesh);ctx.validated=true;
}
template<class Scalar>inline void product(double* out,const Scalar* matrix,const double* in,bool recovery,int count){
 const int rows=recovery?N:M,flowrows=recovery?AF:BF,pa=recovery?P:Q,stride=recovery?FS:HS;
 Kokkos::parallel_for("FS3 compact product",rows*count,KOKKOS_LAMBDA(int k){int e=k/rows,r=k%rows;const Scalar* a=matrix+size_t(e)*stride;const double* x=in+size_t(e)*M;double sum=0.;
  if(r<flowrows){for(int j=0;j<BF;j++)sum+=double(a[r+flowrows*j])*x[j];}
  else{int p=recovery?(r-flowrows)%P:(r-flowrows)/4,s=recovery?(r-flowrows)/P:(r-flowrows)%4;for(int j=0;j<Q;j++)sum+=double(a[flowrows*BF+s*pa*Q+p+pa*j])*x[BF+4*j+s];}out[k]=sum;});
}
inline void multiply(Context& ctx,double* out,double* matrix,const double* in,int e1,int count,meshstruct& mesh,commonstruct& c,int backend,bool isASM=false){
 if(!count)return;auto x=ctx.vecin.data(),y=ctx.vecout.data();packResidual(ctx,x,in,true,e1,count,mesh);
 if(isASM&&ctx.asm_fp32)product(y,reinterpret_cast<const float*>(matrix),x,false,count);else product(y,matrix,x,false,count);unpack(ctx,out,y,true,e1,count,mesh);
}
inline void recover(Context& ctx,double* w,double* f,const double* ve,meshstruct& mesh,commonstruct& c,int backend){
 int count=c.meshsizes.ne1;auto x=ctx.vecin.data(),y=ctx.vecout.data();packResidual(ctx,x,ve,true,0,count,mesh);product(y,f,x,true,count);unpack(ctx,x,y,false,0,count,mesh);ArrayAXPBY(w,w,x,1.,1.,N*count);
}
inline void schwarz(Context& ctx,double* BE,const double* AE,resstruct& res,meshstruct& mesh,tempstruct& tmp,commonstruct& c,int backend){ auto w=ctx.hf.data();auto ec=mesh.elemcon,f2e=mesh.f2e;auto bad=ctx.asm_status.data();Kokkos::deep_copy(ctx.asm_status,0);const bool fp32=ctx.asm_fp32;auto packed=reinterpret_cast<float*>(BE);
 for(int first=0;first<ctx.owned;first+=ctx.batch){int count=std::min(ctx.batch,ctx.owned-first);
  for(int startField=0;startField<2;startField++){
   int fields=startField?4:1,m=startField?Q:BF,q=m*m,comps=startField?1:5;
   Kokkos::parallel_for("FS3 ASM gather",q*count*fields,KOKKOS_LAMBDA(int k){int block=k/q,e=first+block%count,field=startField+block/count,r=k%q%m,col=(k%q)/m,off=field?BF*BF+(field-1)*Q*Q:0;double value=AE[size_t(e)*HS+off+r+m*col];int lf=r/(9*comps);
    if(lf==col/(9*comps)){int f=ec[Q*e+9*lf]/9,a=f2e[4*f],b=f2e[4*f+2],neighbor=e==a?b:a,nlf=e==a?f2e[4*f+3]:f2e[4*f+1];
     if(neighbor>=0){int gr=ec[Q*e+r/comps],gc=ec[Q*e+col/comps],nr=-1,nc=-1;for(int node=0;node<9;node++){int g=ec[Q*neighbor+9*nlf+node];if(g==gr)nr=node;if(g==gc)nc=node;}
      if(nr<0||nc<0)Kokkos::atomic_add(bad,1);else{int rr=comps*(9*nlf+nr)+r%comps,cc=comps*(9*nlf+nc)+col%comps;value+=AE[size_t(neighbor)*HS+off+rr+m*cc];}}
    }w[k]=value;});
   inverse(ctx,c.cublasHandle,w,tmp.tempn,res.ipiv,m,count*fields,backend);
   Kokkos::parallel_for("FS3 ASM inverse store",q*count*fields,KOKKOS_LAMBDA(int k){int block=k/q,e=first+block%count,field=startField+block/count,off=field?BF*BF+(field-1)*Q*Q:0;double value=w[k];size_t idx=size_t(e)*HS+off+k%q;
    if(exasim::composite::invalidInverseValue(value,fp32))Kokkos::atomic_add(bad,1);
    if(fp32)packed[idx]=float(value);else BE[idx]=value;});
  }
 }
 int status=0;TemplateCopytoHost(&status,bad,1,backend);if(status)throw std::runtime_error("FS3 ASM invalid orientation or nonfinite/out-of-range inverse");
}
// Other scalar/index instantiations can still compile the unchanged native path.
// The shape/config gate rejects composite mode before these are reachable.
template<class S,class I> inline void recover(Context&,S*,S*,const S*,meshstructT<S,I>&,commonstructT<S,I>&,int){throw std::runtime_error("unsupported composite scalar/index");}
template<class S,class I> inline void multiply(Context&,S*,S*,const S*,int,int,meshstructT<S,I>&,commonstructT<S,I>&,int,bool=false){throw std::runtime_error("unsupported composite scalar/index");}

#else
// A float-state build retains the native solver; composite mode is rejected by
// validateShape before allocation. Keep the opt-in feature out of its kernels.
struct Context {
 bool enabled=false;
 template<class C> static size_t plannedBytes(const C&){return 0;}
 template<class C> void init(const C&){throw std::runtime_error("compositehdg requires double-state build");}
 void clear(){} size_t bytes()const{return 0;}
};
inline int Hstride(Context&){return HS;}
inline int Fstride(Context&){return FS;}
inline int ASMstorageSlots(Context&){return HS/2;}
template<class S> inline S* H(Context*,S* p,size_t offset){return p+offset;}
template<class S> inline S* F(Context*,S* p,size_t offset){return p+offset;}
template<class... Args> void freeze(Args&&...){throw std::runtime_error("unsupported composite scalar");}
template<class... Args> void clearBatch(Args&&...){throw std::runtime_error("unsupported composite scalar");}
template<class... Args> void condense(Args&&...){throw std::runtime_error("unsupported composite scalar");}
template<class... Args> void recover(Args&&...){throw std::runtime_error("unsupported composite scalar");}
template<class... Args> void multiply(Args&&...){throw std::runtime_error("unsupported composite scalar");}
template<class... Args> void schwarz(Args&&...){throw std::runtime_error("unsupported composite scalar");}
#endif

}
