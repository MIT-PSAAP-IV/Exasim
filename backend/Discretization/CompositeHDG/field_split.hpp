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
#include "boundary_storage.hpp"
struct Context {
 exasim::composite::ModelV1 model{};
 bool validate=false,validated=false;
 bool enabled=false,shared=true,cached_inverse=false,species_batch=true,asm_fp32=false,species_kron=false;
 int dimension=3;
 int species_matrices=4,n=N,m=M,ne=0,owned=0,batch=0;
 View hf,ff,work,vy,ty,vecin,vecout,thermal;
 BoundaryRows boundary;
 Kokkos::View<int*> asm_status,pivots,info;
 Kokkos::View<uintptr_t*> aPointers,cPointers;
 template<class C> static size_t plannedBytes(const C& c){
  const auto d=exasim::composite::dimensions(c.grid.nd);
  const int P=d.P,PF=d.PF,N=d.N,M=d.M,AF=d.AF,BF=d.BF,HS=d.HS,FS=d.FS;

  const size_t batch=c.meshsizes.neb,owned=c.meshsizes.ne1,nf=c.meshsizes.nf;
  size_t bytes=8*(size_t(M)*M*batch+size_t(N)*M*batch+batch*(AF*AF+3*AF*BF+BF*BF+2*AF+BF)+owned*P*9+nf*PF*9+2*size_t(M)*owned+batch*P*d.thermal)+sizeof(int);
#ifdef HAVE_HIP
  bytes+=sizeof(int)*(size_t(BF)*batch+4*batch)+sizeof(uintptr_t)*8*batch;
#endif
  return bytes+BoundaryRows::plannedBytes(c);
 }
 void init(const commonstruct& c){
  enabled=c.compositeConfig.enabled;asm_fp32=enabled;
  if(!enabled)return;
  const auto d=exasim::composite::dimensions(c.grid.nd);
  const int P=d.P,PF=d.PF,N=d.N,M=d.M,AF=d.AF,BF=d.BF,HS=d.HS,FS=d.FS;
  dimension=d.nd;n=d.N;m=d.M;
  if(c.grid.nd!=d.nd||c.grid.npe!=P||c.grid.npf!=d.PF||c.meshsizes.nfe!=2*d.nd||c.components.ncu!=d.NC||c.components.nc!=d.state||c.components.nco!=d.other||c.components.ncw!=1||c.solverparams.preconditioner!=1||c.couplingparams.coupledinterface>0||c.outputparams.debugMode)
    throw std::runtime_error("compositehdg requires P2 quad/hex CNS5air, elemental ASM, no coupled interfaces/debug dumps");
  if(c.backend==2)throw std::runtime_error("compositehdg CUDA is not validated; use HIP or CPU");
  model=exasim::composite::requireModel(*c.driver_abi,c.modelnumber,c.grid.nd);
  validate=std::getenv("EXASIM_COMPOSITE_VALIDATE")!=nullptr;
  ne=c.meshsizes.ne;owned=c.meshsizes.ne1;batch=c.meshsizes.neb;
  checkedCount(uint64_t(HS)*ne,"compact H");checkedCount(uint64_t(FS)*ne,"compact F");
  hf=View("FS batch full H",size_t(M)*M*batch);ff=View("FS batch full F",size_t(N)*M*batch);
  // Sequential field work: A,E,G,S,rv,rh,zv,F. Neither full assembly input
  // is overwritten until both fields have been extracted and condensed.
  work=View("FS sequential field workspace",size_t(batch)*(AF*AF+3*AF*BF+BF*BF+2*AF+BF));
  vy=View("FS volume basis",size_t(owned)*P*9);ty=View("FS trace basis",size_t(c.meshsizes.nf)*PF*9);
  vecin=View("FS packed volume or trace input",size_t(M)*owned);vecout=View("FS packed output",size_t(M)*owned);
  thermal=View("FS thermal basis workspace",size_t(batch)*P*d.thermal);
  asm_status=Kokkos::View<int*>("FS ASM validity",1);
#ifdef HAVE_HIP
  cached_inverse=c.backend==3;
  pivots=Kokkos::View<int*>("FS LU pivots",size_t(BF)*batch);
  info=Kokkos::View<int*>("FS LU info",4*batch);
  aPointers=Kokkos::View<uintptr_t*>("FS LU inputs",4*batch);cPointers=Kokkos::View<uintptr_t*>("FS LU outputs",4*batch);
#endif
 }
 void initBoundary(const commonstruct& c,const meshstruct& mesh){boundary.init(c,mesh);}
 void clear(){boundary.clear();hf=View();ff=View();work=View();vy=View();ty=View();vecin=View();vecout=View();thermal=View();asm_status={};pivots={};info={};aPointers={};cPointers={};}
 size_t bytes()const{return boundary.bytes()+8*(hf.extent(0)+ff.extent(0)+work.extent(0)+vy.extent(0)+ty.extent(0)+vecin.extent(0)+vecout.extent(0)+thermal.extent(0))+sizeof(int)*(asm_status.extent(0)+pivots.extent(0)+info.extent(0))+sizeof(uintptr_t)*(aPointers.extent(0)+cPointers.extent(0));}
};

#include "cached_inverse.hpp"
inline int Hstride(Context& ctx){const auto d=exasim::composite::dimensions(ctx.dimension);return ctx.enabled?d.HS:d.M*d.M;}
inline int Fstride(Context& ctx){const auto d=exasim::composite::dimensions(ctx.dimension);return ctx.enabled?d.FS:d.N*d.M;}
inline int ASMstride(Context& ctx){return Hstride(ctx);}
inline int ASMstorageSlots(Context& ctx){return ctx.asm_fp32?Hstride(ctx)/2:Hstride(ctx);}
inline bool fullLocalEnabled(){return false;}
// Assembly destinations for the current batch. Native H/F use the supplied
// global full-matrix offset; composite hf/ff hold only one batch, starting at
// zero. uEquationHDG assembles and condenses each batch before reusing them.
// condense() persists the results separately at res.H + e1*HS and res.F + e1*FS;
// applying this native offset to hf/ff would overrun the batch workspace.
template<class S> inline S* H(Context* ctx,S* p,size_t offset){if constexpr(std::is_same_v<S,double>) { if(ctx) return ctx->hf.data(); } return p+offset;}
template<class S> inline S* F(Context* ctx,S* p,size_t offset){if constexpr(std::is_same_v<S,double>) { if(ctx) return ctx->ff.data(); } return p+offset;}
extern "C" void dgetrs_(const char*,const int*,const int*,double*,const int*,const int*,double*,const int*,int*);
#include "kernels.hpp"
#include "core_hdg.hpp"
template<class... Args> inline void freeze(Context& ctx, Args&&... args){
 EXASIM_PROFILE_SCOPE("hdg.freeze_basis");
 if(ctx.dimension==2)Kernels<2>::freeze(ctx,std::forward<Args>(args)...);
 else Kernels<3>::freeze(ctx,std::forward<Args>(args)...);
}
template<class... Args> inline void clearBatch(Context& ctx, Args&&... args){
 if(ctx.dimension==2)Kernels<2>::clearBatch(ctx,std::forward<Args>(args)...);
 else Kernels<3>::clearBatch(ctx,std::forward<Args>(args)...);
}
template<class... Args> inline void condense(Context& ctx, Args&&... args){
 if(ctx.dimension==2)Kernels<2>::condense(ctx,std::forward<Args>(args)...);
 else Kernels<3>::condense(ctx,std::forward<Args>(args)...);
}
inline void multiply(Context& ctx,double* out,double* matrix,const double* in,int e1,int count,meshstruct& mesh,commonstruct& c,int backend,bool isASM=false){
 EXASIM_PROFILE_SCOPE(isASM?"hdg.local_asm_apply":"hdg.local_operator");
 if(ctx.dimension==2)CoreHDG<2>::multiply(ctx,out,matrix,in,e1,count,mesh,c,backend,isASM);
 else CoreHDG<3>::multiply(ctx,out,matrix,in,e1,count,mesh,c,backend,isASM);
}
inline void recover(Context& ctx,double* w,double* f,const double* ve,meshstruct& mesh,commonstruct& c,int backend){
 EXASIM_PROFILE_SCOPE("hdg.local_recovery");
 if(ctx.dimension==2)CoreHDG<2>::recover(ctx,w,f,ve,mesh,c,backend);
 else CoreHDG<3>::recover(ctx,w,f,ve,mesh,c,backend);
}
template<class... Args> inline void schwarz(Context& ctx, Args&&... args){
 if(ctx.dimension==2)Kernels<2>::schwarz(ctx,std::forward<Args>(args)...);
 else Kernels<3>::schwarz(ctx,std::forward<Args>(args)...);
}
// Dimension-specific implementation helpers stay on Kernels<ND>; only the
// context-dispatched operations above are exposed at namespace scope.

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
 template<class C,class M> void initBoundary(const C&,const M&){throw std::runtime_error("unsupported composite scalar");}
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
