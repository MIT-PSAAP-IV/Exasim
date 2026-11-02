// Included inside Kernels<ND>; all blocks column-major, bases frozen for this solve.
// The selected full rows use the compact volume elimination, not native D inverse.
static constexpr int BOUNDARY_ROWS=PF*NC;
KOKKOS_INLINE_FUNCTION static int boundaryTraceIndex(int node,int component){return packedIndex(component,node,true);}
KOKKOS_INLINE_FUNCTION static double boundaryF(const double* f,int row,int col){
 if(row<AF && col<BF)return f[row+AF*col];
 if(row>=AF && col>=BF && (row-AF)/P==(col-BF)%4)
  return f[AF*BF+P*Q*((row-AF)/P)+(row-AF)%P+P*((col-BF)/4)];
 return 0.;
}
KOKKOS_INLINE_FUNCTION static double boundaryH(const double* h,int row,int col){
 if(row<BF && col<BF)return h[row+BF*col];
 if(row>=BF && col>=BF && (row-BF)%4==(col-BF)%4)
  return h[BF*BF+Q*Q*((row-BF)%4)+(row-BF)/4+Q*((col-BF)/4)];
 return 0.;
}
static void boundaryPrepare(Context& ctx,const double* C,const double* H,const double* Rh,const meshstruct& mesh,const commonstruct& c,int e1,int count){
 auto& s=ctx.boundary;if(!s.active)return;auto range=s.batch(e1,count);const int first=range.first,nf=range.second;
 auto es=s.elements,fs=s.faces;auto outC=s.C,outH=s.delta,outR=s.rhs;auto vy=ctx.vy,ty=ctx.ty;auto ec=mesh.elemcon;
 Kokkos::parallel_for("COMPOSITE boundary full transformed C",nf*BOUNDARY_ROWS*N,KOKKOS_LAMBDA(int k){
  int r=k%BOUNDARY_ROWS,col=(k/BOUNDARY_ROWS)%N,p=first+k/(BOUNDARY_ROWS*N),e=es[p]-e1,node=PF*fs[p]+r/NC,rc=r%NC;
  int cn=col<AF?col%P:(col-AF)%P,cc=col<AF?col/P:FLOW+(col-AF)/P;
  const double* yr=ty.data()+9*ec[Q*es[p]+node];const double* yc=vy.data()+9*(P*es[p]+cn);
  double v=0.;for(int i=0;i<NC;i++){double l=L(rc,i,yr);if(l!=0.)for(int j=0;j<NC;j++){double t=T(j,cc,yc);if(t!=0.)v+=l*C[size_t(e)*M*N+NC*node+i+M*(cn+P*j)]*t;}}
  outC[size_t(p)*BOUNDARY_ROWS*N+r+BOUNDARY_ROWS*col]=v;
 });
 Kokkos::parallel_for("COMPOSITE boundary full transformed H",nf*BOUNDARY_ROWS*M,KOKKOS_LAMBDA(int k){
  int r=k%BOUNDARY_ROWS,col=(k/BOUNDARY_ROWS)%M,p=first+k/(BOUNDARY_ROWS*M),e=es[p]-e1,node=PF*fs[p]+r/NC,rc=r%NC;
  int cn=col<BF?col/FLOW:(col-BF)/4,cc=col<BF?col%FLOW:FLOW+(col-BF)%4;
  const double* yr=ty.data()+9*ec[Q*es[p]+node];const double* yc=ty.data()+9*ec[Q*es[p]+cn];
  double v=0.;for(int i=0;i<NC;i++){double l=L(rc,i,yr);if(l!=0.)for(int j=0;j<NC;j++){double t=T(j,cc,yc);if(t!=0.)v+=l*H[size_t(e)*M*M+NC*node+i+M*(NC*cn+j)]*t;}}
  outH[size_t(p)*BOUNDARY_ROWS*M+r+BOUNDARY_ROWS*col]=v;
 });
 Kokkos::parallel_for("COMPOSITE boundary original transformed RHS",nf*BOUNDARY_ROWS,KOKKOS_LAMBDA(int k){
  int r=k%BOUNDARY_ROWS,p=first+k/BOUNDARY_ROWS,e=es[p]-e1,node=PF*fs[p]+r/NC,rc=r%NC;const double* yr=ty.data()+9*ec[Q*es[p]+node];
  double v=0.;for(int j=0;j<NC;j++)v+=L(rc,j,yr)*Rh[Q*count*j+Q*e+node];outR[size_t(p)*BOUNDARY_ROWS+r]=v;
 });
}
static void boundaryFinish(Context& ctx,const double* F,const double* H,const double* z,double* rh,const commonstruct& c,int e1,int count){
 auto& s=ctx.boundary;if(!s.active)return;auto range=s.batch(e1,count);const int first=range.first,nf=range.second;
 auto es=s.elements,fs=s.faces;auto fullC=s.C,delta=s.delta,rhs=s.rhs;
 Kokkos::parallel_for("COMPOSITE boundary complete Schur row correction",nf*BOUNDARY_ROWS*M,KOKKOS_LAMBDA(int k){
  int r=k%BOUNDARY_ROWS,col=(k/BOUNDARY_ROWS)%M,p=first+k/(BOUNDARY_ROWS*M),e=es[p]-e1,row=boundaryTraceIndex(PF*fs[p]+r/NC,r%NC);
  double v=delta[size_t(p)*BOUNDARY_ROWS*M+r+BOUNDARY_ROWS*col];for(int j=0;j<N;j++)v+=fullC[size_t(p)*BOUNDARY_ROWS*N+r+BOUNDARY_ROWS*j]*boundaryF(F+size_t(e)*FS,j,col);
  delta[size_t(p)*BOUNDARY_ROWS*M+r+BOUNDARY_ROWS*col]=v-boundaryH(H+size_t(e)*HS,row,col);
 });
 Kokkos::parallel_for("COMPOSITE boundary consistent condensed RHS",nf*BOUNDARY_ROWS,KOKKOS_LAMBDA(int k){
  int r=k%BOUNDARY_ROWS,p=first+k/BOUNDARY_ROWS,e=es[p]-e1,row=boundaryTraceIndex(PF*fs[p]+r/NC,r%NC);double v=rhs[size_t(p)*BOUNDARY_ROWS+r];
  for(int j=0;j<N;j++)v-=fullC[size_t(p)*BOUNDARY_ROWS*N+r+BOUNDARY_ROWS*j]*z[size_t(e)*N+j];rh[size_t(e)*M+row]=v;
 });
}
static void boundaryProduct(Context& ctx,double* y,const double* x,int e1,int count){
 auto& s=ctx.boundary;if(!s.active)return;
 if(e1<0 || count<0 || e1>ctx.owned || count>ctx.owned-e1)throw std::runtime_error("COMPOSITE boundary requires owned matvec range");
 auto es=s.elements,fs=s.faces;auto delta=s.delta;int nf=es.extent(0);
 Kokkos::parallel_for("COMPOSITE boundary operator rows",nf*BOUNDARY_ROWS,KOKKOS_LAMBDA(int k){
  int r=k%BOUNDARY_ROWS,p=k/BOUNDARY_ROWS,e=es[p]-e1;if(e<0 || e>=count)return;int row=boundaryTraceIndex(PF*fs[p]+r/NC,r%NC);
  double v=0.;for(int j=0;j<M;j++)v+=delta[size_t(p)*BOUNDARY_ROWS*M+r+BOUNDARY_ROWS*j]*x[size_t(e)*M+j];y[size_t(e)*M+row]+=v;
 });
}
