// Included inside Kernels<ND>; geometry constants belong to that specialization.
// Same L J T contraction as extractGeneric, with only the nonzero basis entries.
template<bool RT,bool CT>KOKKOS_INLINE_FUNCTION static double columnAction(const double* j,int rp,int cp,int old,int c,const double* y){
 constexpr int nr=RT?M:N;int r=RT?NC*rp+old:rp+P*old;
 if(c==0){double v=0.;for(int s=0;s<5;s++)v+=j[r+nr*(CT?NC*cp+s:cp+P*s)]*y[s];return v;}
 if(c<FLOW)return j[r+nr*(CT?NC*cp+c+4:cp+P*(c+4))];
 int s=species(c-FLOW),cs=CT?NC*cp+s:cp+P*s,cn=CT?NC*cp+3:cp+3*P,ce=CT?NC*cp+NC-1:cp+(NC-1)*P;
 return j[r+nr*cs]-j[r+nr*cn]+y[5+c-FLOW]*j[r+nr*ce];
}
template<bool RT,bool CT>static void extractSparse(Context& ctx,double* out,const double* in,int field,int e1,int count,const meshstruct& mesh){
 constexpr int nr=RT?M:N,nc=CT?M:N;const int comps=field?4:FLOW,base=field?FLOW:0,a=(RT?Q:P)*comps,b=(CT?Q:P)*comps;
 auto vy=ctx.vy,ty=ctx.ty;auto ec=mesh.elemcon;
 Kokkos::parallel_for("COMPOSITE sparse transform",a*b*count,KOKKOS_LAMBDA(int k){int r=k%a,col=(k/a)%b,e=k/(a*b),rp=RT?r/comps:r%P,rc=base+(RT?r%comps:r/P),cp=CT?col/comps:col%P,cc=base+(CT?col%comps:col/P);
  const double* yr=RT?ty.data()+9*ec[(e1+e)*Q+rp]:vy.data()+9*((e1+e)*P+rp);const double* yc=CT?ty.data()+9*ec[(e1+e)*Q+cp]:vy.data()+9*((e1+e)*P+cp);const double* j=in+size_t(e)*nr*nc;double value=0.;
  if(rc>=1&&rc< FLOW-1)value=columnAction<RT,CT>(j,rp,cp,rc+4,cc,yc);
  else{double rows[5],sum=0.;for(int s=0;s<5;s++){rows[s]=columnAction<RT,CT>(j,rp,cp,s,cc,yc);sum+=rows[s];}
   if(rc==0)value=sum;
   else if(rc>=FLOW)value=rows[species(rc-FLOW)]-yr[species(rc-FLOW)]*sum;
   else{value=columnAction<RT,CT>(j,rp,cp,NC-1,cc,yc);for(int s=0;s<4;s++)value+=yr[5+s]*(yr[species(s)]*sum-rows[species(s)]);}}
  out[k]=value;});
}
static void extract(Context& ctx,double* out,const double* in,bool rt,bool ct,int field,int e1,int count,const meshstruct& mesh){
 if(rt){if(ct)extractSparse<true,true>(ctx,out,in,field,e1,count,mesh);else extractSparse<true,false>(ctx,out,in,field,e1,count,mesh);}
 else{if(ct)extractSparse<false,true>(ctx,out,in,field,e1,count,mesh);else extractSparse<false,false>(ctx,out,in,field,e1,count,mesh);}
}
