// Same L J T contraction as extractGeneric, with only the nonzero basis entries.
template<bool RT,bool CT>KOKKOS_INLINE_FUNCTION double columnAction(const double* j,int rp,int cp,int old,int c,const double* y){
 constexpr int nr=RT?M:N;int r=RT?9*rp+old:rp+P*old;
 if(c==0){double v=0.;for(int s=0;s<5;s++)v+=j[r+nr*(CT?9*cp+s:cp+P*s)]*y[s];return v;}
 if(c<5)return j[r+nr*(CT?9*cp+c+4:cp+P*(c+4))];
 int s=species(c-5),cs=CT?9*cp+s:cp+P*s,cn=CT?9*cp+3:cp+3*P,ce=CT?9*cp+8:cp+8*P;
 return j[r+nr*cs]-j[r+nr*cn]+y[c]*j[r+nr*ce];
}
template<bool RT,bool CT>inline void extractSparse(Context& ctx,double* out,const double* in,int field,int e1,int count,const meshstruct& mesh){
 constexpr int nr=RT?M:N,nc=CT?M:N;const int comps=field?4:5,base=field?5:0,a=(RT?Q:P)*comps,b=(CT?Q:P)*comps;
 auto vy=ctx.vy,ty=ctx.ty;auto ec=mesh.elemcon;
 Kokkos::parallel_for("FS3 sparse transform",a*b*count,KOKKOS_LAMBDA(int k){int r=k%a,col=(k/a)%b,e=k/(a*b),rp=RT?r/comps:r%P,rc=base+(RT?r%comps:r/P),cp=CT?col/comps:col%P,cc=base+(CT?col%comps:col/P);
  const double* yr=RT?ty.data()+9*ec[(e1+e)*Q+rp]:vy.data()+9*((e1+e)*P+rp);const double* yc=CT?ty.data()+9*ec[(e1+e)*Q+cp]:vy.data()+9*((e1+e)*P+cp);const double* j=in+size_t(e)*nr*nc;double value=0.;
  if(rc>=1&&rc<=3)value=columnAction<RT,CT>(j,rp,cp,rc+4,cc,yc);
  else{double rows[5],sum=0.;for(int s=0;s<5;s++){rows[s]=columnAction<RT,CT>(j,rp,cp,s,cc,yc);sum+=rows[s];}
   if(rc==0)value=sum;
   else if(rc>=5)value=rows[species(rc-5)]-yr[species(rc-5)]*sum;
   else{value=columnAction<RT,CT>(j,rp,cp,8,cc,yc);for(int s=0;s<4;s++)value+=yr[5+s]*(yr[species(s)]*sum-rows[species(s)]);}}
  out[k]=value;});
}
inline void extract(Context& ctx,double* out,const double* in,bool rt,bool ct,int field,int e1,int count,const meshstruct& mesh){
 if(rt){if(ct)extractSparse<true,true>(ctx,out,in,field,e1,count,mesh);else extractSparse<true,false>(ctx,out,in,field,e1,count,mesh);}
 else{if(ct)extractSparse<false,true>(ctx,out,in,field,e1,count,mesh);else extractSparse<false,false>(ctx,out,in,field,e1,count,mesh);}
}
