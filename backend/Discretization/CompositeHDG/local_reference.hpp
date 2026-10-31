// Independent host LU solve reference for the first actual assembled element.
// Included inside fieldsplit; enabled only for diagnostic runs.
extern "C" void dgetrs_(const char*,const int*,const int*,double*,const int*,const int*,double*,const int*,int*);
struct LocalReference {
 int a,b,field;std::vector<double> aa,ee,gg,ss,rr,hh;
 LocalReference(int av,int bv,int f,double* A,double* E,double* G,double* S,double* rv,double* rh,int backend):a(av),b(bv),field(f),aa(a*a),ee(a*b),gg(b*a),ss(b*b),rr(a),hh(b){
  TemplateCopytoHost(aa.data(),A,aa.size(),backend);TemplateCopytoHost(ee.data(),E,ee.size(),backend);TemplateCopytoHost(gg.data(),G,gg.size(),backend);TemplateCopytoHost(ss.data(),S,ss.size(),backend);TemplateCopytoHost(rr.data(),rv,rr.size(),backend);TemplateCopytoHost(hh.data(),rh,hh.size(),backend);
 }
 static double relativeError(const std::vector<double>& got,const std::vector<double>& ref){double num=0.,den=0.;for(size_t k=0;k<ref.size();k++){double e=got[k]-ref[k];num+=e*e;den+=ref[k]*ref[k];}return std::sqrt(num/std::max(den,1.e-300));}
 void check(double* Fout,double* zv,double* S,double* rh,double* Hcompact,double* Fcompact,int backend){
  auto lu=aa;std::vector<int> piv(a);int status=0;dgetrf_(&a,&a,lu.data(),&a,piv.data(),&status);if(status)throw std::runtime_error("Host reference LU singular");
  std::vector<double> rhs(a*(b+1));for(int k=0;k<a*b;k++)rhs[k]=-ee[k];for(int k=0;k<a;k++)rhs[a*b+k]=rr[k];int nrhs=b+1;char op='N';dgetrs_(&op,&a,&nrhs,lu.data(),&a,piv.data(),rhs.data(),&a,&status);if(status)throw std::runtime_error("Host reference solve failed");
  auto sh=ss,rv=hh;for(int j=0;j<b;j++)for(int k=0;k<a;k++)for(int i=0;i<b;i++)sh[i+b*j]+=gg[i+b*k]*rhs[k+a*j];
  for(int k=0;k<a;k++)for(int i=0;i<b;i++)rv[i]-=gg[i+b*k]*rhs[a*b+k];
  std::vector<double> gf(a*b),gz(a),gs(b*b),gr(b),hc(HS),fc(FS);
  TemplateCopytoHost(gf.data(),Fout,gf.size(),backend);TemplateCopytoHost(gz.data(),zv,gz.size(),backend);TemplateCopytoHost(gs.data(),S,gs.size(),backend);TemplateCopytoHost(gr.data(),rh,gr.size(),backend);TemplateCopytoHost(hc.data(),Hcompact,hc.size(),backend);TemplateCopytoHost(fc.data(),Fcompact,fc.size(),backend);
  std::vector<double> rf(rhs.begin(),rhs.begin()+a*b),rz(rhs.begin()+a*b,rhs.end()),packedH,packedF,expectedH,expectedF;
  if(!field){packedH.assign(hc.begin(),hc.begin()+BF*BF);expectedH=sh;packedF.assign(fc.begin(),fc.begin()+AF*BF);expectedF=rf;}
  else for(int s=0;s<4;s++)for(int j=0;j<Q;j++){
   for(int i=0;i<Q;i++){packedH.push_back(hc[BF*BF+s*Q*Q+i+Q*j]);expectedH.push_back(sh[4*i+s+BS*(4*j+s)]);}
   for(int i=0;i<P;i++){packedF.push_back(fc[AF*BF+s*P*Q+i+P*j]);expectedF.push_back(rf[i+P*s+AS*(4*j+s)]);}
  }
  double er[6]={relativeError(gf,rf),relativeError(gz,rz),relativeError(gs,sh),relativeError(gr,rv),relativeError(packedH,expectedH),relativeError(packedF,expectedF)};
  printf("FS3_LOCAL_REFERENCE field=%d F=%.17g volume_rhs=%.17g S=%.17g trace_rhs=%.17g compact_H=%.17g compact_F=%.17g\n",field,er[0],er[1],er[2],er[3],er[4],er[5]);
  for(double v:er)if(!exasim::composite::finiteValue(v)||v>1.e-7)throw std::runtime_error("3D local condensation disagrees with independent host LU solve");
 }
};
