// Included inside fieldsplit. Host metadata is immutable during Context lifetime.
// Selected rows are owned by the element owning that physical boundary face.
struct BoundaryRows {
 struct Range { int e1,end,first,count; };
 struct Selection { std::vector<Range> ranges; int faces=0; };
 View C,delta,rhs;
 Kokkos::View<int*> elements,faces,faceToRow;
 std::vector<Range> ranges;
 bool active=false;

 template<class Common> static Selection selection(const Common& c) {
  Selection s;
  if(!c.compositeConfig.enabled || c.compositeConfig.boundaryCodes.empty())return s;
  const int blocks=c.meshsizes.nbe1,maxbc=c.meshsizes.maxnbc;
  if(blocks<0 || maxbc<0 || blocks>c.meshsizes.nbe ||
     (blocks && !c.eblks) || (blocks && maxbc && !c.nboufaces))
   throw std::invalid_argument("compositebc: missing boundary block metadata");
  const auto codes=exasim::composite::boundaryCodes(c.compositeConfig.boundaryCodes);
  int previous=0;
  for(int j=0;j<blocks;++j) {
   const int e1=c.eblks[3*j]-1,end=c.eblks[3*j+1];
   if(e1!=previous || end<=e1 || end>c.meshsizes.ne1 || end-e1>c.meshsizes.neb)
    throw std::invalid_argument("compositebc: invalid owned element blocks");
   int n=0;
   for(int code:codes)if(code<=maxbc) {
    const size_t off=size_t(j)*maxbc+code-1;
    const int start=c.nboufaces[off],stop=c.nboufaces[off+1];
    if(start<0 || stop<start)throw std::invalid_argument("compositebc: invalid boundary prefix");
    n=checkedCount(uint64_t(n)+uint64_t(stop-start),"selected faces in batch");
   }
   const auto d=exasim::composite::dimensions(c.grid.nd);
   checkedCount(uint64_t(n)*d.PF*d.NC*d.M,"selected boundary batch rows");
   s.ranges.push_back({e1,end,s.faces,n});
   s.faces=checkedCount(uint64_t(s.faces)+n,"selected owned faces");previous=end;
  }
  if(previous!=c.meshsizes.ne1)throw std::invalid_argument("compositebc: incomplete owned blocks");
  const auto d=exasim::composite::dimensions(c.grid.nd);
  checkedCount(uint64_t(s.faces)*d.PF*d.NC,"selected boundary product rows");
  return s;
 }
 template<class Common> static size_t plannedBytes(const Common& c) {
  const auto s=selection(c);if(!s.faces)return 0;
  const auto d=exasim::composite::dimensions(c.grid.nd);
  const uint64_t perFace=uint64_t(d.PF)*d.NC*(d.N+d.M+1)*sizeof(double)+2*sizeof(int);
  if(uint64_t(s.faces)>uint64_t(std::numeric_limits<ptrdiff_t>::max())/perFace)
   throw std::overflow_error("compositebc: boundary storage exceeds pointer range");
  return size_t(s.faces)*perFace + size_t(c.meshsizes.ne1)*(2*d.nd)*sizeof(int);
 }
 void init(const commonstruct& c,const meshstruct& mesh) {
  if(!c.compositeConfig.enabled || c.compositeConfig.boundaryCodes.empty())return;
  const auto plan=selection(c);ranges=plan.ranges;active=true;
  const auto d=exasim::composite::dimensions(c.grid.nd);
  const auto codes=exasim::composite::boundaryCodes(c.compositeConfig.boundaryCodes);
  // Validate small maps on the host before allocating the large dense row data.
  std::vector<int> es(plan.faces),fs(plan.faces),buf;
  for(size_t j=0;j<ranges.size();++j) {
   int dest=ranges[j].first;
   for(int code:codes)if(code<=c.meshsizes.maxnbc) {
    const size_t off=j*size_t(c.meshsizes.maxnbc)+code-1;
    int start=c.nboufaces[off],stop=c.nboufaces[off+1];
    if(stop>mesh.szboufaces || (stop>start && !mesh.boufaces))
     throw std::invalid_argument("compositebc: boundary map exceeds allocation");
    buf.resize(stop-start);
    if(!buf.empty())TemplateCopytoHost(buf.data(),mesh.boufaces+start,stop-start,c.backend);
    for(int z:buf) {
     if(z<0 || z/(2*d.nd)>=ranges[j].end-ranges[j].e1)
      throw std::invalid_argument("compositebc: face outside its owned batch");
     es[dest]=ranges[j].e1+z/(2*d.nd);fs[dest]=z%(2*d.nd);++dest;
    }
   }
  }
  std::vector<uint64_t> unique;
  for(int i=0;i<plan.faces;++i)unique.push_back(uint64_t(es[i])*(2*d.nd)+fs[i]);
  std::sort(unique.begin(),unique.end());
  if(std::adjacent_find(unique.begin(),unique.end())!=unique.end())
   throw std::invalid_argument("compositebc: duplicate owned boundary face");
  if(plan.faces) {
   const size_t rows=size_t(plan.faces)*d.PF*d.NC;
   C=View("COMPOSITE boundary C rows",rows*d.N);
   delta=View("COMPOSITE boundary Schur correction",rows*d.M);
   rhs=View("COMPOSITE boundary uncondensed RHS",rows);
   std::vector<int> lookup(size_t(c.meshsizes.ne1)*(2*d.nd),-1);
   for(int p=0;p<plan.faces;++p)lookup[size_t(es[p])*(2*d.nd)+fs[p]]=p;
   faceToRow=Kokkos::View<int*>("COMPOSITE boundary face row lookup",lookup.size());
   TemplateCopytoDevice(faceToRow.data(),lookup.data(),int(lookup.size()),c.backend);
   elements=Kokkos::View<int*>("COMPOSITE boundary elements",plan.faces);
   faces=Kokkos::View<int*>("COMPOSITE boundary faces",plan.faces);
   TemplateCopytoDevice(elements.data(),es.data(),plan.faces,c.backend);
   TemplateCopytoDevice(faces.data(),fs.data(),plan.faces,c.backend);
  }
  printf("COMPOSITE_BOUNDARY_ROWS rank=%d owned_faces=%d additional_bytes=%zu codes=",c.mpiRank,plan.faces,bytes());
  for(int code:codes)printf("%d,",code);
  printf("\n");
 }
 std::pair<int,int> batch(int e1,int count)const {
  auto it=std::lower_bound(ranges.begin(),ranges.end(),e1,[](const Range& r,int e){return r.e1<e;});
  if(it==ranges.end() || it->e1!=e1 || it->end-e1!=count)
   throw std::invalid_argument("compositebc: condensation batch contract mismatch");
  return {it->first,it->count};
 }
 size_t bytes()const {
  return sizeof(double)*(C.extent(0)+delta.extent(0)+rhs.extent(0))+
         sizeof(int)*(elements.extent(0)+faces.extent(0)+faceToRow.extent(0));
 }
 void clear(){C={};delta={};rhs={};elements={};faces={};faceToRow={};ranges.clear();active=false;}
};
