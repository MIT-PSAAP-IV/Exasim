#ifdef HAVE_MPI
#include <mpi.h>
MPI_Comm EXASIM_COMM_WORLD = MPI_COMM_NULL;
MPI_Comm EXASIM_COMM_LOCAL = MPI_COMM_NULL;
#endif

#include <Kokkos_Core.hpp>
#include <fstream>
#include <iostream>
#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Discretization/nonmatchingpreprocessgeometry.hpp"
#include "../../backend/Discretization/nonmatchingpreprocessmpi.hpp"
#include "../../backend/Discretization/nonmatchingpreprocess.hpp"

#include <cmath>
#include <atomic>
#include <cstdlib>
#include <new>
#include <cstdio>
#include <string>
#include <vector>

namespace {
std::atomic<long long> trackedAllocations{0};
std::atomic<bool> trackAllocations{false};
}

void* operator new(std::size_t size)
{
    if(trackAllocations.load(std::memory_order_relaxed))
        trackedAllocations.fetch_add(1,std::memory_order_relaxed);
    if(void* pointer=std::malloc(size))return pointer;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer,std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer,std::size_t) noexcept { std::free(pointer); }

namespace {
using namespace exasim::nonmatching;
namespace pp = exasim::nonmatching::preprocess;
int failures = 0;
double maxProjectionError = 0, maxBoxViolation = 0, maxSurfaceGradient = 0;

void check(bool value, const char* message)
{
    if (!value) { std::printf("FAIL %s\n", message); ++failures; }
}

std::string master_path()
{
    const std::string source = __FILE__;
    const std::string marker = "/tests/unit/nonmatching_preprocess.cpp";
    return source.substr(0, source.rfind(marker)) + "/backend/Preprocessing/masternodes.bin";
}

struct Master { Int npf = 0; std::vector<dstype> xpf; };

Master surface_master(int elemtype, int p)
{
    Master out;
    std::vector<dstype> xpe;
    std::vector<int> telem, tface, perm;
    masternodes(xpe, telem, out.xpf, tface, perm, p, 3, elemtype, master_path());
    out.npf = static_cast<Int>(out.xpf.size()/2);
    return out;
}

template <class T, class I>
struct FaceSet {
    I nd = 0, npf = 0;
    std::vector<I> rank, elem, lface, global;
    std::vector<T> nodes;
    I size() const { return static_cast<I>(rank.size()); }
};

void sample_counts()
{
    for (int p=1;p<=4;++p) {
        std::vector<dstype> line(static_cast<std::size_t>(p+1));
        for (int i=0;i<=p;++i) line[i]=static_cast<dstype>(i)/p;
        const auto lm=pp::prepare_face_master<dstype,Int>(line.data(),2,p+1,0,p);
        check(lm.samples()==2*p+1,"line L=2p sample count");
        for(int elemtype=0;elemtype<=1;++elemtype){
            const Master sm=surface_master(elemtype,p);
            const auto m=pp::prepare_face_master<dstype,Int>(sm.xpf.data(),3,sm.npf,elemtype,p);
            const Int expected=elemtype==0?(2*p+1)*(2*p+2)/2:(2*p+1)*(2*p+1);
            check(m.samples()==expected,"surface L=2p sample count");
            check(m.edges()==(elemtype==0?3:4),"surface edge count");
        }
    }
}

void line_projection()
{
    const int p=2; const dstype xpf[3]={0,0.5,1};
    FaceSet<dstype,Int> fs; fs.nd=2;fs.npf=3;fs.rank={0,0};fs.elem={0,1};
    fs.lface={0,0};fs.global={7,3};fs.nodes.resize(12);
    for(int face=0;face<2;++face) for(int a=0;a<3;++a){
        const dstype s=xpf[a]; fs.nodes[a+3*0+6*face]=s;
        fs.nodes[a+3*1+6*face]=(face==0?0.15*s*(1-s):0.35)+0.02*face*s;
    }
    const auto prepared=pp::prepare_faces<dstype,Int>(fs,xpf,0,p);
    const dstype point[2]={0.31,0.11}; pp::ProjectionWorkspace<dstype,Int> work;
    const auto result=pp::search_local_faces(prepared,point,1e-13,100,work);
    check(!result.failed&&result.face==0,"line multistart donor");
    const dstype expectedY=0.15*result.projection.xi[0]*(1-result.projection.xi[0]);
    const dstype distance=std::hypot(result.projection.xi[0]-point[0],expectedY-point[1]);
    maxProjectionError=std::max(maxProjectionError,std::abs(distance-result.projection.distance));
}

void multiple_minimum_tie()
{
    const int p=2;const dstype xpf[3]={0,0.5,1};
    FaceSet<dstype,Int> fs;fs.nd=2;fs.npf=3;fs.rank={0};fs.elem={0};
    fs.lface={0};fs.global={0};fs.nodes.resize(6);
    for(int a=0;a<3;++a){const dstype s=2*xpf[a]-1;fs.nodes[a]=s;fs.nodes[a+3]=s*s;}
    const auto prepared=pp::prepare_faces<dstype,Int>(fs,xpf,0,p);
    const dstype point[2]={0,1};pp::ProjectionWorkspace<dstype,Int> work;
    const auto result=pp::search_local_faces(prepared,point,1e-13,100,work);
    const dstype expected=(1-std::sqrt(0.5))/2;
    check(!result.failed&&result.face==0,"multiple-minimum projection");
    check(std::abs(result.projection.xi[0]-expected)<2e-12,
        "multiple-minimum lexicographic tie");
}

void separated_local_minima()
{
    const int p=4; const dstype xpf[5]={0,0.25,0.5,0.75,1};
    FaceSet<dstype,Int> fs;fs.nd=2;fs.npf=5;fs.rank={0};fs.elem={0};
    fs.lface={0};fs.global={0};fs.nodes.resize(10);
    for(int a=0;a<5;++a){const dstype s=2*xpf[a]-1;fs.nodes[a]=s;
        fs.nodes[a+5]=(s*s-0.4)*(s*s-0.4);}
    const auto prepared=pp::prepare_faces<dstype,Int>(fs,xpf,0,p);
    const dstype point[2]={0.18,0.02};pp::ProjectionWorkspace<dstype,Int> work;
    const auto result=pp::search_local_faces(prepared,point,1e-13,100,work);
    dstype best=std::numeric_limits<dstype>::max(),bestXi=0;
    for(int i=0;i<=200000;++i){const dstype t=static_cast<dstype>(i)/200000;
        const dstype s=2*t-1,y=(s*s-0.4)*(s*s-0.4);
        const dstype d=std::hypot(s-point[0],y-point[1]);
        if(d<best){best=d;bestXi=t;}}
    check(!result.failed&&result.face==0,"separated local minima converged");
    check(std::abs(result.projection.distance-best)<2e-8,
          "separated local minima global distance");
    check(std::abs(result.projection.xi[0]-bestXi)<2e-4,
          "separated local minima selected basin");
}

void surface_orientations()
{
    double reference[2]={-1,-1};
    for(int elemtype=0;elemtype<=1;++elemtype){
        const int p=4,norientation=elemtype==0?6:8;const Master m=surface_master(elemtype,p);
        for(int orientation=0;orientation<norientation;++orientation){
            FaceSet<dstype,Int> fs;fs.nd=3;fs.npf=m.npf;fs.rank={0};fs.elem={0};
            fs.lface={0};fs.global={0};fs.nodes.resize(static_cast<std::size_t>(3)*m.npf);
            for(Int a=0;a<m.npf;++a){const dstype u=m.xpf[a],v=m.xpf[a+m.npf];dstype x=0,y=0;
                if(elemtype==0){const dstype b[3]={1-u-v,u,v};
                    static const int perm[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
                    x=b[perm[orientation][1]];y=b[perm[orientation][2]];}
                else {switch(orientation){case 0:x=u;y=v;break;case 1:x=1-u;y=v;break;
                    case 2:x=u;y=1-v;break;case 3:x=1-u;y=1-v;break;
                    case 4:x=v;y=u;break;case 5:x=1-v;y=u;break;
                    case 6:x=v;y=1-u;break;default:x=1-v;y=1-u;break;}}
                fs.nodes[a]=x;fs.nodes[a+m.npf]=y;
                fs.nodes[a+2*m.npf]=0.09*x*(1-x)+0.04*y*(1-y);}
            const auto prepared=pp::prepare_faces<dstype,Int>(fs,m.xpf.data(),elemtype,p);
            const dstype query[3]={0.23,0.19,0.12};pp::ProjectionWorkspace<dstype,Int> work;
            const auto got=pp::search_local_faces(prepared,query,1e-13,100,work);
            check(!got.failed&&got.face==0,"surface orientation converged");
            if(reference[elemtype]<0)reference[elemtype]=got.projection.distance;
            check(std::abs(got.projection.distance-reference[elemtype])<3e-12,
                  "surface orientation invariant distance");
        }
    }
}

void repeated_search_allocations()
{
    const int p=4;const Master m=surface_master(1,p);
    FaceSet<dstype,Int> fs;fs.nd=3;fs.npf=m.npf;fs.rank={0,0,0};
    fs.elem={0,1,2};fs.lface={0,0,0};fs.global={0,1,2};
    fs.nodes.resize(static_cast<std::size_t>(3)*m.npf*3);
    for(int face=0;face<3;++face)for(Int a=0;a<m.npf;++a){
        const dstype u=m.xpf[a],v=m.xpf[a+m.npf],shift=0.4*face;
        const std::size_t offset=static_cast<std::size_t>(3)*m.npf*face;
        fs.nodes[offset+a]=u+shift;fs.nodes[offset+a+m.npf]=v;
        fs.nodes[offset+a+2*m.npf]=0.08*u*(1-u)+0.05*v*(1-v);}
    const auto prepared=pp::prepare_faces<dstype,Int>(fs,m.xpf.data(),1,p);
    const dstype query[3]={0.23,0.19,0.12};pp::ProjectionWorkspace<dstype,Int> work;
    (void)pp::search_local_faces(prepared,query,1e-13,100,work);
    trackedAllocations.store(0,std::memory_order_relaxed);
    trackAllocations.store(true,std::memory_order_relaxed);
    bool valid=true;
    for(int repeat=0;repeat<100;++repeat){const auto result=
        pp::search_local_faces(prepared,query,1e-13,100,work);
        valid=valid&&!result.failed&&result.face==0;}
    trackAllocations.store(false,std::memory_order_relaxed);
    check(valid,"repeated search result stable");
    check(trackedAllocations.load(std::memory_order_relaxed)==0,
          "zero allocations after projection workspace capacity is established");
}

void screening_boundaries()
{
    const int p=1;
    const dstype xpf[2]={0,1};
    FaceSet<dstype,Int> fs;
    fs.nd=2;fs.npf=2;fs.rank={0,0};fs.elem={0,1};
    fs.lface={0,0};fs.global={0,1};fs.nodes={0,1,0,0,0,1,1,1};
    auto prepared=pp::prepare_faces<dstype,Int>(fs,xpf,0,p);
    const dstype point[2]={0.5,0};
    const dstype margin=pp::face_margin(prepared,1,point);

    prepared.boxMin[1+2]=point[1]+margin;
    prepared.boxMax[1+2]=point[1]+margin;
    pp::ProjectionWorkspace<dstype,Int> equalityWork;
    const auto equality=pp::search_local_faces(prepared,point,1e-13,100,equalityWork);
    const dstype window=128*std::numeric_limits<dstype>::epsilon()*
        std::max({static_cast<dstype>(1),std::abs(equality.feasibleUpper),prepared.cmax[1],
            prepared.cmax[equality.face]});
    check(!equality.failed&&equality.projectedFaces==2&&
          equalityWork.faceLower[1]<=pp::up(equality.feasibleUpper+window),
          "screen face inside U plus tie window remains eligible");

    prepared.boxMin[1+2]=point[1]+margin+1e-6;
    prepared.boxMax[1+2]=point[1]+margin+1e-6;
    pp::ProjectionWorkspace<dstype,Int> pruningWork;
    const auto pruned=pp::search_local_faces(prepared,point,1e-13,100,pruningWork);
    check(!pruned.failed&&pruned.projectedFaces==1&&pruned.prunedFaces==1,
          "decreasing feasible upper prunes a later face");
}

dstype coordinate_ulp(dstype value)
{
    return std::nextafter(value,std::numeric_limits<dstype>::max())-value;
}

void projection_stopping_scale_scan()
{
    const int p=2;
    const dstype xpf[3]={0,0.5,1};
    const dstype translations[8]={0,1,10,100,300,1e3,1e4,1e5};
    const dstype sizes[2]={1,1e-3};
    const dstype relativeDistances[2]={0.03,0};
    for(dstype translation:translations)for(dstype size:sizes)
        for(dstype relativeDistance:relativeDistances){
            FaceSet<dstype,Int> fs;fs.nd=2;fs.npf=3;fs.rank={0};fs.elem={0};
            fs.lface={0};fs.global={0};fs.nodes.resize(6);
            const dstype distance=relativeDistance*size;
            for(int a=0;a<3;++a){fs.nodes[a]=translation+size*xpf[a];fs.nodes[a+3]=distance;}
            const dstype point[2]={translation+0.41*size,0};
            const auto prepared=pp::prepare_faces<dstype,Int>(fs,xpf,0,p);
            pp::ProjectionWorkspace<dstype,Int> work;
            const auto result=pp::search_local_faces(prepared,point,1e-13,100,work);
            const dstype xiBound=std::max<dstype>(5e-10,
                8*coordinate_ulp(translation)/size);
            check(!result.failed&&result.face==0,"translated line projection converged");
            if(!result.failed&&result.face==0){
                const dstype xiError=std::abs(result.projection.xi[0]-0.41);
                check(xiError<=xiBound,"translated line xi follows conditioned S4 bound");
                check(std::abs(result.projection.distance-std::abs(distance))<=
                      512*std::numeric_limits<dstype>::epsilon()*
                          std::max<dstype>(1,std::abs(translation)),
                      "translated line analytic distance");
            }
        }

    const Master m=surface_master(1,p);
    const dstype surfaceTranslations[3]={0,1e3,1e4};
    for(dstype translation:surfaceTranslations){
        FaceSet<dstype,Int> fs;fs.nd=3;fs.npf=m.npf;fs.rank={0};fs.elem={0};
        fs.lface={0};fs.global={0};fs.nodes.resize(static_cast<std::size_t>(3)*m.npf);
        for(Int a=0;a<m.npf;++a){
            fs.nodes[a]=translation+m.xpf[a];
            fs.nodes[a+m.npf]=m.xpf[a+m.npf];
            fs.nodes[a+2*m.npf]=0.03;
        }
        const dstype point[3]={translation+0.41,0.37,0};
        const auto prepared=pp::prepare_faces<dstype,Int>(fs,m.xpf.data(),1,p);
        pp::ProjectionWorkspace<dstype,Int> work;
        const auto result=pp::search_local_faces(prepared,point,1e-13,100,work);
        const dstype xiBound=std::max<dstype>(5e-10,8*coordinate_ulp(translation));
        check(!result.failed&&result.face==0,"translated surface projection converged");
        if(!result.failed&&result.face==0){
            const dstype xiError=std::max(std::abs(result.projection.xi[0]-0.41),
                std::abs(result.projection.xi[1]-0.37));
            check(xiError<=xiBound,"translated surface xi follows conditioned S4 bound");
            check(std::abs(result.projection.distance-0.03)<=
                  512*std::numeric_limits<dstype>::epsilon()*
                      std::max<dstype>(1,std::abs(translation)),
                  "translated surface analytic distance");
        }
    }

    const dstype aspect=1e-4;
    const dstype anisotropicTranslations[3]={0,1,1e3};
    for(int elemtype=0;elemtype<=1;++elemtype){
        const Master anisotropic=surface_master(elemtype,4);
        for(dstype translation:anisotropicTranslations){
            FaceSet<dstype,Int> fs;fs.nd=3;fs.npf=anisotropic.npf;fs.rank={0};fs.elem={0};
            fs.lface={0};fs.global={0};
            fs.nodes.resize(static_cast<std::size_t>(3)*anisotropic.npf);
            for(Int a=0;a<anisotropic.npf;++a){
                fs.nodes[a]=translation+anisotropic.xpf[a];
                fs.nodes[a+anisotropic.npf]=aspect*anisotropic.xpf[a+anisotropic.npf];
                fs.nodes[a+2*anisotropic.npf]=0.03*aspect;
            }
            const dstype reference[2]={elemtype==0?0.23:0.41,elemtype==0?0.19:0.37};
            const dstype point[3]={translation+reference[0],aspect*reference[1],0};
            const auto prepared=pp::prepare_faces<dstype,Int>(fs,anisotropic.xpf.data(),
                elemtype,4);
            pp::ProjectionWorkspace<dstype,Int> work;
            const auto result=pp::search_local_faces(prepared,point,1e-13,100,work);
            const dstype longBound=std::max<dstype>(5e-10,8*coordinate_ulp(translation));
            const dstype thinBound=std::max<dstype>(5e-10,
                8*coordinate_ulp(translation)/aspect);
            check(!result.failed&&result.face==0,"anisotropic surface projection converged");
            if(!result.failed&&result.face==0){
                check(std::abs(result.projection.xi[0]-reference[0])<=longBound,
                    "anisotropic surface long xi follows conditioned S4 bound");
                check(std::abs(result.projection.xi[1]-reference[1])<=thinBound,
                    "anisotropic surface thin xi follows conditioned S4 bound");
                check(std::abs(result.projection.distance-0.03*aspect)<=
                      512*std::numeric_limits<dstype>::epsilon()*
                          std::max<dstype>(1,std::abs(translation)),
                      "anisotropic surface analytic distance");
            }
        }
    }
}

void surface_cases()
{
    constexpr int grid=400;
    int cases=0;
    for(int elemtype=0;elemtype<=1;++elemtype) for(int p=1;p<=4;++p)
        for(const dstype amplitude:{dstype(0.08),dstype(0.3)}){
            const Master m=surface_master(elemtype,p);
            const dstype au=p==1?0:amplitude;
            const dstype av=p==1?0:0.625*amplitude;
            FaceSet<dstype,Int> fs;fs.nd=3;fs.npf=m.npf;fs.rank={0};fs.elem={0};
            fs.lface={0};fs.global={0};fs.nodes.resize(static_cast<std::size_t>(3)*m.npf);
            for(Int a=0;a<m.npf;++a){const dstype u=m.xpf[a],v=m.xpf[a+m.npf];
                fs.nodes[a]=u;fs.nodes[a+m.npf]=v;
                fs.nodes[a+2*m.npf]=au*u*(1-u)+av*v*(1-v);}
            const auto prepared=pp::prepare_faces<dstype,Int>(fs,m.xpf.data(),elemtype,p);
            for(Int a=0;a<m.npf;++a) for(int d=0;d<3;++d){const dstype x=fs.nodes[a+m.npf*d];
                maxBoxViolation=std::max(maxBoxViolation,
                    std::max(prepared.boxMin[d]-x,x-prepared.boxMax[d]));}

            const dstype triangleReference[6][2]={{0.2,0.2},{0.4,0.2},{0.2,0.4},
                {0.6,0.2},{0.1,0.1},{0.3,0.3}};
            const dstype quadReference[6][2]={{0.2,0.2},{0.4,0.2},{0.2,0.4},
                {0.6,0.2},{0.8,0.7},{0.3,0.8}};
            const dstype (*reference)[2]=elemtype==0?triangleReference:quadReference;
            for(int iq=0;iq<6;++iq){
                const dstype ur=reference[iq][0],vr=reference[iq][1];
                const dstype zr=au*ur*(1-ur)+av*vr*(1-vr);
                const dstype zu=au*(1-2*ur),zv=av*(1-2*vr);
                const dstype normalScale=1/std::sqrt(1+zu*zu+zv*zv);
                const dstype offset=(iq%2==0?dstype(0.025):dstype(-0.025));
                const dstype query[3]={ur-offset*zu*normalScale,
                    vr-offset*zv*normalScale,zr+offset*normalScale};
                pp::ProjectionWorkspace<dstype,Int> work;
                const auto got=pp::search_local_faces(prepared,query,1e-13,100,work);
                ++cases;
                if(got.failed||got.face!=0){
                    std::printf("surface projection failed: elemtype %d p %d amplitude %.3e query %d\n",
                        elemtype,p,amplitude,iq);
                    check(false,"surface multistart donor");
                    continue;
                }

                dstype gridDistance2=std::numeric_limits<dstype>::max();
                for(int i=0;i<=grid;++i)for(int j=0;j<=grid;++j){
                    if(elemtype==0&&i+j>grid)continue;
                    const dstype u=static_cast<dstype>(i)/grid;
                    const dstype v=static_cast<dstype>(j)/grid;
                    const dstype z=au*u*(1-u)+av*v*(1-v);
                    const dstype dx=u-query[0],dy=v-query[1],dz=z-query[2];
                    gridDistance2=std::min(gridDistance2,dx*dx+dy*dy+dz*dz);
                }
                const dstype gridDistance=std::sqrt(gridDistance2);
                const dstype scale=std::max({dstype(1),std::abs(query[0]),std::abs(query[1]),
                    std::abs(query[2]),prepared.cmax[0]});
                const dstype distanceAllowance=4096*std::numeric_limits<dstype>::epsilon()*scale;
                const dstype distanceError=std::abs(got.projection.distance-gridDistance);
                maxProjectionError=std::max(maxProjectionError,distanceError);
                check(distanceError<=distanceAllowance,
                    "curved surface distance matches dense-grid reference");

                const dstype u=got.projection.xi[0],v=got.projection.xi[1];
                const dstype z=au*u*(1-u)+av*v*(1-v);
                const dstype residual[3]={u-query[0],v-query[1],z-query[2]};
                const dstype tangent0[3]={1,0,au*(1-2*u)};
                const dstype tangent1[3]={0,1,av*(1-2*v)};
                const dstype gradient[2]={tangent0[0]*residual[0]+tangent0[2]*residual[2],
                    tangent1[1]*residual[1]+tangent1[2]*residual[2]};
                const dstype gradientNorm=std::hypot(gradient[0],gradient[1]);
                maxSurfaceGradient=std::max(maxSurfaceGradient,gradientNorm);
                const dstype tangentNorm=std::max(std::hypot(tangent0[0],tangent0[2]),
                    std::hypot(tangent1[1],tangent1[2]));
                const dstype gradientAllowance=4096*std::numeric_limits<dstype>::epsilon()*
                    std::max({dstype(1),scale*scale,tangentNorm*got.projection.distance});
                check(gradientNorm<=gradientAllowance,
                    "curved surface interior tangential stationarity");
            }
        }
    check(cases==96,"curved surface accuracy case count");
}

#ifdef HAVE_MPI
void distributed_line_search()
{
    int rank=0,size=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(size%2!=0||size<2)return;
    const int ranksPerModel=size/2,model=rank/ranksPerModel,local=rank%ranksPerModel;
    const int p=2;const dstype xpf[3]={0,0.5,1};
    FaceSet<dstype,Int> fs;fs.nd=2;fs.npf=3;fs.rank={rank};fs.elem={local};
    fs.lface={0};fs.global={local};fs.nodes.resize(6);
    const dstype left=static_cast<dstype>(local)/ranksPerModel;
    const dstype right=static_cast<dstype>(local+1)/ranksPerModel;
    for(int a=0;a<3;++a){fs.nodes[a]=left+(right-left)*xpf[a];fs.nodes[a+3]=0.01*model;}
    const std::vector<dstype> point={(left+right)/2,0.01*model};
    const auto donor=pp::locate_donors<dstype,Int>(fs,xpf,0,p,point,rank,model);
    const int expectedRank=(1-model)*ranksPerModel+local;
    check(donor.size()==1&&donor[0].converged,"distributed donor converged");
    check(donor[0].rank==expectedRank&&donor[0].face==local,"distributed donor metadata");
    check(std::abs(donor[0].distance-0.01)<1e-13,"distributed donor distance");
}

void distributed_sparse_search()
{
    int rank=0,size=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(size%2!=0||size<2)return;
    const int ranksPerModel=size/2,model=rank/ranksPerModel,local=rank%ranksPerModel;
    const int p=2;const dstype xpf[3]={0,0.5,1};
    FaceSet<dstype,Int> fs;fs.nd=2;fs.npf=3;
    std::vector<dstype> point;
    if(local==0){
        fs.rank={rank};fs.elem={0};fs.lface={0};fs.global={0};fs.nodes.resize(6);
        for(int a=0;a<3;++a){fs.nodes[a]=xpf[a];fs.nodes[a+3]=0.02*model;}
        point={0.37,0.02*model};
    }
    const auto donor=pp::locate_donors<dstype,Int>(fs,xpf,0,p,point,rank,model);
    if(local==0){
        check(donor.size()==1&&donor[0].converged,"sparse communicator donor converged");
        check(donor[0].rank==(1-model)*ranksPerModel,"sparse communicator donor rank");
    }else check(donor.empty(),"non-interface rank leaves after communicator split");
}

void distributed_rank_tie_line(dstype translation,dstype faceSize)
{
    int rank=0,size=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(size<4||size%2!=0)return;
    const int ranksPerModel=size/2,model=rank/ranksPerModel,local=rank%ranksPerModel;
    const int p=2;const dstype xpf[3]={0,0.5,1};
    FaceSet<dstype,Int> fs;fs.nd=2;fs.npf=3;fs.rank={rank};fs.elem={0};
    fs.lface={0};fs.global={local};fs.nodes.resize(6);
    const dstype delta=8*std::numeric_limits<dstype>::epsilon()*
        std::max<dstype>(1,std::abs(translation));
    const dstype offset=local==1?(model==0?delta:-delta):0.0;
    for(int a=0;a<3;++a){fs.nodes[a]=translation+faceSize*xpf[a];
        fs.nodes[a+3]=0.03*model+offset;}
    const std::vector<dstype> point={translation+0.41*faceSize,0.03*model};
    const auto donor=pp::locate_donors<dstype,Int>(fs,xpf,0,p,point,rank,model);
    check(donor.size()==1&&donor[0].converged,"cross-rank tie converged");
    check(donor[0].face==0&&donor[0].rank==(1-model)*ranksPerModel,
          "cross-rank tolerance tie uses smallest global face");

    FaceSet<dstype,Int> serial;
    serial.nd=2;serial.npf=3;
    serial.nodes.resize(static_cast<std::size_t>(6)*ranksPerModel);
    const int donorModel=1-model;
    for(int q=0;q<ranksPerModel;++q){
        serial.rank.push_back(donorModel*ranksPerModel+q);
        serial.elem.push_back(0);serial.lface.push_back(0);serial.global.push_back(q);
        const dstype qoffset=q==1?(donorModel==0?delta:-delta):0.0;
        for(int a=0;a<3;++a){
            serial.nodes[a+6*q]=translation+faceSize*xpf[a];
            serial.nodes[a+3+6*q]=0.03*donorModel+qoffset;
        }
    }
    const auto prepared=pp::prepare_faces<dstype,Int>(serial,xpf,0,p);
    const dstype query[2]={point[0],point[1]};
    pp::ProjectionWorkspace<dstype,Int> work;
    const auto serialWinner=pp::search_local_faces(prepared,query,1e-13,100,work);
    check(!serialWinner.failed&&serialWinner.face>=0,
          "serial translated tie converged");
    check(donor[0].face==prepared.globalFace[serialWinner.face],
          "serial and distributed translated line tie agree");
    check(std::abs(donor[0].xi[0]-serialWinner.projection.xi[0])<=5e-10,
          "serial and distributed translated line xi agree");
}

void distributed_rank_tie_surface(dstype translation)
{
    int rank=0,size=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(size<4||size%2!=0)return;
    const int ranksPerModel=size/2,model=rank/ranksPerModel,local=rank%ranksPerModel;
    const int p=2;const Master m=surface_master(1,p);
    const dstype delta=8*std::numeric_limits<dstype>::epsilon()*
        std::max<dstype>(1,std::abs(translation));
    auto fill=[&](FaceSet<dstype,Int>& faces,int face,int faceModel,int faceLocal){
        const dstype offset=faceLocal==1?(faceModel==0?delta:-delta):0.0;
        for(Int a=0;a<m.npf;++a){
            const dstype u=m.xpf[a],v=m.xpf[a+m.npf];
            const std::size_t base=static_cast<std::size_t>(3)*m.npf*face;
            faces.nodes[base+a]=translation+u;
            faces.nodes[base+a+m.npf]=v;
            faces.nodes[base+a+2*m.npf]=0.03*faceModel+offset;
        }
    };
    FaceSet<dstype,Int> localFaces;
    localFaces.nd=3;localFaces.npf=m.npf;localFaces.rank={rank};
    localFaces.elem={0};localFaces.lface={0};localFaces.global={local};
    localFaces.nodes.resize(static_cast<std::size_t>(3)*m.npf);
    fill(localFaces,0,model,local);
    const std::vector<dstype> point={translation+0.41,0.37,0.03*model};
    const auto donor=pp::locate_donors<dstype,Int>(localFaces,m.xpf.data(),1,p,
        point,rank,model);

    FaceSet<dstype,Int> serial;
    serial.nd=3;serial.npf=m.npf;
    serial.nodes.resize(static_cast<std::size_t>(3)*m.npf*ranksPerModel);
    const int donorModel=1-model;
    for(int q=0;q<ranksPerModel;++q){
        serial.rank.push_back(donorModel*ranksPerModel+q);
        serial.elem.push_back(0);serial.lface.push_back(0);serial.global.push_back(q);
        fill(serial,q,donorModel,q);
    }
    const auto prepared=pp::prepare_faces<dstype,Int>(serial,m.xpf.data(),1,p);
    const dstype query[3]={point[0],point[1],point[2]};
    pp::ProjectionWorkspace<dstype,Int> work;
    const auto serialWinner=pp::search_local_faces(prepared,query,1e-13,100,work);
    check(donor.size()==1&&donor[0].converged&&!serialWinner.failed&&serialWinner.face>=0,
          "serial and distributed surface tie converged");
    check(donor[0].face==prepared.globalFace[serialWinner.face],
          "serial and distributed translated surface tie agree");
    check(std::abs(donor[0].xi[0]-serialWinner.projection.xi[0])<=5e-10&&
          std::abs(donor[0].xi[1]-serialWinner.projection.xi[1])<=5e-10,
          "serial and distributed translated surface xi agree");
}

void distributed_rank_tie()
{
    auto run=[](const char* name,auto&& function){
        int threw=0;
        try { function(); }
        catch(const std::exception& exception){
            std::printf("FAIL %s threw: %s\n",name,exception.what());
            ++failures;
            threw=1;
        }
        int minimum=0,maximum=0;
        MPI_Allreduce(&threw,&minimum,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
        MPI_Allreduce(&threw,&maximum,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        check(minimum==maximum,"distributed tie exception state is rank uniform");
    };
    run("distributed line tie X=0 h=1",[]{distributed_rank_tie_line(0.0,1.0);});
    run("distributed line tie X=1000 h=1",[]{distributed_rank_tie_line(1.0e3,1.0);});
    run("distributed line tie X=0 h=1e-3",[]{distributed_rank_tie_line(0.0,1.0e-3);});
    run("distributed surface tie X=0",[]{distributed_rank_tie_surface(0.0);});
    run("distributed surface tie X=1000",[]{distributed_rank_tie_surface(1.0e3);});
}

void distributed_failure_is_uniform()
{
    int rank=0,size=1;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(size%2!=0||size<2)return;
    const int ranksPerModel=size/2,model=rank/ranksPerModel,local=rank%ranksPerModel;
    const int p=2;const dstype xpf[3]={0,0.5,1};
    FaceSet<dstype,Int> fs;fs.nd=2;fs.npf=3;fs.rank={rank};fs.elem={local};
    fs.lface={0};fs.global={local};fs.nodes.resize(6);
    const dstype left=static_cast<dstype>(local)/ranksPerModel;
    const dstype right=static_cast<dstype>(local+1)/ranksPerModel;
    for(int a=0;a<3;++a){const dstype s=xpf[a];fs.nodes[a]=left+(right-left)*s;
        fs.nodes[a+3]=0.02*model+0.1*s*(1-s);}
    const std::vector<dstype> point={(left+right)/2,0.02*model+0.2};
    pp::SearchParameters<dstype,Int> parameters;parameters.maximumProjectionIterations=0;
    bool caught=false;
    try { (void)pp::locate_donors<dstype,Int>(fs,xpf,0,p,point,rank,model,parameters); }
    catch(const std::runtime_error& exception) {
        caught=std::string(exception.what()).find("donor face projection failed for owner rank")!=std::string::npos;
    }
    int localCaught=caught?1:0,minimum=0,maximum=0;
    MPI_Allreduce(&localCaught,&minimum,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(&localCaught,&maximum,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    check(minimum==1&&maximum==1,"projection failure reported uniformly");
}
#endif
}

int main(int argc,char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc,&argv);EXASIM_COMM_WORLD=MPI_COMM_WORLD;EXASIM_COMM_LOCAL=MPI_COMM_WORLD;
#endif
    Kokkos::initialize(argc,argv);
    sample_counts();line_projection();multiple_minimum_tie();separated_local_minima();
    surface_cases();surface_orientations();
    repeated_search_allocations();screening_boundaries();projection_stopping_scale_scan();
#ifdef HAVE_MPI
    distributed_line_search();distributed_sparse_search();distributed_rank_tie();
    distributed_failure_is_uniform();
#endif
    std::printf("preprocess geometry: max projection error %.3e, max box violation %.3e\n",maxProjectionError,maxBoxViolation);
    std::printf("preprocess geometry: max curved-surface tangential gradient %.3e\n",
        maxSurfaceGradient);
    check(maxProjectionError<2e-10,"projection against analytic reference");
    check(maxBoxViolation<=1e-13,"Bernstein box enclosure");
    Kokkos::finalize();
#ifdef HAVE_MPI
    int global=0;MPI_Allreduce(&failures,&global,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);MPI_Finalize();failures=global;
#endif
    if(failures==0)std::printf("PASS nonmatching_preprocess\n");
    return failures==0?0:1;
}
