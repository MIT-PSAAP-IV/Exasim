// Small manufactured checkpoint: native H and L operators, no coupled Newton solve.
#include <exasim/operators.hpp>
#include <exasim/export.hpp>
#include "backend/Preprocessing/hdgldgregions.hpp"
#include "backend/Discretization/hdgldginterface.hpp"
#include <iomanip>

#include "models.hpp"

// The shared point mathematics also serves the ordinary generated-provider
// test, which must not include the backend/common definitions in its app TU.
template<int ND,bool Periodic=false>
struct PolynomialDiffusion : exasim::ModelDefaults<PolynomialDiffusion<ND,Periodic>>,
        PolynomialDiffusionMath<ND,Periodic> {
    using Math=PolynomialDiffusionMath<ND,Periodic>;
    using Math::nd;using Math::ncu;using Math::ncw;using Math::nco;
    using Math::nparam;using Math::Nq;
    using Math::flux;using Math::flux_jac_uq;using Math::source;
    using Math::ubou;using Math::fbou;using Math::fbou_hdg;
    using Math::fbou_hdg_jac_uh;using Math::fhat;
    static constexpr auto disc=exasim::Discretization::HDG;
};

static double errorMagnitude(double value,double expected=0)
{
    if(!std::isfinite(value))throw std::runtime_error("nonfinite manufactured checkpoint value");
    return std::abs(value-expected);
}

template<int ND,bool Periodic,class Model=PolynomialDiffusion<ND,Periodic>>
static void prepare(const std::string& root,const std::vector<int>& hRanks,const std::vector<int>& lRanks,int worldSize,bool noInterfaceRank,int preconditioner=0,int rbDim=-1,double initialOffset=0,int timeOrder=0)
{
    PDE pde=exasim::default_pde<Model>(); pde.porder=2;pde.pgauss=6;
    pde.hybrid=1;pde.mpiprocs=1;pde.ncx=ND;pde.ncq=Model::ncu*ND;pde.nch=Model::ncu;pde.preconditioner=preconditioner;
    if(rbDim>=0)pde.RBdim=rbDim;
    if(timeOrder){
        if(timeOrder!=2 && timeOrder!=3)throw std::runtime_error("transient fixture requires DIRK22 or DIRK33");
        pde.tdep=1;pde.torder=pde.nstage=timeOrder;pde.dt.assign(4,1./40);
        pde.saveSolOpt=1;pde.saveSolFreq=1;
    }
    Mesh mesh;mesh.dim=mesh.nd=ND;mesh.nve=1<<ND;mesh.nvf=1<<(ND-1);mesh.nfe=2*ND;
    // At least three cells around a periodic direction avoid the duplicate
    // vertex-set face keys of a two-cell ring; keep the original N=2 3D case.
    const int N=ND==2 || Periodic?4:2,vertices=N+1;
    mesh.ne=ND==2?N*N:N*N*N;mesh.np=ND==2?vertices*vertices:vertices*vertices*vertices;
    mesh.elemtype=1;mesh.nbcm=mesh.nbndexpr=2*ND;
    for(int b=0;b<2*ND;++b){mesh.boundaryConditions.push_back(b+1);mesh.curvedBoundaries.push_back(0);}
    mesh.boundaryPreds={[](const double*x){return x[1]==0;},[](const double*x){return x[0]==1;},
        [](const double*x){return x[1]==1;},[](const double*x){return x[0]==0;}};
    if constexpr(ND==3){mesh.boundaryPreds.push_back([](const double*x){return x[2]==0;});
        mesh.boundaryPreds.push_back([](const double*x){return x[2]==1;});}
    for(int z=0;z<(ND==3?vertices:1);++z)for(int y=0;y<vertices;++y)for(int x=0;x<vertices;++x){
        mesh.p.insert(mesh.p.end(),{double(x)/N,double(y)/N});
        if constexpr(ND==3)mesh.p.push_back(double(z)/N);}
    std::vector<int> labels;
    for(int z=0;z<(ND==3?N:1);++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){
        int v=x+vertices*(y+vertices*z),plane=vertices*vertices;
        mesh.t.insert(mesh.t.end(),{v,v+1,v+vertices+1,v+vertices});
        if constexpr(ND==3)mesh.t.insert(mesh.t.end(),{v+plane,v+plane+1,v+plane+vertices+1,v+plane+vertices});
        mesh.elemGlobalID.push_back(107+13*(x+N*(y+N*z)));labels.push_back(Periodic?x<N/2:(x<N/2 || y<N/2));}
    char yExpression[]="y",zExpression[]="z";char* expressions[]={yExpression,zExpression};
    if constexpr(Periodic){mesh.nprdexpr=1;mesh.nprdcom=ND-1;
        mesh.periodicBoundaries1={4};mesh.periodicBoundaries2={2};
        mesh.periodicExprs1=expressions;mesh.periodicExprs2=expressions;}
    Master master=initializeMaster(pde,mesh);buildMesh(mesh,pde,master);
    mesh.udg.resize(master.npe*Model::Nq*mesh.ne);
    for(int e=0;e<mesh.ne;++e)for(int j=0;j<master.npe;++j){
        double x[ND];for(int a=0;a<ND;++a)x[a]=mesh.xdg[j+master.npe*(a+ND*e)];
        for(int d=0;d<=ND;++d)for(int c=0;c<Model::ncu;++c)
            mesh.udg[j+master.npe*(c+Model::ncu*d+Model::Nq*e)]=Model::value(c,x,d)+(d==0?initialOffset:0);
    }
    if(timeOrder)for(int scheme=0;scheme<2;++scheme)for(int resume=0;resume<2;++resume){
        // Pure counterparts use the native complete-mesh writer and physical
        // boundaries, not a regional H/L boundary relabelled as pure. This
        // fixture is affine: rebuilding nodes is exact and checked in tests.
        PDE pure=pde;Mesh full=mesh;pure.hybrid=scheme;pure.mpiprocs=worldSize;
        pure.discretization=scheme?"hdg":"ldg";
        pure.timestepOffset=resume?2:0;if(resume)pure.dt.assign(2,1./40);
        const std::string name=std::string(scheme?"pureH":"pureL")+(resume?"Resume":"");
        pure.datainpath=root+"/"+name;pure.dataoutpath=root+"/"+name+"-unused-output";
        pdeFinalizeDerived(pure);
        writeBinaryFiles(pure,full,master,ParsedSpec{});
    }
    exasim::hdgldg::Partitioner partitioner=exasim::hdgldg::metisPartitionGraph;
    if(noInterfaceRank)partitioner=[](const exasim::hdgldg::Graph& graph,int parts){
        if(graph.offsets.size()!=13 || parts!=3)throw std::runtime_error("unexpected sparse fixture partition request");
        // Regional H input order is [0..9,12,13]. Rank 0 owns no HL face.
        return std::vector<int>{0,0,0,0,0,0,1,1,0,2,0,2};};
    auto plan=exasim::hdgldg::buildPlan(exasim::hdgldg::topologyFromPreparedMesh(mesh),labels,
        hRanks,lRanks,worldSize,partitioner);
    for(int scheme=0;scheme<2;++scheme){auto region=exasim::hdgldg::extractPreparedRegion(mesh,master,pde,plan,scheme);
        exasim::hdgldg::writePreparedRegion(region,master,plan,root+(scheme?"/H":"/L"));}
}

template<int ND,bool Periodic=false>
static int check(const std::string& root,int rank,int worldSize,bool isH,int localSize,int localRank,
                 const std::vector<int>& hRanks,const std::vector<int>& lRanks,bool noInterfaceRank)
{
        using Model=PolynomialDiffusion<ND,Periodic>;
        int failures=0;
        if(!rank)prepare<ND,Periodic>(root,hRanks,lRanks,worldSize,noInterfaceRank);MPI_Barrier(EXASIM_COMM_WORLD);
        const std::string directory=root+(isH?"/H":"/L");
        CDiscretization disc(directory+"/",directory+"/out","",localSize,localRank,0,0,0,0,ExasimDriverABI{},0,0,0,0,0,ExasimExecutionMode::HDGLDGSetup);
        auto interface=exasim::hdgldg::initializeInterface(disc,directory,0);
        if(noInterfaceRank && rank==0 && !interface.faces.empty())throw std::runtime_error("zero-interface H rank unexpectedly has packets");
        // Empty native work must neither touch residual canaries nor index
        // eblks[-2]. This is also an ASan regression for zero-interior ranks.
        ArraySetValue(disc.res.Rq,17.25,disc.res.szRq);
        ArraySetValue(disc.res.Rh,-9.5,disc.res.szRh);
        GetQ(disc.sol,disc.res,disc.app,disc.master,disc.mesh,disc.tmp,disc.common,
            disc.common.cublasHandle,0,0,0,0,0);
        Kokkos::fence("empty GetQ canaries");
        for(int j=0;j<disc.res.szRq;++j)if(disc.res.Rq[j]!=17.25)throw std::runtime_error("empty GetQ changed Rq");
        for(int j=0;j<disc.res.szRh;++j)if(disc.res.Rh[j]!=-9.5)throw std::runtime_error("empty GetQ changed Rh");
        CResidual<Model> residual(disc);
        auto& c=disc.common;const int npe=c.grid.npe,npf=c.grid.npf,ncu=c.components.ncu;
        std::cerr<<"native blocks rank="<<rank<<" nbe0="<<c.meshsizes.nbe0<<" nbe1="<<c.meshsizes.nbe1
            <<" nbe2="<<c.meshsizes.nbe2<<" ne1="<<c.meshsizes.ne1<<" ne2="<<c.meshsizes.ne2<<'\n';
        double* hRhs=nullptr;
        if(isH){residual.recoverInitialState(0,false);residual.evalQ(0);
            if(disc.res.szRq<c.sizes.ndofuhat)throw std::runtime_error("native Rq cannot hold checkpoint H RHS");
            hRhs=disc.res.Rq;CAssembler<Model> assembler(disc);assembler.hdgAssembleLinearSystem(hRhs,0);}
        exasim::hdgldg::exchangeTrace(disc,interface,0);
        double traceError=0,gradientError=0,fluxError=0,residualError=0;
        int reversed=0,multiple=0,ghostSubscribers=0;
        for(const auto& endpoint:interface.faces)ghostSubscribers+=!endpoint.contributesFlux;
        if(!isH){
            // Both public cold-recovery paths must retain the received lambda,
            // not invoke the deliberately invalid HL physical callback.
            residual.recoverInitialState(0,false);
            residual.evalQ(0);
            for(const auto& endpoint:interface.faces)for(int j=0;j<npf;++j){
                int node=disc.mesh.facecon[2*(j+npf*endpoint.face)];int e=node/npe,k=node%npe;
                double x[ND];for(int a=0;a<ND;++a)x[a]=disc.sol.xdg[k+npe*(a+ND*e)];
                for(int a=0;a<ncu;++a)traceError=std::max(traceError,errorMagnitude(disc.sol.uh[j+npf*(a+ncu*endpoint.face)],Model::value(a,x)));
                reversed+=endpoint.nodeOfH[j]!=j;
            }
            // Poison q to establish that the native residual reconstructs it from u and lambda.
            for(int e=0;e<c.meshsizes.ne;++e)for(int a=3;a<Model::Nq;++a)for(int j=0;j<npe;++j)
                disc.sol.udg[j+npe*(a+Model::Nq*e)]=std::numeric_limits<double>::quiet_NaN();
            // First-ghost u must arrive through the existing LOCAL halo, not
            // through interface packets. The outer u-only layer is native too.
            for(int e=c.meshsizes.ne1;e<c.meshsizes.ne2;++e)for(int a=0;a<3;++a)for(int j=0;j<npe;++j)
                disc.sol.udg[j+npe*(a+Model::Nq*e)]=std::numeric_limits<double>::quiet_NaN();
            ResidualStageContext context{disc.sol,disc.res,disc.app,disc.master,disc.mesh,disc.tmp,c,c.cublasHandle,0};
            RuResidualStaged<Model,ProductionResidualStages<Model>>(context);
            for(const auto& endpoint:interface.faces)for(int j=0;j<npf;++j){
                const int node=disc.mesh.facecon[2*(j+npf*endpoint.face)],e=node/npe,k=node%npe;
                double x[ND];for(int a=0;a<ND;++a)x[a]=disc.sol.xdg[k+npe*(a+ND*e)];
                for(int a=0;a<ncu;++a)traceError=std::max(traceError,errorMagnitude(
                    disc.sol.uh[j+npf*(a+ncu*endpoint.face)],Model::value(a,x)));
            }
            std::vector<int> touches(c.meshsizes.ne,0);
            for(const auto& endpoint:interface.faces){
                if(!endpoint.contributesFlux)continue;
                int f=endpoint.face,e=disc.mesh.f2e[4*f];++touches[e];
                double minimum[ND],maximum[ND],center[ND]={};
                for(int d=0;d<ND;++d){minimum[d]=std::numeric_limits<double>::infinity();maximum[d]=-minimum[d];
                    for(int k=0;k<npe;++k)center[d]+=disc.sol.xdg[k+npe*(d+ND*e)]/npe;}
                for(int j=0;j<npf;++j){int node=disc.mesh.facecon[2*(j+npf*f)],k=node%npe;
                    for(int d=0;d<ND;++d){double x=disc.sol.xdg[k+npe*(d+ND*e)];
                        minimum[d]=std::min(minimum[d],x);maximum[d]=std::max(maximum[d],x);}
                }
                int normalAxis=-1;double measure=1;
                for(int d=0;d<ND;++d){if(maximum[d]==minimum[d])normalAxis=d;else measure*=maximum[d]-minimum[d];}
                if(normalAxis<0)throw std::runtime_error("oracle expects Cartesian planar faces");
                const double normal=minimum[normalAxis]>center[normalAxis]?1:-1;
                // Exact tensor-product p2 moments of an affine normal flux.
                // A linear function's moment for each p2 nodal basis equals
                // its nodal value times the basis integral (1/6,2/3,1/6).
                // Node coordinates, not the native master node ordering, pick weights.
                for(int j=0;j<npf;++j){const int node=disc.mesh.facecon[2*(j+npf*f)],k=node%npe;
                    double x[ND],weight=measure;
                    for(int d=0;d<ND;++d){x[d]=disc.sol.xdg[k+npe*(d+ND*e)];
                        if(d!=normalAxis)weight*=x[d]==minimum[d] || x[d]==maximum[d]?1./6:2./3;}
                    for(int a=0;a<3;++a){double flux=0;for(int b=0;b<3;++b)
                        {double q=Model::value(b,x,normalAxis+1);
                         if constexpr(Periodic)if(normalAxis<2)q+=.25*Model::value(b,x,2-normalAxis);
                         flux+=Model::coefficient(a,b)*normal*q;}
                        fluxError=std::max(fluxError,errorMagnitude(disc.res.Rh[j+npf*(a+3*f)],weight*flux));}
                }
            }
            for(int n:touches)multiple=std::max(multiple,n);
            for(int e=0;e<c.meshsizes.ne2;++e)for(int j=0;j<npe;++j){
                double x[ND];for(int a=0;a<ND;++a)x[a]=disc.sol.xdg[j+npe*(a+ND*e)];
                for(int d=1;d<=ND;++d)for(int a=0;a<3;++a)gradientError=std::max(gradientError,errorMagnitude(disc.sol.udg[j+npe*(a+3*d+Model::Nq*e)],Model::value(a,x,d)));
            }
            for(int j=0;j<c.sizes.ndof1;++j)residualError=std::max(residualError,errorMagnitude(disc.res.Ru[j]));
        }
        exasim::hdgldg::returnFlux(disc,interface,hRhs,0);
        int ghostOnlyH=0;
        if(isH){for(int f=0;f<c.meshsizes.nf;++f){const int e1=disc.mesh.f2e[4*f],e2=disc.mesh.f2e[4*f+2];
                if(!((e1>=0 && e1<c.meshsizes.ne1) || (e2>=0 && e2<c.meshsizes.ne1)))++ghostOnlyH;}
            for(int j=0;j<c.sizes.ndofuhat;++j)residualError=std::max(residualError,errorMagnitude(hRhs[j]));}
        int coverage[2]={reversed,multiple},globalCoverage[2];
        MPI_Allreduce(coverage,globalCoverage,2,MPI_INT,MPI_MAX,EXASIM_COMM_WORLD);
        if(!globalCoverage[0] || globalCoverage[1]<(Periodic?1:2))++failures;
        // A priori floating-point allowance for the small dense recovery products:
        // 48 bounds the fixture's coefficient/derivative scale, npe^2 the dense
        // product length amplification. This is not a spatial acceptance budget.
        const double roundoffBound=512*std::numeric_limits<double>::epsilon()*48*npe*npe;
        if(!std::isfinite(traceError+gradientError+fluxError+residualError) || traceError>1e-12 ||
           std::max({gradientError,fluxError,residualError})>roundoffBound)++failures;
        std::cout<<std::setprecision(17)<<"nd="<<ND<<" periodic="<<int(Periodic)<<" rank="<<rank<<" scheme="<<int(isH)<<" localRank="<<localRank
            <<" trace="<<traceError<<" q="<<gradientError
            <<" flux="<<fluxError<<" residual="<<residualError<<" bound="<<roundoffBound<<" reversed="<<reversed<<" maxHL="<<multiple
            <<" ghostOnlyH="<<ghostOnlyH<<" ghostSubscribers="<<ghostSubscribers
            <<" firstGhostL="<<(isH?0:c.meshsizes.ne2-c.meshsizes.ne1)<<" interfacePackets="<<interface.faces.size()
            <<" emptyQ=PASS\n";
        return failures;
}

#ifndef HDGLDG_FIXTURE_NO_MAIN
int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    bool threeD=false,noInterfaceRank=false,periodic=false,valid=argc>=2 && argc<=4;
    for(int a=2;a<argc;++a){const std::string option=argv[a];
        if(option=="--3d" && !threeD)threeD=true;
        else if(option=="--periodic" && !periodic)periodic=true;
        else if(option=="--no-interface" && !noInterfaceRank)noInterfaceRank=true;
        else valid=false;}
    if(!valid || size<2 || size>4 || (noInterfaceRank && (size!=4 || threeD || periodic))){
        if(!rank)std::cerr<<"usage: mpirun -np {2,3,4} hdgldg_residual /tmp/output [--3d] [--periodic] or [--no-interface]\n";MPI_Finalize();return 2;}
    const std::vector<int> hRanks=noInterfaceRank?std::vector<int>{0,1,2}:(size==4?std::vector<int>{0,2}:std::vector<int>{0});
    const std::vector<int> lRanks=noInterfaceRank?std::vector<int>{3}:(size==2?std::vector<int>{1}:(size==3?std::vector<int>{1,2}:std::vector<int>{1,3}));
    const bool isH=std::find(hRanks.begin(),hRanks.end(),rank)!=hRanks.end();
    EXASIM_COMM_WORLD=MPI_COMM_WORLD;MPI_Comm_split(MPI_COMM_WORLD,isH?1:0,rank,&EXASIM_COMM_LOCAL);
    int localRank=0,localSize=0;MPI_Comm_rank(EXASIM_COMM_LOCAL,&localRank);MPI_Comm_size(EXASIM_COMM_LOCAL,&localSize);
    Kokkos::initialize(argc,argv);int failures=0;
    try {
        if(threeD)failures=periodic?check<3,true>(argv[1],rank,size,isH,localSize,localRank,hRanks,lRanks,noInterfaceRank):
                                  check<3>(argv[1],rank,size,isH,localSize,localRank,hRanks,lRanks,noInterfaceRank);
        else failures=periodic?check<2,true>(argv[1],rank,size,isH,localSize,localRank,hRanks,lRanks,noInterfaceRank):
                               check<2>(argv[1],rank,size,isH,localSize,localRank,hRanks,lRanks,noInterfaceRank);
    }
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';MPI_Abort(EXASIM_COMM_WORLD,1);}
    int total=0;MPI_Allreduce(&failures,&total,1,MPI_INT,MPI_SUM,EXASIM_COMM_WORLD);
    Kokkos::finalize();MPI_Comm_free(&EXASIM_COMM_LOCAL);MPI_Finalize();
    if(!rank)std::cout<<"hdgldg residual failures="<<total<<'\n';return total?1:0;
}
#endif
