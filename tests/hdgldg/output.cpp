// Native boundary output, checked with explicit face/node storage indices.
#define HDGLDG_SOLVER_NO_MAIN
#include "solver.cpp"
#include "backend/Solution/solutionwriter.h"

static std::vector<double> readBoundary(const std::string& path,int npf,int faces,int components)
{
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file || file.tellg()!=std::streamoff(sizeof(double)*(3+npf*faces*components)))
        throw std::runtime_error("wrong native boundary file extent: "+path);
    file.seekg(0);double dimensions[3];file.read(reinterpret_cast<char*>(dimensions),sizeof(dimensions));
    if(dimensions[0]!=npf || dimensions[1]!=faces || dimensions[2]!=components)
        throw std::runtime_error("wrong native boundary dimensions: "+path);
    std::vector<double> values(npf*faces*components);file.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(double));
    return values;
}

static int outputCheck(const std::string& root,int rank,int size,bool rejectAuxiliary)
{
    const std::vector<int> h=size==4?std::vector<int>{0,2}:std::vector<int>{0};
    const std::vector<int> l=size==2?std::vector<int>{1}:size==3?std::vector<int>{1,2}:std::vector<int>{1,3};
    const bool isH=std::find(h.begin(),h.end(),rank)!=h.end();
    MPI_Comm_split(EXASIM_COMM_WORLD,isH?1:0,rank,&EXASIM_COMM_LOCAL);
    int lr=0,ls=0;MPI_Comm_rank(EXASIM_COMM_LOCAL,&lr);MPI_Comm_size(EXASIM_COMM_LOCAL,&ls);
    if(!rank)prepare<2,false>(root,h,l,size,false,1,0);
    MPI_Barrier(EXASIM_COMM_WORLD);int failures=0,counts[4]={};
    {
        const std::string dir=root+(isH?"/H":"/L");
        CDiscretization d(dir+"/",dir+"/boundary","",ls,lr,0,0,0,0,TestABI<SolverDiffusion>::make(),0,0,0,0,0,ExasimExecutionMode::HDGLDGSetup);
        auto interface=exasim::hdgldg::initializeInterface(d,dir,0);
        CResidual<> residual(d);CAssembler<> assembler(d);CPreconditioner<> preconditioner(d,0);CSolver<> solver(d,0);
        // Capability rejection is tested before any unsupported numerical
        // execution, without constructing a malformed auxiliary-field file.
        if(rejectAuxiliary)d.common.components.ncw=1;
        auto system=exasim::hdgldg::initializeSystem(d,residual,assembler,preconditioner,solver,interface,0);
        exasim::hdgldg::refreshState(system);
        auto& c=d.common;const int npe=c.grid.npe,npf=c.grid.npf,ncu=c.components.ncu,nc=c.components.nc;
        c.outputparams.saveSolBouFreq=1;c.timestate.currentstep=0;c.outputparams.compudgavg=0;
        c.qoiparams.ibslist={1,2,3,4,interface.boundary};c.qoiparams.ibs=1;
        int faces=0;std::vector<double> expectedU,expectedTrace,expectedX;
        for(int block=0;block<c.meshsizes.nbf;++block){const int first=c.fblks[3*block]-1,last=c.fblks[3*block+1],ib=c.fblks[3*block+2];
            if(!c.qoiparams.isSaveBoundary(ib))continue;
            faces+=last-first;counts[(isH?0:2)+(ib==interface.boundary?1:0)]+=last-first;
            // The file concatenates selected blocks; within each block its
            // records are [npf*blockFaces,components], not native trace order.
            for(int a=0;a<nc;++a)for(int f=first;f<last;++f)for(int j=0;j<npf;++j){
                const int id=d.mesh.facecon[2*(j+npf*f)];
                expectedU.push_back(d.sol.udg[id%npe+npe*(a+nc*(id/npe))]);}
            for(int a=0;a<ncu;++a)for(int f=first;f<last;++f)for(int j=0;j<npf;++j)
                expectedTrace.push_back(d.sol.uh[isH?a+ncu*(j+npf*f):j+npf*(a+ncu*f)]);
            for(int a=0;a<2;++a)for(int f=first;f<last;++f)for(int j=0;j<npf;++j){
                const int id=d.mesh.facecon[2*(j+npf*f)];
                expectedX.push_back(d.sol.xdg[id%npe+npe*(a+2*(id/npe))]);}}
        {
            CVisualization vis(d,0);CSolutionWriter<> writer(d,residual,vis,solver);
            writer.openBoundaryFiles(c.fileout);writer.SaveNodesOnBoundary(0);writer.SaveSolutionsOnBoundary(0);
        }
        const std::string suffix="_np"+std::to_string(lr)+".bin",prefix=c.fileout;
        const auto u=readBoundary(prefix+"bouudg"+suffix,npf,faces,nc);
        const auto trace=readBoundary(prefix+"bouuhat"+suffix,npf,faces,ncu);
        const auto x=readBoundary(prefix+"bouxdg"+suffix,npf,faces,2);
        const auto normal=readBoundary(prefix+"boundg"+suffix,npf,faces,2);
        int exactDifferences=0;double normalError=0;
        for(std::size_t k=0;k<u.size();++k)exactDifferences+=u[k]!=expectedU[k] || !std::isfinite(u[k]);
        for(std::size_t k=0;k<trace.size();++k)exactDifferences+=trace[k]!=expectedTrace[k] || !std::isfinite(trace[k]);
        for(std::size_t k=0;k<x.size();++k)exactDifferences+=x[k]!=expectedX[k] || !std::isfinite(x[k]);
        int offset=0;
        for(int block=0;block<c.meshsizes.nbf;++block){const int first=c.fblks[3*block]-1,last=c.fblks[3*block+1],ib=c.fblks[3*block+2];
            if(!c.qoiparams.isSaveBoundary(ib))continue;const int nn=npf*(last-first);
            for(int f=first;f<last;++f){const int id=d.mesh.facecon[2*npf*f],element=id/npe;double center[2]={};
                for(int a=0;a<2;++a)for(int k=0;k<npe;++k)center[a]+=d.sol.xdg[k+npe*(a+2*element)]/npe;
                for(int j=0;j<npf;++j){const int index=j+npf*(f-first);double norm=0,outward=0;
                    for(int a=0;a<2;++a){const double n=normal[offset+index+nn*a];
                        failures+=!std::isfinite(n);norm+=n*n;outward+=n*(x[offset+index+nn*a]-center[a]);}
                    normalError=std::max(normalError,std::abs(norm-1));failures+=!(outward>0);}}
            offset+=2*nn;}
        failures+=exactDifferences!=0 || normalError>128*std::numeric_limits<double>::epsilon();
        std::cout<<"BOUNDARY_OUTPUT rank="<<rank<<" scheme="<<(isH?"H":"L")<<" faces="<<faces
            <<" exactDifferences="<<exactDifferences<<" normalError="<<normalError<<" failures="<<failures<<'\n';
    }
    int global[4];MPI_Allreduce(counts,global,4,MPI_INT,MPI_SUM,EXASIM_COMM_WORLD);
    for(int value:global)failures+=value==0;
    if(!rank)std::cout<<"BOUNDARY_COVERAGE Hb="<<global[0]<<" HL_H="<<global[1]<<" Lb="<<global[2]<<" HL_L="<<global[3]<<'\n';
    MPI_Comm_free(&EXASIM_COMM_LOCAL);EXASIM_COMM_LOCAL=MPI_COMM_WORLD;return failures;
}

int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);EXASIM_COMM_WORLD=MPI_COMM_WORLD;int rank=0,size=0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);Kokkos::initialize(argc,argv);int failures=0;
    try{if((argc!=2 && argc!=3) || size<2 || size>4)throw std::runtime_error("usage: output /tmp/root [--reject-w] (np2/3/4)");
        const bool rejectAuxiliary=argc==3 && std::string(argv[2])=="--reject-w";
        if(argc==3 && !rejectAuxiliary)throw std::runtime_error("unknown output fixture option");
        failures=outputCheck(argv[1],rank,size,rejectAuxiliary);}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);}
    int total=0;MPI_Allreduce(&failures,&total,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);Kokkos::finalize();MPI_Finalize();return total?1:0;
}
