// Check the MPI residual with the same physics as the serial test.
// Each rank owns one element and stores one halo element.
#include <mpi.h>
MPI_Comm EXASIM_COMM_WORLD = MPI_COMM_NULL;
MPI_Comm EXASIM_COMM_LOCAL = MPI_COMM_NULL;
#define EXASIM_ENZYME_RESIDUAL_FIXTURE_ONLY
#include "enzyme_residual.cpp"

struct MpiFixture : Fixture {
    int rank;
    std::array<Int,6> blocks{1,1,0,2,2,0};
    std::array<Int,1> neighbor{}, counts{1};
    std::array<Int,2> sendIndices{0,1}, receiveIndices{4,5};
    std::array<dstype,2> send{}, receive{};
    std::array<MPI_Request,2> requests{};
    std::array<MPI_Status,2> statuses{};
    explicit MpiFixture(int rank_) : rank(rank_) {
        neighbor[0]=1-rank;
        c.mpiRank=rank; c.mpiProcs=2;
        c.meshsizes.ne1=1; c.sizes.ndof1=2;
        c.meshsizes.nbe=2; c.meshsizes.nbe0=0;
        c.meshsizes.nbe1=1; c.meshsizes.nbe2=2;
        c.eblks=blocks.data();
        c.nnbsd=c.nelemsend=c.nelemrecv=1;
        c.nbsd=neighbor.data(); c.elemsendpts=c.elemrecvpts=counts.data();
        c.requests=requests.data(); c.statuses=statuses.data();
        mesh.elemsendind=sendIndices.data(); mesh.elemrecvind=receiveIndices.data();
        tmp.buffsend=send.data(); tmp.buffrecv=receive.data();
        if(rank==1) {
            // Put the right element first on rank 1. Reverse its interface normal.
            x={0.5,1,0,0.5};
            for(int i=0;i<2;++i)std::swap(elemg[i],elemg[i+2]);
            facecon={0,3,1,1,2,2};
            find1={0,2,1,4,3,6}; find2={5,7,1,4,3,6}; findx={0,1,2};
            faceg={0.5,-1,1,1,0,1,-1,1,1};
            cgelcon={1,2,0,1}; cols={1,0,1,0};
        }
        const std::array<dstype,4> gaussX{elemg[0],elemg[1],elemg[2],elemg[3]};
        elemg={gaussX[0],gaussX[1],1,1,0.25,0.25,
               gaussX[2],gaussX[3],1,1,0.25,0.25};
        reset();
    }
    void reset() {
        Fixture::reset();
        if(rank==1) {
            for(int i=0;i<4;++i) {
                std::swap(udg[i],udg[i+4]);
                std::swap(odg[i],odg[i+4]);
                std::swap(sdg[i],sdg[i+4]);
            }
            for(int i=0;i<2;++i) {
                std::swap(w[i],w[i+2]);
                std::swap(history[i],history[i+2]);
            }
            std::swap(uh[1],uh[2]);
        }
    }
};

template<class Model> static void verifyMPI(MpiFixture& f,int wave,bool frozen) {
    f.c.timeparams.tdep=1; f.c.timeparams.wave=wave;
    f.c.physicsparams.frozenAVflag=frozen ? 1:0;
    // Change the base state from the stored state to test the primal halo update.
    std::array<dstype,2> u=f.rank==0 ? std::array<dstype,2>{0.23,0.31}
                                    : std::array<dstype,2>{0.73,0.81};
    std::array<dstype,2> v=f.rank==0 ? std::array<dstype,2>{0.4,-0.3}
                                    : std::array<dstype,2>{0.2,0.5};
    std::array<dstype,2> result{}, repeated{}, plus{}, minus{}, probe{};
    auto evaluate=[&](const std::array<dstype,2>& state,std::array<dstype,2>& out) {
        f.reset();
        ArrayInsert(f.sol.udg,state.data(),2,2,1,0,2,0,1,0,1);
        Residual<Model>(f.sol,f.res,f.app,f.master,f.mesh,f.tmp,f.c,f.c.cublasHandle,0);
        Kokkos::fence();
        for(int i=0;i<2;++i)out[i]=f.Ru[i];
    };
    f.reset();
    const auto originalU=f.udg,originalO=f.odg;
    const auto originalW=f.w;
    const auto originalH=f.uh;
    auto jvp=[&](std::array<dstype,2>& direction,std::array<dstype,2>& out) {
        MatVec<Model>(out.data(),f.sol,f.res,f.app,f.master,f.mesh,f.tmp,f.c,
            f.c.cublasHandle,direction.data(),u.data(),f.Ru.data(),0);
        Kokkos::fence();
    };
    jvp(v,result);
    int failures=(f.udg!=originalU || f.odg!=originalO || f.w!=originalW || f.uh!=originalH);
    auto other=v; other[0]*=-2;
    jvp(other,repeated); jvp(v,repeated);
    failures += result!=repeated;
    const dstype eps=1e-6;
    for(int i=0;i<2;++i)probe[i]=u[i]+eps*v[i]; evaluate(probe,plus);
    for(int i=0;i<2;++i)probe[i]=u[i]-eps*v[i]; evaluate(probe,minus);
    dstype localError=0,error=0;
    for(int i=0;i<2;++i) {
        if (!std::isfinite(result[i]) || !std::isfinite(plus[i]) || !std::isfinite(minus[i]))
            ++failures;
        else
            localError=std::max(localError,std::abs(result[i]-(plus[i]-minus[i])/(2*eps)));
    }
    MPI_Allreduce(&localError,&error,1,mpi_type<dstype>(),MPI_MAX,MPI_COMM_WORLD);
    f.reset(); v.fill(0); jvp(v,result);
    for(auto value:result)failures += value!=0;
    int totalFailures=0;
    MPI_Allreduce(&failures,&totalFailures,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    if(f.rank==0)
        std::printf("MPI residual wave=%d frozenAV=%d material=%d max JVP error %.3e\n",
            wave,frozen,f.app.materialdb_nprop,(double)error);
    require(error<2e-7,"MPI residual JVP disagrees with central differences");
    require(totalFailures==0,"MPI JVP finite-value, state, repeatability, or zero-direction check failed");
}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    EXASIM_COMM_WORLD=EXASIM_COMM_LOCAL=MPI_COMM_WORLD;
    Kokkos::initialize(argc,argv);
    int rank=0,ranks=0,code=0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    try {
        require(ranks==2,"The MPI residual test requires two ranks");
        MpiFixture f(rank);
        for(int typed=0;typed<2;++typed)for(int wave=0;wave<2;++wave)for(bool frozen:{false,true}) {
            if(typed)verifyMPI<PdeModel>(f,wave,frozen);
            else verifyMPI<exasim::detail::AbiAdapter>(f,wave,frozen);
        }
        f.app.materialdb_nprop=1;
        verifyMPI<PdeModel>(f,0,false);
        verifyMPI<exasim::detail::AbiAdapter>(f,0,false);
    } catch(const std::exception& e) {
        std::fprintf(stderr,"rank %d: %s\n",rank,e.what()); code=1;
    }
    Kokkos::finalize(); MPI_Finalize(); return code;
}
