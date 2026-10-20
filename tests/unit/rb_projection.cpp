#include <Kokkos_Core.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

#include "backend/Common/common.h"
#include "backend/Common/cpuimpl.h"
#include "backend/Common/kokkosimpl.h"
#include "backend/Common/pblas.h"
#include "backend/Discretization/ioutilities.hpp"

// Actual RB projection, insertion, trial rollback, and GMRES; only the PDE
// assembler/residual and allocation fixture are mocked for this CPU test.
namespace exasim { namespace nonmatching { template<class T, class I> struct NonmatchingRuntime; } }
template<class T, class I> class CDiscretizationT {
public:
    commonstructT<T,I> common{};
    solstructT<T,I> sol{};
    resstructT<T,I> res{};
    appstructT<T,I> app{};
    masterstructT<T,I> master{};
    meshstructT<T,I> mesh{};
    tempstructT<T,I> tmp{};
    // This fixture has no coupled interface; match the production accessor.
    exasim::nonmatching::NonmatchingRuntime<T,I>* getNonmatchingRuntime() noexcept { return nullptr; }
};
using CDiscretization=CDiscretizationT<dstype,Int>;
struct ProjectionModel {};
template<class M, class T=dstype, class I=Int> class CAssembler {
public:
    CDiscretizationT<T,I>& disc;
    T diagonal[2]={1,1};
    bool mutateAuxiliary=false;
    int calls=0;
    T lastDirectionNorm=-1;
    void evalMatVec(T* result,T* v,T*,T*,I) {
        ++calls;
        for(int k=0;k<2;++k) result[k]=diagonal[k]*v[k];
        if(mutateAuxiliary) disc.sol.wdg[0]=999;
    }
    void evalMatVec(T* result,T* v,T* u,T* b,I,I backend) {
        evalMatVec(result,v,u,b,backend);
    }
    void evalMatVecWithNorm(T* result,T* v,T* u,T* b,T norm,I backend) {
        lastDirectionNorm=norm;
        evalMatVec(result,v,u,b,backend);
    }
};
template<class M, class T=dstype, class I=Int> class CResidual {
public:
    CDiscretizationT<T,I>& disc;
    T diagonal[2]={1,1};
    bool nonfiniteTrial=false;
    vector<T> incomingSeeds;
    void evalResidual(T* residual,T* u,I) {
        incomingSeeds.push_back(disc.sol.wdg[0]);
        for(int k=0;k<2;++k) {
            disc.sol.udg[k]=u[k];
            residual[k]=diagonal[k]*u[k];
        }
        disc.sol.wdg[0]=100+u[0]+2*u[1];
        if(nonfiniteTrial && u[0]<0) {
            residual[0]=std::numeric_limits<T>::quiet_NaN();
            disc.sol.wdg[0]=std::numeric_limits<T>::quiet_NaN();
        }
    }
};
#include "backend/Preconditioning/preconditioner.cpp"
#include "backend/Solver/solver.h"
#include "backend/Solver/getpoly.cpp"
#include "backend/Solver/gmres.cpp"
#include "backend/Solver/ptcsolver.cpp"
template<class M,class T,class I>
CSolver<M,T,I>::CSolver(CDiscretization&,Int,ExasimExecutionMode) {}
template<class M,class T,class I>
CSolver<M,T,I>::~CSolver() {}

struct Fixture {
    CDiscretization disc;
    CSolver<ProjectionModel> solver;
    CAssembler<ProjectionModel> assembler;
    CResidual<ProjectionModel> residual;
    double u[2]{},x[2]{},b[2]{},r[2]{},v[8]{},small[128]{};
    double tempn[128]{},tempg[128]{},udg[2]{},ru[2]{},w[1]{17};
    int ipiv[16]{};
    Fixture():solver(disc,0),assembler{disc},residual{disc} {
        disc.common.sizes.ndof1=2; disc.common.sizes.ndofuhat=2;
        disc.common.grid.npe=1; disc.common.components.ncu=1;
        disc.common.components.nc=1; disc.common.components.ncw=1;
        disc.common.meshsizes.ne1=2; disc.common.spatialScheme=0;
        disc.common.mpiRank=1; disc.common.ppdegree=0;
        disc.common.solverparams.RBdim=2;
        disc.common.solverparams.preconditioner=0;
        disc.common.solverparams.gmresRestart=2;
        disc.common.solverparams.linearSolverMaxIter=8;
        disc.common.solverparams.linearSolverTol=1e-10;
        disc.common.solverparams.nonlinearSolverTol=1e-14;
        disc.sol.udg=udg; disc.sol.wdg=w; disc.sol.szwdg=1;
        disc.res.Ru=ru; disc.tmp.tempn=tempn; disc.tmp.tempg=tempg;
        solver.sys.u=u; solver.sys.x=x; solver.sys.b=b; solver.sys.r=r;
        solver.sys.v=v; solver.sys.tempmem=small; solver.sys.ipiv=ipiv;
    }
    void basis(CPreconditioner<ProjectionModel>& p,double a00=1,double a10=0,
               double a01=0,double a11=1) {
        const double values[]={a00,a10,a01,a11};
        for(int k=0;k<4;++k) p.precond.W[k]=p.precond.U[k]=values[k];
        solver.state.RBcurrentdim=2; solver.state.RBremovedind=0;
    }
    void project(CPreconditioner<ProjectionModel>& p,int scheme) {
        disc.common.spatialScheme=scheme;
        if(scheme==0) p.ComputeInitialGuessAndPreconditioner(solver.sys,solver.state,assembler,disc,0);
        else p.ComputeInitialGuessAndPreconditioner(solver.sys,solver.state,assembler,disc,2,1,0);
    }
};

static int failures=0;
static void check(bool condition,const char* message) {
    if(!condition) { cerr<<"FAIL: "<<message<<'\n'; ++failures; }
}
static bool close(double a,double b,double scale=1) {
    return is_finite_bitwise(a) && abs(a-b)<=1e-10*scale;
}

static void testGram() {
    double H[]={1e-160,.25,.25,1e160};
    double rhs[]={1.25e-80,-2.5e80},work[2]{};
    check(SolveReducedBasisGram(H,rhs,work,2),"equilibrated Gram solve accepts differently scaled independent columns");
    check(close(rhs[0],2e80,2e80) && close(rhs[1],-3e-80,3e-80),"equilibrated Gram solve returns correct unscaled coefficients");
    check(!SolveReducedBasisGram(H,rhs,work,0),"empty Gram system is not used");
    for(int mode=0;mode<6;++mode) {
        double A[]={1,0,0,1},b[]={1,2};
        if(mode==0) A[1]=A[2]=1;
        if(mode==1) A[1]=A[2]=1-1e-12;
        if(mode==2) A[0]=0;
        if(mode==3) A[1]=std::numeric_limits<double>::quiet_NaN();
        if(mode==4) b[1]=std::numeric_limits<double>::infinity();
        if(mode==5) A[3]=-1;
        check(!SolveReducedBasisGram(A,b,work,2),"invalid or numerically dependent Gram system is rejected");
    }
}

static void testProjection(int scheme) {
    {
        Fixture f; CPreconditioner<ProjectionModel> p(f.disc,0);
        f.basis(p); f.assembler.diagonal[0]=2; f.assembler.diagonal[1]=3;
        p.precond.U[0]=2; p.precond.U[3]=3;
        f.b[0]=4; f.b[1]=-6;
        f.project(p,scheme);
        const double sign=scheme==0 ? -1 : 1;
        check(close(f.x[0],sign*2) && close(f.x[1],-sign*2),"LDG/HDG projection preserves the correct RHS sign");
        check(f.solver.state.RBcurrentdim==2,"independent basis remains active");
        if(scheme==0) check(f.assembler.lastDirectionNorm==1,"LDG normalized basis uses known unit direction norm");
    }
    for(int mode=0;mode<5;++mode) {
        Fixture f; CPreconditioner<ProjectionModel> p(f.disc,0);
        f.basis(p,1,0,1,mode==1 ? 1e-6 : 0);
        if(mode>=2) f.basis(p);
        if(mode==2) p.precond.U[0]=std::numeric_limits<double>::quiet_NaN();
        if(mode==3) p.precond.W[3]=std::numeric_limits<double>::infinity();
        f.b[0]=mode==4 ? std::numeric_limits<double>::quiet_NaN() : 1;
        f.b[1]=2; f.x[0]=7; f.x[1]=8;
        f.project(p,scheme);
        check(f.solver.state.RBcurrentdim==0 && f.solver.state.RBremovedind==0,"unusable RB basis resets both recycling counters");
        check(f.x[0]==0 && f.x[1]==0,"unusable RB basis returns a clean zero GMRES guess");
    }
    if(scheme==1) {
        Fixture f; CPreconditioner<ProjectionModel> p(f.disc,0);
        f.basis(p,1e-80,0,0,1e80); f.b[0]=4; f.b[1]=-6;
        f.project(p,scheme);
        check(close(f.x[0],4) && close(f.x[1],-6),"raw differently scaled HDG basis yields scale-invariant projection");
    }
}

static void testInsertion() {
    for(bool explicitSize:{false,true}) {
        Fixture f; CPreconditioner<ProjectionModel> p(f.disc,0);
        f.x[0]=3e-16; f.x[1]=4e-16;
        if(explicitSize) f.solver.updateRB(f.disc,p,2,0);
        else f.solver.updateRB(f.disc,p,0);
        check(close(p.precond.W[0],.6) && close(p.precond.W[1],.8),"LDG insertion stores a unit direction even for tiny corrections");
        check(f.x[0]==3e-16 && f.x[1]==4e-16,"LDG basis normalization does not alter the accepted correction");
        check(f.solver.state.RBcurrentdim==1 && f.solver.state.RBremovedind==1,"LDG basis insertion advances counters");
        f.x[0]=0; f.x[1]=0; f.solver.updateRB(f.disc,p,0);
        check(f.solver.state.RBcurrentdim==1 && f.solver.state.RBremovedind==1,"zero update is not inserted");
        f.x[0]=std::numeric_limits<double>::quiet_NaN(); f.solver.updateRB(f.disc,p,0);
        check(f.solver.state.RBcurrentdim==1 && f.solver.state.RBremovedind==1,"nonfinite update is not inserted");
        f.x[0]=1; f.x[1]=0; f.solver.updateRB(f.disc,p,0);
        check(f.solver.state.RBcurrentdim==2 && f.solver.state.RBremovedind==0,"full basis replacement index wraps correctly");
        f.disc.common.solverparams.RBdim=0; f.solver.updateRB(f.disc,p,0);
        check(f.solver.state.RBcurrentdim==2 && f.solver.state.RBremovedind==0,"disabled RB insertion is a no-op");
    }
    {
        Fixture f; CPreconditioner<ProjectionModel> p(f.disc,0);
        f.disc.common.spatialScheme=1; f.x[0]=3; f.x[1]=4;
        f.solver.updateRB(f.disc,p,2,0);
        check(p.precond.W[0]==3 && p.precond.W[1]==4,"HDG insertion retains its established raw-step representation");
    }
}

static void testTrialRollback() {
    for(bool nonfiniteTrial:{false,true}) {
        Fixture f; CPreconditioner<ProjectionModel> p(f.disc,0);
        f.basis(p); f.u[0]=1; f.u[1]=1;
        f.assembler.diagonal[0]=10; f.residual.diagonal[0]=10;
        f.assembler.mutateAuxiliary=true; f.residual.nonfiniteTrial=nonfiniteTrial;
        f.disc.common.solverparams.linearSolverMaxIter=0;
        ofstream output;
        f.solver.linearSolve(f.residual,f.assembler,f.disc,p,output,0,0);
        check(f.solver.state.RBcurrentdim==0 && f.solver.state.RBremovedind==0,"stale-image nonlinear trial is rejected and clears RB");
        check(f.x[0]==0 && f.x[1]==0,"rejected trial leaves cold GMRES guess");
        check(f.w[0]==103 && f.udg[0]==1 && f.udg[1]==1 && f.b[0]==10 && f.b[1]==1,"rejected RB trial restores base auxiliary state, solution, and RHS");
        check(f.residual.incomingSeeds.size()==3 && f.residual.incomingSeeds[1]==103 && f.residual.incomingSeeds[2]==103,"trial and base reevaluation start from identical accepted auxiliary seed");
    }
    {
        Fixture f; CPreconditioner<ProjectionModel> p(f.disc,0);
        f.basis(p); f.u[0]=1; f.u[1]=1; f.assembler.mutateAuxiliary=true;
        ofstream output;
        int status=f.solver.linearSolve(f.residual,f.assembler,f.disc,p,output,0,0);
        check(status==1 && f.x[0]==-1 && f.x[1]==-1,"exact nonlinear RB trial can return its correction without GMRES");
        check(f.w[0]==103,"early successful RB trial also rolls auxiliary state back");
    }
}

int main(int argc,char** argv) {
    Kokkos::initialize(argc,argv);
    testGram(); testProjection(0); testProjection(1); testInsertion(); testTrialRollback();
    Kokkos::finalize();
    if(!failures) cout<<"RB projection, insertion, and rollback regressions passed\n";
    return failures ? 1 : 0;
}
