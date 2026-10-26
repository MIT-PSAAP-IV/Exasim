#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>
using std::partial_sum;
using std::ofstream;
using std::string;
using std::cout;
using std::endl;
#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/pblas.h"

// Include the actual nonlinear control flow. Only the PDE residual, linear
// direction generator and output/exit boundary are mocked. Capturing the exit
// permits checking that failing w is saved before snapshot destruction.
struct TestExit : std::runtime_error { using std::runtime_error::runtime_error; };
static std::vector<char> outputEvents;
static double savedW = 0;
template<class T> std::string NumberToString(T value)
{ std::ostringstream s; s<<value; return s.str(); }
template<class T> void writearray(std::ofstream&, T*, Int) {}
template<class T> void writearray2file(std::string name,T* data,Int n,Int)
{
    Kokkos::fence();
    const bool auxiliary=name.find("_w_np")!=std::string::npos;
    outputEvents.push_back(auxiliary?'w':'u');
    if(auxiliary && n>0) savedW=data[0];
}
static double SolutionBenchmarkStart(Int) { return 0; }
static double SolutionBenchmarkStop(double,Int) { return 0; }
template<class... A> void hdgGetQ(A&&...);
template<class... A> void hdgGetDUDG(A&&...);
template<class M,class... A> void GetW(A&&...);

template<class T,class I> class CDiscretizationT {
public:
    commonstructT<T,I> common{};
    solstructT<T,I> sol{};
    resstructT<T,I> res{};
    appstructT<T,I> app{};
    masterstructT<T,I> master{};
    meshstructT<T,I> mesh{};
    tempstructT<T,I> tmp{};
};
using CDiscretization=CDiscretizationT<dstype,Int>;
struct TestModel {};
enum class TestMode { ColdRetry, EpsilonRetry, TinyDamping, FatalTrial, FailedSearch, Converged, PartialStep };

template<class M,class T=dstype,class I=Int> class CResidual {
public:
    CDiscretization& disc;
    TestMode mode;
    int calls=0;
    std::vector<double> trialSeeds;
    void evalResidual(T* result,T* u,Int) {
        Kokkos::fence();
        if(calls>0) trialSeeds.push_back(disc.sol.wdg[0]);
        ++calls;
        disc.sol.udg[0]=u[0];
        result[0]=u[0];
        if(mode==TestMode::TinyDamping && u[0]<1)
            result[0]=(1-u[0] <= 1e-9) ? 0.5 : 2;
        disc.sol.wdg[0]=100+u[0];
        if(mode==TestMode::FatalTrial && calls>1) {
            ArraySetNaN(result,1);
            ArraySetNaN(disc.sol.wdg,1);
        }
    }
};
template<class M,class T=dstype,class I=Int> class CAssembler {};
template<class M,class T=dstype,class I=Int> class CPreconditioner {
public:
    int builds=0;
    bool poisonAuxiliary=false;
    void ComputeLDGPreconditioner(CDiscretization& disc,T*,T*,Int) {
        ++builds;
        if(poisonAuxiliary) disc.sol.wdg[0]=-999;
    }
};
template<class M,class T=dstype,class I=Int> class CSolver {
public:
    sysstructT<T,I> sys{};
    solverstatestruct state{};
    TestMode mode;
    int calls=0,updates=0;
    std::vector<int> dimensions,removedIndices;
    std::vector<double> epsilons,linearSeeds;
    int linearSolve(CResidual<M,T,I>&,CAssembler<M,T,I>&,CDiscretization& disc,
                    CPreconditioner<M,T,I>&,std::ofstream&,Int,Int) {
        Kokkos::fence();
        dimensions.push_back(state.RBcurrentdim);
        removedIndices.push_back(state.RBremovedind);
        epsilons.push_back(disc.common.solverparams.matvecTol);
        linearSeeds.push_back(disc.sol.wdg[0]);
        ++calls;
        if(mode==TestMode::ColdRetry && calls==1) sys.x[0]=0; // no progress
        else if(mode==TestMode::EpsilonRetry && calls==1) sys.x[0]=1;
        else if(mode==TestMode::FatalTrial || mode==TestMode::FailedSearch) sys.x[0]=1;
        else sys.x[0]=(mode==TestMode::PartialStep ? -0.5 : -1.0)*sys.u[0];
        // A linear solve is permitted to make unaccepted auxiliary probes.
        disc.sol.wdg[0]=-77;
        return 0;
    }
    void updateRB(CDiscretization&,CPreconditioner<M,T,I>&,Int) { ++updates; }
};
template<class M> class CSolutionWriter {
public:
    void crashDump(Int) { outputEvents.push_back('c'); }
};

#undef error
#define error(message) throw TestExit(message)
#include "../../backend/Solution/nonlinearsolver.cpp"

static int failures=0;
static void require(bool okay,const char* message)
{
    if(!okay) { std::fprintf(stderr,"FAIL: %s\n",message); ++failures; }
}

struct Fixture {
    CDiscretization disc;
    CResidual<TestModel> residual;
    CAssembler<TestModel> assembler;
    CPreconditioner<TestModel> preconditioner;
    CSolver<TestModel> solver;
    CSolutionWriter<TestModel> writer;
    CNonlinearSolver<TestModel> nonlinear;
    Kokkos::View<double*> u{"u",1},x{"x",1},r{"r",1},v{"v",1},
                          udg{"udg",1},w{"w",1};
    std::ofstream output;
    Fixture(TestMode mode):residual{disc,mode},
        nonlinear(disc,residual,assembler,preconditioner,solver,writer) {
        solver.mode=mode;
        auto& c=disc.common;
        c.grid.npe=1; c.components.ncu=1; c.components.nc=1; c.components.ncw=1;
        c.meshsizes.ne1=1; c.sizes.ndof1=1; c.sizes.ndofudg1=1;
        c.spatialScheme=0; c.solverparams.preconditioner=0;
        c.solverparams.nonlinearSolverMaxIter=1; c.solverparams.nonlinearSolverTol=1e-14;
        c.solverparams.matvecTol=1e-6; c.solverparams.RBdim=10;
        c.fileout="captured_test_output";
        c.mpiRank=1; // suppress expected line-search progress chatter
        solver.state.RBcurrentdim=mode==TestMode::ColdRetry ? 2 : 0;
        solver.state.RBremovedind=mode==TestMode::ColdRetry ? 2 : 0;
        solver.sys.u=u.data(); solver.sys.x=x.data(); solver.sys.r=r.data(); solver.sys.v=v.data();
        disc.sol.udg=udg.data(); disc.sol.wdg=w.data(); disc.sol.szwdg=1;
        ArraySetValue(u.data(),mode==TestMode::Converged ? 1e-16 : 1.0,1);
        ArraySetValue(w.data(),7.0,1);
    }
    SolveStatus run() { return nonlinear.PTCsolver(output,0); }
};

int main(int argc,char** argv)
{
    Kokkos::initialize(argc,argv);
    {
        // The transient block preconditioner is cached through Newton
        // corrections and later DIRK stages.
        for(int stage : {0,1}) {
            Fixture f(TestMode::PartialStep);
            f.disc.common.solverparams.preconditioner=1;
            f.disc.common.solverparams.nonlinearSolverMaxIter=3;
            f.disc.common.timeparams.tdep=1;
            f.disc.common.timestate.currentstage=stage;
            auto status=f.run();
            require(status.finite && !status.converged && status.iterations==3,
                    "cached preconditioner preserves nonlinear iteration budget");
            require(f.preconditioner.builds==(stage==0 ? 1 : 0),
                    "transient preconditioner builds once at the first DIRK stage");
            require(f.solver.linearSeeds==std::vector<double>({101,100.5,100.25}),
                    "cached preconditioner preserves linear-solve auxiliary seeds");
            require(f.u(0)==0.125 && f.w(0)==100.125,
                    "cached preconditioner preserves the accepted solution");
        }
        {
            Fixture f(TestMode::Converged);
            auto status=f.run();
            require(status.converged && status.finite && status.iterations==0,
                    "already converged state returns without Newton iteration");
            require(f.solver.calls==0 && f.solver.updates==0,"already converged state must skip linear solver/RB insertion");
        }
        for(int stage : {0,1}) {
            Fixture f(TestMode::Converged);
            f.disc.common.solverparams.preconditioner=1;
            f.disc.common.timeparams.tdep=1;
            f.disc.common.timestate.currentstage=stage;
            auto status=f.run();
            require(status.converged && f.solver.calls==0,"converged DIRK stage skips GMRES");
            require(f.preconditioner.builds==(stage==0 ? 1 : 0),
                    "converged first DIRK stage must retain preconditioner setup for later stages");
        }
        {
            Fixture f(TestMode::ColdRetry);
            auto status=f.run();
            require(status.converged && status.finite,"cold RB retry converges");
            require(f.solver.calls==2,"equal residual outside tolerance must reject RB direction");
            if(f.solver.calls==2) {
                require(f.solver.dimensions[0]==2 && f.solver.dimensions[1]==0 &&
                        f.solver.removedIndices[1]==0,"cold retry clears both RB counters");
                require(f.solver.epsilons[0]==1e-6 && f.solver.epsilons[1]==1e-6,
                        "cold retry precedes any finite-difference epsilon reduction");
            }
            for(double seed:f.residual.trialSeeds)
                require(seed==101,"every nonlinear trial must start from saved w seed");
            for(double seed:f.solver.linearSeeds)
                require(seed==101,"retry linear solver must see restored base w");
            require(f.solver.updates==1,"only accepted direction enters RB");
            Kokkos::fence();
            require(f.w(0)==100,"accepted auxiliary state is committed");
            require(f.disc.common.solverparams.matvecTol==1e-6,"cold retry preserves configured epsilon");
        }
        {
            Fixture f(TestMode::EpsilonRetry);
            auto status=f.run();
            require(status.converged && f.solver.calls==2,"finite no-RB failure retries successfully");
            if(f.solver.calls==2)
                require(std::abs(f.solver.epsilons[1]-1e-7)<1e-22,"no-RB retry reduces epsilon locally");
            require(f.disc.common.solverparams.matvecTol==1e-6,"epsilon restored after successful retry");
        }
        {
            Fixture f(TestMode::TinyDamping);
            auto status=f.run();
            require(status.finite && !status.converged,"tiny damped finite decrease remains a valid nonlinear step");
            require(f.solver.calls==1 && f.solver.updates==0,"near-zero damped step must not enter RB");
        }
        {
            outputEvents.clear(); savedW=0;
            Fixture f(TestMode::FatalTrial);
            bool exited=false;
            try { f.run(); } catch(const TestExit&) { exited=true; }
            require(exited && f.solver.calls==1,"nonfinite nonlinear trial must exit, not retry");
            require(outputEvents==std::vector<char>({'u','w','c'}),"crash output saves u and w before writer crash dump");
            require(!is_finite_bitwise(savedW),"crash output must retain failing w before rollback");
        }
        {
            outputEvents.clear();
            Fixture f(TestMode::FailedSearch);
            bool exited=false;
            try { f.run(); } catch(const TestExit& e) {
                exited=std::string(e.what()).find("line search failed")!=std::string::npos;
            }
            require(exited && f.solver.calls==4,"exhausted finite line search retains bounded crash path");
            require(f.disc.common.solverparams.matvecTol==1e-6,"epsilon restored before exhausted line-search exit");
            require(outputEvents==std::vector<char>({'u','w','c'}),"finite failure still saves all crash output");
        }
    }
    Kokkos::finalize();
    std::printf("nonlinear RB retry: %d failures\n",failures);
    return failures ? 1 : 0;
}
