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

// Exercise both actual GMRES implementations and orthogonalizers. Only the
// matrix action, preconditioner, and discretization allocation are fixtures.
template<class T, class I> class CDiscretizationT {
public:
    commonstructT<T, I> common{};
};
using CDiscretization = CDiscretizationT<dstype, Int>;
struct GmresModel {};
template<class M, class T=dstype, class I=Int> class CAssembler {
public:
    T matrix[4] = {1, 0, 0, 1}; // column major
    int calls = 0;
    bool inconsistentAction = false;
    bool nonfiniteAction = false;
    void evalMatVec(T* result, T* v, T*, T*, I) {
        ++calls;
        T scale = inconsistentAction && std::abs(v[0]*v[0]+v[1]*v[1]-1) > 1e-8 ? 2 : 1;
        T a = scale*(matrix[0]*v[0]+matrix[2]*v[1]);
        T b = scale*(matrix[1]*v[0]+matrix[3]*v[1]);
        result[0] = nonfiniteAction ? bitwise_quiet_nan<T>() : a;
        result[1] = b;
    }
    void evalMatVec(T* r, T* v, T* u, T* b, I, I backend) {
        evalMatVec(r, v, u, b, backend);
    }
    void evalMatVecWithNorm(T* r, T* v, T* u, T* b, T, I backend) {
        evalMatVec(r, v, u, b, backend);
    }
};
template<class M, class T=dstype, class I=Int> class CPreconditioner {
public:
    bool singular = false;
    void ApplyPreconditioner(T* v, sysstructT<T,I>&, CDiscretizationT<T,I>&, I) {
        if (singular) ArraySetValue(v, T(0), 2);
    }
    void ApplyPreconditioner(T* v, sysstructT<T,I>& s, CDiscretizationT<T,I>& d, I, I backend) {
        ApplyPreconditioner(v, s, d, backend);
    }
};
#include "backend/Solver/solver.h"
#include "backend/Solver/getpoly.cpp"
#include "backend/Solver/gmres.cpp"
template<class M, class T, class I>
CSolver<M,T,I>::CSolver(CDiscretization&, Int, ExasimExecutionMode) {}
template<class M, class T, class I>
CSolver<M,T,I>::~CSolver() {}

struct Fixture {
    CDiscretization disc;
    CSolver<GmresModel> solver;
    CAssembler<GmresModel> assembler;
    CPreconditioner<GmresModel> prec;
    double u[2]{}, x[2]{}, b[2]{}, r[2]{}, v[8]{}, small[128]{};
    double lam[128]{}, randvect[2]{1,2}, q[2]{}, p[2]{};
    int ipiv[8]{};
    Fixture():solver(disc, 0) {
        disc.common.grid.npe=1;
        disc.common.components.ncu=1;
        disc.common.meshsizes.ne1=2;
        disc.common.mpiRank=1; // keep regression output short
        disc.common.ppdegree=0;
        disc.common.solverparams.gmresRestart=2;
        disc.common.solverparams.linearSolverMaxIter=8;
        disc.common.solverparams.linearSolverTol=1e-10;
        disc.common.solverparams.nonlinearSolverTol=1e-30;
        solver.sys.u=u; solver.sys.x=x; solver.sys.b=b; solver.sys.r=r;
        solver.sys.v=v; solver.sys.tempmem=small;
        solver.sys.lam=lam; solver.sys.randvect=randvect;
        solver.sys.q=q; solver.sys.p=p; solver.sys.ipiv=ipiv;
    }
    int run(bool hdg) {
        return hdg ? solver.gmres(assembler,disc,prec,2,1,0)
                   : solver.gmres(assembler,disc,prec,0);
    }
};

static int failures = 0;
static void check(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
static bool close(double x, double y, double scale=1) {
    return is_finite_bitwise(x) && std::abs(x-y) <= 1e-10*scale;
}
static void testVariant(bool hdg, int orthogonalization) {
    const double sign = hdg ? 1 : -1;
    {
        Fixture f;
        f.disc.common.solverparams.gmresOrthogMethod=orthogonalization;
        f.b[0]=1; f.b[1]=2; f.x[0]=sign; f.x[1]=2*sign;
        int it=f.run(hdg);
        check(it==0 && close(f.x[0],sign) && close(f.x[1],2*sign), "exact recycled guess is preserved");
        check(f.solver.state.linearSolverRelError==0, "exact recycled guess reports zero error");
        check(is_finite_bitwise(f.solver.state.linearSolverTolFactor), "exact guess leaves finite tolerance factor");
    }
    {
        Fixture f;
        f.disc.common.solverparams.gmresOrthogMethod=orthogonalization;
        f.b[0]=1e-13; f.b[1]=2e-13; f.x[0]=sign*0.5e-13;
        f.run(hdg);
        check(close(f.x[0],sign*1e-13,1e-13) && close(f.x[1],sign*2e-13,1e-13), "tiny recycled guess is included in initial residual");
    }
    {
        Fixture f;
        f.disc.common.solverparams.gmresOrthogMethod=orthogonalization;
        f.b[0]=1; f.b[1]=2;
        int it=f.run(hdg);
        check(it==1 && close(f.x[0],sign) && close(f.x[1],2*sign), "happy Arnoldi breakdown solves identity in one iteration");
        check(f.assembler.calls>=2, "recurrence convergence is checked with actual operator");
    }
    {
        Fixture f;
        f.b[0]=1; f.b[1]=2;
        for(double& a:f.assembler.matrix) a=0;
        int it=f.run(hdg);
        check(it==1 && close(f.x[0],0) && close(f.x[1],0), "null Arnoldi breakdown preserves finite iterate");
        check(f.solver.state.linearSolverRelError>=1, "null breakdown is not reported converged");
    }
    {
        Fixture f;
        f.b[0]=1; f.b[1]=2; f.assembler.matrix[3]=0;
        f.disc.common.ppdegree=2;
        f.run(hdg);
        check(is_finite_bitwise(f.x[0]) && is_finite_bitwise(f.x[1]), "singular projected polynomial matrix is recoverable");
        check(f.solver.state.linearSolverRelError>0.1, "inconsistent singular system is not reported converged");
    }
    {
        Fixture f;
        f.x[0]=sign;
        f.solver.state.linearSolverRelError=0.5;
        check(f.run(hdg)==0 && close(f.x[0],0), "zero RHS returns zero correction");
        check(f.solver.state.linearSolverRelError==0, "zero RHS resets stale solver error");
    }
    {
        Fixture f;
        f.b[0]=1; f.b[1]=2; f.prec.singular=true;
        check(f.run(hdg)==0 && close(f.x[0],0), "singular preconditioner does not create NaN correction");
        check(!is_finite_bitwise(f.solver.state.linearSolverRelError), "singular preconditioner is not false convergence");
    }
    {
        Fixture f;
        f.b[0]=1; f.b[1]=2; f.assembler.nonfiniteAction=true;
        f.run(hdg);
        check(!is_finite_bitwise(f.x[0]) && !is_finite_bitwise(f.solver.state.linearSolverRelError), "nonfinite operator remains visible to nonlinear crash handling");
    }
    for(int mode=0; mode<2; ++mode) {
        Fixture f;
        f.b[0]=1; f.b[1]=2;
        if(mode==0) f.b[0]=bitwise_infinity<double>();
        else f.x[0]=bitwise_quiet_nan<double>();
        check(f.run(hdg)==0 && !is_finite_bitwise(f.x[0]), "nonfinite RHS or warm start is rejected before Arnoldi");
    }
    {
        Fixture f;
        f.b[0]=1; f.b[1]=2; f.assembler.inconsistentAction=true;
        int it=f.run(hdg);
        check(it==f.disc.common.solverparams.linearSolverMaxIter, "inconsistent operator does not falsely stop at recurrence convergence");
        check(f.solver.state.linearSolverRelError>0.1, "reported error uses actual residual for inconsistent action");
    }
    {
        Fixture f;
        f.disc.common.solverparams.gmresOrthogMethod=orthogonalization;
        f.assembler.matrix[0]=2; f.assembler.matrix[1]=-.2;
        f.assembler.matrix[2]=.5; f.assembler.matrix[3]=3;
        f.b[0]=sign; f.b[1]=sign*(-6.2);
        if(hdg) f.disc.common.couplingparams.ndofuhatinterface=1;
        f.run(hdg);
        check(close(f.x[0],1) && close(f.x[1],-2), "LDG/HDG signs and weighted HDG inner product solve nonsymmetric matrix");
    }
    {
        Fixture f;
        f.disc.common.solverparams.gmresRestart=1;
        f.disc.common.solverparams.linearSolverMaxIter=60;
        f.disc.common.solverparams.gmresOrthogMethod=orthogonalization;
        f.assembler.matrix[0]=2; f.assembler.matrix[1]=-.2;
        f.assembler.matrix[2]=.5; f.assembler.matrix[3]=3;
        f.b[0]=sign; f.b[1]=sign*(-6.2);
        int it=f.run(hdg);
        check(it>2 && it<60 && close(f.x[0],1) && close(f.x[1],-2), "restarted GMRES preserves correction accumulation and iteration count");
    }
    for(int mode=0; mode<4; ++mode) {
        Fixture f;
        f.b[0]=1; f.b[1]=2;
        f.disc.common.ppdegree=mode==2 ? 3 : 2;
        if(mode==1) f.randvect[0]=f.randvect[1]=0;
        if(mode==3) f.disc.common.solverparams.gmresRestart=3;
        const int configuredDegree=f.disc.common.ppdegree;
        f.run(hdg);
        check(close(f.x[0],sign) && close(f.x[1],2*sign), "degenerate polynomial setup falls back to ordinary GMRES");
        check(f.disc.common.ppdegree==configuredDegree, "polynomial fallback does not mutate configured degree");
    }
    {
        Fixture f;
        f.b[0]=sign; f.b[1]=sign*(-6.2);
        f.assembler.matrix[0]=2; f.assembler.matrix[1]=-.2;
        f.assembler.matrix[2]=.5; f.assembler.matrix[3]=3;
        f.disc.common.ppdegree=2;
        f.run(hdg);
        check(close(f.x[0],1) && close(f.x[1],-2), "full-rank polynomial preconditioner still solves correctly");
    }
}

int main(int argc, char** argv) {
    Kokkos::initialize(argc,argv);
    for(bool hdg:{false,true})
        for(int orthogonalization:{0,1}) testVariant(hdg,orthogonalization);
    Kokkos::finalize();
    if(!failures) std::cout << "GMRES recycling and LDG/HDG regressions passed\n";
    return failures ? 1 : 0;
}
