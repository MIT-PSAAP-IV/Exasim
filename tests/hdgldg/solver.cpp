// Independent small block-system oracle for the coupled solver signs.
// Dense arrays below are TEST-ONLY; production borrows native solver storage.
#define HDGLDG_DERIVATIVE_NO_MAIN
#include "derivative.cpp"
#include "backend/Solver/hdgldgsolver.hpp"
#include "tests/models/navierstokes2d.hpp"
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

template<int N>
static std::array<double,N> denseSolve(std::array<double,N*N> a,std::array<double,N> b)
{
    for(int col=0;col<N;++col){int pivot=col;
        for(int row=col+1;row<N;++row)if(std::abs(a[row+N*col])>std::abs(a[pivot+N*col]))pivot=row;
        if(std::abs(a[pivot+N*col])<.1)throw std::runtime_error("oracle matrix unexpectedly singular");
        for(int j=col;j<N;++j)std::swap(a[col+N*j],a[pivot+N*j]);std::swap(b[col],b[pivot]);
        for(int row=col+1;row<N;++row){const double factor=a[row+N*col]/a[col+N*col];
            for(int j=col;j<N;++j)a[row+N*j]-=factor*a[col+N*j];b[row]-=factor*b[col];}
    }
    for(int i=N-1;i>=0;--i){for(int j=i+1;j<N;++j)b[i]-=a[i+N*j]*b[j];b[i]/=a[i+N*i];}return b;
}

struct BlockOracle {
    std::array<double,4> A{4,1,1,3},B{1,.2,-.4,.7},C{.5,-.3,.1,.8};
    std::array<double,4> F{3,.1,.2,2},D{.4,-.02,.05,.3},G{.2,.1,-.1,.4};
    std::array<double,4> E{.3,-.2,.2,.1},L{2,.1,.1,1.8};
    std::array<double,16> reduced{};
    std::array<double,36> full{};
    BlockOracle(){
        for(int j=0;j<2;++j){const auto response=denseSolve<2>(A,{B[2*j],B[1+2*j]});
            for(int i=0;i<2;++i){double schur=F[i+2*j];
                for(int k=0;k<2;++k)schur-=C[i+2*k]*response[k];
                reduced[i+4*j]=schur+D[i+2*j];reduced[i+4*(j+2)]=G[i+2*j];
                reduced[i+2+4*j]=E[i+2*j];reduced[i+2+4*(j+2)]=L[i+2*j];
                full[i+6*j]=A[i+2*j];full[i+6*(j+2)]=B[i+2*j];
                full[i+2+6*j]=C[i+2*j];full[i+2+6*(j+2)]=F[i+2*j]+D[i+2*j];
                full[i+2+6*(j+4)]=G[i+2*j];full[i+4+6*(j+2)]=E[i+2*j];
                full[i+4+6*(j+4)]=L[i+2*j];}}
    }
    std::array<double,6> correction(const std::array<double,6>& residual)const {
        const auto local=denseSolve<2>(A,{residual[0],residual[1]});
        std::array<double,4> rhs{};
        for(int i=0;i<2;++i){rhs[i]=-residual[i+2];for(int k=0;k<2;++k)rhs[i]+=C[i+2*k]*local[k];
            rhs[i+2]=-residual[i+4];}
        const auto independent=denseSolve<4>(reduced,rhs);
        std::array<double,2> hRhs{-residual[0],-residual[1]};
        for(int i=0;i<2;++i)for(int k=0;k<2;++k)hRhs[i]-=B[i+2*k]*independent[k];
        const auto h=denseSolve<2>(A,hRhs);
        return {h[0],h[1],independent[0],independent[1],independent[2],independent[3]};
    }
};

static int blockCheck(int rank,int size)
{
    const BlockOracle oracle;
    const std::array<double,6> residual{.7,-.2,.4,-.8,.9,.3};
    const auto correction=oracle.correction(residual);
    std::array<double,6> rhs;for(int i=0;i<6;++i)rhs[i]=-residual[i];
    const auto independent=denseSolve<6>(oracle.full,rhs);
    double error=0,closure=0;
    for(int i=0;i<6;++i){error=std::max(error,std::abs(correction[i]-independent[i]));double row=residual[i];
        for(int j=0;j<6;++j)row+=oracle.full[i+6*j]*correction[j];closure=std::max(closure,std::abs(row));}
    // Every reduced unknown is owned exactly once. At np3 the H rank owns
    // two slots and each L rank one; np4 uses noncontiguous H ranks {0,2}.
    std::vector<int> owned;
    if(size==2)owned=rank==0?std::vector<int>{0,1}:std::vector<int>{2,3};
    else if(size==3)owned=rank==0?std::vector<int>{0,1}:std::vector<int>{rank+1};
    else owned=rank==0?std::vector<int>{0}:rank==2?std::vector<int>{1}:
        rank==1?std::vector<int>{2}:std::vector<int>{3};
    const std::array<double,4> v{.3,-.7,1.1,.4};
    double local=0;int localCount=owned.size();for(int i:owned)local+=v[i]*v[i];
    double normSquared=0;int count=0;MPI_Allreduce(&local,&normSquared,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(&localCount,&count,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    double exact=0;for(double value:v)exact+=value*value;
    const double bound=256*std::numeric_limits<double>::epsilon();
    int failures=count!=4 || std::abs(normSquared-exact)>bound || error>bound || closure>bound;
    // Normalizing a tiny nonzero direction must not select the zero branch.
    for(double scale:{0.,1.,1e-20,1e20}){
        std::array<double,4> direct{},normalized{};double norm=std::abs(scale)*std::sqrt(normSquared);
        for(int i=0;i<4;++i)for(int j=0;j<4;++j){direct[i]+=oracle.reduced[i+4*j]*(scale*v[j]);
            if(norm>0)normalized[i]+=oracle.reduced[i+4*j]*(scale*v[j]/norm);}
        for(int i=0;i<4;++i)failures+=std::abs(direct[i]-norm*normalized[i])>bound*std::max(1.,std::abs(direct[i]));
    }
    if(!rank)std::cout<<"BLOCK correctionError="<<error<<" closure="<<closure<<" owned="<<count<<" failures="<<failures<<'\n';
    return failures;
}

struct SolverDiffusion : PolynomialDiffusion<2> {
    KOKKOS_INLINE_FUNCTION static void initu(double out[],const double x[],const double[],const double[])
    {for(int c=0;c<3;++c)out[c]=value(c,x);}
};

struct NonlinearSolverDiffusion : NonlinearDiffusion<2> {
    using Base=NonlinearDiffusion<2>;
    KOKKOS_INLINE_FUNCTION static void initu(double out[],const double x[],const double[],const double[])
    {for(int c=0;c<3;++c)out[c]=value(c,x);}
    KOKKOS_INLINE_FUNCTION static void source(double out[],const double x[],const double[],
        const double[],const double[],const double[],const double[],double)
    {
        double exactU[9],linearFlux[6],linearSource[3];
        for(int dir=0;dir<3;++dir)for(int c=0;c<3;++c)exactU[c+3*dir]=value(c,x,dir);
        PolynomialDiffusion<2>::source(linearSource,x,exactU,nullptr,nullptr,nullptr,nullptr,0);
        TraceDiffusion<2>::flux(linearFlux,x,exactU,nullptr,nullptr,nullptr,nullptr,0);
        const double xy[3]={.13,.27,-.31};
        for(int c=0;c<3;++c){double mixed=0,chain=0;
            for(int a=0;a<3;++a)mixed+=.5*coefficient(c,a)*xy[a];
            for(int dir=0;dir<2;++dir)chain+=exactU[c+3*(dir+1)]*linearFlux[c+3*dir];
            out[c]=(1+exactU[c]*exactU[c])*(linearSource[c]-mixed)-2*exactU[c]*chain;}
    }
    KOKKOS_INLINE_FUNCTION static void fbou(double out[],int ib,const double x[],const double u[],
        const double[],const double[],const double h[],const double n[],const double tau[],const double[],const double[],double)
    {
        double f[6];Base::flux(f,x,u,nullptr,nullptr,nullptr,nullptr,0);
        for(int c=0;c<3;++c){out[c]=tau[0]*(u[c]-h[c]);
            for(int dir=0;dir<2;++dir)out[c]+=f[c+3*dir]*n[dir];
            if(ib==5)out[c]=std::numeric_limits<double>::quiet_NaN();}
    }
    KOKKOS_INLINE_FUNCTION static void fhat(double out[],const double x[],const double u1[],const double u2[],
        const double[],const double[],const double[],const double[],const double[],const double n[],
        const double tau[],const double[],const double[],double)
    {
        double f1[6],f2[6];Base::flux(f1,x,u1,nullptr,nullptr,nullptr,nullptr,0);
        Base::flux(f2,x,u2,nullptr,nullptr,nullptr,nullptr,0);
        for(int c=0;c<3;++c){out[c]=tau[0]*(u1[c]-u2[c]);
            for(int dir=0;dir<2;++dir)out[c]+=.5*(f1[c+3*dir]+f2[c+3*dir])*n[dir];}
    }
};

// Reuse the generated compressible NS flux/Jacobian, not its placeholder LDG
// boundaries. This is a constant-state Dirichlet smoke test, NOT a NACA case.
struct CompressibleSmoke : GeneratedModel {
    static constexpr int nparam=0;
    KOKKOS_INLINE_FUNCTION static double value(int c,const double[],int derivative=0)
    {const double state[4]={1,.1,0,.255};return derivative?0:state[c];}
    KOKKOS_INLINE_FUNCTION static void initu(double out[],const double x[],const double[],const double[])
    {for(int c=0;c<4;++c)out[c]=value(c,x);}
#define NS_VOLUME_ARGS const double x[],const double u[],const double v[],const double w[],const double[],const double eta[],double t
    KOKKOS_INLINE_FUNCTION static void flux(double f[],NS_VOLUME_ARGS)
    {const double mu[8]={1.4,100,.72,.25,1,.1,0,.255};GeneratedModel::flux(f,x,u,v,w,mu,eta,t);}
    KOKKOS_INLINE_FUNCTION static void flux_jac_uq(double f[],NS_VOLUME_ARGS)
    {const double mu[8]={1.4,100,.72,.25,1,.1,0,.255};GeneratedModel::flux_jac_uq(f,x,u,v,w,mu,eta,t);}
#undef NS_VOLUME_ARGS
#define NS_BOUNDARY_ARGS int ib,const double x[],const double u[],const double[],const double[],const double h[],const double n[],const double tau[],const double[],const double[],double
    KOKKOS_INLINE_FUNCTION static void ubou(double out[],NS_BOUNDARY_ARGS)
    {for(int c=0;c<4;++c)out[c]=ib==5?std::numeric_limits<double>::quiet_NaN():value(c,x);}
    KOKKOS_INLINE_FUNCTION static void fbou(double out[],NS_BOUNDARY_ARGS)
    {double f[8];flux(f,x,u,nullptr,nullptr,nullptr,nullptr,0);
        for(int c=0;c<4;++c){out[c]=tau[0]*(u[c]-h[c]);for(int d=0;d<2;++d)out[c]+=f[c+4*d]*n[d];
            if(ib==5)out[c]=std::numeric_limits<double>::quiet_NaN();}}
    KOKKOS_INLINE_FUNCTION static void fbou_hdg(double out[],NS_BOUNDARY_ARGS)
    {for(int c=0;c<4;++c)out[c]=ib==5?std::numeric_limits<double>::quiet_NaN():tau[0]*(value(c,x)-h[c]);}
    KOKKOS_INLINE_FUNCTION static void fbou_hdg_jac_uq(double out[],NS_BOUNDARY_ARGS)
    {for(int k=0;k<4*Nq;++k)out[k]=0;}
    KOKKOS_INLINE_FUNCTION static void fbou_hdg_jac_uh(double out[],NS_BOUNDARY_ARGS)
    {for(int a=0;a<4;++a)for(int c=0;c<4;++c)out[c+4*a]=a==c?-tau[0]:0;}
#undef NS_BOUNDARY_ARGS
    KOKKOS_INLINE_FUNCTION static void fhat(double out[],const double x[],const double u1[],const double u2[],
        const double[],const double[],const double[],const double[],const double[],const double n[],
        const double tau[],const double[],const double[],double)
    {double f1[8],f2[8];flux(f1,x,u1,nullptr,nullptr,nullptr,nullptr,0);flux(f2,x,u2,nullptr,nullptr,nullptr,nullptr,0);
        for(int c=0;c<4;++c){out[c]=tau[0]*(u1[c]-u2[c]);for(int d=0;d<2;++d)out[c]+=.5*(f1[c+4*d]+f2[c+4*d])*n[d];}}
};

template<bool Nonlinear>
static int nativeCheck(const std::string& root,int rank,int size,int invalidControls)
{
    using Model=std::conditional_t<Nonlinear,NonlinearSolverDiffusion,SolverDiffusion>;
    const std::vector<int> hRanks=size==4?std::vector<int>{0,2}:std::vector<int>{0};
    const std::vector<int> lRanks=size==2?std::vector<int>{1}:size==3?std::vector<int>{1,2}:std::vector<int>{1,3};
    const bool isH=std::find(hRanks.begin(),hRanks.end(),rank)!=hRanks.end();
    MPI_Comm_split(EXASIM_COMM_WORLD,isH?1:0,rank,&EXASIM_COMM_LOCAL);
    int localRank=0,localSize=0;MPI_Comm_rank(EXASIM_COMM_LOCAL,&localRank);MPI_Comm_size(EXASIM_COMM_LOCAL,&localSize);
    if(!rank)prepare<2,false>(root,hRanks,lRanks,size,false,1,0);
    MPI_Barrier(EXASIM_COMM_WORLD);int failures=0;
    {
        const std::string directory=root+(isH?"/H":"/L");
        CDiscretization d(directory+"/",directory+"/out","",localSize,localRank,0,0,0,0,
            TestABI<Model>::make(),0,0,0,0,0,ExasimExecutionMode::HDGLDGSetup);
        auto interface=exasim::hdgldg::initializeInterface(d,directory,0);
        CResidual<Model> residual(d);CAssembler<Model> assembler(d);
        if(invalidControls==1 && rank==1)++d.common.solverparams.gmresRestart;
        if(invalidControls==2)d.common.solverparams.nonlinearSolverTol=-1;
        CPreconditioner<Model> preconditioner(d,0);CSolver<Model> solver(d,0);
        std::cout<<"CONTROLS rank="<<rank<<" q="<<d.common.components.ncq<<" w="<<d.common.components.ncw
            <<" wave="<<d.common.timeparams.wave<<" AV="<<d.common.physicsparams.ncAV<<" pp="<<d.common.ppdegree
            <<" RB="<<d.common.solverparams.RBdim<<" preconditioner="<<d.common.solverparams.preconditioner
            <<" interface="<<d.common.couplingparams.ndofuhatinterface<<" order="<<d.common.solverparams.matvecOrder
            <<" FDstep="<<d.common.solverparams.matvecTol<<std::endl;
        auto system=exasim::hdgldg::initializeSystem(d,residual,assembler,preconditioner,solver,interface,0);
        const auto& c=d.common;const int npe=c.grid.npe,npf=c.grid.npf,ncu=c.components.ncu,nc=c.components.nc;
        Int globalOwned=0;MPI_Allreduce(&system.N,&globalOwned,1,mpi_type<Int>(),MPI_SUM,EXASIM_COMM_WORLD);
        // The fixed 4x4 fixture has 32 unique H faces and four L elements.
        failures+=globalOwned!=ncu*(32*npf+4*npe);
        // Retain the fixture's existing Newton/GMRES/FD settings. Perturb the
        // primal state, not the dependent gradient, before the linearization.
        if(!Nonlinear || !isH)for(int e=0;e<c.meshsizes.ne1;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)
            d.sol.udg[j+npe*(a+nc*e)]+=.01*(a+1);
        if(!Nonlinear && isH)for(int k=0;k<c.sizes.ndofuhat;++k)d.sol.uh[k]+=.015;
        exasim::hdgldg::linearize(system);
        std::vector<double> baseU(d.sol.udg,d.sol.udg+d.sol.szudg),baseTrace(d.sol.uh,d.sol.uh+d.sol.szuh);
        std::vector<double> baseSysU(solver.sys.u,solver.sys.u+solver.sys.szu);
        std::vector<double> baseSysX(solver.sys.x,solver.sys.x+solver.sys.szx);
        std::vector<double> baseSysB(solver.sys.b,solver.sys.b+solver.sys.szb);
        int cacheCount=isH?npe*ncu*c.meshsizes.ne1:0;
        if(!isH)for(const auto& face:interface.faces)if(face.contributesFlux)cacheCount+=npf*ncu;
        std::vector<double> baseCache(d.res.Mass2,d.res.Mass2+cacheCount);
        std::vector<double> direction(system.N),product(system.N),repeat(system.N),zero(system.N),scaled(system.N),alias(system.N);
        for(int k=0;k<system.N;++k)direction[k]=Nonlinear && isH?0:.2+.001*(k+rank);
        std::vector<double> baseRhs(solver.sys.b,solver.sys.b+system.N);
        exasim::hdgldg::matvec(system,product.data(),direction.data());
        int stateDifferences=0;
        auto checkState=[&](){for(int k=0;k<d.sol.szudg;++k)stateDifferences+=d.sol.udg[k]!=baseU[k];
            for(int k=0;k<d.sol.szuh;++k)stateDifferences+=d.sol.uh[k]!=baseTrace[k];
            for(int k=0;k<solver.sys.szu;++k)stateDifferences+=solver.sys.u[k]!=baseSysU[k];
            for(int k=0;k<solver.sys.szx;++k)stateDifferences+=solver.sys.x[k]!=baseSysX[k];
            for(int k=0;k<solver.sys.szb;++k)stateDifferences+=solver.sys.b[k]!=baseSysB[k];
            for(int k=0;k<cacheCount;++k)stateDifferences+=d.res.Mass2[k]!=baseCache[k];};
        checkState();exasim::hdgldg::matvec(system,repeat.data(),direction.data());checkState();
        for(int k=0;k<system.N;++k)failures+=product[k]!=repeat[k] || !std::isfinite(product[k]);
        exasim::hdgldg::matvec(system,zero.data(),zero.data());checkState();
        for(double value:zero)failures+=value!=0;
        alias=direction;exasim::hdgldg::matvec(system,alias.data(),alias.data());checkState();
        for(int k=0;k<system.N;++k)failures+=alias[k]!=product[k];
        double scaleError=0;
        for(double factor:{1e-20,1e20}){for(int k=0;k<system.N;++k)scaled[k]=factor*direction[k];
            exasim::hdgldg::matvec(system,scaled.data(),scaled.data());checkState();
            for(int k=0;k<system.N;++k)scaleError=std::max(scaleError,std::abs(scaled[k]/factor-product[k]));}
        auto restore=[&](){std::copy(baseU.begin(),baseU.end(),d.sol.udg);std::copy(baseTrace.begin(),baseTrace.end(),d.sol.uh);};
        auto perturb=[&](double step){
            if(isH){int offset=0;for(const auto& face:system.owned){for(int j=0;j<npf;++j)for(int a=0;a<ncu;++a)
                d.sol.uh[a+ncu*(face.nodes[j]+npf*face.face)]+=step*direction[offset+a+ncu*j];offset+=npf*ncu;}}
            else for(int e=0;e<c.meshsizes.ne1;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)
                d.sol.udg[j+npe*(a+nc*e)]+=step*direction[j+npe*(a+ncu*e)];};
        double finiteError=0,truncationBound=0,localSquared=0,globalSquared=0;
        for(double value:direction)localSquared+=value*value;
        MPI_Allreduce(&localSquared,&globalSquared,1,MPI_DOUBLE,MPI_SUM,EXASIM_COMM_WORLD);
        const double directionNorm=std::sqrt(globalSquared),h[2]={Nonlinear?.01:1e-4,Nonlinear?.005:1e-5};
        std::vector<double> centered[2],quadratic(system.N);
        for(int level=0;level<2;++level){const double step=Nonlinear?h[level]/directionNorm:h[level];std::vector<double> plus(system.N);centered[level].resize(system.N);
            for(int sign:{1,-1}){restore();perturb(sign*step);exasim::hdgldg::linearize(system);
                if(sign==1)std::copy(solver.sys.b,solver.sys.b+system.N,plus.begin());
                else for(int k=0;k<system.N;++k){centered[level][k]=-(plus[k]-solver.sys.b[k])/(2*step);
                    if(Nonlinear && level==0)quadratic[k]=-directionNorm*(plus[k]+solver.sys.b[k]-2*baseRhs[k])/(2*h[level]*h[level]);}}}
        restore();exasim::hdgldg::linearize(system);
        // Same predeclared arithmetic allowance as the residual checkpoint,
        // amplified by cancellation at the smallest reference difference.
        const double bound=2*512*std::numeric_limits<double>::epsilon()*(Nonlinear?256:48)*npe*npe/1e-5;
        for(int k=0;k<system.N;++k){double reference=centered[1][k],truncation=0;
            if constexpr(Nonlinear){
                // Along this L-only direction q is linear and the flux cubic.
                // Richardson removes the centered h^2*c term exactly. The
                // configured forward product has eps*b+eps^2*c truncation.
                const double cubic=(centered[0][k]-centered[1][k])/(h[0]*h[0]-h[1]*h[1]);
                reference=(4*centered[1][k]-centered[0][k])/3;
                const double eps=c.solverparams.matvecTol;
                truncation=eps*std::abs(quadratic[k])+eps*eps*std::abs(cubic);}
            const double error=std::abs(reference-product[k]);finiteError=std::max(finiteError,error);
            truncationBound=std::max(truncationBound,truncation);failures+=!std::isfinite(error) || error>bound+truncation;}
        failures+=stateDifferences!=0 || !std::isfinite(scaleError) || scaleError>bound;
        std::cout<<std::setprecision(17)<<"NATIVE_JV rank="<<rank<<" owned="<<system.N<<" centeredError="<<finiteError
            <<" scaleError="<<scaleError<<" stateDifferences="<<stateDifferences<<" roundoffBound="<<bound<<" truncationBound="<<truncationBound<<" failures="<<failures<<'\n';
        // This linear exact fixture starts at x_exact+error. Deliberately
        // reverse its exact Newton correction: every trial residual equals
        // (1+alpha)*R_base, so no positive alpha is an acceptable descent.
        if constexpr(!Nonlinear){if(isH){for(int k=0;k<system.N;++k)solver.sys.x[k]=.015;
            for(int e=0;e<c.meshsizes.ne1;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)
                d.res.Minv2[j+npe*(a+ncu*e)]=.01*(a+1);}
        else for(int e=0;e<c.meshsizes.ne1;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)
            solver.sys.x[j+npe*(a+ncu*e)]=.01*(a+1);
        const double before=exasim::hdgldg::nonlinearNorm(system);double updated=0,alpha=0;
        const bool accepted=exasim::hdgldg::backtrack(system,before,updated,alpha);
        // x is the deliberately wrong correction; it is not rollback state.
        std::copy(solver.sys.x,solver.sys.x+solver.sys.szx,baseSysX.begin());
        const int previousDifferences=stateDifferences;checkState();
        const int rollbackDifferences=stateDifferences-previousDifferences;
        failures+=accepted || rollbackDifferences!=0;
        std::cout<<"ROLLBACK rank="<<rank<<" accepted="<<accepted<<" differences="<<rollbackDifferences
            <<" alpha="<<alpha<<" baseNorm="<<before<<" lastNorm="<<updated<<'\n';}
        const bool converged=exasim::hdgldg::solve(system);
        const double norm=exasim::hdgldg::nonlinearNorm(system);
        double solutionError=0;
        for(int e=0;e<c.meshsizes.ne1;++e)for(int j=0;j<npe;++j){double x[2];
            for(int a=0;a<2;++a)x[a]=d.sol.xdg[j+npe*(a+2*e)];
            for(int a=0;a<ncu;++a)solutionError=std::max(solutionError,
                std::abs(d.sol.udg[j+npe*(a+nc*e)]-Model::value(a,x)));}
        failures+=!converged || !std::isfinite(norm+solutionError) || norm>c.solverparams.nonlinearSolverTol;
        std::cout<<"NATIVE_SOLVE rank="<<rank<<" converged="<<converged<<" rawNorm="<<norm
            <<" solutionError="<<solutionError<<" failures="<<failures<<'\n';
    }
    MPI_Comm_free(&EXASIM_COMM_LOCAL);return failures;
}

static int compressibleCheck(const std::string& root,int rank,int size)
{
    using Model=CompressibleSmoke;
    const std::vector<int> hRanks=size==4?std::vector<int>{0,2}:std::vector<int>{0};
    const std::vector<int> lRanks=size==2?std::vector<int>{1}:size==3?std::vector<int>{1,2}:std::vector<int>{1,3};
    const bool isH=std::find(hRanks.begin(),hRanks.end(),rank)!=hRanks.end();
    MPI_Comm_split(EXASIM_COMM_WORLD,isH?1:0,rank,&EXASIM_COMM_LOCAL);
    int localRank=0,localSize=0;MPI_Comm_rank(EXASIM_COMM_LOCAL,&localRank);MPI_Comm_size(EXASIM_COMM_LOCAL,&localSize);
    if(!rank)prepare<2,false,Model>(root,hRanks,lRanks,size,false,1,0);
    MPI_Barrier(EXASIM_COMM_WORLD);int failures=0;
    {
        // Pointwise finite differences check the reused generated Jacobian's
        // packing before using it in a native block or condensation.
        double x[2]={.2,.3},u[12]={},jac[96],plus[8],minus[8],pointError=0;
        for(int c=0;c<4;++c)u[c]=Model::value(c,x);
        Model::flux_jac_uq(jac,x,u,nullptr,nullptr,nullptr,nullptr,0);
        const double pointStep=1e-5;
        for(int a=0;a<12;++a){const double saved=u[a];u[a]=saved+pointStep;Model::flux(plus,x,u,nullptr,nullptr,nullptr,nullptr,0);
            u[a]=saved-pointStep;Model::flux(minus,x,u,nullptr,nullptr,nullptr,nullptr,0);u[a]=saved;
            for(int c=0;c<8;++c)pointError=std::max(pointError,std::abs((plus[c]-minus[c])/(2*pointStep)-jac[c+8*a]));}
        // At rho=1, |u|=.1, p=.1 and Re=100, this bounds callback
        // cancellation plus the O(step^2) centered third-derivative term.
        const double pointBound=256*std::numeric_limits<double>::epsilon()/pointStep+64*pointStep*pointStep;
        failures+=!std::isfinite(pointError) || pointError>pointBound;
        const std::string directory=root+(isH?"/H":"/L");
        CDiscretization d(directory+"/",directory+"/out","",localSize,localRank,0,0,0,0,
            TestABI<Model>::make(),0,0,0,0,0,ExasimExecutionMode::HDGLDGSetup);
        auto interface=exasim::hdgldg::initializeInterface(d,directory,0);
        CResidual<Model> residual(d);CAssembler<Model> assembler(d);CPreconditioner<Model> preconditioner(d,0);CSolver<Model> solver(d,0);
        auto system=exasim::hdgldg::initializeSystem(d,residual,assembler,preconditioner,solver,interface,0);
        const auto& c=d.common;const int npe=c.grid.npe,nc=c.components.nc;
        for(int e=0;e<c.meshsizes.ne1;++e)for(int a=0;a<4;++a)for(int j=0;j<npe;++j)d.sol.udg[j+npe*(a+nc*e)]+=.001*(a+1);
        if(isH)for(int k=0;k<c.sizes.ndofuhat;++k)d.sol.uh[k]+=.001;
        const bool converged=exasim::hdgldg::solve(system);const double norm=exasim::hdgldg::nonlinearNorm(system);
        double minimumDensity=1e100,minimumPressure=1e100,solutionError=0;
        for(int e=0;e<c.meshsizes.ne1;++e)for(int j=0;j<npe;++j){double state[4];
            for(int a=0;a<4;++a){state[a]=d.sol.udg[j+npe*(a+nc*e)];solutionError=std::max(solutionError,std::abs(state[a]-Model::value(a,x)));}
            minimumDensity=std::min(minimumDensity,state[0]);
            minimumPressure=std::min(minimumPressure,.4*(state[3]-.5*(state[1]*state[1]+state[2]*state[2])/state[0]));}
        failures+=!converged || !std::isfinite(norm+solutionError) || norm>c.solverparams.nonlinearSolverTol || minimumDensity<=0 || minimumPressure<=0;
        std::cout<<std::setprecision(17)<<"COMPRESSIBLE rank="<<rank<<" pointJacError="<<pointError<<" pointBound="<<pointBound
            <<" converged="<<converged<<" rawNorm="<<norm<<" density="<<minimumDensity<<" pressure="<<minimumPressure
            <<" solutionError="<<solutionError<<" failures="<<failures<<'\n';
    }
    MPI_Comm_free(&EXASIM_COMM_LOCAL);return failures;
}

#ifndef HDGLDG_SOLVER_NO_MAIN
int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);EXASIM_COMM_WORLD=MPI_COMM_WORLD;int rank=0,size=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    Kokkos::initialize(argc,argv);
    int failures=0;
    try{if(size<2 || size>4)throw std::runtime_error("solver oracle requires np2/3/4");failures=blockCheck(rank,size);
        bool nonlinear=false,compressible=false,prepareOnly=false;int invalidControls=0,timeOrder=0;
        for(int a=2;a<argc;++a){const std::string option=argv[a];if(option=="--nonlinear")nonlinear=true;
            else if(option=="--mismatched-controls")invalidControls=1;
            else if(option=="--invalid-tolerance")invalidControls=2;
            else if(option=="--compressible")compressible=true;
            else if(option=="--prepare-only")prepareOnly=true;
            else if(option=="--dirk22")timeOrder=2;
            else if(option=="--dirk33")timeOrder=3;
            else throw std::runtime_error("unknown solver oracle option");}
        if(argc>1 && prepareOnly){
            const std::vector<int> h=size==4?std::vector<int>{0,2}:std::vector<int>{0};
            const std::vector<int> l=size==2?std::vector<int>{1}:size==3?std::vector<int>{1,2}:std::vector<int>{1,3};
            if(!rank)prepare<2,false>(argv[1],h,l,size,false,1,0,.01,timeOrder);
            MPI_Barrier(MPI_COMM_WORLD);
        } else if(argc>1)failures+=compressible?compressibleCheck(argv[1],rank,size):nonlinear?nativeCheck<true>(argv[1],rank,size,invalidControls):nativeCheck<false>(argv[1],rank,size,invalidControls);}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);}
    int total=0;MPI_Allreduce(&failures,&total,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    Kokkos::finalize();MPI_Finalize();return total?1:0;
}
#endif
