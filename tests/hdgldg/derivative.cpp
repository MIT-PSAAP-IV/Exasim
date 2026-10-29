// Independent centered residual differences. Test-owned vectors are intentional;
// production derivative construction uses only the existing native workspace.
#define HDGLDG_FIXTURE_NO_MAIN
#include "residual.cpp"
#include "backend/Discretization/hdgldgpreconditioner.hpp"
#include "backend/Discretization/hdgldgmatrix.hpp"

// Only the pre-existing ABI-based native BJ builder needs these adapters.
// Value/Jacobian kernels are reused unchanged; boundary derivatives below are
// the explicit Dirichlet formulas of the fixture, not production hooks.
template<class M>
struct TestABI {
    static void boundaryJac(bool state,double* f,double* fu,double*,double* fh,
        const double* x,const double* u,const double*,const double*,const double* h,
        const double* normal,const double* tau,const double*,const double*,double,
        int,int ib,int ng,int nc,int ncu,int nd,int,int,int)
    {
        for(int i=0;i<ng;++i){
            double xp[M::nd],up[M::Nq],hp[M::ncu],np[M::nd],flux[M::ncu*M::nd],jac[M::ncu*M::nd*M::Nq];
            for(int d=0;d<nd;++d){xp[d]=x[i+ng*d];np[d]=normal[i+ng*d];}
            for(int c=0;c<nc;++c)up[c]=u[i+ng*c];
            for(int c=0;c<ncu;++c)hp[c]=h[i+ng*c];
            M::flux(flux,xp,up,nullptr,nullptr,nullptr,nullptr,0);
            M::flux_jac_uq(jac,xp,up,nullptr,nullptr,nullptr,nullptr,0);
            for(int c=0;c<ncu;++c){
                f[i+ng*c]=state?M::value(c,xp):tau[0]*(up[c]-hp[c]);
                if(!state)for(int d=0;d<nd;++d)f[i+ng*c]+=flux[c+ncu*d]*np[d];
                if(ib==2*nd+1)f[i+ng*c]=std::numeric_limits<double>::quiet_NaN();
                for(int a=0;a<nc;++a){double derivative=(!state && a==c)?tau[0]:0;
                    if(!state)for(int d=0;d<nd;++d)derivative+=jac[c+ncu*d+ncu*nd*a]*np[d];
                    fu[i+ng*(c+ncu*a)]=derivative;}
                for(int a=0;a<ncu;++a)fh[i+ng*(c+ncu*a)]=!state && a==c?-tau[0]:0;
            }
        }
    }
#define JAC_ARGUMENTS double* f,double* fu,double* fw,double* fh,const double* x,const double* u,const double* o,const double* w,const double* h,const double* n,const double* tau,const double* eta,const double* mu,double t,int model,int ib,int ng,int nc,int ncu,int nd,int ncx,int nco,int ncw
#define JAC_FORWARD f,fu,fw,fh,x,u,o,w,h,n,tau,eta,mu,t,model,ib,ng,nc,ncu,nd,ncx,nco,ncw
    static void ubouJac(JAC_ARGUMENTS){boundaryJac(true,JAC_FORWARD);}
    static void fbouJac(JAC_ARGUMENTS){boundaryJac(false,JAC_FORWARD);}
#undef JAC_ARGUMENTS
#undef JAC_FORWARD
    static ExasimDriverABI make(){ExasimDriverABI abi;
        abi.hdgjac.HdgFlux=&exasim::hdg_flux_kernel<M>;
        abi.hdgjac.HdgSource=&exasim::hdg_source_kernel<M>;
        abi.volume.KokkosFlux=&exasim::flux_kernel<M>;
        abi.volume.KokkosSource=&exasim::source_kernel<M>;
        abi.volume.KokkosTdfunc=&exasim::tdfunc_kernel<M>;
        abi.boundary.KokkosUbou=&exasim::ubou_kernel<M>;
        abi.boundary.KokkosFbou=&exasim::fbou_kernel<M>;
        abi.boundary.KokkosUbouJac=&ubouJac;
        abi.boundary.KokkosFbouJac=&fbouJac;return abi;}
};

struct CaptureInverse : PrecondStageTable {
    std::vector<double> raw;
    CaptureInverse(){inverse=[](ResidualStageContext& c,double* A,Int n,Int batch){
        auto& capture=*const_cast<CaptureInverse*>(static_cast<const CaptureInverse*>(PrecondStagesOf(c.common)));
        const std::size_t offset=A-c.res.K,count=std::size_t(n)*n*batch;
        if(offset>capture.raw.size() || count>capture.raw.size()-offset)
            throw std::runtime_error("native BJ capture range exceeds owned blocks");
        std::copy(A,A+count,capture.raw.begin()+offset);
        Inverse(c.handle,A,c.res.H,c.res.ipiv,n,batch,c.backend);
    };}
};

static int inverseCheck()
{
    const double originals[8]={4,1,2,3,2,-1,.5,3};
    double a[10],b[8],work[4];Int pivots[2];
    a[0]=17;a[9]=-19;std::copy(originals,originals+8,a+1);std::copy(originals,originals+8,b);
    Inverse(0,b,work,pivots,2,2,0);
    InverseWithWorkspace(0,a+1,work,pivots,2,2,0,nullptr,0,4,2);
    int failures=a[0]!=17 || a[9]!=-19;
    for(int k=0;k<8;++k)failures+=a[k+1]!=b[k];
    double error=0;for(int e=0;e<2;++e)for(int j=0;j<2;++j)for(int i=0;i<2;++i){double product=0;
        for(int k=0;k<2;++k)product+=originals[i+2*k+4*e]*a[1+k+2*j+4*e];
        error=std::max(error,std::abs(product-(i==j?1.:0.)));}
    failures+=error>32*std::numeric_limits<double>::epsilon();
    InverseWithWorkspace<double>(0,nullptr,nullptr,nullptr,2,0,0,nullptr,0,0,0);
    std::cout<<"INVERSE byteEqual="<<(failures==0)<<" identityError="<<error<<'\n';return failures;
}

// Spatial cross diffusion makes the derivative between distinct HL faces
// nonzero; isotropic tensor-product diffusion would cancel those blocks.
template<int ND>
struct TraceDiffusion : PolynomialDiffusion<ND> {
    using Base=PolynomialDiffusion<ND>;
    KOKKOS_INLINE_FUNCTION static void flux(double f[],const double[],const double u[],
        const double[],const double[],const double[],const double[],double)
    {
        for(int d=0;d<ND;++d)for(int c=0;c<3;++c){f[c+3*d]=0;
            for(int a=0;a<3;++a)f[c+3*d]+=Base::coefficient(c,a)*
                (u[3+3*d+a]+(d<2?.25*u[3+3*(1-d)+a]:0));}
    }
    KOKKOS_INLINE_FUNCTION static void flux_jac_uq(double f[],const double[],const double[],
        const double[],const double[],const double[],const double[],double)
    {
        for(int k=0;k<3*ND*Base::Nq;++k)f[k]=0;
        for(int d=0;d<ND;++d)for(int c=0;c<3;++c)for(int a=0;a<3;++a){
            f[c+3*d+(3*ND)*(3+3*d+a)]=Base::coefficient(c,a);
            if(d<2)f[c+3*d+(3*ND)*(3+3*(1-d)+a)]=.25*Base::coefficient(c,a);}
    }
    KOKKOS_INLINE_FUNCTION static void fbou(double out[],int ib,const double x[],const double u[],
        const double[],const double[],const double h[],const double n[],const double tau[],
        const double[],const double[],double)
    {
        double f[3*ND];flux(f,x,u,nullptr,nullptr,nullptr,nullptr,0);
        for(int c=0;c<3;++c){out[c]=tau[0]*(u[c]-h[c]);
            for(int d=0;d<ND;++d)out[c]+=f[c+3*d]*n[d];
            if(ib==2*ND+1)out[c]=std::numeric_limits<double>::quiet_NaN();}
    }
    KOKKOS_INLINE_FUNCTION static void fhat(double out[],const double x[],const double u1[],const double u2[],
        const double[],const double[],const double[],const double[],const double[],const double n[],
        const double tau[],const double[],const double[],double)
    {
        double f1[3*ND],f2[3*ND];flux(f1,x,u1,nullptr,nullptr,nullptr,nullptr,0);
        flux(f2,x,u2,nullptr,nullptr,nullptr,nullptr,0);
        for(int c=0;c<3;++c){out[c]=tau[0]*(u1[c]-u2[c]);
            for(int d=0;d<ND;++d)out[c]+=.5*(f1[c+3*d]+f2[c+3*d])*n[d];}
    }
};

template<int ND>
struct NonlinearDiffusion : TraceDiffusion<ND> {
    using Base=TraceDiffusion<ND>;
    KOKKOS_INLINE_FUNCTION static void flux(double f[],const double x[],const double u[],
        const double v[],const double w[],const double mu[],const double eta[],double t)
    {
        Base::flux(f,x,u,v,w,mu,eta,t);
        for(int d=0;d<ND;++d)for(int c=0;c<3;++c)f[c+3*d]*=1+u[c]*u[c];
    }
    KOKKOS_INLINE_FUNCTION static void flux_jac_uq(double f[],const double x[],const double u[],
        const double v[],const double w[],const double mu[],const double eta[],double t)
    {
        Base::flux_jac_uq(f,x,u,v,w,mu,eta,t);
        double value[3*ND];Base::flux(value,x,u,v,w,mu,eta,t);
        for(int d=0;d<ND;++d)for(int c=0;c<3;++c){
            for(int k=0;k<Base::Nq;++k)f[c+3*d+(3*ND)*k]*=1+u[c]*u[c];
            f[c+3*d+(3*ND)*c]=2*u[c]*value[c+3*d];
        }
    }
};

template<int ND,bool Nonlinear>
static int derivativeCheck(const std::string& root,int rank,int size)
{
    using Model=std::conditional_t<Nonlinear,NonlinearDiffusion<ND>,TraceDiffusion<ND>>;
    // Check the callback against the actual native HdgFlux packing before
    // using any assembled matrices: output index is fastest, input outer.
    // This is the established layout in tests/models/poisson2d.hpp.
    double pointError=0;
    {
        double x[ND],u[Model::Nq],jac[3*ND*Model::Nq],plus[3*ND],minus[3*ND];
        for(int k=0;k<ND;++k)x[k]=.2+.1*k;
        for(int k=0;k<Model::Nq;++k)u[k]=.3+.07*k;
        Model::flux_jac_uq(jac,x,u,nullptr,nullptr,nullptr,nullptr,0);
        for(int k=0;k<Model::Nq;++k){const double saved=u[k],step=1e-5;
            u[k]=saved+step;Model::flux(plus,x,u,nullptr,nullptr,nullptr,nullptr,0);
            u[k]=saved-step;Model::flux(minus,x,u,nullptr,nullptr,nullptr,nullptr,0);u[k]=saved;
            for(int out=0;out<3*ND;++out)
                pointError=std::max(pointError,std::abs((plus[out]-minus[out])/(2*step)-jac[out+3*ND*k]));
        }
    }
    if(pointError>1e-8)throw std::runtime_error("point flux Jacobian does not match native packing");
    if(!rank)std::cout<<std::setprecision(17)<<"POINT_JAC error="<<pointError<<'\n';
    const std::vector<int> hRanks=size==4?std::vector<int>{0,2}:std::vector<int>{0};
    const std::vector<int> lRanks=size==2?std::vector<int>{1}:(size==3?std::vector<int>{1,2}:std::vector<int>{1,3});
    const bool isH=std::find(hRanks.begin(),hRanks.end(),rank)!=hRanks.end();
    MPI_Comm_split(EXASIM_COMM_WORLD,isH?1:0,rank,&EXASIM_COMM_LOCAL);
    int localRank=0,localSize=0;MPI_Comm_rank(EXASIM_COMM_LOCAL,&localRank);MPI_Comm_size(EXASIM_COMM_LOCAL,&localSize);
    if(!rank)prepare<ND,false>(root,hRanks,lRanks,size,false,1);
    MPI_Barrier(EXASIM_COMM_WORLD);
    int failures=0;
    {
        const std::string directory=root+(isH?"/H":"/L");
        CDiscretization d(directory+"/",directory+"/out","",localSize,localRank,0,0,0,0,
            TestABI<Model>::make(),0,0,0,0,0,ExasimExecutionMode::HDGLDGSetup);
        auto interface=exasim::hdgldg::initializeInterface(d,directory,0);
        CResidual<Model> residual(d);
        const auto& c=d.common;
        const int npe=c.grid.npe,npf=c.grid.npf,ncu=c.components.ncu,nfe=c.meshsizes.nfe;
        const int nc=c.components.nc,ncf=npf*ncu,columns=ncf*nfe;
        if(isH)residual.recoverInitialState(0,false);
        exasim::hdgldg::exchangeTrace(d,interface,0);
        if(!isH)residual.evalResidual(0);
        std::vector<double> baseU(d.sol.udg,d.sol.udg+d.sol.szudg);
        std::vector<double> baseTrace(d.sol.uh,d.sol.uh+d.sol.szuh);
        auto restore=[&](){std::copy(baseU.begin(),baseU.end(),d.sol.udg);
            std::copy(baseTrace.begin(),baseTrace.end(),d.sol.uh);};
        auto evaluate=[&](){exasim::hdgldg::exchangeTrace(d,interface,0);if(!isH)residual.evalResidual(0);};
        double maximumError[2]={},maximumRatio[2]={},crossFace=0,analyticScale=0,qMapError=0;
        std::vector<std::vector<double>> finiteRows(interface.traceFaces.size());
        std::size_t entries=0,crossEntries=0;int reversed=0;
        const double steps[2]={1e-4,1e-5};
        // The explicit polynomial has |u|<4.5 and |Kq|<8 on the unit box;
        // 256 bounds the optional nonlinear point flux. Dense-recovery
        // accumulation uses the same conservative operation allowance as
        // the residual fixture, amplified by centered-difference cancellation.
        const double fluxRoundoff=512*std::numeric_limits<double>::epsilon()*256*npe*npe;
        for(const auto& row:interface.traceFaces){
            restore();evaluate();
            std::vector<double> analytic;
            if(rank==row.lRank){
                std::vector<double> savedE;
                if(std::getenv("HDGLDG_TEST_FREEZE_Q")){savedE.assign(d.res.E,d.res.E+d.res.szE);ArraySetValue(d.res.E,0.,d.res.szE);}
                double* matrix=exasim::hdgldg::buildTraceRow<Model>(d,row.face,row.boundarySlot,row.element,0);
                analytic.assign(matrix,matrix+ncf*columns);
                if(!savedE.empty())std::copy(savedE.begin(),savedE.end(),d.res.E);
                if(std::getenv("HDGLDG_TEST_DROP_CROSS"))for(int column=0;column<columns;++column)
                    if(column/ncf!=row.key[3])std::fill(analytic.begin()+ncf*column,analytic.begin()+ncf*(column+1),0.);
                finiteRows[&row-interface.traceFaces.data()].resize(ncf*columns);
                for(int j=0;j<npf;++j)reversed+=row.nodeOfH[j]!=j;
            }
            for(const auto& column:interface.traceFaces)if(column.key[2]==row.key[2])
              for(int node=0;node<npf;++node)for(int component=0;component<ncu;++component){
                int localColumn=-1;
                if(rank==row.lRank){
                    const int faceNode=column.nodeOfH[node],lf=column.key[3];
                    for(int j=0;j<npf;++j)if(d.mesh.elemcon[j+npf*(lf+nfe*row.element)]%npf==faceNode)
                        localColumn=component+ncu*(j+npf*lf);
                    if(localColumn<0)throw std::runtime_error("derivative oracle column has no local face node");
                }
                for(int stepIndex=0;stepIndex<2;++stepIndex){
                    const double step=steps[stepIndex];
                    std::vector<double> plus(ncf),plusQ(npe*c.components.ncq);
                    for(int sign : {1,-1}){
                        restore();
                        if(rank==column.hRank)d.sol.uh[component+ncu*(node+npf*column.face)]+=sign*step;
                        evaluate();
                        if(rank!=row.lRank)continue;
                        if(sign==1){
                            for(int j=0;j<npf;++j)for(int a=0;a<ncu;++a)
                                plus[a+ncu*j]=d.res.Rh[j+npf*(a+ncu*row.face)];
                            for(int a=ncu;a<nc;++a)for(int j=0;j<npe;++j)
                                plusQ[j+npe*(a-ncu)]=d.sol.udg[j+npe*(a+nc*row.element)];
                        }else{
                            double qSlope=0;
                            for(int a=ncu;a<nc;++a)for(int j=0;j<npe;++j){
                                const double slope=(plusQ[j+npe*(a-ncu)]-d.sol.udg[j+npe*(a+nc*row.element)])/(2*step);
                                qSlope=std::max(qSlope,std::abs(slope));
                                const double expected=(a-ncu)%ncu==component?-d.res.E[j+npe*(localColumn/ncu)+
                                    npe*npf*nfe*(row.element+(a-ncu)/ncu*c.meshsizes.ne)]:0;
                                qMapError=std::max(qMapError,std::abs(slope-expected));
                            }
                            // For the cubic nonlinear flux, centered truncation
                            // is step^2 times its cubic coefficient. |basis|<=1,
                            // face measure<=1 and max row sum(K)<3; nodal q
                            // interpolation contributes at most npe.
                            const double bound=2*fluxRoundoff/step+(Nonlinear?3*npe*qSlope*step*step:0);
                            for(int j=0;j<npf;++j)for(int a=0;a<ncu;++a){
                                const int r=a+ncu*j;
                                const double finite=(plus[r]-d.res.Rh[j+npf*(a+ncu*row.face)])/(2*step);
                                if(stepIndex==1)finiteRows[&row-interface.traceFaces.data()][r+ncf*localColumn]=finite;
                                const double exact=analytic[r+ncf*localColumn];
                                const double error=errorMagnitude(finite,exact);
                                maximumError[stepIndex]=std::max(maximumError[stepIndex],error);
                                maximumRatio[stepIndex]=std::max(maximumRatio[stepIndex],error/bound);
                                analyticScale=std::max(analyticScale,std::abs(exact));++entries;
                                if(row.key!=column.key){crossFace=std::max(crossFace,std::abs(exact));++crossEntries;}
                                if(error>bound){if(failures<4)std::cerr<<"D mismatch rowFace="<<row.key[3]
                                    <<" columnFace="<<column.key[3]<<" nodeH="<<node<<" component="<<component
                                    <<" row="<<r<<" localColumn="<<localColumn<<" FD="<<finite<<" analytic="<<exact<<'\n';
                                    ++failures;}
                            }
                        }
                    }
                }
            }
        }
        restore();evaluate();
        if constexpr(!Nonlinear){
            // Capture native pre-inverse blocks, then independently perturb
            // one volume unknown while lambda and every other element stay fixed.
            std::vector<double> volume(std::size_t(npe)*ncu*c.meshsizes.ne);
            if(!isH){for(int e=0;e<c.meshsizes.ne;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)
                volume[j+npe*(a+ncu*e)]=d.sol.udg[j+npe*(a+nc*e)];
                CaptureInverse capture;capture.raw.resize(std::size_t(npe*ncu)*(npe*ncu)*c.meshsizes.ne1,
                    std::numeric_limits<double>::quiet_NaN());SetPrecondStages(d.common,&capture);
                if(localSize>1)mpiBlockJacobianLDG(d.res.K,volume.data(),d.sol,d.res,d.app,d.driver_abi,d.master,d.mesh,d.tmp,d.common,c.cublasHandle,0);
                else BlockJacobianLDG(d.res.K,volume.data(),d.sol,d.res,d.app,d.driver_abi,d.master,d.mesh,d.tmp,d.common,c.cublasHandle,0);
                SetPrecondStages(d.common,nullptr);
                for(double x:capture.raw)if(!std::isfinite(x))throw std::runtime_error("native BJ capture contains an unwritten/nonfinite block");
                const int n=npe*ncu;double bjError=0,bjOpposite=0,inverseError=0;int bjPrinted=0;
                for(int e=0;e<c.meshsizes.ne1;++e)for(int col=0;col<n;++col)for(int row=0;row<n;++row){double product=0;
                    for(int k=0;k<n;++k)product+=capture.raw[row+n*(k+n*e)]*d.res.K[k+n*(col+n*e)];
                    inverseError=std::max(inverseError,std::abs(product-(row==col?1.:0.)));}
                const int owned=c.meshsizes.ne1;
                std::vector<int> counts(localSize);MPI_Allgather(&owned,1,MPI_INT,counts.data(),1,MPI_INT,EXASIM_COMM_LOCAL);
                for(int owner=0;owner<localSize;++owner)for(int e=0;e<counts[owner];++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j){
                    const int col=j+npe*a;const double step=1e-5;
                    std::vector<double> plus(n);
                    for(int sign:{1,-1}){restore();if(localRank==owner)d.sol.udg[j+npe*(a+nc*e)]+=sign*step;residual.evalResidual(0);
                        if(localRank!=owner)continue;
                        if(sign==1)std::copy(d.res.Ru+n*e,d.res.Ru+n*(e+1),plus.begin());
                        else for(int row=0;row<n;++row){const double fd=(plus[row]-d.res.Ru[row+n*e])/(2*step);
                            const double raw=capture.raw[row+n*(col+n*e)];
                            if(std::abs(raw-fd)>2*fluxRoundoff/1e-5 && bjPrinted++<4)
                                std::cerr<<"BJ mismatch rank="<<rank<<" e="<<e<<" row="<<row<<" col="<<col<<" raw="<<raw<<" FD="<<fd<<'\n';
                            bjError=std::max(bjError,std::abs(raw-fd));bjOpposite=std::max(bjOpposite,std::abs(raw+fd));}}
                }
                std::cout<<"BJ rank="<<rank<<" directError="<<bjError<<" oppositeError="<<bjOpposite<<" inverseError="<<inverseError<<'\n';
                if(!std::isfinite(bjError+inverseError) || bjError>2*fluxRoundoff/1e-5 || inverseError>fluxRoundoff)++failures;
            }
        }
        restore();evaluate();
        if constexpr(!Nonlinear){
            // Independent H Schwarz oracle: native S_H restriction plus the
            // centered residual derivative above, then multiply by production K.
            const int m=ncf*nfe;std::vector<double> expected;
            if(isH){CAssembler<Model> assembler(d);assembler.hdgAssembleLinearSystem(d.res.Rq,0);
                expected.assign(d.res.H,d.res.H+std::size_t(m)*m*c.meshsizes.ne);
                ElementalAdditiveSchwarz(expected.data(),d.res.H,d.mesh.f2e,d.mesh.elemcon,npf,nfe,ncu,c.meshsizes.nf);}
            for(const auto& row:interface.traceFaces)for(const auto& col:interface.traceFaces)if(row.key[2]==col.key[2]){
                std::vector<double> packet(ncf*ncf);
                if(rank==row.lRank){const auto& fd=finiteRows[&row-interface.traceFaces.data()];
                    for(int j=0;j<npf;++j)for(int a=0;a<ncu;++a){int local=-1;
                        for(int k=0;k<npf;++k)if(d.mesh.elemcon[k+npf*(col.key[3]+nfe*row.element)]%npf==col.nodeOfH[j])local=a+ncu*(k+npf*col.key[3]);
                        if(local<0)throw std::runtime_error("Schwarz oracle column has no local node");
                        for(int i=0;i<npf;++i)for(int b=0;b<ncu;++b)
                            packet[b+ncu*i+ncf*(a+ncu*j)]=fd[b+ncu*row.nodeOfH[i]+ncf*local];}
                    MPI_Send(packet.data(),packet.size(),MPI_DOUBLE,row.hRank,19,EXASIM_COMM_WORLD);
                }else if(rank==row.hRank){MPI_Recv(packet.data(),packet.size(),MPI_DOUBLE,row.lRank,19,EXASIM_COMM_WORLD,MPI_STATUS_IGNORE);
                    if(row.key[0]==col.key[0])for(int j=0;j<npf;++j)for(int a=0;a<ncu;++a)for(int i=0;i<npf;++i)for(int b=0;b<ncu;++b){
                        const int r=b+ncu*(row.nodeOfH[i]+npf*row.key[1]),s=a+ncu*(col.nodeOfH[j]+npf*col.key[1]);
                        expected[r+m*(s+m*row.element)]+=packet[b+ncu*i+ncf*(a+ncu*j)];}}
            }
            std::vector<double> volume(std::size_t(npe)*ncu*c.meshsizes.ne);
            if(!isH)for(int e=0;e<c.meshsizes.ne;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)
                volume[j+npe*(a+ncu*e)]=d.sol.udg[j+npe*(a+nc*e)];
            exasim::hdgldg::buildPreconditioner<Model>(d,interface,isH?nullptr:volume.data(),0);
            {
                CPreconditioner<Model> preconditioner(d,0,ExasimExecutionMode::Postprocess);
                sysstruct system;
                const int count=isH?c.sizes.ndofuhat:c.sizes.ndof1;
                std::vector<double> direct(count),adapted(count);
                for(int k=0;k<count;++k)direct[k]=adapted[k]=.2+.001*k;
                if(isH)preconditioner.ApplyPreconditioner(direct.data(),system,d,1,0);
                else preconditioner.ApplyPreconditioner(direct.data(),system,d,0);
                exasim::hdgldg::applyNativePreconditioner(preconditioner,system,d,interface,adapted.data(),0);
                int differences=0;for(int k=0;k<count;++k)differences+=direct[k]!=adapted[k] || !std::isfinite(adapted[k]);
                std::cout<<"APPLY rank="<<rank<<" byteDifferences="<<differences<<'\n';failures+=differences;
            }
            if(isH){double productError=0;
                for(int e=0;e<c.meshsizes.ne1;++e)for(int j=0;j<m;++j)for(int i=0;i<m;++i){double product=0;
                    for(int k=0;k<m;++k)product+=expected[i+m*(k+m*e)]*d.res.K[k+m*(j+m*e)];
                    productError=std::max(productError,std::abs(product-(i==j?1.:0.)));}
                std::cout<<"SCHWARZ rank="<<rank<<" inverseProductError="<<productError<<'\n';
                if(!std::isfinite(productError) || productError>2*fluxRoundoff/1e-5)++failures;
            }
        }
        double globalCross=0;MPI_Allreduce(&crossFace,&globalCross,1,MPI_DOUBLE,MPI_MAX,EXASIM_COMM_WORLD);
        if(globalCross<.01)++failures;
        std::cout<<std::setprecision(17)<<"DERIVATIVE nd="<<ND<<" nonlinear="<<Nonlinear<<" rank="<<rank
            <<" entries="<<entries<<" crossEntries="<<crossEntries<<" reversed="<<reversed
            <<" error1="<<maximumError[0]<<" error2="<<maximumError[1]
            <<" boundRatio1="<<maximumRatio[0]<<" boundRatio2="<<maximumRatio[1]
            <<" scale="<<analyticScale<<" cross="<<crossFace<<" qMapError="<<qMapError<<" failures="<<failures<<'\n';
    }
    MPI_Comm_free(&EXASIM_COMM_LOCAL);
    return failures;
}

#ifndef HDGLDG_DERIVATIVE_NO_MAIN
int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);EXASIM_COMM_WORLD=MPI_COMM_WORLD;
    int rank=0,size=0;MPI_Comm_rank(EXASIM_COMM_WORLD,&rank);MPI_Comm_size(EXASIM_COMM_WORLD,&size);
    bool threeD=false,nonlinear=false,valid=argc>=2 && argc<=4;
    for(int a=2;a<argc;++a){const std::string option=argv[a];
        if(option=="--3d" && !threeD)threeD=true;
        else if(option=="--nonlinear" && !nonlinear)nonlinear=true;
        else valid=false;}
    if(!valid || size<2 || size>4){if(!rank)std::cerr<<"usage: hdgldg_derivative /tmp/output [--3d] [--nonlinear]\n";MPI_Finalize();return 2;}
    Kokkos::initialize(argc,argv);int failures=inverseCheck();
    try{if(threeD)failures+=nonlinear?derivativeCheck<3,true>(argv[1],rank,size):derivativeCheck<3,false>(argv[1],rank,size);
        else failures+=nonlinear?derivativeCheck<2,true>(argv[1],rank,size):derivativeCheck<2,false>(argv[1],rank,size);}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';MPI_Abort(EXASIM_COMM_WORLD,1);}
    int total=0;MPI_Allreduce(&failures,&total,1,MPI_INT,MPI_SUM,EXASIM_COMM_WORLD);
    Kokkos::finalize();MPI_Finalize();if(!rank)std::cout<<"hdgldg derivative failures="<<total<<'\n';return total?1:0;
}
#endif
