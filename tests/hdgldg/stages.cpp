// Direct, independent DIRK algebra and element mass/source checks. All saved
// arrays below are test-owned; the coupled implementation borrows native arenas.
#define HDGLDG_SOLVER_NO_MAIN
#include "solver.cpp"

static std::array<double,9> inverseButcher(int order)
{
    std::array<double,9> a{},d{};
    // Native DIRK33 defines gamma with this finite decimal, rather than the
    // higher-precision root .435866521508459. Invert its actual Butcher A.
    const double g=order==2?1-1/std::sqrt(2.):.4358665215;
    a[0]=g;a[order+1]=g;
    if(order==2)a[1]=1-g;
    else {a[1]=(1-g)/2;a[2]=-(6*g*g-16*g+1)/4;
        a[2+order]=(6*g*g-20*g+5)/4;a[2+2*order]=g;}
    // Forward substitution, independently of the native inverse-A literals.
    for(int col=0;col<order;++col)for(int row=0;row<order;++row){
        double b=row==col?1:0;
        for(int k=0;k<row;++k)b-=a[row+order*k]*d[k+order*col];
        d[row+order*col]=b/a[row+order*row];}
    return d;
}

template<class S>
static int massCheck(S& s,double& error)
{
    auto& d=s.d;const auto& c=d.common;
    const int npe=c.grid.npe,ncu=c.components.ncu,nc=c.components.nc;
    const int ne=c.meshsizes.ne1,count=npe*ncu*ne;
    const std::vector<double> u(d.sol.udg,d.sol.udg+d.sol.szudg);
    const std::vector<double> source(d.sol.sdgg,d.sol.sdgg+d.sol.szsdgg);
    auto evaluate=[&](){std::vector<double> result(count);for(int b=0;b<c.meshsizes.nbe1;++b){
        const int first=c.eblks[3*b]-1,last=c.eblks[3*b+1];
        if(s.interface.scheme)uEquationElemBlock<SolverDiffusion>(d.sol,d.res,d.app,d.master,d.mesh,d.tmp,d.common,c.cublasHandle,b,0);
        else RuElemBlock<SolverDiffusion>(d.sol,d.res,d.app,d.master,d.mesh,d.tmp,d.common,c.cublasHandle,first,last,0);
        const double* r=s.interface.scheme?d.res.Ru:d.res.Rue;
        // HDG ApplyXxJac packs [npe,blockElements,ncu]; LDG ApplyXx4
        // instead produces the native volume order [npe,ncu,element].
        for(int e=first;e<last;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)
            result[j+npe*(a+ncu*e)]=r[s.interface.scheme?npe*ncu*first+j+npe*(e-first+(last-first)*a):j+npe*(a+ncu*e)];}
        return result;};
    const auto base=evaluate();
    // Constant changes isolate the mass term even with nonzero spatial q.
    for(int e=0;e<ne;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)
        d.sol.udg[j+npe*(a+nc*e)]+=.125*(a+1);
    const auto shifted=evaluate();std::copy(u.begin(),u.end(),d.sol.udg);
    for(int b=0;b<c.meshsizes.nbe;++b){const int first=c.eblks[3*b]-1,last=c.eblks[3*b+1];
        for(int e=first;e<last;++e)for(int a=0;a<ncu;++a)for(int g=0;g<c.grid.nge;++g)
            d.sol.sdgg[c.grid.nge*ncu*first+g+c.grid.nge*(e-first+(last-first)*a)]+=.25*(a+1);}
    const auto sourced=evaluate();std::copy(source.begin(),source.end(),d.sol.sdgg);
    int failures=0;error=0;
    // Integral of each quadratic Lagrange basis on a uniform tensor cell:
    // h*(1/6,2/3,1/6). Coordinates identify endpoints/midpoints, independently
    // of the solver's quadrature arrays and local node enumeration.
    for(int e=0;e<ne;++e){double lo[2]={1e30,1e30},hi[2]={-1e30,-1e30};
        for(int j=0;j<npe;++j)for(int k=0;k<2;++k){const double x=d.sol.xdg[j+npe*(k+2*e)];lo[k]=std::min(lo[k],x);hi[k]=std::max(hi[k],x);}
        for(int j=0;j<npe;++j){double moment=1;
            for(int k=0;k<2;++k){const double x=d.sol.xdg[j+npe*(k+2*e)];
                moment*=(hi[k]-lo[k])*(std::abs(x-.5*(lo[k]+hi[k]))<1e-12?2./3:1./6);}
            for(int a=0;a<ncu;++a){const int index=j+npe*(a+ncu*e);
                error=std::max(error,std::abs(shifted[index]-base[index]+c.timestate.dtfactor*.125*(a+1)*moment));
                error=std::max(error,std::abs(sourced[index]-base[index]-.25*(a+1)*moment));}}}
    // Fixed before measurement: floating-point accumulation over npe^2
    // mass entries, including coefficient products and subtractive probes.
    const double bound=512*std::numeric_limits<double>::epsilon()*npe*npe*std::max(1.,c.timestate.dtfactor*(ncu+1));
    failures+=!std::isfinite(error) || error>bound;
    std::cout<<"STAGE_MASS scheme="<<s.interface.scheme<<" error="<<error<<" bound="<<bound<<'\n';
    return failures;
}

static int stageCheck(const std::string& root,int rank,int size,int order)
{
    using Model=SolverDiffusion;
    const std::vector<int> h=size==4?std::vector<int>{0,2}:std::vector<int>{0};
    const std::vector<int> l=size==2?std::vector<int>{1}:size==3?std::vector<int>{1,2}:std::vector<int>{1,3};
    const bool isH=std::find(h.begin(),h.end(),rank)!=h.end();
    MPI_Comm_split(EXASIM_COMM_WORLD,isH?1:0,rank,&EXASIM_COMM_LOCAL);
    int lr=0,ls=0;MPI_Comm_rank(EXASIM_COMM_LOCAL,&lr);MPI_Comm_size(EXASIM_COMM_LOCAL,&ls);
    if(!rank)prepare<2,false>(root,h,l,size,false,1,0,0,order);
    MPI_Barrier(EXASIM_COMM_WORLD);int failures=0;
    {
        const std::string directory=root+(isH?"/H":"/L");
        CDiscretization d(directory+"/",directory+"/out","",ls,lr,0,0,0,0,TestABI<Model>::make(),0,0,0,0,0,ExasimExecutionMode::HDGLDGSetup);
        auto interface=exasim::hdgldg::initializeInterface(d,directory,0);
        CResidual<Model> residual(d);CAssembler<Model> assembler(d);CPreconditioner<Model> preconditioner(d,0);CSolver<Model> solver(d,0);
        auto s=exasim::hdgldg::initializeSystem(d,residual,assembler,preconditioner,solver,interface,0);
        // The public transient driver initializes the native coefficient arrays
        // once before beginStep; the direct fixture follows that same contract.
        TimestepCoefficents(d.common);
        auto& c=d.common;auto& sys=solver.sys;const int npe=c.grid.npe,ncu=c.components.ncu,nc=c.components.nc,ne2=c.meshsizes.ne2;
        // Model checkpoint loading: owned primal data are exact, first ghosts
        // are deliberately invalid. beginStep must exchange BEFORE snapshot.
        int poisoned=0;
        for(int e=c.meshsizes.ne1;e<ne2;++e)for(int a=0;a<nc;++a)for(int j=0;j<npe;++j){
            d.sol.udg[j+npe*(a+nc*e)]=std::numeric_limits<double>::quiet_NaN();++poisoned;}
        exasim::hdgldg::beginStep(s,0);
        double historyError=0;int historyNonfinite=0;
        for(int e=0;e<ne2;++e)for(int j=0;j<npe;++j){double x[2];for(int k=0;k<2;++k)x[k]=d.sol.xdg[j+npe*(k+2*e)];
            for(int a=0;a<ncu;++a){const double v=sys.udgprev[j+npe*(a+ncu*e)];
                historyNonfinite+=!std::isfinite(v);historyError=std::max(historyError,std::abs(v-Model::value(a,x)));}}
        failures+=historyNonfinite;
        failures+=historyError!=0;
        const std::vector<double> initial(sys.udgprev,sys.udgprev+npe*ncu*ne2);
        const auto inverse=inverseButcher(order);double sourceError=0,timeError=0;
        std::vector<std::vector<double>> values;
        // Artificial, known stage values isolate source recurrence from Newton.
        for(int stage=0;stage<order;++stage){
            exasim::hdgldg::prepareStage(s,stage,.125);
            const double dt=c.dt[0],a=inverse[stage+order*stage]/dt;
            const double gamma=order==2?1-1/std::sqrt(2.):.4358665215;
            const double timeFraction=stage==order-1?1:stage==0?gamma:(1+gamma)/2;
            timeError=std::max(timeError,std::abs(c.timestate.time-(.125+dt*timeFraction)));
            failures+=std::abs(c.timestate.dtfactor-a)>1e-11;
            for(int e=0;e<ne2;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j){const int index=j+npe*(a+ncu*e);
                double expected=0;for(int k=0;k<=stage;++k)expected+=inverse[stage+order*k]*initial[index]/dt;
                for(int k=0;k<stage;++k)expected-=inverse[stage+order*k]*values[k][index]/dt;
                sourceError=std::max(sourceError,std::abs(d.sol.sdg[index]-expected));}
            values.push_back(initial);
            for(int e=0;e<ne2;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j){const int index=j+npe*(a+ncu*e);
                values.back()[index]+=.03125*(stage+1)*(a+1);
                d.sol.udg[j+npe*(a+nc*e)]=values.back()[index];}
            const std::vector<double> traceCache(sys.u,sys.u+sys.szu);
            exasim::hdgldg::commitStage(s);
            if(isH)for(int k=0;k<sys.szu;++k)failures+=sys.u[k]!=traceCache[k];
        }
        const double algebraBound=512*std::numeric_limits<double>::epsilon()*npe*npe*256/c.dt[0];
        failures+=!std::isfinite(sourceError) || sourceError>algebraBound || timeError>5e-11*c.dt[0];
        // Reset to the exact stationary spatial field, but retain an authentic
        // first-stage history: S_stage=a*u_exact. q recovery is done natively.
        for(int e=0;e<ne2;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)
            d.sol.udg[j+npe*(a+nc*e)]=initial[j+npe*(a+ncu*e)];
        exasim::hdgldg::beginStep(s,0);exasim::hdgldg::prepareStage(s,0,0);
        double massError=0;failures+=massCheck(s,massError);
        for(int e=0;e<c.meshsizes.ne1;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)d.sol.udg[j+npe*(a+nc*e)]+=.01*(a+1);
        if(isH)for(int k=0;k<c.sizes.ndofuhat;++k)d.sol.uh[k]+=.015;
        exasim::hdgldg::linearize(s);
        const std::vector<double> u(d.sol.udg,d.sol.udg+d.sol.szudg),trace(d.sol.uh,d.sol.uh+d.sol.szuh);
        const std::vector<double> sdg(d.sol.sdg,d.sol.sdg+d.sol.szsdg),sdgg(d.sol.sdgg,d.sol.sdgg+d.sol.szsdgg);
        const std::vector<double> previous(sys.udgprev,sys.udgprev+sys.szudgprev),utmp(sys.utmp,sys.utmp+sys.szutmp);
        const double time=c.timestate.time,factor=c.timestate.dtfactor;
        std::vector<double> direction(s.N),product(s.N),plus(s.N);for(int k=0;k<s.N;++k)direction[k]=.2+.001*(k+rank);
        exasim::hdgldg::matvec(s,product.data(),direction.data());double productError=0;
        const double step=1e-5;
        for(int sign:{1,-1}){
            std::copy(u.begin(),u.end(),d.sol.udg);std::copy(trace.begin(),trace.end(),d.sol.uh);
            if(isH){int offset=0;for(const auto& f:s.owned){for(int j=0;j<c.grid.npf;++j)for(int a=0;a<ncu;++a)
                d.sol.uh[a+ncu*(f.nodes[j]+c.grid.npf*f.face)]+=sign*step*direction[offset+a+ncu*j];offset+=c.grid.npf*ncu;}}
            else for(int e=0;e<c.meshsizes.ne1;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)
                d.sol.udg[j+npe*(a+nc*e)]+=sign*step*direction[j+npe*(a+ncu*e)];
            exasim::hdgldg::linearize(s);
            if(sign==1)std::copy(sys.b,sys.b+s.N,plus.begin());
            else for(int k=0;k<s.N;++k)productError=std::max(productError,std::abs(product[k]+(plus[k]-sys.b[k])/(2*step)));}
        std::copy(u.begin(),u.end(),d.sol.udg);std::copy(trace.begin(),trace.end(),d.sol.uh);exasim::hdgldg::linearize(s);
        const double productBound=2*512*std::numeric_limits<double>::epsilon()*npe*npe*std::max(48.,factor*(ncu+1))/step;
        failures+=!std::isfinite(productError) || productError>productBound;
        if(isH){for(int k=0;k<s.N;++k)sys.x[k]=.015;for(int e=0;e<c.meshsizes.ne1;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)d.res.Minv2[j+npe*(a+ncu*e)]=.01*(a+1);}
        else for(int e=0;e<c.meshsizes.ne1;++e)for(int a=0;a<ncu;++a)for(int j=0;j<npe;++j)sys.x[j+npe*(a+ncu*e)]=.01*(a+1);
        double updated=0,alpha=0;const bool accepted=exasim::hdgldg::backtrack(s,exasim::hdgldg::nonlinearNorm(s),updated,alpha);
        failures+=accepted;int frozenDifferences=0,rollbackDifferences=0;
        for(int k=0;k<d.sol.szudg;++k)rollbackDifferences+=d.sol.udg[k]!=u[k];
        for(int k=0;k<d.sol.szuh;++k)rollbackDifferences+=d.sol.uh[k]!=trace[k];
        for(int k=0;k<d.sol.szsdg;++k)frozenDifferences+=d.sol.sdg[k]!=sdg[k];
        for(int k=0;k<d.sol.szsdgg;++k)frozenDifferences+=d.sol.sdgg[k]!=sdgg[k];
        for(int k=0;k<sys.szudgprev;++k)frozenDifferences+=sys.udgprev[k]!=previous[k];
        for(int k=0;k<sys.szutmp;++k)frozenDifferences+=sys.utmp[k]!=utmp[k];
        frozenDifferences+=time!=c.timestate.time || factor!=c.timestate.dtfactor;
        failures+=frozenDifferences!=0 || rollbackDifferences!=0;
        std::cout<<std::setprecision(17)<<"STAGE rank="<<rank<<" order="<<order<<" poisoned="<<poisoned
            <<" historyError="<<historyError<<" historyNonfinite="<<historyNonfinite<<" sourceError="<<sourceError<<" algebraBound="<<algebraBound
            <<" timeError="<<timeError<<" productError="<<productError<<" productBound="<<productBound
            <<" frozenDifferences="<<frozenDifferences<<" rollbackDifferences="<<rollbackDifferences<<" failures="<<failures<<'\n';
    }
    MPI_Comm_free(&EXASIM_COMM_LOCAL);EXASIM_COMM_LOCAL=MPI_COMM_WORLD;return failures;
}

int main(int argc,char** argv)
{
    MPI_Init(&argc,&argv);EXASIM_COMM_WORLD=MPI_COMM_WORLD;int rank=0,size=0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);Kokkos::initialize(argc,argv);int failures=0;
    try {if(argc!=3 || size<2 || size>4)throw std::runtime_error("usage: stages /tmp/directory 2|3 (np2/3/4)");
        const int order=std::stoi(argv[2]);if(order!=2 && order!=3)throw std::runtime_error("only DIRK22/33 supported");
        failures=stageCheck(argv[1],rank,size,order);}
    catch(const std::exception& e){std::cerr<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);}
    int total=0;MPI_Allreduce(&failures,&total,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);Kokkos::finalize();MPI_Finalize();return total?1:0;
}
