// [nonmatching] Phase 13 focused mixed tet/hex descriptor, rectangular algebra,
// and MPI-protocol tests. Reuse accepted Phase-11 fixture builders only; the
// rectangular references below are independent long-double loops.

#include <array>

#define main phase13_embedded_nonmatching_3d_main
#include "nonmatching_3d.cpp"
#undef main

namespace phase13 {

struct Options { int split = -1; std::string mode; };

Options parse(int argc, char** argv, int size)
{
    Options result;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--split") == 0) {
            if (++i >= argc) throw std::runtime_error("--split requires a rank count");
            char* end = nullptr;
            const long value = std::strtol(argv[i], &end, 10);
            if (!end || *end || value <= 0 || value >= size)
                throw std::runtime_error("--split must satisfy 0 < split < world size");
            result.split = static_cast<int>(value);
        }
        else if (std::strncmp(argv[i], "--", 2) == 0) {
            if (!result.mode.empty()) throw std::runtime_error("exactly one mode is required");
            result.mode = argv[i] + 2;
        }
        else throw std::runtime_error(std::string("unexpected argument: ") + argv[i]);
    }
    if (result.split < 0) throw std::runtime_error("every Phase 13 fixture requires --split k");
    if (result.mode.empty()) throw std::runtime_error("a Phase 13 mode is required");
    return result;
}

bool rejection_mode(const std::string& mode) { return mode.rfind("reject-", 0) == 0; }

void verify_exact_exchange_payload(const nonmatchingdata& host,
        const exasim::nonmatching::NonmatchingRuntime<dstype,Int>& runtime,int rank)
{
    std::vector<dstype> local;
    for(int direction=0;direction<2;++direction) {
        const auto& h=host.dir[direction]; const auto& d=runtime.dir[direction];
        if(!d.view.isdonor) continue;
        const Int rows=d.view.ncu12*d.view.ownerNpf;
        for(Int slot=0;slot<d.view.nslots;++slot) {
            local.push_back(direction); local.push_back(h.slotrank[slot]);
            local.push_back(h.slotface[slot]); local.push_back(rank); local.push_back(rows);
            for(Int row=0;row<rows;++row)
                local.push_back(d.Ri.data[row+static_cast<std::size_t>(rows)*slot]);
        }
    }
    int size=0,count=static_cast<int>(local.size()); MPI_Comm_size(MPI_COMM_WORLD,&size);
    std::vector<int> counts(size),displs(size,0);
    MPI_Allgather(&count,1,MPI_INT,counts.data(),1,MPI_INT,MPI_COMM_WORLD);
    int total=0; for(int sender=0;sender<size;++sender) { displs[sender]=total; total+=counts[sender]; }
    std::vector<dstype> all(std::max(total,1));
    MPI_Allgatherv(local.data(),count,MPI_DOUBLE,all.data(),counts.data(),displs.data(),
        MPI_DOUBLE,MPI_COMM_WORLD);

    for(int direction=0;direction<2;++direction) {
        const auto& h=host.dir[direction]; const auto& d=runtime.dir[direction];
        if(!d.view.isowner) continue;
        const Int rows=d.view.ncu12*d.view.ownerNpf;
        Int block=0;
        for(std::size_t neighbor=0;neighbor<d.recvnb.size();++neighbor)
            for(Int entry=0;entry<d.recvcount[neighbor];++entry,++block) {
                const Int own=h.recvface[block]; const Int globalFace=h.ownfaceglobal[own];
                const int expectedSender=d.recvnb[neighbor]; const dstype* expected=nullptr;
                for(int sender=0;sender<size && !expected;++sender) {
                    int offset=displs[sender],end=offset+counts[sender];
                    while(offset<end) {
                        if(offset+5>end) throw std::runtime_error("truncated exchange reference header");
                        const Int recordRows=static_cast<Int>(all[offset+4]);
                        if(recordRows<=0 || offset+5+recordRows>end)
                            throw std::runtime_error("invalid exchange reference extent");
                        if(static_cast<int>(all[offset])==direction &&
                           static_cast<int>(all[offset+1])==rank &&
                           static_cast<Int>(all[offset+2])==globalFace &&
                           static_cast<int>(all[offset+3])==expectedSender) {
                            if(recordRows!=rows)
                                throw std::runtime_error("exchange reference owner-row mismatch");
                            expected=&all[offset+5]; break;
                        }
                        offset+=5+recordRows;
                    }
                }
                if(!expected) throw std::runtime_error("missing exact compact-exchange reference block");
                for(Int row=0;row<rows;++row)
                    if(d.recvbuf.data[row+static_cast<std::size_t>(rows)*block]!=expected[row])
                        throw std::runtime_error("compact exchange payload differs from slot reference");
            }
        if(block*d.view.ncu12*d.view.ownerNpf!=d.recvbuf.size)
            throw std::runtime_error("compact exchange reference block count is incorrect");
    }
}

void descriptor_case(const Options& options, int rank)
{
    using namespace exasim::nonmatching;
    const int model = rank < options.split ? 0 : 1;
    const std::string reject = rejection_mode(options.mode)
            ? options.mode.substr(std::strlen("reject-")) : std::string();
    auto communicator = preprocess::make_interface_communicator<Int>(true, rank, model);
    const int firstOrder = reject.empty() ? 1 : 2;
    const int lastOrder = reject.empty() ? 4 : 2;
    for (int baseOrder = firstOrder; baseOrder <= lastOrder; ++baseOrder) {
        const int p = reject == "porder" && model == 1 ? baseOrder + 1 : baseOrder;
        SetupMaster3D data = setup_master(model, p);
        if (reject == "pgauss-unequal" && model == 1) data.ndims[4] += 2;
        if (reject == "pgauss-unsupported") data.ndims[4] = 3;
        if (reject == "weight" && model == 1) data.gwf[0] = -data.gwf[0];
        if (reject == "node" && model == 0) data.gpf[0] = -1.0e-3;
        if (reject == "measure" && model == 1)
            for (dstype& value : data.gwf) value *= 1.01;
        if (reject == "inconsistent" && model == 1 && rank == options.split)
            data.gwf[0] += 1.0e-3;
        if (reject == "family" && model == 1) data.ndims[1] = 2;

        masterstruct master{};
        master.ndims = data.ndims.data(); master.xpf = data.xpf.data();
        master.gpf = data.gpf.data(); master.gwf = data.gwf.data();
        const Int ncu12 = reject == "ncu12" && model == 1 ? 2 : 1;
        const auto compatibility = check_face_master_compatibility<dstype, Int>(
                master, data.nfe, model, communicator, ncu12);
        if (!reject.empty())
            throw std::runtime_error("expected mixed descriptor rejection did not occur");
        const Int tetNpf = (baseOrder + 1) * (baseOrder + 2) / 2;
        const Int hexNpf = (baseOrder + 1) * (baseOrder + 1);
        if (!compatibility.mixedTopology3D || !compatibility.mixedFamily ||
                compatibility.model[0].elemtype != 0 ||
                compatibility.model[1].elemtype != 1 ||
                compatibility.model[0].npf != tetNpf ||
                compatibility.model[1].npf != hexNpf || tetNpf == hexNpf ||
                compatibility.model[0].ngf <= 0 || compatibility.model[1].ngf <= 0)
            throw std::runtime_error("mixed descriptor roles are incorrect");

        // Regression for the production role check: frozen directions both
        // carry this rank's local master sizes, whereas sidecar direction k
        // carries owner k and donor 1-k.  The owner-only direction must not
        // be compared with its opposite-family donor dimensions.
        nonmatchingdata host{}; host.interfacetype=1;
        OwnerTestData<dstype,Int> sidecar;
        for(int direction=0;direction<2;++direction) {
            auto& frozen=host.dir[direction]; frozen.direction=direction;
            frozen.nd=3; frozen.npe=data.ndims[5]; frozen.npf=data.ndims[6];
            frozen.nfe=data.nfe; frozen.ngf=data.ndims[8]; frozen.ncu=1;
            frozen.ncu12=1;
            frozen.isowner=direction==model; frozen.isdonor=direction!=model;
            auto& role=sidecar.dir[direction]; role.enabled=1; role.rectangular=1;
            role.ownerElemtype=compatibility.model[direction].elemtype;
            role.ownerNpf=compatibility.model[direction].npf;
            role.ownerNgf=compatibility.model[direction].ngf;
            role.donorElemtype=compatibility.model[1-direction].elemtype;
            role.donorNpe=compatibility.model[1-direction].npe;
            role.donorNpf=compatibility.model[1-direction].npf;
            role.donorNfe=compatibility.model[1-direction].nfe;
        }
        appstruct app{}; meshstruct mesh{}; commonstruct common{};
        common.components.ncu=1; common.components.nc=1; common.components.ncx=3;
        NonmatchingRuntime<dstype,Int> runtime;
        refresh_nonmatching_runtime(runtime,host,app,mesh,common,0,&sidecar);
        if(runtime.dir[model].view.ownerNpf!=compatibility.model[model].npf ||
                runtime.dir[1-model].view.donorNpf!=compatibility.model[model].npf)
            throw std::runtime_error("mixed runtime role dimensions are incorrect");
    }
}

void mixed_rebuild_case(const Options& options,int rank,int p,bool reverseFamilies)
{
    using namespace exasim::nonmatching;
    const int model=rank<options.split?0:1;
    const int modelFirst=model==0?0:options.split;
    int worldSize=1; MPI_Comm_size(MPI_COMM_WORLD,&worldSize);
    const int actualModelRanks=model==0?options.split:worldSize-options.split;
    const int part=rank-modelFirst;
    const int elemtype=reverseFamilies?1-model:model;
    SetupMaster3D masterData=setup_master(elemtype,p);
    SetupRank3D data;
    setup_rank_3d(data,masterData,model,part,actualModelRanks,0.0);
    if(part>0) {
        std::fill(data.bf.begin(),data.bf.end(),1);
        data.mesh.szintfaces=0;
    }
    nonmatchingdata host;
    commonstruct common{};
    common.components.nc=1; common.components.ncu=1; common.components.ncq=0;
    common.components.ncx=3; common.meshsizes.ne=1; common.meshsizes.ne1=1;
    common.meshsizes.nfe=masterData.nfe;
    common.couplingparams.coupledboundarycondition=2;
    std::unique_ptr<NonmatchingRuntime<dstype,Int>> runtime;
    std::unique_ptr<OwnerTestData<dstype,Int>> sidecar;
    DonorSearchResult<dstype,Int> search;
    const double setupStart=MPI_Wtime();
    const bool ran=rebuild_nonmatching_interface(data.app,data.master,data.mesh,data.sol,
        common,worldSize,rank,model,host,runtime,0,&search,DonorSearchParams(),true,&sidecar);
    const double setupSeconds=MPI_Wtime()-setupStart;
    if(!ran || !runtime || !sidecar)
        throw std::runtime_error("mixed transactional rebuild did not publish its caches");
    if(part>0) {
        if(sidecar->dir[0].rectangular || sidecar->dir[1].rectangular ||
                runtime->dir[0].view.isowner || runtime->dir[0].view.isdonor ||
                runtime->dir[1].view.isowner || runtime->dir[1].view.isdonor)
            throw std::runtime_error("empty-interface rank published mixed role data");
        // The active ranks use these collectives for exact-payload and global
        // cost references after their point-to-point exchange.  Empty ranks
        // contribute empty records and zero metrics in the same order.
        verify_exact_exchange_payload(host,*runtime,rank);
        long long zeroSums[5]={0,0,0,0,0},ignoredSums[5]={0,0,0,0,0};
        long long zeroPeak=0,ignoredPeak=0; double zeroTimes[2]={0,0},ignoredTimes[2]={0,0};
        MPI_Reduce(zeroSums,ignoredSums,5,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
        MPI_Reduce(&zeroPeak,&ignoredPeak,1,MPI_LONG_LONG,MPI_MAX,0,MPI_COMM_WORLD);
        MPI_Reduce(zeroTimes,ignoredTimes,2,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        free_nonmatching_data(host); runtime.reset(); sidecar.reset();
        CPUFREE(common.requests); CPUFREE(common.statuses);
        return;
    }
    const auto& owner=runtime->dir[model];
    const auto& donor=runtime->dir[1-model];
    if(!owner.view.rectangular || !donor.view.rectangular ||
            owner.view.ownerElemtype!=elemtype || owner.view.ownerNpf!=masterData.npf ||
            donor.view.donorElemtype!=elemtype || donor.view.donorNpe!=masterData.npe ||
            donor.view.donorNpf!=masterData.npf || donor.view.donorNfe!=masterData.nfe)
        throw std::runtime_error("mixed transactional rebuild published incorrect role dimensions");
    const auto& frozenDonor=host.dir[1-model];
    if(frozenDonor.Ri || frozenDonor.Hi || frozenDonor.sendbuf ||
            frozenDonor.szRi!=0 || frozenDonor.szHi!=0 || frozenDonor.szsendbuf!=0)
        throw std::runtime_error("mixed transactional rebuild used frozen rectangular work arrays");
    const Int rows=donor.view.ncu12*donor.view.ownerNpf;
    const Int columns=donor.view.ncu*donor.view.donorNpf*donor.view.donorNfe;
    if(donor.Ri.size!=rows*donor.view.nslots ||
            donor.Hi.size!=rows*columns*donor.view.npairs ||
            donor.ptest.size!=donor.view.ownerNpf*donor.view.npts)
        throw std::runtime_error("mixed transactional rebuild allocated incorrect rectangular extents");
    for(Int point=0;point<donor.view.npts;++point) {
        long double partition=0;
        for(Int i=0;i<donor.view.ownerNpf;++i)
            partition+=donor.ptest.data[i+donor.view.ownerNpf*point];
        if(std::abs(partition-1.0L)>5e-12L)
            throw std::runtime_error("transported mixed owner tests fail partition of unity");
    }

    for(Int i=0;i<donor.Ri.size;++i)
        donor.Ri.data[i]=1.0+0.01*rank+0.0001*i;
    for(Int i=0;i<owner.recvbuf.size;++i) owner.recvbuf.data[i]=-91.0;
    const long long allocations=runtime->allocations;
    const double runtimeStart=MPI_Wtime();
    Int requestCounter=0;
    begin_coupling_slot_exchange(*runtime,common,requestCounter);
    Int expectedRequests=0;
    for(int direction=0;direction<2;++direction)
        expectedRequests+=static_cast<Int>(runtime->dir[direction].sendnb.size()+
                                           runtime->dir[direction].recvnb.size());
    if(requestCounter!=expectedRequests)
        throw std::runtime_error("mixed compact exchange request count is incorrect");
    if(requestCounter>0) MPI_Waitall(requestCounter,common.requests,common.statuses);
    for(Int request=0;request<requestCounter;++request)
        if(common.requests[request]!=MPI_REQUEST_NULL)
            throw std::runtime_error("mixed compact exchange left an active request");
    for(Int i=0;i<owner.recvbuf.size;++i)
        if(owner.recvbuf.data[i]==-91.0)
            throw std::runtime_error("mixed compact exchange left an owner block untouched");
    verify_exact_exchange_payload(host,*runtime,rank);
    if(runtime->allocations!=allocations)
        throw std::runtime_error("mixed compact exchange allocated at runtime");
    const double runtimeSeconds=MPI_Wtime()-runtimeStart;

    const long long localMpiBytes=static_cast<long long>(sizeof(dstype))*rows*
        std::accumulate(donor.sendcount.begin(),donor.sendcount.end(),0LL);
    const long long localPersistent=runtime_direction_bytes(runtime->dir[0])+
        runtime_direction_bytes(runtime->dir[1]);
    const long long localHost=host_direction_bytes(host.dir[0])+host_direction_bytes(host.dir[1]);
    const long long localPtest=static_cast<long long>(sizeof(dstype))*donor.ptest.capacity;
    const long long localPoints=donor.view.npts;
    long long sums[5]={localMpiBytes,localPersistent,localHost,localPtest,localPoints};
    long long globalSums[5]={0,0,0,0,0},peakPersistent=0;
    double localTimes[2]={setupSeconds,runtimeSeconds},criticalTimes[2]={0,0};
    MPI_Reduce(sums,globalSums,5,MPI_LONG_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&localPersistent,&peakPersistent,1,MPI_LONG_LONG,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(localTimes,criticalTimes,2,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) {
        std::printf("PHASE13 REAL COST p %d ordering %c split %d setup-critical %.6e "
                    "runtime-critical %.6e points-global %lld allocations %lld host-bytes-global %lld "
                    "persistent-bytes-global %lld persistent-bytes-peak-rank %lld "
                    "ptest-bytes-global %lld mpi-bytes-global %lld\n",
                    p,reverseFamilies?'B':'A',options.split,criticalTimes[0],criticalTimes[1],
                    globalSums[4],runtime->allocations,globalSums[2],globalSums[1],peakPersistent,
                    globalSums[3],globalSums[0]);
    }
    free_nonmatching_data(host); runtime.reset(); sidecar.reset();
    CPUFREE(common.requests); CPUFREE(common.statuses);
}

void rectangular_helper_case()
{
    using namespace exasim::nonmatching;
    constexpr std::size_t npe=3,ncu12=2,ncu=2,ownerNpf=3,donorNpf=2,donorNfe=2,pairs=2;
    const std::size_t ndf=donorNpf*donorNfe,rows=ncu12*ownerNpf,columns=ncu*ndf;
    const std::size_t M=ownerNpf*npe,L=M*pairs,P=L*ncu12;
    std::vector<dstype> G(ownerNpf*npe*pairs*ncu12*ncu),E(npe*ndf*pairs);
    std::vector<dstype> H(rows*columns*pairs);
    for(std::size_t i=0;i<G.size();++i) G[i]=0.01*(1+i%17);
    for(std::size_t i=0;i<E.size();++i) E[i]=-0.02*(1+i%11);
    for(std::size_t i=0;i<H.size();++i) H[i]=0.03*(1+i%7);
    std::vector<long double> reference(H.begin(),H.end());
    const long double scalar=0.7L;
    for(std::size_t e=0;e<pairs;++e) for(std::size_t j=0;j<ndf;++j)
      for(std::size_t n=0;n<ncu;++n) for(std::size_t i=0;i<ownerNpf;++i)
       for(std::size_t m=0;m<ncu12;++m) {
        long double sum=0;
        for(std::size_t k=0;k<npe;++k)
            sum+=scalar*G[i+ownerNpf*k+M*e+L*m+P*n]*E[k+npe*j+npe*ndf*e];
        reference[(m+ncu12*i)+rows*(n+ncu*j)+rows*columns*e]-=sum;
      }
    schurMatrixGintMinvERect(H.data(),G.data(),E.data(),static_cast<dstype>(scalar),
        npe,ncu12,ncu,ownerNpf,donorNpf,donorNfe,pairs,H.size());
    Kokkos::fence();
    dstype error=0,scale=0;
    for(std::size_t i=0;i<H.size();++i) {
        error=std::max(error,std::abs(H[i]-static_cast<dstype>(reference[i])));
        scale=std::max(scale,std::abs(static_cast<dstype>(reference[i])));
    }
    if(error>1e-14*std::max<dstype>(1,scale))
        throw std::runtime_error("rectangular G*E helper differs from dense reference");

    // Exercise the production rectangular residual, q-free K/H condensation,
    // and runtime matvec kernels.  References deliberately use long-double
    // scalar loops and only the public setup-array layouts.
    NonmatchingRuntime<dstype,Int> runtime;
    runtime.interfacetype=1;
    auto& direction=runtime.dir[0]; auto& v=direction.view;
    v.direction=0; v.isdonor=1; v.rectangular=1; v.nd=3;
    v.ncu=2; v.ncu12=2; v.ncq=0; v.ownerNpf=3;
    v.donorNpe=2; v.donorNpf=2; v.donorNfe=2;
    v.npe=2; v.npf=2; v.nfe=2; v.npts=4; v.nslots=1; v.npairs=2;
    const Int testRows=v.ncu12*v.ownerNpf;
    const Int volumeColumns=v.donorNpe*v.ncu;
    const Int traceColumns=v.donorNpf*v.donorNfe*v.ncu;
    auto resize=[&](auto& buffer,Int count){ buffer.resize(count,0); };
    resize(direction.slotptr,2); direction.slotptr.data[0]=0; direction.slotptr.data[1]=4;
    resize(direction.pairptr,3); direction.pairptr.data[0]=0;
    direction.pairptr.data[1]=2; direction.pairptr.data[2]=4;
    resize(direction.pairSlotPtr,2); direction.pairSlotPtr.data[0]=0;
    direction.pairSlotPtr.data[1]=2;
    resize(direction.pairelem,2); direction.pairelem.data[0]=0; direction.pairelem.data[1]=1;
    resize(direction.pa,4); direction.pa.data[0]=0; direction.pa.data[1]=1;
    direction.pa.data[2]=0; direction.pa.data[3]=1;
    resize(direction.pwJ,4); resize(direction.ptest,v.ownerNpf*v.npts);
    resize(direction.pphi,v.donorNpe*v.npts); resize(direction.ppsi,v.donorNpf*v.npts);
    resize(direction.pd,v.npts*v.nd); resize(direction.pdphibar,v.donorNpe*v.nd*v.npts);
    resize(direction.flux,v.npts*v.ncu12);
    resize(direction.fluxUdg,v.npts*v.ncu12*v.ncu);
    resize(direction.fluxUhg,v.npts*v.ncu12*v.ncu);
    resize(direction.Ri,testRows*v.nslots); resize(direction.Hi,testRows*traceColumns*v.npairs);
    resize(direction.K,testRows*volumeColumns*v.npairs);
    resize(direction.Fpair,volumeColumns*traceColumns*v.npairs);
    resize(direction.Rupair,volumeColumns*v.npairs);
    resize(direction.pairResidual,testRows*v.npairs);
    resize(direction.vpair,traceColumns*v.npairs); resize(direction.rpair,testRows*v.npairs);
    for(Int i=0;i<direction.pwJ.size;++i) direction.pwJ.data[i]=0.4+0.1*i;
    for(Int i=0;i<direction.ptest.size;++i) direction.ptest.data[i]=0.05*(1+i%7);
    for(Int i=0;i<direction.pphi.size;++i) direction.pphi.data[i]=0.07*(1+i%5);
    for(Int i=0;i<direction.ppsi.size;++i) direction.ppsi.data[i]=-0.03*(1+i%6);
    for(Int i=0;i<direction.pd.size;++i) direction.pd.data[i]=0.02*(1+i%4);
    for(Int i=0;i<direction.pdphibar.size;++i) direction.pdphibar.data[i]=0.01*(1+i%9);
    for(Int i=0;i<direction.flux.size;++i) direction.flux.data[i]=0.11*(1+i%8);
    for(Int i=0;i<direction.fluxUdg.size;++i) direction.fluxUdg.data[i]=0.013*(1+i%11);
    for(Int i=0;i<direction.fluxUhg.size;++i) direction.fluxUhg.data[i]=-0.017*(1+i%13);
    direction.bind_view();

    std::vector<long double> expectedRi(testRows,0.0L);
    for(Int row=0;row<testRows;++row) {
        const Int out=row%v.ncu12, i=row/v.ncu12;
        for(Int point=0;point<v.npts;++point)
            expectedRi[row]-=static_cast<long double>(direction.pwJ.data[point])*
                direction.ptest.data[i+v.ownerNpf*point]*
                direction.flux.data[point+v.npts*out];
    }
    form_rectangular_slot_residual(direction); Kokkos::fence();
    for(Int i=0;i<testRows;++i)
        if(std::abs(static_cast<long double>(direction.Ri.data[i])-expectedRi[i])>
                1e-12L*std::max(1.0L,std::abs(expectedRi[i])))
            throw std::runtime_error("rectangular production residual differs from dense reference");

    std::vector<dstype> localF(static_cast<std::size_t>(volumeColumns)*traceColumns*2);
    std::vector<dstype> localRu(static_cast<std::size_t>(volumeColumns)*2);
    for(std::size_t i=0;i<localF.size();++i) localF[i]=0.004*(1+i%17);
    for(std::size_t i=0;i<localRu.size();++i) localRu[i]=-0.006*(1+i%7);
    resstruct residual{}; residual.F=localF.data(); residual.Ru=localRu.data();
    commonstruct common{}; common.meshsizes.ne=2;

    std::vector<long double> expectedK(static_cast<std::size_t>(testRows)*volumeColumns*2,0.0L);
    std::vector<long double> expectedH(static_cast<std::size_t>(testRows)*traceColumns*2,0.0L);
    for(Int pair=0;pair<v.npairs;++pair) {
        for(Int col=0;col<volumeColumns;++col) for(Int row=0;row<testRows;++row) {
            const Int out=row%v.ncu12, i=row/v.ncu12, j=col%v.donorNpe, c=col/v.donorNpe;
            long double value=0;
            for(Int point=direction.pairptr.data[pair];point<direction.pairptr.data[pair+1];++point) {
                const long double W=static_cast<long double>(direction.pwJ.data[point])*
                    direction.ptest.data[i+v.ownerNpf*point];
                long double derivative=static_cast<long double>(direction.fluxUdg.data[
                    point+v.npts*(out+v.ncu12*c)])*direction.pphi.data[j+v.donorNpe*point];
                for(Int d=0;d<v.nd;++d)
                    derivative+=static_cast<long double>(direction.fluxUhg.data[
                        point+v.npts*(out+v.ncu12*c)])*direction.pd.data[point+v.npts*d]*
                        direction.pdphibar.data[j+v.donorNpe*(d+v.nd*point)];
                value+=W*derivative;
            }
            expectedK[row+testRows*(col+volumeColumns*pair)]=value;
        }
        for(Int col=0;col<traceColumns;++col) for(Int row=0;row<testRows;++row) {
            const Int out=row%v.ncu12, i=row/v.ncu12, c=col%v.ncu;
            const Int faceNode=col/v.ncu, j=faceNode%v.donorNpf, a=faceNode/v.donorNpf;
            long double value=0;
            for(Int point=direction.pairptr.data[pair];point<direction.pairptr.data[pair+1];++point)
                if(direction.pa.data[point]==a)
                    value+=static_cast<long double>(direction.pwJ.data[point])*
                        direction.ptest.data[i+v.ownerNpf*point]*
                        direction.fluxUhg.data[point+v.npts*(out+v.ncu12*c)]*
                        direction.ppsi.data[j+v.donorNpf*point];
            expectedH[row+testRows*(col+traceColumns*pair)]=value;
        }
    }
    for(Int pair=0;pair<v.npairs;++pair) {
        const Int element=direction.pairelem.data[pair];
        for(Int col=0;col<traceColumns;++col) for(Int row=0;row<testRows;++row)
            for(Int k=0;k<volumeColumns;++k)
                expectedH[row+testRows*(col+traceColumns*pair)]+=
                    expectedK[row+testRows*(k+volumeColumns*pair)]*
                    localF[k+volumeColumns*(col+traceColumns*element)];
        for(Int row=0;row<testRows;++row)
            for(Int k=0;k<volumeColumns;++k)
                expectedRi[row]-=expectedK[row+testRows*(k+volumeColumns*pair)]*
                    localRu[k+volumeColumns*element];
    }
    form_rectangular_pairs(direction,residual,common,0,0); Kokkos::fence();
    auto compare=[&](const dstype* actual,const std::vector<long double>& wanted,const char* label) {
        for(std::size_t i=0;i<wanted.size();++i)
            if(std::abs(static_cast<long double>(actual[i])-wanted[i])>
                    1e-12L*std::max(1.0L,std::abs(wanted[i])))
                throw std::runtime_error(std::string("rectangular production ")+label+
                        " differs from dense reference");
    };
    compare(direction.K.data,expectedK,"K"); compare(direction.Hi.data,expectedH,"H");
    compare(direction.Ri.data,expectedRi,"condensed residual");

    std::vector<dstype> trace(static_cast<std::size_t>(traceColumns)*2);
    for(std::size_t i=0;i<trace.size();++i) trace[i]=0.009*(1+i%19);
    std::vector<long double> expectedProduct(testRows,0.0L);
    for(Int pair=0;pair<v.npairs;++pair) for(Int row=0;row<testRows;++row)
        for(Int col=0;col<traceColumns;++col)
            expectedProduct[row]+=expectedH[row+testRows*(col+traceColumns*pair)]*
                trace[col+traceColumns*direction.pairelem.data[pair]];
    nonmatching_matvec_products(runtime,trace.data(),traceColumns,2,0,0); Kokkos::fence();
    compare(direction.Ri.data,expectedProduct,"matvec");
}

void duplicate_owner_scatter_case()
{
    using namespace exasim::nonmatching;
    NonmatchingRuntime<dstype,Int> runtime;
    runtime.interfacetype=1;
    auto& direction=runtime.dir[0]; auto& v=direction.view;
    v.direction=0; v.isowner=1; v.rectangular=1; v.ncu=1; v.ncu12=2;
    v.npf=2; v.ownerNpf=2; v.nownfaces=1;
    direction.ownerTarget.resize(1,0); direction.ownerTarget.data[0]=0;
    direction.faceBlockPtr.resize(2,0); direction.faceBlockPtr.data[0]=0;
    direction.faceBlockPtr.data[1]=2;
    direction.faceBlocks.resize(2,0); direction.faceBlocks.data[0]=0;
    direction.faceBlocks.data[1]=1;
    direction.recvbuf.resize(8,0);
    for(Int i=0;i<direction.recvbuf.size;++i) direction.recvbuf.data[i]=0.125*(i+1);
    direction.bind_view();
    std::vector<Int> fluxmap={0,0}; appstruct app{}; app.interfacefluxmap=fluxmap.data();
    meshstruct mesh{}; commonstruct common{}; common.components.ncu=1;
    std::vector<dstype> residual(2,0.0);
    assemble_coupling_slot_owner(residual.data(),runtime,mesh,app,common);
    Kokkos::fence();
    for(Int node=0;node<2;++node) {
        long double expected=0;
        for(Int block=0;block<2;++block) for(Int out=0;out<2;++out)
            expected+=direction.recvbuf.data[out+2*(node+2*block)];
        if(std::abs(static_cast<long double>(residual[node])-expected)>1e-14L)
            throw std::runtime_error("duplicate flux-map owner scatter differs from dense reference");
    }
}

void slot_protocol_case(const Options& options,int rank)
{
    using namespace exasim::nonmatching;
    const int model=rank<options.split?0:1;
    auto communicator=preprocess::make_interface_communicator<Int>(true,rank,model);
    nonmatchinginterface donor{},owner{};
    std::vector<Int> sendnb,sendcount,slotrank,slotface,recvnb,recvcount,recvface,ownfaceglobal;
    Int advertised=6,local=6;
    const int sender=0,receiver=options.split;
    if(rank==sender) {
        sendnb={receiver}; sendcount={1}; slotrank={receiver}; slotface={17};
        donor.nsendnb=1; donor.nslots=1;
        if(options.mode=="reject-slotcount") sendcount[0]=2;
        if(options.mode=="reject-slotface") slotface[0]=18;
        if(options.mode=="reject-extent") advertised=5;
    }
    if(rank==receiver) {
        recvnb={sender}; recvcount={1}; recvface={0}; ownfaceglobal={17};
        owner.nrecvnb=1; owner.nownfaces=1;
    }
    donor.sendnb=sendnb.data(); donor.sendcount=sendcount.data();
    donor.slotrank=slotrank.data(); donor.slotface=slotface.data();
    owner.recvnb=recvnb.data(); owner.recvcount=recvcount.data();
    owner.recvface=recvface.data(); owner.ownfaceglobal=ownfaceglobal.data();
    verify_slot_lists_and_extents(communicator,donor,owner,advertised,local);
    if(rejection_mode(options.mode)) throw std::runtime_error("expected slot rejection did not occur");
}

void compact_exchange_case(const Options& options, int rank)
{
    using namespace exasim::nonmatching;
    NonmatchingRuntime<dstype, Int> runtime;
    runtime.interfacetype = 1;
    const Int ownerNpf[2] = {6, 9};
    for (int direction = 0; direction < 2; ++direction) {
        const int ownerRank = direction == 0 ? 0 : options.split;
        const int donorRank = direction == 0 ? options.split : 0;
        auto& d = runtime.dir[direction];
        d.view.direction = direction;
        d.view.rectangular = 1;
        d.view.ownerNpf = ownerNpf[direction];
        d.view.npf = ownerNpf[direction];
        d.view.ncu12 = 2;
        if (rank == donorRank) {
            d.view.isdonor = 1;
            d.view.nslots = 1;
            d.sendnb = {ownerRank};
            d.sendcount = {1};
            d.Ri.resize(2 * ownerNpf[direction], 0);
            for (Int i = 0; i < d.Ri.size; ++i)
                d.Ri.data[i] = static_cast<dstype>(100 * direction + i + 1);
        }
        if (rank == ownerRank) {
            d.view.isowner = 1;
            d.recvnb = {donorRank};
            d.recvcount = {1};
            d.recvbuf.resize(2 * ownerNpf[direction], 0);
            std::fill(d.recvbuf.data, d.recvbuf.data + d.recvbuf.size, -1.0);
        }
        d.bind_view();
    }

    std::vector<MPI_Request> requests(2, MPI_REQUEST_NULL);
    std::vector<MPI_Status> statuses(2);
    commonstruct common{};
    common.requests = requests.data();
    common.statuses = statuses.data();
    Int requestCounter = 0;
    begin_coupling_slot_exchange(runtime, common, requestCounter);
    if (rank == 0 || rank == options.split) {
        if (requestCounter != 2)
            throw std::runtime_error("compact exchange request count is incorrect");
        MPI_Waitall(requestCounter, common.requests, common.statuses);
        for (Int request = 0; request < requestCounter; ++request)
            if (common.requests[request] != MPI_REQUEST_NULL)
                throw std::runtime_error("compact exchange left an active MPI request");
        const int receivedDirection = rank == 0 ? 0 : 1;
        const auto& received = runtime.dir[receivedDirection].recvbuf;
        for (Int i = 0; i < received.size; ++i) {
            const dstype expected = static_cast<dstype>(100 * receivedDirection + i + 1);
            if (received.data[i] != expected)
                throw std::runtime_error("compact exchange payload differs from owner-sized block");
        }
    }
    else if (requestCounter != 0) {
        throw std::runtime_error("empty-interface rank posted a compact exchange request");
    }

    // MPI permits a short send into a larger receive.  Demonstrate that the
    // operation succeeds and leaves the unmatched tail untouched; D7's
    // extent handshake is what turns this otherwise-silent case into a setup
    // rejection for production mixed coupling.
    constexpr int extent=6;
    std::vector<dstype> shortBuffer(extent,-77.0);
    std::vector<dstype> shortPayload(extent-1,3.25);
    MPI_Request shortRequest=MPI_REQUEST_NULL; MPI_Status shortStatus{};
    if(rank==0) MPI_Isend(shortPayload.data(),extent-1,mpi_type<dstype>(),
                         options.split,9130+options.split,MPI_COMM_WORLD,&shortRequest);
    if(rank==options.split) MPI_Irecv(shortBuffer.data(),extent,mpi_type<dstype>(),
                         0,9130+options.split,MPI_COMM_WORLD,&shortRequest);
    if(shortRequest!=MPI_REQUEST_NULL) MPI_Wait(&shortRequest,&shortStatus);
    if(rank==options.split) {
        int received=0; MPI_Get_count(&shortStatus,mpi_type<dstype>(),&received);
        if(received!=extent-1 || shortBuffer.back()!=-77.0)
            throw std::runtime_error("plain MPI short-receive demonstration did not retain its tail");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank==0) {
        const long long mpiBytes=static_cast<long long>(sizeof(dstype))*
                2*(ownerNpf[0]+ownerNpf[1]);
        long long persistentBytes=0,ptestBytes=0;
        for(int direction=0;direction<2;++direction) {
            const auto& d=runtime.dir[direction];
            persistentBytes+=static_cast<long long>(sizeof(dstype))*
                    (d.Ri.capacity+d.Hi.capacity+d.recvbuf.capacity+
                     d.vpair.capacity+d.rpair.capacity);
            ptestBytes+=static_cast<long long>(sizeof(dstype))*d.ptest.capacity;
        }
        std::printf("PHASE13 COST split %d allocations %lld persistent-bytes %lld "
                    "ptest-bytes %lld mpi-bytes %lld\n",options.split,
                    runtime.allocations,persistentBytes,ptestBytes,mpiBytes);
    }
}

std::vector<std::vector<dstype>> partition_result(int split,int rank,int size)
{
    using namespace exasim::nonmatching;
    constexpr Int globalFaces=8;
    const int model=rank<split?0:1,first=model==0?0:split;
    const int modelRanks=model==0?split:size-split,part=rank-first;
    const Int quotient=globalFaces/modelRanks,remainder=globalFaces%modelRanks;
    const Int ne=quotient+(part<remainder?1:0);
    const Int firstFace=part*quotient+std::min<Int>(part,remainder);
    SetupMaster3D masterData=setup_master(model,2); SetupRank3D data;
    setup_multi_rank_3d(data,masterData,model,part,modelRanks,0.0,1,0,0,1,ne);
    const dstype stagger=model==1?-0.5:0.0;
    for(Int e=0;e<ne;++e) {
        data.elempart[e]=firstFace+e;
        for(Int j=0;j<masterData.npe;++j) {
            const std::size_t index=j+static_cast<std::size_t>(masterData.npe)*(1+3*e);
            const dstype a=data.xdg[index]-(2.0*part+e+stagger);
            data.xdg[index]=(firstFace+e+a)/globalFaces;
        }
    }
    commonstruct common{}; common.components.nc=1; common.components.ncu=1;
    common.components.ncq=0; common.components.ncx=3; common.meshsizes.ne=ne;
    common.meshsizes.ne0=ne; common.meshsizes.ne1=ne; common.meshsizes.nfe=masterData.nfe;
    common.couplingparams.coupledboundarycondition=2;
    nonmatchingdata host; std::unique_ptr<NonmatchingRuntime<dstype,Int>> runtime;
    std::unique_ptr<OwnerTestData<dstype,Int>> sidecar;
    if(!rebuild_nonmatching_interface(data.app,data.master,data.mesh,data.sol,common,size,rank,
            model,host,runtime,0,static_cast<DonorSearchResult<dstype,Int>*>(nullptr),
            DonorSearchParams(),true,&sidecar))
        throw std::runtime_error("fixed-global partition cache setup failed");
    auto& donor=runtime->dir[1-model];
    for(Int point=0;point<donor.view.npts;++point)
        donor.flux.data[point]=donor.view.px[point+donor.view.npts]+
            2.0*donor.view.px[point+2*donor.view.npts];
    form_runtime_slot_residual(donor,data.master); Kokkos::fence();
    Int requests=0; begin_coupling_slot_exchange(*runtime,common,requests);
    if(requests>0) MPI_Waitall(requests,common.requests,common.statuses);

    const auto& owner=runtime->dir[model]; const auto& ownerHost=host.dir[model];
    std::vector<dstype> local;
    const Int rows=owner.view.ncu12*owner.view.ownerNpf;
    for(Int own=0;own<owner.view.nownfaces;++own) {
        std::vector<long double> values(rows,0.0L);
        for(Int entry=owner.faceBlockPtr.data[own];entry<owner.faceBlockPtr.data[own+1];++entry) {
            const Int block=owner.faceBlocks.data[entry];
            for(Int row=0;row<rows;++row)
                values[row]+=owner.recvbuf.data[row+static_cast<std::size_t>(rows)*block];
        }
        local.push_back(model); local.push_back(ownerHost.ownfaceglobal[own]);
        local.push_back(rows);
        for(long double value:values) local.push_back(static_cast<dstype>(value));
    }
    int count=static_cast<int>(local.size()); std::vector<int> counts(size),displs(size,0);
    MPI_Allgather(&count,1,MPI_INT,counts.data(),1,MPI_INT,MPI_COMM_WORLD);
    int total=0; for(int r=0;r<size;++r) { displs[r]=total; total+=counts[r]; }
    std::vector<dstype> all(std::max(total,1));
    MPI_Allgatherv(local.data(),count,MPI_DOUBLE,all.data(),counts.data(),displs.data(),
        MPI_DOUBLE,MPI_COMM_WORLD);
    std::vector<std::vector<dstype>> records;
    for(int offset=0;offset<total;) {
        if(offset+3>total) throw std::runtime_error("fixed-global partition header is truncated");
        const Int recordRows=static_cast<Int>(all[offset+2]);
        if(recordRows<=0 || offset+3+recordRows>total)
            throw std::runtime_error("fixed-global partition row extent is invalid");
        records.emplace_back(all.begin()+offset,all.begin()+offset+3+recordRows);
        offset+=3+recordRows;
    }
    std::sort(records.begin(),records.end(),[](const auto& a,const auto& b) {
        return a[0]<b[0] || (a[0]==b[0] && a[1]<b[1]);
    });
    free_nonmatching_data(host); runtime.reset(); sidecar.reset();
    CPUFREE(common.requests); CPUFREE(common.statuses);
    return records;
}

void partition_comparison_case(const Options& options,int rank,int size)
{
    const std::vector<std::vector<dstype>> reference=partition_result(options.split,rank,size);
    std::vector<int> comparisons;
    const int complement=size-options.split;
    if(complement>0 && complement<size && complement!=options.split)
        comparisons.push_back(complement);
    if(size>3 && options.split!=3 && complement!=3) comparisons.push_back(3);
    for(int split:comparisons) {
        const std::vector<std::vector<dstype>> candidate=partition_result(split,rank,size);
        if(candidate.size()!=reference.size())
            throw std::runtime_error("fixed-global partition result extent changed");
        for(std::size_t record=0;record<reference.size();++record) {
            if(candidate[record].size()!=reference[record].size() ||
                    candidate[record][0]!=reference[record][0] ||
                    candidate[record][1]!=reference[record][1] ||
                    candidate[record][2]!=reference[record][2])
                throw std::runtime_error("fixed-global partition ordering changed");
            for(std::size_t row=3;row<reference[record].size();++row) {
                const dstype scale=std::max<dstype>(1.0,std::abs(reference[record][row]));
                if(std::abs(candidate[record][row]-reference[record][row])>1e-12*scale)
                    throw std::runtime_error("fixed-global partition residual row changed");
            }
        }
    }
}

struct ReciprocalFlux3D : exasim::ModelDefaults<ReciprocalFlux3D> {
    static constexpr int nd=3,ncu=1,ncq=3,ncw=0,nparam=1,nfint=1;
    static constexpr auto disc=exasim::Discretization::HDG;
    static constexpr bool has_external_coupling=true;
    KOKKOS_INLINE_FUNCTION static void fint(double f[],int,const double[],const double uq[],
        const double[],const double[],const double uh[],const double n[],const double tau[],
        const double[],const double[],double)
    { f[0]=uq[1]*n[0]+uq[2]*n[1]+uq[3]*n[2]+tau[0]*(uq[0]-uh[0]); }
    KOKKOS_INLINE_FUNCTION static void fint_jac_uq(double j[],int,const double[],const double[],
        const double[],const double[],const double[],const double n[],const double tau[],
        const double[],const double[],double)
    { j[0]=tau[0]; j[1]=n[0]; j[2]=n[1]; j[3]=n[2]; }
    KOKKOS_INLINE_FUNCTION static void fint_jac_uh(double j[],int,const double[],const double[],
        const double[],const double[],const double[],const double[],const double tau[],
        const double[],const double[],double)
    { j[0]=-tau[0]; }
};

void map_tet_face_to_triangle(SetupRank3D& data,const SetupMaster3D& masterData,
        Int element,dstype sourceShift,const std::array<dstype,2>& target0,
        const std::array<dstype,2>& target1,const std::array<dstype,2>& target2)
{
    std::array<Int,3> vertex={0,1,2}; long double bestArea=-1;
    for(Int i=0;i<masterData.npf;++i) for(Int j=i+1;j<masterData.npf;++j)
      for(Int k=j+1;k<masterData.npf;++k) {
        const Int vi=masterData.perm[i],vj=masterData.perm[j],vk=masterData.perm[k];
        const std::size_t base=static_cast<std::size_t>(masterData.npe)*3*element;
        const long double ai=data.xdg[base+vi+masterData.npe]-sourceShift;
        const long double bi=data.xdg[base+vi+2*masterData.npe];
        const long double aj=data.xdg[base+vj+masterData.npe]-sourceShift;
        const long double bj=data.xdg[base+vj+2*masterData.npe];
        const long double ak=data.xdg[base+vk+masterData.npe]-sourceShift;
        const long double bk=data.xdg[base+vk+2*masterData.npe];
        const long double area=std::abs((aj-ai)*(bk-bi)-(ak-ai)*(bj-bi));
        if(area>bestArea) { bestArea=area; vertex={vi,vj,vk}; }
      }
    if(!(bestArea>1e-14L)) throw std::runtime_error("tet face vertices are degenerate");
    const std::size_t base=static_cast<std::size_t>(masterData.npe)*3*element;
    const long double a0=data.xdg[base+vertex[0]+masterData.npe]-sourceShift;
    const long double b0=data.xdg[base+vertex[0]+2*masterData.npe];
    const long double a1=data.xdg[base+vertex[1]+masterData.npe]-sourceShift;
    const long double b1=data.xdg[base+vertex[1]+2*masterData.npe];
    const long double a2=data.xdg[base+vertex[2]+masterData.npe]-sourceShift;
    const long double b2=data.xdg[base+vertex[2]+2*masterData.npe];
    const long double determinant=(a1-a0)*(b2-b0)-(a2-a0)*(b1-b0);
    for(Int node=0;node<masterData.npe;++node) {
        const long double a=data.xdg[base+node+masterData.npe]-sourceShift;
        const long double b=data.xdg[base+node+2*masterData.npe];
        const long double lambda1=((a-a0)*(b2-b0)-(a2-a0)*(b-b0))/determinant;
        const long double lambda2=((a1-a0)*(b-b0)-(a-a0)*(b1-b0))/determinant;
        data.xdg[base+node+masterData.npe]=target0[0]+
            lambda1*(target1[0]-target0[0])+lambda2*(target2[0]-target0[0]);
        data.xdg[base+node+2*masterData.npe]=target0[1]+
            lambda1*(target1[1]-target0[1])+lambda2*(target2[1]-target0[1]);
    }
}

void conservation_case(const Options& options,int rank)
{
    using namespace exasim::nonmatching;
    int size=1; MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(size!=2 || options.split!=1)
        throw std::runtime_error("the coupled conservation fixture requires the registered 1+1 split");
    for(int reverse=0;reverse<2;++reverse) for(int fine=0;fine<2;++fine)
      for(int p=1;p<=4;++p) {
        const int model=rank<options.split?0:1;
        const int elemtype=reverse?1-model:model;
        const Int ne=elemtype==0?(fine?8:2):(fine?8:2);
        SetupMaster3D masterData=setup_master(elemtype,p); SetupRank3D data;
        setup_multi_rank_3d(data,masterData,model,0,1,0.0,1,3,0,1,ne);

        if(elemtype==0) {
            const dstype stagger=model==1?-0.5:0.0;
            const Int squares=fine?4:1;
            for(Int e=0;e<2*squares;++e) {
                const Int square=e/2; const dstype scale=fine?0.5:1.0;
                const dstype y0=fine?0.5*(square%2):0.0;
                const dstype z0=fine?0.5*(square/2):0.0;
                if(e%2==0)
                    map_tet_face_to_triangle(data,masterData,e,e+stagger,
                        {y0,z0},{y0+scale,z0},{y0,z0+scale});
                else
                    map_tet_face_to_triangle(data,masterData,e,e+stagger,
                        {y0+scale,z0+scale},{y0,z0+scale},{y0+scale,z0});
            }
        }
        else {
            const dstype stagger=model==1?-0.5:0.0;
            for(Int e=0;e<ne;++e) for(Int j=0;j<masterData.npe;++j) {
                const std::size_t index=j+static_cast<std::size_t>(masterData.npe)*(1+3*e);
                data.xdg[index]-=stagger;
            }
            for(Int e=1;e<ne;++e) data.bf[masterData.nfe*e]=1;
            data.intfaces.resize(1); data.mesh.intfaces=data.intfaces.data();
            data.mesh.szintfaces=1;
        }

        // setup_multi_rank_3d leaves model 1 with negative orientation.  That
        // is harmless for search/algebra fixtures, but it reverses the face
        // node orientation and hence the physical outward normal.  Reflect
        // one tangential coordinate here so this conservation-only mesh is
        // right-handed without changing its interface as a geometric set.
        if(model==1) for(Int e=0;e<ne;++e) for(Int j=0;j<masterData.npe;++j) {
            const std::size_t base=j+static_cast<std::size_t>(masterData.npe)*3*e;
            std::swap(data.xdg[base+masterData.npe],data.xdg[base+2*masterData.npe]);
        }

        const InterfaceFaceSet<dstype,Int> ownFaces=collect_own_interface_faces(
            data.app,data.master,data.mesh,data.sol,rank,0);
        DonorSearchResult<dstype,Int> ownerPoints;
        own_quadrature_points(ownerPoints,ownFaces,masterData.shapfgt.data(),masterData.ngf);
        std::vector<dstype> ownerWeights,ownerNormals;
        owner_point_geometry(ownerWeights,ownerNormals,ownerPoints,ownFaces,
            masterData.shapfgt.data(),masterData.gwf.data());
        dstype localFaceMeasure=0;
        for(const dstype weight:ownerWeights) localFaceMeasure+=std::abs(weight);
        const dstype localFaceSize=std::sqrt(localFaceMeasure);
        dstype globalFaceSize=0;
        MPI_Allreduce(&localFaceSize,&globalFaceSize,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        const dstype geometryTolerance=1000*std::numeric_limits<dstype>::epsilon()*
            std::max(static_cast<dstype>(1),globalFaceSize);
        const dstype expectedNormal=model==0?1.0:-1.0;
        dstype localNormalError=0;
        for(Int point=0;point<ownerPoints.npts;++point) {
            localNormalError=std::max(localNormalError,
                std::abs(ownerNormals[point]-expectedNormal));
            localNormalError=std::max(localNormalError,
                std::abs(ownerNormals[point+ownerPoints.npts]));
            localNormalError=std::max(localNormalError,
                std::abs(ownerNormals[point+2*ownerPoints.npts]));
        }
        dstype globalNormalError=0;
        MPI_Allreduce(&localNormalError,&globalNormalError,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        if(globalNormalError>geometryTolerance)
            throw std::runtime_error("coupled conservation owner normal is not outward");

        const Int ndf=masterData.npf*masterData.nfe,nc=4;
        data.udg.assign(static_cast<std::size_t>(masterData.npe)*nc*ne,0.0);
        data.uh.assign(static_cast<std::size_t>(ndf)*ne,0.0);
        auto integerPower=[](dstype value,int exponent) {
            dstype result=1; for(int k=0;k<exponent;++k) result*=value; return result;
        };
        for(Int e=0;e<ne;++e) {
            for(Int j=0;j<masterData.npe;++j) {
                const std::size_t base=j+static_cast<std::size_t>(masterData.npe)*3*e;
                const dstype x=data.xdg[base],y=data.xdg[base+masterData.npe];
                const dstype z=data.xdg[base+2*masterData.npe],linear=1+y+2*z;
                const dstype h=integerPower(linear,p-1);
                const dstype derivative=p>1?(p-1)*integerPower(linear,p-2):0.0;
                data.udg[j+masterData.npe*(0+nc*e)]=x*h;
                data.udg[j+masterData.npe*(1+nc*e)]=h;
                data.udg[j+masterData.npe*(2+nc*e)]=x*derivative;
                data.udg[j+masterData.npe*(3+nc*e)]=2*x*derivative;
            }
            for(Int faceNode=0;faceNode<ndf;++faceNode) {
                const Int volumeNode=masterData.perm[faceNode];
                const std::size_t base=volumeNode+static_cast<std::size_t>(masterData.npe)*3*e;
                const dstype x=data.xdg[base],y=data.xdg[base+masterData.npe];
                const dstype z=data.xdg[base+2*masterData.npe];
                data.uh[data.elemcon[faceNode+ndf*e]]=x*integerPower(1+y+2*z,p-1);
            }
        }
        data.sol.udg=data.udg.data(); data.sol.uh=data.uh.data();
        std::vector<dstype> tau(1,1.0),param(1,1.0),uinf(1,0.0);
        data.app.tau=tau.data(); data.app.physicsparam=param.data(); data.app.uinf=uinf.data();
        commonstruct common{}; common.components.nc=nc; common.components.ncu=1;
        common.components.ncq=3; common.components.ncx=3; common.grid.nd=3;
        common.grid.npe=masterData.npe; common.grid.npf=masterData.npf;
        common.grid.ngf=masterData.ngf; common.meshsizes.ne=ne;
        common.meshsizes.ne0=ne; common.meshsizes.ne1=ne; common.meshsizes.nfe=masterData.nfe;
        common.spatialScheme=1; common.couplingparams.coupledboundarycondition=2;
        common.couplingparams.coupledcondition=1; common.couplingparams.coupledinterface=1;
        common.couplingparams.interfacetype=1; common.timestate.dtfactor=1;
        nonmatchingdata host; std::unique_ptr<NonmatchingRuntime<dstype,Int>> runtime;
        std::unique_ptr<OwnerTestData<dstype,Int>> sidecar;
        DonorSearchResult<dstype,Int> search;
        if(!rebuild_nonmatching_interface(data.app,data.master,data.mesh,data.sol,common,size,rank,
                model,host,runtime,0,&search,
                DonorSearchParams(),true,&sidecar))
            throw std::runtime_error("coupled conservation cache setup failed");
        dstype localDistance=0; Int localOffFace=0;
        // Only the tet-owner direction is nested in a single donor hex face.
        // The reverse hex-owner direction crosses the tet diagonal and is not
        // subject to the Phase-13 scale-relative nesting threshold.
        if(elemtype==0) for(Int point=0;point<search.npts;++point) {
            localDistance=std::max(localDistance,search.dist[point]);
            if(search.dist[point]>geometryTolerance) ++localOffFace;
        }
        dstype globalDistance=0; Int globalOffFace=0;
        MPI_Allreduce(&localDistance,&globalDistance,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        MPI_Allreduce(&localOffFace,&globalOffFace,1,mpi_type<Int>(),MPI_SUM,MPI_COMM_WORLD);
        if(globalOffFace!=0 || globalDistance>geometryTolerance)
            throw std::runtime_error("coupled conservation nested faces are not coincident");
        if(rank==0)
            std::printf("PHASE13 CONSERVATION GEOMETRY ordering %c fine %d p %d "
                        "normal-error %.3e nested-max-distance %.3e nested-off-face %lld "
                        "geometry-tolerance %.3e\n",
                        reverse?'B':'A',fine,p,globalNormalError,globalDistance,
                        static_cast<long long>(globalOffFace),geometryTolerance);

        const Int n=masterData.npe,trace=ndf;
        std::vector<dstype> C(static_cast<std::size_t>(n)*n*ne*3,0.0);
        std::vector<dstype> E(static_cast<std::size_t>(n)*trace*ne*3,0.0);
        std::vector<dstype> F(static_cast<std::size_t>(n)*trace*ne,0.0);
        std::vector<dstype> Ru(static_cast<std::size_t>(n)*ne,0.0);
        resstruct residual{}; residual.C=C.data(); residual.E=E.data();
        residual.F=F.data(); residual.Ru=Ru.data(); tempstruct tmp{};
        compute_nonmatching_coupling<ReciprocalFlux3D>(*runtime,data.sol,residual,data.app,
            data.master,data.mesh,tmp,common,0,true,0);
        long double local[4]={0,0,0,0};
        const auto& donor=runtime->dir[1-model];
        for(Int i=0;i<donor.Ri.size;++i) {
            local[1-model]+=donor.Ri.data[i]; local[2+(1-model)]+=std::abs(donor.Ri.data[i]);
        }
        long double total[4]={0,0,0,0};
        MPI_Allreduce(local,total,4,MPI_LONG_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        const long double tolerance=1000.0L*std::numeric_limits<dstype>::epsilon()*
            std::max(total[2]+total[3],static_cast<long double>(std::numeric_limits<dstype>::min()));
        if(std::abs(total[0]+total[1])>tolerance)
            throw std::runtime_error("mixed action-reaction conservation gate failed");
        {
            SetupMaster3D hex=setup_master(1,p); long double donorIntegral=0;
            for(Int g=0;g<hex.ngf;++g) {
                const long double y=hex.gpf[g],z=hex.gpf[g+hex.ngf];
                long double h=1,linear=1+y+2*z;
                for(int degree=0;degree<p-1;++degree) h*=linear;
                donorIntegral+=static_cast<long double>(hex.gwf[g])*h;
            }
            const int tetOwnerDirection=reverse?1:0;
            // Production supplies -pnl to fint and then forms Ri=-integral(flux),
            // so the residual total has the sign of the owner's outward normal.
            const long double donorReference=tetOwnerDirection==0?donorIntegral:-donorIntegral;
            if(std::abs(total[tetOwnerDirection]-donorReference)>tolerance)
                throw std::runtime_error("mixed transfer-consistency gate failed");
        }
        free_nonmatching_data(host); runtime.reset(); sidecar.reset();
        CPUFREE(common.requests); CPUFREE(common.statuses);
      }
}

void host_guard_case()
{
    exasim::nonmatching::OwnerTestData<dstype,Int> sidecar;
    sidecar.dir[0].rectangular=sidecar.dir[1].rectangular=1;
    exasim::nonmatching::reject_rectangular_host_path(&sidecar,"compute_nonmatching_coupling");
    throw std::runtime_error("expected host rejection did not occur");
}

} // namespace phase13

int main(int argc,char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc,&argv); EXASIM_COMM_WORLD=MPI_COMM_WORLD;
    int rank=0,size=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
#else
    const int rank=0,size=1;
#endif
    Kokkos::initialize(argc,argv);
    int failed=0;
    try {
        const auto options=phase13::parse(argc,argv,size);
        if(options.mode=="setup") {
            phase13::descriptor_case(options,rank);
            phase13::mixed_rebuild_case(options,rank,1,false);
            phase13::mixed_rebuild_case(options,rank,4,false);
            phase13::mixed_rebuild_case(options,rank,1,true);
            phase13::mixed_rebuild_case(options,rank,4,true);
            if(size>2 && options.split!=size-options.split) {
                auto complement=options; complement.split=size-options.split;
                phase13::descriptor_case(complement,rank);
                phase13::mixed_rebuild_case(complement,rank,1,false);
                phase13::mixed_rebuild_case(complement,rank,4,false);
                phase13::mixed_rebuild_case(complement,rank,1,true);
                phase13::mixed_rebuild_case(complement,rank,4,true);
            }
        }
        else if(options.mode=="operators") {
            phase13::rectangular_helper_case();
            phase13::duplicate_owner_scatter_case();
            model_operator_case<WCoupled3D>("mixed-q-w-nonlinear-A",0,2,rank,size,
                0.0,options.split,true,false);
            model_operator_case<WCoupled3D>("mixed-q-w-nonlinear-B",0,2,rank,size,
                0.0,options.split,true,true);
            model_operator_case<QFreeCoupled3D>("mixed-q-free-A",0,2,rank,size,
                0.0,options.split,true,false);
            model_operator_case<QFreeCoupled3D>("mixed-q-free-B",0,2,rank,size,
                0.0,options.split,true,true);
            if(failures) throw std::runtime_error("mixed independent operator checks failed");
        }
        else if(options.mode=="conservation") phase13::conservation_case(options,rank);
        else if(options.mode=="cost") {
            phase13::mixed_rebuild_case(options,rank,1,false);
            phase13::mixed_rebuild_case(options,rank,4,false);
            phase13::mixed_rebuild_case(options,rank,1,true);
            phase13::mixed_rebuild_case(options,rank,4,true);
            cost_runtime_case(0,rank,size,2,options.split);
            cost_runtime_case(1,rank,size,2,options.split);
            cost_runtime_case(0,rank,size,8,options.split);
            cost_runtime_case(1,rank,size,8,options.split);
            if(failures) throw std::runtime_error("Phase 13 cost diagnostics failed");
        }
        else if(options.mode=="mpi") {
            phase13::partition_comparison_case(options,rank,size);
            phase13::mixed_rebuild_case(options,rank,2,false);
            phase13::mixed_rebuild_case(options,rank,2,true);
            model_operator_case<WCoupled3D>("mixed-mpi-A",0,2,rank,size,
                0.0,options.split,true,false);
            model_operator_case<WCoupled3D>("mixed-mpi-B",0,2,rank,size,
                0.0,options.split,true,true);
            phase13::slot_protocol_case(options,rank);
            phase13::compact_exchange_case(options,rank);
            int alternateSplit=-1;
            if(size==4) alternateSplit=3;
            else if(size==8) alternateSplit=3;
            if(alternateSplit>0 && alternateSplit!=options.split) {
                auto alternate=options; alternate.split=alternateSplit;
                phase13::mixed_rebuild_case(alternate,rank,2,false);
                phase13::mixed_rebuild_case(alternate,rank,2,true);
                model_operator_case<WCoupled3D>("mixed-mpi-alternate-A",0,2,rank,size,
                    0.0,alternate.split,true,false);
                model_operator_case<WCoupled3D>("mixed-mpi-alternate-B",0,2,rank,size,
                    0.0,alternate.split,true,true);
                phase13::slot_protocol_case(alternate,rank);
                phase13::compact_exchange_case(alternate,rank);
            }
            if(failures) throw std::runtime_error("mixed MPI operator checks failed");
        }
        else if(options.mode=="reject-slotcount" ||
                options.mode=="reject-slotface" || options.mode=="reject-extent")
            phase13::slot_protocol_case(options,rank);
        else if(options.mode=="reject-host") phase13::host_guard_case();
        else if(phase13::rejection_mode(options.mode)) phase13::descriptor_case(options,rank);
        else throw std::runtime_error("unknown Phase 13 mode");
        if(rank==0) std::printf("PASS Phase 13 %s split %d\n",options.mode.c_str(),options.split);
    }
    catch(const std::exception& exception) {
        std::fprintf(stderr,"[Rank %d] ERROR: %s\n",rank,exception.what());
        failed=1;
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD); int globalFailed=0;
    MPI_Allreduce(&failed,&globalFailed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD); failed=globalFailed;
#endif
    Kokkos::finalize();
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return failed?1:0;
}
