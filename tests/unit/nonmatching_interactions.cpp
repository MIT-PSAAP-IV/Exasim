// Focused Phase 4 graph-construction fixture.  Numerical operator accuracy
// remains covered by the Phase 13/14 production-kernel tests; this executable
// adds real multi-object setup, per-edge metadata/tags, compact MPI capacity,
// and graph-selection coverage without registering an unmeasured CTest.

#ifdef HAVE_MPI
#include <mpi.h>
extern MPI_Comm EXASIM_COMM_WORLD;
extern MPI_Comm EXASIM_COMM_LOCAL;

namespace phase4_mpi_counts {
int worldAllreduces = 0;
int worldSplits = 0;
}

inline int phase4_counted_allreduce(const void* sendbuf, void* recvbuf, int count,
        MPI_Datatype datatype, MPI_Op op, MPI_Comm communicator)
{
    if (communicator == EXASIM_COMM_WORLD)
        ++phase4_mpi_counts::worldAllreduces;
    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, communicator);
}

inline int phase4_counted_comm_split(MPI_Comm communicator, int color, int key,
        MPI_Comm* result)
{
    if (communicator == EXASIM_COMM_WORLD)
        ++phase4_mpi_counts::worldSplits;
    return PMPI_Comm_split(communicator, color, key, result);
}

#define MPI_Allreduce phase4_counted_allreduce
#define MPI_Comm_split phase4_counted_comm_split
#endif
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define main phase4_embedded_nonmatching_3d_main
#include "nonmatching_3d.cpp"
#undef main
#ifdef HAVE_MPI
#undef MPI_Allreduce
#undef MPI_Comm_split
#endif

namespace {

using Interaction = exasim::nonmatching::NonmatchingInteraction<dstype, Int>;
using Record = exasim::nonmatching::InteractionRecord<Int>;

struct ConservativeGraph3D : exasim::ModelDefaults<ConservativeGraph3D> {
    static constexpr int nd=3,ncu=1,ncq=0,ncw=0,nparam=1,nfint=1;
    static constexpr auto disc=exasim::Discretization::HDG;
    static constexpr bool has_external_coupling=true;
    KOKKOS_INLINE_FUNCTION static void fint(double f[],int,const double x[],const double[],
        const double[],const double[],const double[],const double nl[],const double[],
        const double[],const double[],double)
    { f[0]=nl[0]*(1.0+x[1]+0.25*x[2]); }
    KOKKOS_INLINE_FUNCTION static void fint_jac_uq(double j[],int,const double[],const double[],
        const double[],const double[],const double[],const double[],const double[],
        const double[],const double[],double) { j[0]=0.0; }
    KOKKOS_INLINE_FUNCTION static void fint_jac_uh(double j[],int,const double[],const double[],
        const double[],const double[],const double[],const double[],const double[],
        const double[],const double[],double) { j[0]=0.0; }
};

struct GraphInputs {
    std::vector<Int> table, meshSizes, boundaries, flags;
    std::vector<dstype> dt, solver;
};

std::vector<Int> parallel_table()
{
    return {1,2,2,
            11,0,1,1,1,1,1,0,1,1,1,0,
            11,0,1,1,2,2,1,0,2,2,1,0};
}

std::vector<Int> single_table()
{
    return {1,2,1, 11,0,1,1,1,1,1,0,1,1,1,0};
}

std::vector<Int> isolated_table()
{
    auto table=parallel_table(); table[1]=3; return table;
}

std::vector<Int> inconsistent_width_table()
{
    return {1,2,2,
            11,0,1,1,1,1,1,0,1,1,1,0,
            12,0,1,1,2,2,1,0,2,2,2,0,0};
}

std::vector<Int> unequal_width_table()
{
    return {1,2,1,
            12,0,1,1,1,1,1,0,1,1,2,0,1};
}

std::vector<Int> chain_table()
{
    return {1,4,3,
            11,0,1,1,1,1,1,0,1,1,1,0,
            11,1,2,1,2,2,1,0,1,1,1,0,
            11,2,3,1,2,3,1,0,1,1,1,0};
}

std::vector<Int> star_table()
{
    return {1,4,3,
            11,0,1,1,1,1,1,0,1,1,1,0,
            11,0,2,1,2,2,1,0,1,1,1,0,
            11,0,3,1,3,3,1,0,1,1,1,0};
}

std::vector<double> as_double(const std::vector<Int>& values)
{
    return std::vector<double>(values.begin(), values.end());
}

void require(bool value, const char* text, int& failures)
{
    if (!value) {
        std::fprintf(stderr, "FAIL Phase 4 interactions: %s\n", text);
        ++failures;
    }
}

void attach_graph_inputs(SetupRank3D& rankData, GraphInputs& graph,
        const std::vector<Int>& table, Int localDegree)
{
    graph.table = table;
    graph.meshSizes.assign(30, 0);
    graph.meshSizes[27] = localDegree;
    graph.boundaries.resize(static_cast<std::size_t>(localDegree));
    for (Int i=0;i<localDegree;++i) graph.boundaries[i]=i+2;
    graph.flags.assign(21, 0);
    graph.dt = {0.1};
    graph.solver = {1.0e-8, 1.0e-10, 1.0e-8, 0.0};
    rankData.nsize[4] = static_cast<Int>(graph.dt.size());
    rankData.nsize[7] = static_cast<Int>(graph.solver.size());
    rankData.app.interactions = graph.table.data();
    rankData.app.szinteractions = static_cast<Int>(graph.table.size());
    rankData.app.flag = graph.flags.data();
    rankData.app.dt = graph.dt.data();
    rankData.app.solversparam = graph.solver.data();
    rankData.mesh.nsize = graph.meshSizes.data();
    rankData.mesh.boundaryConditions = graph.boundaries.data();
}

void setup_graph_rank(SetupRank3D& rankData, SetupMaster3D& masterData,
        Int localDomain, Int localPart, const std::vector<Record>& records,
        Int localNcu, Int localNfint)
{
    std::vector<Int> sides;
    for (const auto& record:records) {
        if(record.endpoint[0].domain==localDomain) sides.push_back(0);
        else if(record.endpoint[1].domain==localDomain) sides.push_back(1);
    }
    const Int ne=std::max<Int>(1,static_cast<Int>(sides.size()));
    setup_multi_rank_3d(rankData,masterData,0,localPart,1,0.0,
                        localNcu,0,0,localNfint,ne);
    rankData.problem[28]=rankData.problem[29]=rankData.problem[30]=0;
    rankData.problem[34]=0;
    rankData.bf.assign(static_cast<std::size_t>(masterData.nfe)*ne,1);
    if(localPart==0) {
        SetupRank3D sideData[2];
        setup_multi_rank_3d(sideData[0],masterData,0,0,1,0.0,1,0,0,1,1);
        setup_multi_rank_3d(sideData[1],masterData,1,0,1,0.0,1,0,0,1,1);
        for(Int node=0;node<masterData.npe;++node) {
            const std::size_t base=static_cast<std::size_t>(node);
            sideData[1].xdg[base+masterData.npe]+=0.5;
            // setup_multi_rank_3d mirrors model 1 and therefore gives it a
            // negative element orientation.  Reflect the aligned tangential
            // coordinates so both endpoint fixtures are right-handed.  The
            // square interface remains the same geometric set.
            std::swap(sideData[1].xdg[base+masterData.npe],
                      sideData[1].xdg[base+2*masterData.npe]);
        }
        for(std::size_t edge=0;edge<sides.size();++edge) {
            const auto& source=sideData[sides[edge]].xdg;
            std::copy(source.begin(),source.begin()+3*masterData.npe,
                rankData.xdg.begin()+static_cast<std::ptrdiff_t>(3*masterData.npe*edge));
            rankData.bf[masterData.nfe*edge]=static_cast<Int>(edge)+2;
        }
    }
    rankData.mesh.bf=rankData.bf.data();
    rankData.mesh.intfaces=nullptr;
    rankData.mesh.szintfaces=0;
}

void parser_shapes(int& failures)
{
    const std::vector<double> chain = {
        1,3,2, 11,0,1,1,1,1,1,0,1,1,1,0,
        11,1,2,1,2,2,1,0,1,1,1,0};
    const std::vector<double> star = {
        1,4,3, 11,0,1,1,1,1,1,0,1,1,1,0,
        11,0,2,1,2,1,1,0,1,1,1,0,
        11,0,3,1,3,1,1,0,1,1,1,0};
    require(exasim::nonmatching::parse_interaction_table<Int>(chain.data(), chain.size()).size() == 2,
            "chain parser", failures);
    require(exasim::nonmatching::parse_interaction_table<Int>(star.data(), star.size()).size() == 3,
            "star parser", failures);
    const auto parallel = as_double(parallel_table());
    require(exasim::nonmatching::parse_interaction_table<Int>(parallel.data(), parallel.size()).size() == 2,
            "parallel-edge parser", failures);
    bool tagRejected=false;
    try { (void)exasim::nonmatching::checked_interaction_tag<Int>(10,1,27120); }
    catch (const std::out_of_range&) { tagRejected=true; }
    require(tagRejected,"checked tag overflow helper",failures);
}

} // namespace

int main(int argc, char** argv)
{
#ifndef HAVE_MPI
    (void)argc; (void)argv;
    std::fprintf(stderr, "non-matching coupling (construction): interfacetype 1 requires MPI\n");
    return 1;
#else
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    EXASIM_COMM_LOCAL = MPI_COMM_WORLD;
    Kokkos::initialize(argc, argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Int split = -1;
    std::string mode = "--graph";
    std::string shape = "parallel";
    std::vector<Int> domainRanks;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--split") == 0 && i + 1 < argc)
            split = static_cast<Int>(std::stoi(argv[++i]));
        else if (std::strcmp(argv[i], "--shape") == 0 && i + 1 < argc)
            shape = argv[++i];
        else if (std::strcmp(argv[i], "--domain-ranks") == 0 && i + 1 < argc) {
            std::string text=argv[++i]; std::size_t begin=0;
            while(begin<=text.size()) {
                const std::size_t comma=text.find(',',begin);
                domainRanks.push_back(static_cast<Int>(std::stoi(
                    text.substr(begin,comma==std::string::npos?comma:comma-begin))));
                if(comma==std::string::npos) break;
                begin=comma+1;
            }
        }
        else mode = argv[i];
    }
    int failures = 0;
    MPI_Comm localCommunicator=MPI_COMM_NULL;
    std::vector<Int> table = shape=="chain" ? chain_table() :
            (shape=="star" ? star_table() :
             (shape=="isolated" ? isolated_table() :
              (shape=="single" ? single_table() :
               (shape=="unequal" ? unequal_width_table() : parallel_table()))));
    if(mode=="--reject-width") table=inconsistent_width_table();
    const Int domainCount=table[1];
    if(domainRanks.empty() && domainCount==2 && split>0)
        domainRanks={split,static_cast<Int>(size)-split};
    Int rankSum=0; for(Int count:domainRanks) rankSum+=count;
    if (domainRanks.size()!=static_cast<std::size_t>(domainCount) || rankSum!=size ||
            std::any_of(domainRanks.begin(),domainRanks.end(),[](Int count){return count<=0;})) {
        if (rank == 0) std::fprintf(stderr,
            "Phase 4 requires positive --domain-ranks counts summing to np (or --split for two domains)\n");
        failures = 1;
    }
    else {
        parser_shapes(failures);
        std::vector<Int> offsets(static_cast<std::size_t>(domainCount)+1,0);
        for(Int d=0;d<domainCount;++d) offsets[d+1]=offsets[d]+domainRanks[d];
        Int domain=0; while(domain+1<domainCount && rank>=offsets[domain+1]) ++domain;
        const Int localPart=rank-offsets[domain];
        MPI_Comm_split(MPI_COMM_WORLD,static_cast<int>(domain),rank,&localCommunicator);
        EXASIM_COMM_LOCAL=localCommunicator;
        const auto tableDouble=as_double(table);
        const std::vector<Record> expectedRecords=
            exasim::nonmatching::parse_interaction_table<Int>(tableDouble.data(),tableDouble.size());
        Int localNcu=1,localNfint=1;
        for(const auto& record:expectedRecords) {
            Int side=-1;
            if(record.endpoint[0].domain==domain) side=0;
            else if(record.endpoint[1].domain==domain) side=1;
            if(side<0) continue;
            localNfint=static_cast<Int>(record.endpoint[1-side].fluxmap.size());
            for(const Int component:record.endpoint[side].fluxmap)
                localNcu=std::max(localNcu,component+1);
        }
        SetupMaster3D masterData = setup_master(1, 1);
        SetupRank3D rankData;
        setup_graph_rank(rankData,masterData,domain,localPart,expectedRecords,
                         localNcu,localNfint);
        Int localDegree=0;
        for(const auto& record:expectedRecords)
            localDegree+=record.endpoint[0].domain==domain||record.endpoint[1].domain==domain?1:0;
        GraphInputs graph;
        attach_graph_inputs(rankData, graph, table, localDegree);
        if(mode=="--reject-overlap" && shape=="parallel") {
            graph.table[19]=1; graph.table[23]=1;
        }
        if(mode=="--reject-selector" && shape=="parallel") graph.table[19]=3;
        if(mode=="--reject-map" && shape=="parallel") {
            graph.table[22]=1; graph.table[26]=1;
        }
        if(mode=="--reject-boundary" && !graph.boundaries.empty() && rank==0)
            graph.boundaries.back()=0;

        commonstruct common{};
        common.mpiRank = rank;
        common.components.nc = localNcu; common.components.ncu = localNcu;
        common.components.ncx = 3;
        common.grid.nd = 3; common.grid.npe = masterData.npe;
        common.grid.npf = masterData.npf; common.grid.ngf = masterData.ngf;
        const Int ne=std::max<Int>(1,localDegree);
        common.meshsizes.ne = ne; common.meshsizes.ne0 = ne; common.meshsizes.ne1 = ne;
        common.meshsizes.nfe = masterData.nfe;
        common.spatialScheme = 1;
        common.timeparams.wave = 0;
        common.timestate.dtfactor = 1;
        common.timestate.time = 0;
        if(mode=="--reject-controls" && rank==0) graph.solver[0]=2.0e-8;
        if (mode == "--reject-table")
            graph.table[8] = rank == 0 ? 2 : graph.table[8];
        const std::vector<Record> records = exasim::nonmatching::validate_interaction_graph(
                rankData.app, rankData.mesh, common, domain, offsets);

        std::vector<Interaction> interactions;
        phase4_mpi_counts::worldAllreduces=0;
        phase4_mpi_counts::worldSplits=0;
        exasim::nonmatching::setup_nonmatching_interactions(
                records,domain,rankData.app,rankData.master,rankData.mesh,
                rankData.sol,common,static_cast<Int>(size),static_cast<Int>(rank),
                offsets[domain],0,interactions);
        const int setupWorldAllreduces=phase4_mpi_counts::worldAllreduces;
        const int setupWorldSplits=phase4_mpi_counts::worldSplits;
        require(interactions.size() == static_cast<std::size_t>(localDegree),
                "one local object per incident edge", failures);
        for (std::size_t k = 0; k < interactions.size(); ++k) {
            const auto& interaction = interactions[k];
            const auto& expectedRecord=records[static_cast<std::size_t>(interaction.ordinal())];
            const auto& expected=expectedRecord.endpoint[static_cast<std::size_t>(interaction.side())];
            const auto& expectedPeer=expectedRecord.endpoint[
                    static_cast<std::size_t>(1-interaction.side())];
            require(k==0 || interactions[k-1].ordinal()<interaction.ordinal(),
                    "strict local ordinal order", failures);
            require(interaction.side()>=0 && interaction.side()<=1 &&
                    interaction.peerDomain()!=domain,"side/peer", failures);
            require(interaction.interfaceSelector()==expected.interfaceSelector &&
                    interaction.boundary()==graph.boundaries[
                            static_cast<std::size_t>(expected.interfaceSelector-1)] &&
                    interaction.condition()==expected.condition,
                    "selector/boundary/condition", failures);
            require(interaction.fluxmap()==expected.fluxmap,"flux map",failures);
            require(interaction.receivedWidth()==static_cast<Int>(expected.fluxmap.size()) &&
                    interaction.donatedWidth()==static_cast<Int>(expectedPeer.fluxmap.size()),
                    "directional widths",failures);
            require(interaction.runtimeData()->tag[0] == 27100 + 2*interaction.ordinal() &&
                    interaction.runtimeData()->tag[1] == 27101 + 2*interaction.ordinal(),
                    "per-edge tags", failures);
        }
        if(mode=="--collectives") {
            const int expected=static_cast<int>(records.size());
            require(setupWorldAllreduces==expected,
                    "one world handshake per global interaction",failures);
            require(setupWorldSplits==expected,
                    "one world communicator split per global interaction",failures);
            require(static_cast<Int>(interactions.size())==localDegree,
                    "non-endpoint rank entered pairwise rebuild",failures);
            int allreduceMin=0,allreduceMax=0,splitMin=0,splitMax=0;
            PMPI_Allreduce(&setupWorldAllreduces,&allreduceMin,1,MPI_INT,MPI_MIN,
                           MPI_COMM_WORLD);
            PMPI_Allreduce(&setupWorldAllreduces,&allreduceMax,1,MPI_INT,MPI_MAX,
                           MPI_COMM_WORLD);
            PMPI_Allreduce(&setupWorldSplits,&splitMin,1,MPI_INT,MPI_MIN,
                           MPI_COMM_WORLD);
            PMPI_Allreduce(&setupWorldSplits,&splitMax,1,MPI_INT,MPI_MAX,
                           MPI_COMM_WORLD);
            require(allreduceMin==expected && allreduceMax==expected &&
                    splitMin==expected && splitMax==expected,
                    "collective counts differ between world ranks",failures);
            if(rank==0) std::printf(
                    "PHASE4_COLLECTIVES edges=%d handshakes=[%d,%d] splits=[%d,%d]\n",
                    expected,allreduceMin,allreduceMax,splitMin,splitMax);
        }
        const bool graphActive=exasim::nonmatching::validate_interactions(
                    common, interactions, mode=="--reject-backend"?2:0,
                    "phase4 fixture", true);
        require(graphActive==!interactions.empty(),"graph runtime selection",failures);
        if(mode=="--lifecycle") {
            MPI_Request* grownRequests=common.requests;
            std::vector<Interaction> empty;
            exasim::nonmatching::ensure_nonmatching_request_capacity(common,empty);
            require(common.requests==nullptr || common.requests!=grownRequests,
                    "request storage did not shrink transactionally",failures);
            exasim::nonmatching::ensure_nonmatching_request_capacity(common,interactions);
            if(!interactions.empty() && interactions.front().requestCount()>0)
                require(common.requests!=nullptr,"request storage did not regrow",failures);
        }
        if(mode=="--equivalence" && shape=="single") {
            commonstruct legacy{};
            legacy.components=common.components; legacy.grid=common.grid;
            legacy.meshsizes=common.meshsizes; legacy.spatialScheme=common.spatialScheme;
            legacy.timeparams=common.timeparams; legacy.timestate=common.timestate;
            legacy.couplingparams.coupledinterface=1;
            legacy.couplingparams.coupledcondition=1;
            legacy.couplingparams.coupledboundarycondition=2;
            legacy.couplingparams.interfacetype=1;
            rankData.problem[28]=1; rankData.problem[29]=1;
            rankData.problem[30]=2; rankData.problem[34]=1;
            Interaction legacyInteraction;
            require(Interaction::setupLegacy(rankData.app,rankData.master,rankData.mesh,
                rankData.sol,legacy,static_cast<Int>(size),static_cast<Int>(rank),
                offsets[domain],0,legacyInteraction),"legacy equivalence setup",failures);
            require(interactions.size()==1 && legacyInteraction.requestCount()==
                interactions.front().requestCount(),"one-edge request equivalence",failures);
            if(interactions.size()==1) for(int direction=0;direction<2;++direction) {
                const auto& graphDirection=interactions.front().setupData()->dir[direction];
                const auto& legacyDirection=legacyInteraction.setupData()->dir[direction];
                require(graphDirection.npts==legacyDirection.npts &&
                        graphDirection.nslots==legacyDirection.nslots &&
                        graphDirection.npairs==legacyDirection.npairs &&
                        graphDirection.nsendnb==legacyDirection.nsendnb &&
                        graphDirection.nrecvnb==legacyDirection.nrecvnb,
                        "one-edge setup metadata equivalence",failures);
                if(graphDirection.npts==legacyDirection.npts && graphDirection.npts>0) {
                    require(std::equal(graphDirection.pe,graphDirection.pe+graphDirection.npts,
                                       legacyDirection.pe) &&
                            std::equal(graphDirection.pa,graphDirection.pa+graphDirection.npts,
                                       legacyDirection.pa) &&
                            std::equal(graphDirection.pg,graphDirection.pg+graphDirection.npts,
                                       legacyDirection.pg),
                            "one-edge point-map equivalence",failures);
                }
            }
            require(interactions.front().runtimeData()->tag[0]==27100 &&
                    interactions.front().runtimeData()->tag[1]==27101,
                    "one-edge tag equivalence",failures);
            if(interactions.size()==1) {
                const auto* graphRuntime=interactions.front().runtimeData();
                const auto* legacyRuntime=legacyInteraction.runtimeData();
                require(graphRuntime->condition==legacyRuntime->condition,
                        "one-edge condition equivalence",failures);
                auto compareBuffer=[&](const auto& graphBuffer,
                                       const auto& legacyBuffer,
                                       const char* label) {
                    const bool sameSize=graphBuffer.size==legacyBuffer.size;
                    bool sameValues=sameSize;
                    if(sameValues && graphBuffer.size>0)
                        sameValues=std::equal(graphBuffer.data,
                            graphBuffer.data+graphBuffer.size,legacyBuffer.data);
                    require(sameSize&&sameValues,label,failures);
                };
                for(int direction=0;direction<2;++direction) {
                    const auto& graphDirection=graphRuntime->dir[direction];
                    const auto& legacyDirection=legacyRuntime->dir[direction];
#define COMPARE_RUNTIME_BUFFER(name) compareBuffer(graphDirection.name, \
                        legacyDirection.name,"one-edge runtime " #name " equivalence")
                    COMPARE_RUNTIME_BUFFER(pe); COMPARE_RUNTIME_BUFFER(pa);
                    COMPARE_RUNTIME_BUFFER(pslot); COMPARE_RUNTIME_BUFFER(pg);
                    COMPARE_RUNTIME_BUFFER(slotptr); COMPARE_RUNTIME_BUFFER(pairslot);
                    COMPARE_RUNTIME_BUFFER(pairelem); COMPARE_RUNTIME_BUFFER(pairptr);
                    COMPARE_RUNTIME_BUFFER(recvface); COMPARE_RUNTIME_BUFFER(ownface);
                    COMPARE_RUNTIME_BUFFER(pairSlotPtr); COMPARE_RUNTIME_BUFFER(ownerTarget);
                    COMPARE_RUNTIME_BUFFER(faceBlockPtr); COMPARE_RUNTIME_BUFFER(faceBlocks);
                    COMPARE_RUNTIME_BUFFER(px); COMPARE_RUNTIME_BUFFER(pnl);
                    COMPARE_RUNTIME_BUFFER(pwJ); COMPARE_RUNTIME_BUFFER(pphi);
                    COMPARE_RUNTIME_BUFFER(pphibar); COMPARE_RUNTIME_BUFFER(ppsi);
                    COMPARE_RUNTIME_BUFFER(pd); COMPARE_RUNTIME_BUFFER(pdphibar);
                    COMPARE_RUNTIME_BUFFER(ptest);
#undef COMPARE_RUNTIME_BUFFER
                    require(graphDirection.sendnb==legacyDirection.sendnb &&
                            graphDirection.sendcount==legacyDirection.sendcount &&
                            graphDirection.recvnb==legacyDirection.recvnb &&
                            graphDirection.recvcount==legacyDirection.recvcount,
                            "one-edge runtime neighbor equivalence",failures);
                }
            }
            CPUFREE(legacy.requests); CPUFREE(legacy.statuses);
            rankData.problem[28]=rankData.problem[29]=rankData.problem[30]=0;
            rankData.problem[34]=0;
        }
        if ((mode == "--operators" || mode == "--conservation") && shape=="parallel") {
            const Int npe = masterData.npe, npf = masterData.npf;
            const Int nfe = masterData.nfe;
            const Int traceSize = npf*nfe*ne;
            rankData.udg.resize(static_cast<std::size_t>(npe)*ne);
            rankData.uh.resize(traceSize);
            for (std::size_t i = 0; i < rankData.udg.size(); ++i)
                rankData.udg[i] = 0.2 + 0.001*static_cast<dstype>(i%31);
            for (std::size_t i = 0; i < rankData.uh.size(); ++i)
                rankData.uh[i] = -0.1 + 0.002*static_cast<dstype>(i%29);
            rankData.sol.udg = rankData.udg.data();
            rankData.sol.uh = rankData.uh.data();
            std::vector<dstype> tau(1,1.0), parameters(1,1.0), uinf(1,0.0);
            rankData.app.tau=tau.data(); rankData.app.physicsparam=parameters.data();
            rankData.app.uinf=uinf.data();
            std::vector<dstype> C(static_cast<std::size_t>(npe)*npe*ne*3,0.0);
            std::vector<dstype> E(static_cast<std::size_t>(npe)*npf*nfe*ne*3,0.0);
            std::vector<dstype> F(static_cast<std::size_t>(npe)*npf*nfe*ne,0.0);
            std::vector<dstype> Ru(static_cast<std::size_t>(npe)*ne,0.0);
            for (std::size_t i=0;i<C.size();++i) C[i]=0.01*(1+i%7);
            for (std::size_t i=0;i<E.size();++i) E[i]=0.001*(1+i%11);
            for (std::size_t i=0;i<F.size();++i) F[i]=0.0001*(1+i%13);
            for (std::size_t i=0;i<Ru.size();++i) Ru[i]=0.0002*(1+i%17);
            resstruct residual{}; residual.C=C.data(); residual.E=E.data();
            residual.F=F.data(); residual.Ru=Ru.data();
            tempstruct temporary{};

            auto exchangeAndAssemble = [&](const std::vector<std::size_t>& which) {
                std::vector<dstype> assembled(traceSize,0.0);
                Int requests=0;
                for (std::size_t k : which) interactions[k].beginExchange(common,requests);
                if (requests>0) MPI_Waitall(requests,common.requests,common.statuses);
                for (std::size_t k : which)
                    interactions[k].assembleOwner(assembled.data(),rankData.mesh,
                                                   rankData.app,common);
                return assembled;
            };
            auto checkSum = [&](const char* label) {
                const auto edge0=exchangeAndAssemble({0});
                const auto edge1=exchangeAndAssemble({1});
                const auto both=exchangeAndAssemble({0,1});
                dstype localError=0,localScale=0;
                for (std::size_t i=0;i<both.size();++i) {
                    const dstype expected=edge0[i]+edge1[i];
                    localError=std::max(localError,std::abs(both[i]-expected));
                    localScale=std::max(localScale,std::abs(expected));
                }
                dstype error=0,scale=0;
                MPI_Allreduce(&localError,&error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
                MPI_Allreduce(&localScale,&scale,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
                if(rank==0) std::printf("PHASE4_%s abs=%.17e scale=%.17e\n",label,error,scale);
                require(error==0,"multi-edge result is not the exact sum of one-edge runs",failures);
            };
            for(auto& interaction:interactions)
                interaction.compute<OrientationCoupled3D>(rankData.sol,residual,rankData.app,
                    rankData.master,rankData.mesh,temporary,common,0,false,0);
            checkSum("RESIDUAL");

            for(auto& interaction:interactions)
                interaction.compute<OrientationCoupled3D>(rankData.sol,residual,rankData.app,
                    rankData.master,rankData.mesh,temporary,common,0,true,0);
            std::vector<dstype> ve(traceSize);
            for(std::size_t i=0;i<ve.size();++i) ve[i]=0.03+0.0007*(i%23);
            for(auto& interaction:interactions)
                interaction.matvecProducts(ve.data(),npf*nfe,ne,0,0);
            checkSum("JVP");
            if(mode=="--conservation") {
                for(auto& interaction:interactions)
                    interaction.compute<ConservativeGraph3D>(rankData.sol,residual,
                        rankData.app,rankData.master,rankData.mesh,temporary,common,0,false,0);
                dstype totalSides[2]={0,0};
                for(std::size_t edge=0;edge<interactions.size();++edge) {
                    const auto assembled=exchangeAndAssemble({edge});
                    dstype localTotal=0;
                    for(dstype value:assembled) localTotal+=value;
                    dstype sideTotals[2]={0,0},mine[2]={0,0};
                    mine[domain]=localTotal;
                    MPI_Allreduce(mine,sideTotals,2,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                    const dstype imbalance=std::abs(sideTotals[0]+sideTotals[1]);
                    const dstype scale=std::abs(sideTotals[0])+std::abs(sideTotals[1]);
                    totalSides[0]+=sideTotals[0]; totalSides[1]+=sideTotals[1];
                    if(rank==0) std::printf(
                        "PHASE4_CONSERVATION edge=%zu side0=%.17e side1=%.17e imbalance=%.17e scale=%.17e\n",
                        edge,sideTotals[0],sideTotals[1],imbalance,scale);
                    require(imbalance<=1024*std::numeric_limits<dstype>::epsilon()*
                            std::max<dstype>(1.0,scale),
                            "per-edge action-reaction balance",failures);
                }
                const dstype totalImbalance=std::abs(totalSides[0]+totalSides[1]);
                const dstype totalScale=std::abs(totalSides[0])+std::abs(totalSides[1]);
                if(rank==0) std::printf(
                    "PHASE4_CONSERVATION total-imbalance=%.17e total-scale=%.17e\n",
                    totalImbalance,totalScale);
                require(totalImbalance<=1024*std::numeric_limits<dstype>::epsilon()*
                        std::max<dstype>(1.0,totalScale),
                        "total action-reaction balance",failures);
            }
        }
        interactions.clear();
        CPUFREE(common.requests); CPUFREE(common.statuses);
        if(mode=="--operators" && shape=="parallel") {
            const int earlier=::failures;
            model_operator_case<NLCoupled3D>("phase4-q-present",1,1,rank,size,0.0,
                                              domainRanks[0],false,false);
            model_operator_case<QFreeCoupled3D>("phase4-q-free",1,1,rank,size,0.0,
                                                 domainRanks[0],false,false);
            model_operator_case<WCoupled3D>("phase4-ncw",1,1,rank,size,0.0,
                                             domainRanks[0],false,false);
            failures+=::failures-earlier;
        }
    }
    int globalFailures = 0;
    MPI_Allreduce(&failures, &globalFailures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0 && globalFailures == 0)
        std::printf("PASS Phase 4 multiple interactions mode=%s shape=%s\n",
                    mode.c_str(),shape.c_str());
    Kokkos::finalize();
    if(localCommunicator!=MPI_COMM_NULL) MPI_Comm_free(&localCommunicator);
    MPI_Finalize();
    return globalFailures == 0 ? 0 : 1;
#endif
}
