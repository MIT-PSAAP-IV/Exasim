// Focused one-time CPU preprocessing test. No coupled solver is enabled.
#include <exasim/operators.hpp>
#include <exasim/export.hpp>
#include "backend/Preprocessing/hdgldgregions.hpp"
#include <filesystem>
#include <iostream>

using namespace exasim::hdgldg;
static int failures = 0;
static void require(bool ok, const char* message)
{ if (!ok) { ++failures; std::cerr << message << '\n'; } }
static std::vector<std::vector<double>> readPayload(const std::string& path)
{
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    const auto bytes=in.tellg(); in.seekg(0);
    std::vector<double> data(static_cast<std::size_t>(bytes)/sizeof(double));
    in.read(reinterpret_cast<char*>(data.data()),bytes);
    if(!in || data.empty()) throw std::runtime_error("cannot read native payload");
    const int count=static_cast<int>(data[0]); std::size_t offset=count+1;
    std::vector<std::vector<double>> arrays;
    for(int i=0;i<count;++i){
        const auto size=static_cast<std::size_t>(data[i+1]);
        if(offset+size>data.size()) throw std::runtime_error("truncated native payload");
        arrays.emplace_back(data.begin()+offset,data.begin()+offset+size);offset+=size;
    }
    require(offset==data.size(),"native payload sizes do not match length");
    return arrays;
}

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    Kokkos::initialize(argc, argv);
    try {
        PDE pde{}; pde.porder=2; pde.pgauss=4; pde.hybrid=1;
        pde.ncu=3; pde.ncq=6; pde.nc=9; pde.ncv=2; pde.ncw=1;
        pde.ncx=2; pde.nd=2; pde.mpiprocs=1;
        Mesh mesh; mesh.dim=mesh.nd=2; mesh.nve=4; mesh.nvf=2;
        mesh.nfe=4; mesh.ne=16; mesh.np=25; mesh.elemtype=1;
        mesh.nbcm=4; mesh.nbndexpr=4;
        mesh.boundaryConditions={1,2,3,4}; mesh.curvedBoundaries={0,0,0,0};
        mesh.boundaryPreds={[](const double*x){return x[1]==0;},
            [](const double*x){return x[0]==1;},[](const double*x){return x[1]==1;},
            [](const double*x){return x[0]==0;}};
        for(int y=0;y<5;++y)for(int x=0;x<5;++x){mesh.p.push_back(x/4.);mesh.p.push_back(y/4.);}
        for(int y=0;y<4;++y)for(int x=0;x<4;++x){
            int v=x+5*y; mesh.t.insert(mesh.t.end(),{v,v+1,v+6,v+5});
            mesh.elemGlobalID.push_back(107+13*(x+4*y));
        }
        Master master=initializeMaster(pde,mesh);
        const Mesh original=mesh;
        buildMesh(mesh,pde,master); // Exactly once: regions reuse prepared nodes/topology.
        for(auto field : {&mesh.udg,&mesh.vdg,&mesh.wdg}){
            const int components=field==&mesh.udg?9:(field==&mesh.vdg?2:1);
            field->resize(master.npe*components*mesh.ne);
            for(int e=0;e<mesh.ne;++e)for(int j=0;j<master.npe*components;++j)
                (*field)[j+master.npe*components*e]=10000.*mesh.elemGlobalID[e]+j;
        }
        const auto topology=topologyFromPreparedMesh(mesh);
        require(selectBoundaryLayers(topology,{1},1)==std::vector<int>({1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0}),"native boundary layers");
        mesh.boundaryConditions={7,3,7,4};
        const auto repeatedBC=topologyFromPreparedMesh(mesh);
        require(selectBoundaryLayers(repeatedBC,{7},1)==std::vector<int>({1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1}),"repeated model BC seed union");
        mesh.boundaryConditions={1,2,3,4};
        std::vector<int> labels(16);
        for(int e=0;e<16;++e) labels[e]=(e%4<2 || e/4<2);
        const auto plan=buildPlan(topology,labels,{1,3},{0,2},4,metisPartitionGraph);
        for(int scheme=0;scheme<2;++scheme){
            auto region=extractPreparedRegion(mesh,master,pde,plan,scheme);
            const auto& selection=plan.regions[scheme];
            for(int rank=0;rank<2;++rank)for(int e:region.partitions[rank].elempart){
                require(labels[selection.elements[e]]==scheme,"foreign scheme ghost");
                for(int j=0;j<master.npe*2;++j)
                    require(region.mesh.xdg[j+master.npe*2*e]==mesh.xdg[j+master.npe*2*selection.elements[e]],"prepared high-order nodes changed");
                for(int j=0;j<master.npe*9;++j)
                    require(region.mesh.udg[j+master.npe*9*e]==10000.*selection.originalIDs[e]+j,"aligned UDG lost");
                for(int j=0;j<master.npe*2;++j)
                    require(region.mesh.vdg[j+master.npe*2*e]==10000.*selection.originalIDs[e]+j,"aligned VDG lost");
                for(int j=0;j<master.npe;++j)
                    require(region.mesh.wdg[j+master.npe*e]==10000.*selection.originalIDs[e]+j,"aligned WDG lost");
            }
            for(int rank=0;rank<2;++rank)for(const auto& received:region.partitions[rank].elemrecv){
                const int sender=received[0];
                const auto& source=region.partitions.at(sender);
                require(region.partitions[rank].elempart.at(received[1])==received[2],"native receive index mismatch");
                bool found=false;
                for(const auto& sent:source.elemsend)
                    if(sent[0]==rank && source.elempart.at(sent[1])==received[2])found=true;
                require(found,"native send receive pair absent");
            }
            writePreparedRegion(region,master,plan,std::string(argv[1])+"/"+(scheme?"H":"L"));
            const std::string dir=std::string(argv[1])+"/"+(scheme?"H":"L");
            const auto app=readPayload(dir+"/app.bin");
            require(app.at(2).at(0)==scheme,"app.bin native scheme differs from regional scheme");
            for(int rank=0;rank<2;++rank){
                const auto solution=readPayload(dir+"/sol"+std::to_string(rank+1)+".bin");
                const auto meshPayload=readPayload(dir+"/mesh"+std::to_string(rank+1)+".bin");
                const auto& order=region.partitions[rank].elempart;
                require(meshPayload.at(9).size()==order.size(),"native mesh order extent");
                for(std::size_t e=0;e<order.size();++e){
                    require(meshPayload[9][e]==order[e],"native mesh regional element IDs");
                    require(solution.at(2).at(master.npe*9*e)==10000.*selection.originalIDs[order[e]],"serialized UDG order wrong");
                    require(solution.at(3).at(master.npe*2*e)==10000.*selection.originalIDs[order[e]],"serialized VDG order wrong");
                    require(solution.at(4).at(master.npe*e)==10000.*selection.originalIDs[order[e]],"serialized WDG order wrong");
                }
            }
            require(std::filesystem::file_size(std::string(argv[1])+"/"+(scheme?"H":"L")+"/mesh1.bin")>0,"native mesh absent");
        }
        mesh.uhat={1};
        bool rejected=false;
        try{extractPreparedRegion(mesh,master,pde,plan,0);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"supplied trace silently dropped");
        pde.discretization="hdg"; // Exercise extraction from either pure input convention.
        Mesh periodic=original;
        char coordinate[]="y"; char* expressions[]={coordinate};
        periodic.nprdexpr=1; periodic.nprdcom=1;
        periodic.periodicBoundaries1={4}; periodic.periodicBoundaries2={2};
        periodic.periodicExprs1=expressions; periodic.periodicExprs2=expressions;
        buildMesh(periodic,pde,master);
        const auto periodicTopology=topologyFromPreparedMesh(periodic);
        const auto layers=selectBoundaryLayers(periodicTopology,{1},1);
        require(layers==std::vector<int>({1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0}),"periodic faces seeded as walls");
        const auto periodicPlan=buildPlan(periodicTopology,labels,{1},{0},2);
        require(periodicPlan.interfaces.size()>plan.interfaces.size(),"periodic HL seams lost");
        for(int scheme=0;scheme<2;++scheme){
            auto region=extractPreparedRegion(periodic,master,pde,periodicPlan,scheme);
            for(const auto& vertices:region.interfaceVertexIDs){
                std::vector<int> h(vertices.begin(),vertices.begin()+periodic.nvf);
                std::vector<int> l(vertices.begin()+periodic.nvf,vertices.end());
                std::sort(h.begin(),h.end());std::sort(l.begin(),l.end());
                require(h==l,"retained face orientation vertices disagree");
            }
            writePreparedRegion(region,master,periodicPlan,std::string(argv[1])+"/periodic/"+(scheme?"H":"L"));
            const auto app=readPayload(std::string(argv[1])+"/periodic/"+(scheme?"H":"L")+"/app.bin");
            require(app.at(2).at(0)==scheme,"serial region app.bin scheme wrong");
        }
        Mesh cube; cube.nd=cube.dim=3;cube.nve=8;cube.nvf=4;cube.nfe=6;
        cube.ne=8;cube.np=27;cube.elemtype=1;cube.nbcm=cube.nbndexpr=6;
        cube.boundaryConditions={1,2,3,4,5,6};cube.curvedBoundaries.assign(6,0);
        cube.boundaryPreds={[](const double*x){return x[0]==0;},[](const double*x){return x[0]==1;},
          [](const double*x){return x[1]==0;},[](const double*x){return x[1]==1;},
          [](const double*x){return x[2]==0;},[](const double*x){return x[2]==1;}};
        for(int z=0;z<3;++z)for(int y=0;y<3;++y)for(int x=0;x<3;++x)
            cube.p.insert(cube.p.end(),{x/2.,y/2.,z/2.});
        for(int z=0;z<2;++z)for(int y=0;y<2;++y)for(int x=0;x<2;++x){
            int v=x+3*y+9*z;
            cube.t.insert(cube.t.end(),{v,v+1,v+4,v+3,v+9,v+10,v+13,v+12});
        }
        PDE pde3=pde;pde3.nd=pde3.ncx=3;pde3.ncq=9;pde3.nc=12;
        Master master3=initializeMaster(pde3,cube);buildMesh(cube,pde3,master3);
        auto topology3=topologyFromPreparedMesh(cube);
        auto plan3=buildPlan(topology3,{1,0,1,0,1,0,1,0},{1},{0,2},3,metisPartitionGraph);
        require(plan3.interfaces.size()==4,"3D native interface face count");
        for(int scheme=0;scheme<2;++scheme){
            auto region=extractPreparedRegion(cube,master3,pde3,plan3,scheme);
            writePreparedRegion(region,master3,plan3,std::string(argv[1])+"/cube/"+(scheme?"H":"L"));
            auto app=readPayload(std::string(argv[1])+"/cube/"+(scheme?"H":"L")+"/app.bin");
            require(app.at(2).at(0)==scheme,"3D native scheme mismatch");
            for(const auto& partition:region.partitions)for(int e:partition.elempart)
                require(plan3.labels[plan3.regions[scheme].elements[e]]==scheme,"3D foreign ghost");
        }
    } catch(const std::exception& error) {++failures;std::cerr<<error.what()<<'\n';}
    Kokkos::finalize();
    std::cout<<"native regional preprocessing failures="<<failures<<'\n';
    return failures?1:0;
}
