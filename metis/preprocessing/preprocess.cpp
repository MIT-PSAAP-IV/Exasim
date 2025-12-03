// cmake -S . -B build
// cmake --build build -j4

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

#define HAVE_MPI

#include <metis.h>
#include <parmetis.h>

#include "structs.hpp"
#include "TextParser.hpp"

#include "tinyexpr.cpp"
#include "helpers.cpp"
#include "readpdeapp.cpp"
#include "readmesh.cpp"
#include "makemesh.cpp"
#include "makemaster.cpp"
#include "domaindecomposition.cpp"
#include "connectivity.cpp"
#include "writebinaryfiles.cpp"
#include "parmetisexasim.cpp"

int main(int argc, char** argv)
{
    // ----------------------------------------------------
    // 0. Initialize MPI
    // ----------------------------------------------------
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 2) {
        if (rank==0) std::cerr << "Usage: ./parseinput <pdeapp.txt>\n";
        return 1;
    }    

    if (std::filesystem::exists(argv[1])) {
        if (rank==0) std::cout << "Generating Exasim's input files for this text file ("<< argv[1] << ") ... \n\n";
    } else {
        error("Error: Input file does not exist.\n");        
    }          
           
    InputParams params = parseInputFile(argv[1], rank);                           
    PDE pde = initializePDE(params, rank);    
     
    ParsedSpec spec = TextParser::parseFile(make_path(pde.datapath, pde.modelfile));        
    spec.exasimpath = pde.exasimpath;    

    if (size == 1) {
      Mesh mesh = initializeMesh(params, pde);        
      Master master = initializeMaster(pde, mesh);                                    
      writeBinaryFiles(pde, mesh, master, spec);
    }
    else {
      Mesh mesh;
      readParMeshFromFile(make_path(pde.datapath, pde.meshfile), mesh, MPI_COMM_WORLD);                       
      mesh.boundaryConditions = params.boundaryConditions;
      mesh.curvedBoundaries = params.curvedBoundaries;
      mesh.periodicBoundaries1 = params.periodicBoundaries1;
      mesh.periodicBoundaries2 = params.periodicBoundaries2;
      mesh.cartGridPart = params.cartGridPart;
      mesh.interfaceConditions = params.interfaceConditions;
          
      assignVectorToCharArray(params.boundaryExprs, &mesh.boundaryExprs);
      assignVectorToCharArray(params.curvedBoundaryExprs, &mesh.curvedBoundaryExprs);
      assignVectorToCharArray(params.periodicExprs1, &mesh.periodicExprs1);
      assignVectorToCharArray(params.periodicExprs2, &mesh.periodicExprs2);
          
      mesh.nbndexpr = params.boundaryExprs.size();
      mesh.nbcm = params.boundaryConditions.size();
      mesh.nprdexpr = params.periodicBoundaries1.size();    
      mesh.nprdcom = (mesh.nprdexpr == 0) ? 0 : params.periodicExprs1.size()/mesh.nprdexpr;
      if (mesh.nbndexpr != mesh.nbcm) 
          error("boundaryconditions and boundaryexpressions are not the same size. Exiting.\n");
                      
      ensure_dir(pde.datainpath);
      ensure_dir(pde.dataoutpath);
      
      for (const auto& vec : spec.vectors) {
          const std::string& name = vec.first;
          int size = vec.second;
          if (name == "uhat") pde.ncu = size;
          if (name == "v") pde.ncv = size;
          if (name == "w") pde.ncw = size;
          if (name == "uq") pde.nc = size;        
      }
      
      for (int i=0; i<spec.functions.size(); i++) {
          if (spec.functions[i].name == "VisScalars") pde.nsca = spec.functions[i].outputsize;
          if (spec.functions[i].name == "VisVectors") pde.nvec = spec.functions[i].outputsize/pde.nd;
          if (spec.functions[i].name == "VisTensors") pde.nten = spec.functions[i].outputsize/(pde.nd*pde.nd);
          if (spec.functions[i].name == "QoIboundary") pde.nsurf = spec.functions[i].outputsize;
          if (spec.functions[i].name == "QoIvolume") pde.nvqoi = spec.functions[i].outputsize;
      }
      
      mesh.dim = mesh.nd;
      pde.nve = mesh.nve; pde.np = mesh.np; pde.ne = mesh.ne; pde.elemtype = mesh.elemtype;                
      pde.nd = mesh.dim; pde.ncx = mesh.dim;
      if (pde.model=="ModelC" || pde.model=="modelC") {
          pde.wave = 0;
          pde.nc = pde.ncu;
      } else if (pde.model=="ModelD" || pde.model=="modelD") {     
          pde.wave = 0;
          pde.nc = (pde.ncu)*(pde.nd+1);
      } else if (pde.model=="ModelW" || pde.model=="modelW") {
          pde.tdep = 1;
          pde.wave = 1;
          pde.nc = (pde.ncu)*(pde.nd+1);
      }
      pde.ncq = pde.nc - pde.ncu;
      pde.nch  = pde.ncu;                    

      Master master = initializeMaster(pde, mesh, rank);    
      
      if (rank==0) {
        writepde(pde, make_path(pde.datainpath, "app.bin"));
        writemaster(master, make_path(pde.datainpath, "master.bin"));    
      }
      MPI_Barrier(MPI_COMM_WORLD);

      std::vector<idx_t> elmdist = buildElmdistFromLocalCount(mesh.ne, MPI_COMM_WORLD);

      std::vector<idx_t>  epart_local;
      partitionMeshParMETIS(epart_local, mesh.t, elmdist, mesh.nve, mesh.nvf, size, MPI_COMM_WORLD);
            
      Mesh mesh_in;
      mesh_in.nd = mesh.nd;
      mesh_in.nve = mesh.nve;
      mesh_in.ne = mesh.ne;
      mesh_in.np = mesh.np;
      mesh_in.t.resize(mesh.nve*mesh.ne);
      mesh_in.p.resize(mesh.nd*mesh.np);
      for (int i = 0; i < mesh.nve*mesh.ne; i++) mesh_in.t[i] = mesh.t[i];
      for (int i = 0; i < mesh.nd*mesh.np; i++) mesh_in.p[i] = mesh.p[i];

      migrateMeshWithParMETIS(mesh_in, epart_local, mesh, MPI_COMM_WORLD);      

      mesh.localfaces.resize(mesh.nvf * mesh.nfe);           
      getelemface(mesh.localfaces.data(), mesh.dim, mesh.elemtype);    

      mesh.t2t.resize(mesh.nfe*mesh.ne);
      mke2e_global(mesh.t2t.data(), mesh.t.data(), mesh.localfaces.data(), 
                   mesh.elemGlobalID.data(), mesh.ne, mesh.nve, mesh.nvf, mesh.nfe, rank);      
      DMD dmd;

      mke2e_fill_first_neighbors(mesh.t2t.data(), mesh.t.data(), mesh.localfaces.data(),
                 mesh.elemGlobalID.data(), mesh.nodeGlobalID.data(), mesh.ne, mesh.nve, 
                mesh.nvf, mesh.nfe, MPI_COMM_WORLD, dmd.nbinfo);

      dmd.numneigh = static_cast<int>(dmd.nbinfo.size() / 6); 

      ElementClassification elemclass;      
      classifyElementsWithE2EAndNbinfo(mesh.t2t.data(), mesh.elemGlobalID.data(), mesh.nfe, 
                                       mesh.ne, dmd.nbinfo.data(), dmd.numneigh, elemclass, rank);
      
      buildElempartFromClassification(elemclass, dmd, rank);
      buildElem2CpuFromClassification(elemclass, dmd, MPI_COMM_WORLD);      
      buildElemRecv(dmd, MPI_COMM_WORLD);
      buildElemsend(mesh, dmd, MPI_COMM_WORLD);
    }
    
    if (rank==0) std::cout << "\n******** Done with generating input files for EXASIM ********\n";
  
    MPI_Finalize();
    return 0;
}

