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

#include "comparestructs.cpp"
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
      pde.nve = mesh.nve; pde.np = mesh.np_global; pde.ne = mesh.ne_global; pde.elemtype = mesh.elemtype;                
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

      std::vector<idx_t>  epart_local;
      std::vector<idx_t> elmdist = buildElmdistFromLocalCount(mesh.ne, MPI_COMM_WORLD);      
      
      if (pde.partitionfile == "")  {
        partitionMeshParMETIS(epart_local, mesh.t, elmdist, mesh.nve, mesh.nvf, size, MPI_COMM_WORLD);
      }
      else {        
        vector<double> tm;
        readarrayfrombinaryfile(make_path(pde.datapath, pde.partitionfile), tm);      
        int n1 = elmdist[rank];
        int n2 = elmdist[rank+1];
        int n = n2 - n1;
        epart_local.resize(n);
        for (int j = 0; j < n; j++)
          epart_local[j] = (idx_t) (tm[n1+j]-1);   
      }

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

      //std::vector<int> t2t_local(mesh.nfe*mesh.ne);      
      mesh.t2t.resize(mesh.nfe*mesh.ne);
      mke2e_global(mesh.t2t.data(), mesh.t.data(), mesh.localfaces.data(), 
                   mesh.elemGlobalID.data(), mesh.ne, mesh.nve, mesh.nvf, mesh.nfe, rank);     

      DMD dmd;

      mke2e_fill_first_neighbors(mesh.t2t.data(), mesh.t.data(), mesh.localfaces.data(),
                 mesh.elemGlobalID.data(), mesh.nodeGlobalID.data(), mesh.ne, mesh.nve, 
                mesh.nvf, mesh.nfe, MPI_COMM_WORLD, dmd.nbinfo);

      int nboufaces = setboundaryfaces(mesh.t2t.data(), mesh.t.data(), mesh.localfaces.data(), mesh.p.data(),    
          mesh.boundaryExprs, mesh.dim, mesh.nve, mesh.nvf, mesh.nfe, mesh.ne, mesh.nbndexpr);

      // setperiodicfaces(mesh.t2t.data(), mesh.t.data(), mesh.localfaces.data(), mesh.p.data(),    
      //     mesh.elemGlobalID.data(), mesh.periodicBoundaries1.data(), mesh.periodicBoundaries2.data(),
      //     mesh.periodicExprs1, mesh.periodicExprs2, mesh.dim, mesh.nve, mesh.nvf, mesh.nfe, mesh.ne, 
      //     mesh.nprdexpr, mesh.nprdcom, nboufaces, MPI_COMM_WORLD, dmd.nbinfo);
                  
      dmd.numneigh = static_cast<int>(dmd.nbinfo.size() / 6); 
      ElementClassification elemclass;      
      classifyElementsWithE2EAndNbinfo(mesh.t2t.data(), mesh.elemGlobalID.data(), mesh.nfe, 
                                       mesh.ne, dmd.nbinfo.data(), dmd.numneigh, elemclass, rank);

      buildElempartFromClassification(elemclass, dmd, rank);
      buildElem2CpuFromClassification(elemclass, dmd, MPI_COMM_WORLD);      
      buildElemRecv(dmd, MPI_COMM_WORLD);
      buildElemsend(mesh, dmd, MPI_COMM_WORLD);
      
      dmd.bf.resize(mesh.nfe * mesh.ne);
      for (int i = 0; i < mesh.ne; i++) {
        int k = dmd.elempart_local[i];
        for (int j = 0; j < mesh.nfe; j++)
          dmd.bf[j + mesh.nfe*i] = (mesh.t2t[j + mesh.nfe*k] < 0) ? -mesh.t2t[j + mesh.nfe*k] : 0;
      }

      
      
      // if (rank==0) {
      //   //print2iarray(t2t_local.data(), mesh.nfe, mesh.ne);
      //   //print2iarray(mesh.t.data(), mesh.nve, mesh.ne);
      //   //print2darray(mesh.p.data(), mesh.nd, mesh.np);
      //   print2iarray(dmd.bf.data(), mesh.nfe, mesh.ne);
      // }
      
      InputParams params1 = parseInputFile(argv[1], 0);                           
      PDE pde1 = initializePDE(params1, 0);         
      ParsedSpec spec1 = TextParser::parseFile(make_path(pde1.datapath, pde1.modelfile));        
      spec1.exasimpath = pde1.exasimpath;    
      Mesh mesh1 = initializeMesh(params1, pde1);        
      Master master1 = initializeMaster(pde1, mesh1);                                    
      vector<DMD> dmd1 = buildMeshDMD(pde1, mesh1, master1, spec1, rank); 
            
      comparePDE(pde1, pde, true, 1e-10);
      compareMaster(master1, master, true, 1e-10);        

      std::vector<int> intelem;
      intelem.reserve(mesh1.ne / size);
      for (int e = 0; e < mesh1.ne; ++e) {
          if (mesh1.elem2cpu[e] == rank) intelem.push_back(e);
      }
      vector<int> t2ti(mesh.nfe*mesh.ne); 
      select_columns(t2ti.data(), mesh1.t2t.data(), intelem.data(), mesh.nfe, mesh.ne); 
      for (int i = 0; i < mesh.nfe*mesh.ne; i++) t2ti[i] = (t2ti[i] < 0) ? mesh.t2t[i] : t2ti[i];
      //print2iarray(t2ti.data(), mesh.nfe, mesh.ne);
      if (compareVecInt(mesh.t2t, t2ti, "t2t", true)) cout<<"Rank: "<<rank<<", t2t arrays are identical"<<endl;

      int n0 = dmd1[rank].elempartpts[0];
      int n1 = dmd1[rank].elempartpts[1];
      int n2 = dmd1[rank].elempartpts[2];
      vector<int> part0(n0, 0);
      vector<int> part1(n1, 0);
      vector<int> part2(n2, 0);
      for (int i=0; i<n0; i++) part0[i] = dmd1[rank].elempart[i];
      for (int i=0; i<n1; i++) part1[i] = dmd1[rank].elempart[n0+i];
      for (int i=0; i<n2; i++) part2[i] = dmd1[rank].elempart[n0+n1+i];

      int m0 = dmd.elempartpts[0];
      int m1 = dmd.elempartpts[1];
      int m2 = dmd.elempartpts[2];
      vector<int> qart0(m0, 0);
      vector<int> qart1(m1, 0);
      vector<int> qart2(m2, 0);
      for (int i=0; i<m0; i++) qart0[i] = dmd.elempart[i];
      for (int i=0; i<m1; i++) qart1[i] = dmd.elempart[m0+i];
      for (int i=0; i<m2; i++) qart2[i] = dmd.elempart[m0+m1+i];      
      
      if (compareVecInt(qart0, part0, "interiorGlobal", true)) 
        cout<<"Rank: "<<rank<<", interiorGlobal arrays are identical"<<endl;

      if (compareVecInt(qart1, part1, "interfaceGlobal", true)) 
        cout<<"Rank: "<<rank<<", interfaceGlobal arrays are identical"<<endl;

      if (compareVecInt(qart2, part2, "exteriorGlobal", true)) 
        cout<<"Rank: "<<rank<<", exteriorGlobal arrays are identical"<<endl;
      
      for (int i=0; i<n0; i++) part0[i] = dmd1[rank].elem2cpu[i];
      for (int i=0; i<n1; i++) part1[i] = dmd1[rank].elem2cpu[n0+i];
      for (int i=0; i<n2; i++) part2[i] = dmd1[rank].elem2cpu[n0+n1+i];
      for (int i=0; i<m0; i++) qart0[i] = dmd.elem2cpu[i];
      for (int i=0; i<m1; i++) qart1[i] = dmd.elem2cpu[m0+i];
      for (int i=0; i<m2; i++) qart2[i] = dmd.elem2cpu[m0+m1+i];      
      
      if (compareVecIntSorted(qart0, part0, "interior elem2cpu", true)) 
        cout<<"Rank: "<<rank<<", interior elem2cpu arrays are identical"<<endl;

      if (compareVecIntSorted(qart1, part1, "interface elem2cpu", true)) 
        cout<<"Rank: "<<rank<<", interface elem2cpu arrays are identical"<<endl;

      if (compareVecIntSorted(qart2, part2, "exterior elem2cpu", true)) 
        cout<<"Rank: "<<rank<<", exterior elem2cpu arrays are identical"<<endl;
      
      if (compareArray3(dmd.elemrecv, dmd1[rank].elemrecv, "elemrecv", true))
        cout<<"Rank: "<<rank<<", elemrecv arrays are identical"<<endl;

      if (compareArray3(dmd.elemsend, dmd1[rank].elemsend, "elemsend", true))
        cout<<"Rank: "<<rank<<", elemsend arrays are identical"<<endl;

      if (compareVecInt(dmd.nbsd, dmd1[rank].nbsd, "nbsd", true)) 
        cout<<"Rank: "<<rank<<", nbsd arrays are identical"<<endl;

      vector<int> tm(mesh.nfe*mesh.ne, 0); 
      for (int i = 0; i < mesh.nfe*mesh.ne; i++) tm[i] = dmd1[rank].bf[i];
      if (compareVecInt(dmd.bf, tm, "bf", true)) 
        cout<<"Rank: "<<rank<<", bf arrays are identical"<<endl;

      if (rank==0) {        
        //print2iarray(mesh.t2t.data(), mesh.nfe, mesh.ne);        
        //print2iarray(t2ti.data(), mesh.nfe, mesh.ne);        
        print2iarray(dmd.bf.data(), mesh.nfe, mesh.ne);        
        print2iarray(tm.data(), mesh.nfe, mesh.ne);
        // vector<int> fi(mesh.nfe*mesh.ne);      
        // select_columns(fi.data(), mesh1.f.data(), dmd1[rank].elempart.data(), mesh.nfe, mesh.ne);       
        // print2iarray(fi.data(), mesh.nfe, mesh.ne);

        //printElemRecv(dmd.elemrecv);
        //printElemRecv(dmd1[rank].elemrecv);
        //printElemRecv(dmd.elemsend, "elemsend");
        //printElemRecv(dmd1[rank].elemsend, "elemsend");
      }

      // if (rank==0) {
      //   print2iarray(part1.data(), 1, part1.size());  
      //   print2iarray(elemclass.interfaceGlobal.data(), 1, elemclass.interfaceGlobal.size());  
      //   print2iarray(part2.data(), 1, part2.size());  
      //   print2iarray(elemclass.neighborElemGlobal.data(), 1, elemclass.neighborElemGlobal.size());  
      // }

      //print2iarray(dmd1[rank].elempartpts.data(), 1, dmd1[rank].elempartpts.size());  

      // print2iarray(mesh.t2t.data(), mesh.nfe, mesh.ne);
      // vector<int> t2ti(mesh.nfe*mesh.ne); 
      // select_columns(t2ti.data(), mesh1.t2t.data(), dmd1[rank].elempart.data(), mesh.nfe, mesh.ne); 
      // print2iarray(t2ti.data(), mesh.nfe, mesh.ne);

      // compareDMD(dmd1[rank], dmd, true);
      // print2iarray(dmd1[rank].elempart.data(), 1, dmd1[rank].elempart.size());
      // print2iarray(dmd.elempart.data(), 1, dmd.elempart.size());
      //print2iarray(mesh1.elem2cpu.data(), 1, mesh1.elem2cpu.size());
    
       //printvector(epart_local, "epart_local", MPI_COMM_WORLD);

    }
    
    if (rank==0) std::cout << "\n******** Done with generating input files for EXASIM ********\n";
  
    MPI_Finalize();
    return 0;
}

