/*************************************************************************

// Build the built-in model shared library from Exasim/backend/Model/BuiltIn 
// 
//   make serial
// 
// Build the static libraries from Exasim/build
// 
//   cmake -DEXASIM_LIB=ON -DEXASIM_MPI=ON -DEXASIM_NOMPI=ON -DWITH_PARMETIS=ON -DWITH_TEXT2CODE=ON -DWITH_BUILTINMODEL=ON ../install
//   cmake --build .

Build the static libraries and the built-in model shared library

  cmake -S Exasim -B Exasim-build \
    -DEXASIM_FRONTENDS=ON \
    -DCMAKE_INSTALL_PREFIX=/path/to/prefix
  cmake --build Exasim-build -j8
  cmake --install Exasim-build

Build this exasimapp.cpp from Exasim/apps/builtinlibrary 

  cmake -B build -DExasim_DIR=/path/to/prefix
  cmake --build build

Run an APP with an input file that uses a built-in model ID:

  mpirun -np 4 build/exasimapp /path/to/pdeapp.txt

Poisson APPs:

  mpirun -np 4 build/exasimapp ../poisson/poisson2d/pdeapp.txt  
  mpirun -np 4 build/exasimapp ../poisson/lshape/pdeapp.txt  
  mpirun -np 4 build/exasimapp ../poisson/poisson3d/pdeapp.txt
  mpirun -np 4 build/exasimapp ../poisson/orion/pdeapp.txt
  mpirun -np 4 build/exasimapp ../poisson/isoq3d/pdeapp.txt
  mpirun -np 4 build/exasimapp ../poisson/cone/pdeapp.txt

Navier-Stokes APPs: 

  mpirun -np 4 build/exasimapp ../navierstokes/nsmach8/pdeapp.txt  
  mpirun -np 4 build/exasimapp ../navierstokes/naca0012steady/pdeapp.txt
  mpirun -np 4 build/exasimapp ../navierstokes/naca0012unsteady/pdeapp.txt
  mpirun -np 4 build/exasimapp ../navierstokes/isoq/pdeapp.txt
  mpirun -np 4 build/exasimapp ../navierstokes/sharpb2/pdeapp.txt
  mpirun -np 4 build/exasimapp ../navierstokes/orion/pdeapp.txt
  mpirun -np 4 build/exasimapp ../navierstokes/reactingsharpb2/pdeapp.txt  

FSP-1 APPs: 

  basic_parallel_coupling -> builtinmodelID = 7
  poisson2d (pdemodel1)   -> builtinmodelID = 8
  poisson2d (pdemodel2)   -> builtinmodelID = 9
  isoq3d_poisson          -> builtinmodelID = 10  
  isoq2d and isoq2d_cht   -> builtinmodelID = 11

**************************************************************************/

#include "ExasimSolverSetup.hpp"
#include "builtinlibprovider.hpp"

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    MPI_Comm comm = MPI_COMM_NULL;
#endif

    ExasimSolver solver;

    if (argc > 1 && std::string(argv[1]) == "postprocess") {
        std::vector<char*> shifted;
        shifted.push_back(argv[0]);
        for (int i = 2; i < argc; i++)
            shifted.push_back(argv[i]);
        return RunExasimPostprocess(solver, static_cast<int>(shifted.size()), shifted.data(), comm);
    }

    if (argc > 1 && std::string(argv[1]) == "solve") {
        std::vector<char*> shifted;
        shifted.push_back(argv[0]);
        for (int i = 2; i < argc; i++)
            shifted.push_back(argv[i]);
        return RunExasimSolver(solver, static_cast<int>(shifted.size()), shifted.data(), comm);
    }

    return RunExasimSolver(solver, argc, argv, comm);
}
