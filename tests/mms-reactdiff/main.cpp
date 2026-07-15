// MMS reaction-diffusion smoke: solve via exasim::petsc::solve_steady and verify the
// computed field against the exact manufactured solution u0=sin(pi x)sin(pi y),
// u1=0.5*sin(pi x)sin(pi y). Returns 0 (pass) / 1 (fail).
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <exasim/operators.hpp>
#include <exasim/export.hpp>
#include <exasim/petsc.hpp>
using exasim::ModelDefaults;
#include "generated/my_model.hpp"
int main(int argc,char**argv){
  MPI_Init(&argc,&argv); PETSC_COMM_WORLD=MPI_COMM_WORLD; PetscInitialize(&argc,&argv,0,0);
  if(!Kokkos::is_initialized())Kokkos::initialize(argc,argv); EXASIM_COMM_WORLD=MPI_COMM_WORLD;
  int rc=1;
  // Backend chosen at compile time (matches the emit-app scaffold): _CUDA->2, _HIP->3, else 0.
#if defined(_CUDA)
  const int backend = 2;
#elif defined(_HIP)
  const int backend = 3;
#else
  const int backend = 0;
#endif
  { const char* filein = argc>1?argv[1]:"datain/";
    CSolution<PdeModel> model(filein,"dataout/out","",1,0,0,0,backend,100);
    model.disc.common.nomodels=1;
    std::vector<Int> ncarr={model.disc.common.components.nc};
    std::vector<dstype*> udgarr={&model.disc.sol.udg[0]};
    model.disc.common.ncarray=ncarr.data(); model.disc.sol.udgarray=udgarr.data();
    int reason = exasim::petsc::solve_steady<PdeModel>(model, MPI_COMM_WORLD);
    auto& d=model.disc;
    const Int npe=d.common.grid.npe, ne=d.common.meshsizes.ne, ncx=d.common.components.ncx, nc=d.common.components.nc;
    // Copy to host first so the check works whether udg/xdg live on host (backend 0) or device
    // (backend 2/3); TemplateCopytoHost is a plain copy on the CPU backend.
    std::vector<dstype> udgh(npe*nc*ne), xdgh(npe*ncx*ne);
    TemplateCopytoHost(udgh.data(), d.sol.udg, (Int)udgh.size(), backend);
    TemplateCopytoHost(xdgh.data(), d.sol.xdg, (Int)xdgh.size(), backend);
    const dstype* udg=udgh.data(); const dstype* xdg=xdgh.data();
    double emax=0;
    for(Int e=0;e<ne;++e) for(Int p=0;p<npe;++p){
      double x=xdg[p+npe*(0+ncx*e)], y=xdg[p+npe*(1+ncx*e)];
      double S=std::sin(M_PI*x)*std::sin(M_PI*y);
      double u0=udg[p+npe*(0+nc*e)], u1=udg[p+npe*(1+nc*e)];
      emax=std::max(emax,std::max(std::fabs(u0-S),std::fabs(u1-0.5*S)));
    }
    const double TOL=2e-2;
    printf("[mms] SNESConvergedReason=%d  max|u-u_exact|=%.3e  (tol %.0e)  %s\n",
           reason, emax, TOL, (reason>0 && emax<TOL)?"PASS":"FAIL");
    rc = (reason>0 && emax<TOL)?0:1;
  }
  Kokkos::finalize(); PetscFinalize(); MPI_Finalize(); return rc;
}
