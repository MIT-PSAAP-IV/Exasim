#include <exasim/drivers.hpp>
#include <exasim/detail/driver_dispatch.hpp>

/*
  wequation.cpp

  This file contains functions for solving the auxiliary variable equation (w-equation) in the context of a discontinuous Galerkin (DG) or hybridized DG (HDG) discretization. The w-equation is typically used for time-dependent or differential-algebraic equation (DAE) systems, where auxiliary variables are introduced to facilitate the solution process.

  Functions:

  1. void wEquation<M>(dstype *wdg, dstype *xdg, dstype *udg, dstype *odg, dstype *wsrc, dstype *tempg, appstruct &app, commonstruct &common, Int ng, Int backend)
    - Solves for the auxiliary variable w (wdg) given the solution variables (udg), coordinates (xdg), other variables (odg), and source term (wsrc).
    - Handles both wave-type equations (explicit update) and general DAE systems (Newton iteration).
    - Uses temporary storage (tempg) for intermediate calculations.
    - Supports up to five auxiliary variables (ncw <= 5).

  2. void wEquation<M>(dstype *wdg, dstype *wdg_udg, dstype *xdg, dstype *udg, dstype *odg, dstype *wsrc, dstype *tempg, appstruct &app, commonstruct &common, Int ng, Int backend)
    - Extended version of wEquation that also computes the sensitivity of w with respect to udg (wdg_udg).
    - Uses Newton iteration for nonlinear systems and computes the Jacobian of the source term.
    - Supports up to five auxiliary variables (ncw <= 5).

  3. void GetW<M>(dstype *w, solstruct &sol, tempstruct &tmp, appstruct &app, commonstruct &common, Int backend)
    - Loops over element blocks and computes the auxiliary variable w for each element.
    - Extracts element-wise data from global arrays, solves the w-equation, and writes the results back to the global array.

  Notes:
  - The functions rely on several utility routines for matrix operations, source term evaluation, and error handling.
  - The Newton iteration in wEquation checks for convergence and throws an error if the solution does not converge within the specified tolerance.
  - The code is designed for flexibility in the number of auxiliary variables and supports both explicit and implicit time integration schemes.
*/
#ifndef __WEQUATION
#define __WEQUATION

template <class T=dstype, class I=Int>
static void ReportNanInHdgSourcewonlyOutput(const char* field, const T* data,
      const T* xdg, const T* udg, const T* odg, const T* wdg,
      Int ng, Int ncomp, Int nc, Int nco, Int ncw, Int nd, Int mpiRank, Int iter)
{
    using dstype=T;
    for (Int comp = 0; comp < ncomp; ++comp) {
        for (Int i = 0; i < ng; ++i) {
            dstype value = data[i + ng * comp];
            if (IS_NAN(value)) {
                std::cout << "Rank = " << mpiRank
                     << ", Iter = " << iter
                     << ", stage = HdgSourcewonly"
                     << ", field = " << field
                     << ", gausspoint = " << i
                     << ", component = " << comp
                     << ", x = (";
                for (Int d = 0; d < nd; ++d) {
                    if (d > 0)
                        std::cout << ", ";
                    std::cout << xdg[i + ng * d];
                }
                std::cout << "), w = (";
                for (Int k = 0; k < ncw; ++k) {
                    if (k > 0)
                        std::cout << ", ";
                    std::cout << wdg[i + ng * k];
                }
                std::cout << "), u = (";
                for (Int k = 0; k < nc; ++k) {
                    if (k > 0)
                        std::cout << ", ";
                    std::cout << udg[i + ng * k];
                }
                std::cout << ")";
                if (nco > 0) {
                    std::cout << ", o = (";
                    for (Int k = 0; k < nco; ++k) {
                        if (k > 0)
                            std::cout << ", ";
                        std::cout << odg[i + ng * k];
                    }
                    std::cout << ")";
                }
                std::cout << ", value = " << value << std::endl;
                //error("NaN detected in HdgSourcewonly");
            }
        }
    }
}

template <class M, class T=dstype, class I=Int>
inline void wEquation(T *wdg, T *xdg, T *udg, T *odg, T *wsrc, 
      T *tempg, appstructT<T,I> &app, commonstructT<T,I> &common, Int ng, Int backend)
{
    using dstype=T;        
    Int ncu = common.components.ncu; // number of compoments of (u)
    Int nd = common.grid.nd; // spatial dimension
    Int nc = common.components.nc; // number of compoments of (u, q)
    Int ncw = common.components.ncw;// number of compoments of (w)
    Int nco = common.components.nco;// number of compoments of (o)
    Int ncx = common.components.ncx;// number of compoments of (xdg)        
    //Int npe = common.grid.npe; // number of nodes on master element    
    Int modelnumber = common.modelnumber;
    if ((modelnumber <= 0) && (common.builtinmodelID > 0)) modelnumber = common.builtinmodelID;
    dstype time = common.timestate.time;                
    dstype *uinf = app.uinf;
    dstype *physicsparam = app.physicsparam;    

    if (common.timeparams.wave==1) {
        // dw/dt = u -> (dtfactor * w - wsrc) = u -> w = (1/dtfactor) * (u + wsrc)
        dstype scalar = one/common.timestate.dtfactor;
        ArrayAXPBY(wdg, udg, wsrc, scalar, scalar, ng*ncw);
    }        
    else {         
        // fix bug here
        dstype *s = tempg; // temporary array    
        dstype *s_wdg = &tempg[ng*ncw]; // temporary array 
        // use Newton to solve the nonlinear system  alpha * dw/dt + beta w = S(w, u, q) to obtain w for given (u, q)             
        for (int iter=0; iter<20; iter++) {
          // alpha * dw/dt + beta w = sourcew(u,q,w) -> alpha (dtfactor * w - wsrc) + beta w = sourcew(u,q,w) 
          // ->  (alpha * dtfactor + beta) w - alpha * wsrc - sourcew(u,q,w) = 0              

          // calculate the source term Sourcew(xdg, udg, odg, wdg)
          EXASIM_LEGACY_W_CALL(HdgSourcewonly, s, s_wdg, xdg, udg, odg, wdg, uinf, physicsparam, time, modelnumber, ng, nc, ncu, nd, ncx, nco, ncw);
          // if (backend <= 1) {
          //     ReportNanInHdgSourcewonlyOutput("f", s, xdg, udg, odg, wdg, ng, ncw, nc, nco, ncw, nd, common.mpiRank, iter);
          //     ReportNanInHdgSourcewonlyOutput("f_wdg", s_wdg, xdg, udg, odg, wdg, ng, ncw*ncw, nc, nco, ncw, nd, common.mpiRank, iter);
          // }
                    
          // alpha*dirkd/dt + beta
          dstype scalar = common.timeparams.dae_alpha*common.timestate.dtfactor + common.timeparams.dae_beta;

          // calculate residual vector = sourcew(u,q,w) + alpha * wsrc - (alpha * dtfactor + beta) w
          ArrayAdd3Vectors(s, s, wsrc, wdg, one, common.timeparams.dae_alpha, -scalar, ng*ncw);                           

          // compute jacobian matrix = (alpha * dtfactor + beta) - s_wdg
          ArrayAXPB(s_wdg, s_wdg, minusone, scalar, ng*ncw*ncw);                

          // check convergence
          dstype nrm = NORM(common.cublasHandle, ng*ncw, s, backend);
          if (nrm < 1e-6) {
            // if (common.mpiRank==2) {
            //   std::cout << std::fixed << std::setprecision(15);
            //   std::cout<<common.timeparams.dae_alpha<<"  "<<common.timeparams.dae_beta<<"  "<<scalar<<std::endl;
            //   std::cout<<"Iter = "<<iter<<", norm = "<<nrm<<", s[0] = "<<s[0]<<std::endl;
            //   std::cout<<wdg[0];
            //   for (int m=0; m<8; m++)
            //     std::cout<<"   "<<udg[ng*m];
            //   std::cout<<std::endl;
            // }
            break;              
          }
          
          // solve the linear system jacobian * dw = residual
          if (ncw==1) {                
            SmallMatrixSolve11(s, s_wdg, ng, 1);
          }
          else if (ncw==2) {
            SmallMatrixSolve22(s, s_wdg, ng, 1);
          }
          else if (ncw==3) {
            SmallMatrixSolve33(s, s_wdg, ng, 1);
          }
          else if (ncw==4) {
            SmallMatrixSolve44(s, s_wdg, ng, 1);
          }
          else if (ncw==5) {
            SmallMatrixSolve55(s, s_wdg, ng, 1);
          }
          else {
            error("DAE functionality supports at most five variables.");
          }              

          // update w = w + dw
          ArrayAXPBY(wdg, wdg, s, one, one, ng*ncw);          
        }                        
    }            
}

template <class M, class T=dstype, class I=Int>
inline void wEquation(T *wdg, T *wdg_udg, T *xdg, T *udg, T *odg, T *wsrc, 
       T *tempg, appstructT<T,I> &app, commonstructT<T,I> &common, Int ng, Int backend)
{
    using dstype=T;        
    Int ncu = common.components.ncu; // number of compoments of (u)
    Int nd = common.grid.nd; // spatial dimension
    Int nc = common.components.nc; // number of compoments of (u, q)
    Int ncw = common.components.ncw;// number of compoments of (w)
    Int nco = common.components.nco;// number of compoments of (o)
    Int ncx = common.components.ncx;// number of compoments of (xdg)        
    //Int npe = common.grid.npe; // number of nodes on master element    
    Int modelnumber = common.modelnumber;
    if ((modelnumber <= 0) && (common.builtinmodelID > 0)) modelnumber = common.builtinmodelID;
    dstype time = common.timestate.time;                
    dstype *uinf = app.uinf;
    dstype *physicsparam = app.physicsparam;

    if (common.timeparams.wave==1) {
        // dw/dt = u -> (dtfactor * w - wsrc) = u -> w = (1/dtfactor) * (u + wsrc)
        dstype scalar = one/common.timestate.dtfactor;
        ArrayAXPBY(wdg, udg, wsrc, scalar, scalar, ng*ncw);
        ArraySetValue(wdg_udg, scalar, ng*ncw*nc);
    }        
    else {   
         // fix bug here
        dstype *s = tempg; // temporary array    
        dstype *s_wdg = &tempg[ng*ncw]; // temporary array              
        // use Newton to solve the nonlinear system  alpha * dw/dt + beta w = S(w, u, q) to obtain w for given (u, q)                
        for (int iter=0; iter<20; iter++) {
          // alpha * dw/dt + beta w = sourcew(u,q,w) -> alpha (dtfactor * w - wsrc) + beta w = sourcew(u,q,w) 
          // ->  (alpha * dtfactor + beta) w - alpha * wsrc - sourcew(u,q,w) = 0              

          // calculate the source term Sourcew(xdg, udg, odg, wdg)
          EXASIM_LEGACY_W_CALL(HdgSourcewonly, s, s_wdg, xdg, udg, odg, wdg, uinf, physicsparam, time, modelnumber, ng, nc, ncu, nd, ncx, nco, ncw);            
          
          // alpha*dirkd/dt + beta
          dstype scalar = common.timeparams.dae_alpha*common.timestate.dtfactor + common.timeparams.dae_beta;

          // calculate residual vector = sourcew(u,q,w) + alpha * wsrc - (alpha * dtfactor + beta) w
          ArrayAdd3Vectors(s, s, wsrc, wdg, one, common.timeparams.dae_alpha, -scalar, ng*ncw);                           

          // compute jacobian matrix = (alpha * dtfactor + beta) - s_wdg
          ArrayAXPB(s_wdg, s_wdg, minusone, scalar, ng*ncw*ncw);                

          // solve the linear system jacobian * dw = residual
          if (ncw==1) {                
            SmallMatrixSolve11(s, s_wdg, ng, 1);
          }
          else if (ncw==2) {
            SmallMatrixSolve22(s, s_wdg, ng, 1);
          }
          else if (ncw==3) {
            SmallMatrixSolve33(s, s_wdg, ng, 1);
          }
          else if (ncw==4) {
            SmallMatrixSolve44(s, s_wdg, ng, 1);
          }
          else if (ncw==5) {
            SmallMatrixSolve55(s, s_wdg, ng, 1);
          }
          else {
            error("DAE functionality supports at most five variables.");
          }              

          // update w = w + dw
          ArrayAXPBY(wdg, wdg, s, one, one, ng*ncw);

          // check convergence
          dstype nrm = NORM(common.cublasHandle, ng*ncw, s, backend);
          if (nrm < 1e-8) {     
            // wdg_udg is actually s_udg 
            EXASIM_LEGACY_W_CALL(HdgSourcew, s, wdg_udg, s_wdg, xdg, udg, odg, wdg, uinf, physicsparam, time, modelnumber, ng, nc, ncu, nd, ncx, nco, ncw);
            
            // fix bug here
            // compute jacobian matrix = (alpha * dtfactor + beta) - s_wdg
            ArrayAXPB(s_wdg, s_wdg, minusone, scalar, ng*ncw*ncw);                
            
            // w_udg = -inverse(s_wdg) * s_udg
            if (ncw==1) {                
              SmallMatrixSolve11(wdg_udg, s_wdg, ng, nc);
            }
            else if (ncw==2) {
              SmallMatrixSolve22(wdg_udg, s_wdg, ng, nc);
            }
            else if (ncw==3) {
              SmallMatrixSolve33(wdg_udg, s_wdg, ng, nc);
            }
            else if (ncw==4) {
              SmallMatrixSolve44(wdg_udg, s_wdg, ng, nc);
            }
            else if (ncw==5) {
              SmallMatrixSolve55(wdg_udg, s_wdg, ng, nc);
            }
            else {
              error("DAE functionality supports at most three dependent variables.");
            }                          
            break;              
          }
          else {
            if (iter==20) {
              error("Newton in wequation does not converge to 1e-8.");
            }
          } 
        }                        
    }            
}

template <class M, class T=dstype, class I=Int>
inline void GetW(T *w, solstructT<T,I> &sol, tempstructT<T,I> &tmp, appstructT<T,I> &app, commonstructT<T,I> &common, Int backend)
{
    using dstype=T;
  for (Int j=0; j<common.meshsizes.nbe; j++) {         
      Int e1 = common.eblks[3*j]-1;
      Int e2 = common.eblks[3*j+1];
      Int ns = e2-e1;        
      Int ng = common.grid.npe*ns;
      Int ncw = common.components.ncw;
      Int ncx = common.components.ncx;
      Int nc = common.components.nc;
      Int nco = common.components.nco;
      dstype* wdg = &tmp.tempn[0];
      dstype* xdg = &tmp.tempn[ng*ncw];
      dstype* udg = &tmp.tempn[ng*(ncw+ncx)];
      dstype* odg = &tmp.tempn[ng*(ncw+ncx+nc)];
      dstype* sdg = &tmp.tempn[ng*(ncw+ncx+nc+nco)];
      GetElemNodes(wdg, w, common.grid.npe, ncw, 0, ncw, e1, e2);
      GetElemNodes(xdg, sol.xdg, common.grid.npe, ncx, 0, ncx, e1, e2);
      GetElemNodes(udg, sol.udg, common.grid.npe, nc, 0, nc, e1, e2);
      GetElemNodes(odg, sol.odg, common.grid.npe, nco, 0, nco, e1, e2);
      GetElemNodes(sdg, sol.wsrc, common.grid.npe, ncw, 0, ncw, e1, e2);
      wEquation<M>(wdg, xdg, udg, odg, sdg, tmp.tempg, app, common, ng, common.backend);
      PutElemNodes(w, wdg, common.grid.npe, ncw, 0, ncw, e1, e2);
  }   
}

#endif
