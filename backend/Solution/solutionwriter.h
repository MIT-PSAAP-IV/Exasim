/**
 * @class CSolutionWriter
 * @brief Owns the solution output (the binary solution/boundary streams + the QoI text file)
 *        and all of the read/write/output-field logic that was previously interleaved into
 *        CSolution.
 *
 * CSolution split (writer half): the output state (9 ofstreams) and the Save / Read / Get /
 * evalOutput / evalMonitor methods + the crash-dump are an I/O concern, distinct from the time-
 * integration orchestration and the nonlinear solve that remain in CSolution. The writer holds
 * references to the discretization, residual, visualization and solver it reads from; method
 * bodies are unchanged (the referenced members keep the same names: disc / residual / vis / solv
 * and the out* streams).
 */
#ifndef __SOLUTIONWRITER_H__
#define __SOLUTIONWRITER_H__

#include <exasim/detail/abi_adapter.hpp>

template <class, class> class CDiscretizationT; using CDiscretization = CDiscretizationT<::dstype, ::Int>;  // forward declarations (CSolutionWriter only holds references)
template <class M, class T, class I> class CResidual;
class CVisualization;   // model-free (no driver calls) -> stays non-templated
template <class M, class T, class I> class CSolver;

// Templated on the user Model type M (default = AbiAdapter): M threads to its QoI chain kernels
// (qoiElement<M>/qoiFace<M>) and to the typed members (residual/solv); vis is model-free.
template <class M = exasim::detail::AbiAdapter>
class CSolutionWriter {
public:
    CDiscretization& disc;
    CResidual<M>& residual;
    CVisualization& vis;
    CSolver<M>& solv;

    ofstream outsol;       // storing solutions
    ofstream outwdg;
    ofstream outuhat;
    ofstream outbouxdg;
    ofstream outboundg;
    ofstream outbouudg;
    ofstream outbouwdg;
    ofstream outbouuhat;
    ofstream outbousurf;     // SurfaceQuantities on the ibs boundaries (face nodes or Gauss points)
    ofstream outbousurfgeo;  // [x, n, dA] at the face Gauss points of outbousurf (saveSolBouLoc == 1)
    ofstream outqoi;

    dstype* surfbuf = nullptr;  // device scratch for evalSurfaceQuantities (largest boundary block)

    CSolutionWriter(CDiscretization& disc_, CResidual<M>& residual_, CVisualization& vis_, CSolver<M>& solv_)
        : disc(disc_), residual(residual_), vis(vis_), solv(solv_) {}

    ~CSolutionWriter() {
        if (outsol.is_open()) { outsol.close(); }
        if (outwdg.is_open()) { outwdg.close(); }
        if (outuhat.is_open()) { outuhat.close(); }
        if (outbouxdg.is_open()) { outbouxdg.close(); }
        if (outboundg.is_open()) { outboundg.close(); }
        if (outbouudg.is_open()) { outbouudg.close(); }
        if (outbouwdg.is_open()) { outbouwdg.close(); }
        if (outbouuhat.is_open()) { outbouuhat.close(); }
        if (outbousurf.is_open()) { outbousurf.close(); }
        if (outbousurfgeo.is_open()) { outbousurfgeo.close(); }
        if (outqoi.is_open()) { outqoi.close(); }
        if (surfbuf) TemplateFree(surfbuf, disc.common.backend);
    }

    // open the output streams and write the initial solution (was the CSolution ctor body)
    void setup(bool postprocessOnly);

    // crash teardown: optionally write a "_CRASH" Paraview, then close all output streams
    // (the caller dumps the raw solution .bin(s) before calling this)
    void crashDump(Int backend);

    // close and reopen the output streams under a new fileout prefix (for parameter sweeps)
    void ResetOutputFiles(const std::string& fileout);

    // PTC convergence-monitor field / output field (model Monitor/Output drivers)
    void evalMonitor(dstype* output, dstype* udg, dstype* wdg, Int nc, Int backend);
    void evalOutput(dstype* output, Int backend);

    // save solutions / QoI / Paraview / boundary fields in files
    void SaveSolutions(Int backend);
    void SaveQoI(Int backend);
    void SaveParaview(Int backend, std::string fname_modifier = "", bool force_tdep_write = false);

    // Write the ParaView output for an EXPLICIT 1-based step, without disturbing the
    // solver's own step counter.
    //
    // SaveParaview names the file outvis<modifier>_<currentstep+timestepOffset+1>, so a
    // caller that wants to name a specific step has to set common.timestate.currentstep =
    // step-1, write, and put it back — otherwise the scratch index leaks into a later
    // solve/postprocess that reads currentstep. That save/restore dance was written out by
    // hand in ExasimSolver::SaveParaviewStep AND copied into the CHEFSI
    // isoq2d_cht-petsc-fluid app. It belongs here, once.
    void SaveParaviewAt(Int step, Int backend, std::string fname_modifier = "");
    void SaveSolutionsOnBoundary(Int backend);
    void SaveNodesOnBoundary(Int backend);

    // open the outbou*_np<rank>.bin streams (and write outbouinfo) under the prefix base
    void openBoundaryFiles(const std::string& base);
    // face-node coordinates [nn, ncx] at buf and unit normals [nn, nd] at buf + nn*ncx for the
    // faces [f1, f2); uses nn*(ncx+3*nd+1) entries of buf
    void faceNodeGeometry(dstype* buf, Int f1, Int f2, Int backend);
    // evaluate SurfaceQuantities on faces [f1, f2) at face nodes or Gauss points
    // (common.qoiparams.saveSolBouLoc); returns the [np*nf, nsurfq] result inside surfbuf
    dstype* evalSurfaceQuantities(Int f1, Int f2, Int backend);
    void SaveOutputCG(Int backend);

    // read solutions / a saved record from the appended solution files
    void ReadSolutions(Int backend);
    void GetSolutions(Int step, Int backend);
};

#endif
