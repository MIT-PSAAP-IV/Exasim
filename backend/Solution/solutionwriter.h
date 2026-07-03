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
    ofstream outqoi;

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
        if (outqoi.is_open()) { outqoi.close(); }
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
    void SaveSolutionsOnBoundary(Int backend);
    void SaveNodesOnBoundary(Int backend);
    void SaveOutputCG(Int backend);

    // read solutions / a saved record from the appended solution files
    void ReadSolutions(Int backend);
    void GetSolutions(Int step, Int backend);
};

#endif
