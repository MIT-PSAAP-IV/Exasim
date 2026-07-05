/**
 * @file driver_abi.hpp
 * @brief Stable low-level provider ABI for Exasim model modules.
 *
 * This header adapts the small exported-ABI pattern from StaticLibraryABI to
 * Exasim. Provider modules expose one versioned DriverABI object containing
 * typed kernel entry points. The reusable core backend can then build
 * higher-level driver wrappers on top of this low-level table without
 * compiling provider code into the core static library.
 */
#ifndef __EXASIM_DRIVER_ABI_H__
#define __EXASIM_DRIVER_ABI_H__

#include <cstdint>
#include <Kokkos_Core.hpp>

#ifdef USE_FLOAT
using dstype = float;
#else
using dstype = double;
#endif

inline constexpr std::uint32_t kExasimDriverABIVersion = 2;  // v2: kernels grouped into per-concern sub-structs + model-owned ModelSizes

// Model dimension constants carried by the model (PR #33): lets preprocessing obtain
// ncu/ncv/ncw/nsca/nvec/nten/nsurf/nvqoi from the compiled model instead of pdeapp.txt.
struct ModelSizes {
    int ncu  = 0;
    int nco  = 0;   // "other DG" == pdeapp.txt's ncv
    int ncw  = 0;
    int nsca = 0;
    int nvec = 0;
    int nten = 0;
    int nsurf = 0;
    int nvqoi = 0;
};

struct ExasimDriverABI {
    using KokkosElementFn =
        void (*)(dstype* f, const dstype* xdg, const dstype* udg,
                 const dstype* odg, const dstype* wdg, const dstype* uinf,
                 const dstype* param, dstype time, int modelnumber, int ng,
                 int nc, int ncu, int nd, int ncx, int nco, int ncw);

    using KokkosGlobalElementFn =
        void (*)(dstype* f, const dstype* xdg, const dstype* udg,
                 const dstype* odg, const dstype* wdg, const dstype* uinf,
                 const dstype* param, dstype time, int modelnumber, int ng,
                 int nc, int ncu, int nd, int ncx, int nco, int ncw, int nce,
                 int npe, int ne);

    using KokkosBoundaryFn =
        void (*)(dstype* f, const dstype* xdg, const dstype* udg,
                 const dstype* odg, const dstype* wdg, const dstype* uhg,
                 const dstype* nlg, const dstype* tau, const dstype* uinf,
                 const dstype* param, dstype time, int modelnumber, int ib,
                 int ng, int nc, int ncu, int nd, int ncx, int nco, int ncw);

    using KokkosBoundaryJacFn =
        void (*)(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg,
                 const dstype* xdg, const dstype* udg, const dstype* odg,
                 const dstype* wdg, const dstype* uhg, const dstype* nlg,
                 const dstype* tau, const dstype* uinf, const dstype* param,
                 dstype time, int modelnumber, int ib, int ng, int nc,
                 int ncu, int nd, int ncx, int nco, int ncw);

    using KokkosFaceCoupledFn =
        void (*)(dstype* f, const dstype* xdg, const dstype* udg1,
                 const dstype* udg2, const dstype* odg1, const dstype* odg2,
                 const dstype* wdg1, const dstype* wdg2, const dstype* uhg,
                 const dstype* nlg, const dstype* tau, const dstype* uinf,
                 const dstype* param, dstype time, int modelnumber, int ng,
                 int nc, int ncu, int nd, int ncx, int nco, int ncw);

    using KokkosInitFn =
        void (*)(dstype* f, const dstype* xdg, const dstype* uinf,
                 const dstype* param, int modelnumber, int ng, int ncx,
                 int nfield, int npe, int ne);

    using CpuInitFn =
        void (*)(dstype* f, const dstype* xdg, const dstype* uinf,
                 const dstype* param, int modelnumber, int ng, int ncx,
                 int nfield_runtime, int npe, int ne);

    using HdgElementJacFn =
        void (*)(dstype* f, dstype* f_udg, dstype* f_wdg,
                 const dstype* xdg, const dstype* udg, const dstype* odg,
                 const dstype* wdg, const dstype* uinf, const dstype* param,
                 dstype time, int modelnumber, int ng, int nc, int ncu,
                 int nd, int ncx, int nco, int ncw);

    using HdgSourcewOnlyFn =
        void (*)(dstype* f, dstype* f_wdg, const dstype* xdg,
                 const dstype* udg, const dstype* odg, const dstype* wdg,
                 const dstype* uinf, const dstype* param, dstype time,
                 int modelnumber, int ng, int nc, int ncu, int nd, int ncx,
                 int nco, int ncw);

    using HdgBoundaryJacFn =
        void (*)(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg,
                 const dstype* xdg, const dstype* udg, const dstype* odg,
                 const dstype* wdg, const dstype* uhg, const dstype* nlg,
                 const dstype* tau, const dstype* uinf, const dstype* param,
                 dstype time, int modelnumber, int ib, int ng, int nc,
                 int ncu, int nd, int ncx, int nco, int ncw);

    using HdgBoundaryStateFn =
        void (*)(dstype* f, const dstype* xdg, const dstype* udg,
                 const dstype* odg, const dstype* wdg, const dstype* uhg,
                 const dstype* nlg, const dstype* tau, const dstype* uinf,
                 const dstype* param, dstype time, int modelnumber, int ib,
                 int ng, int nc, int ncu, int nd, int ncx, int nco, int ncw);

    using HdgBoundaryExternalJacFn =
        void (*)(dstype* f, dstype* f_udg, dstype* f_wdg, dstype* f_uhg,
                 const dstype* xdg, const dstype* udg, const dstype* odg,
                 const dstype* wdg, const dstype* uhg, const dstype* nlg,
                 const dstype* uext, const dstype* tau, const dstype* uinf,
                 const dstype* param, dstype time, int modelnumber, int ib,
                 int ng, int nc, int ncu, int nd, int ncx, int nco, int ncw);

    using HdgBoundaryExternalStateFn =
        void (*)(dstype* f, const dstype* xdg, const dstype* udg,
                 const dstype* odg, const dstype* wdg, const dstype* uhg,
                 const dstype* nlg, const dstype* uext, const dstype* tau,
                 const dstype* uinf, const dstype* param, dstype time,
                 int modelnumber, int ib, int ng, int nc, int ncu, int nd,
                 int ncx, int nco, int ncw);

    std::uint32_t abi_version = 0;
    std::uint32_t struct_size = 0;

    // Model dimension constants (PR #33): model-owned sizes, read by preprocessing
    // instead of pdeapp.txt. 0 = not specified / use default.
    int ncu  = 0;
    int nco  = 0;   // "other DG" == pdeapp.txt's ncv
    int ncw  = 0;
    int nsca = 0;
    int nvec = 0;
    int nten = 0;
    int nsurf = 0;
    int nvqoi = 0;

    // The model's kernel dispatch table, grouped by concern to mirror the compile-time model
    // decomposition (the ModelDefaults mixins / is_*_model_v traits). Each sub-struct is a
    // self-contained, COPYABLE unit: a consumer can compose a model from sub-parts of different
    // providers (e.g. one model's `volume` physics with another's `boundary`), as long as the
    // parts share the same dimensional contract (nc/ncu/nd, which come from app.bin). (ABI v2.)
    struct VolumeDriverABI {        // flux/source/sourcew/tdfunc/avfield (the volume terms)
        KokkosElementFn       KokkosFlux    = nullptr;
        KokkosElementFn       KokkosSource  = nullptr;
        KokkosGlobalElementFn KokkosSourcew = nullptr;
        KokkosElementFn       KokkosTdfunc  = nullptr;
        KokkosGlobalElementFn KokkosAvfield = nullptr;
    } volume;
    struct EoSDriverABI {           // equation of state
        KokkosGlobalElementFn KokkosEoS   = nullptr;
        KokkosGlobalElementFn KokkosEoSdu = nullptr;
        KokkosGlobalElementFn KokkosEoSdw = nullptr;
    } eos;
    struct BoundaryDriverABI {      // boundary terms + LDG jacobians
        KokkosBoundaryFn    KokkosFbou    = nullptr;
        KokkosBoundaryFn    KokkosUbou    = nullptr;
        KokkosBoundaryJacFn KokkosFbouJac = nullptr;
        KokkosBoundaryJacFn KokkosUbouJac = nullptr;
    } boundary;
    struct InterfaceDriverABI {     // fhat/uhat/stab
        KokkosFaceCoupledFn KokkosFhat = nullptr;
        KokkosFaceCoupledFn KokkosUhat = nullptr;
        KokkosFaceCoupledFn KokkosStab = nullptr;
    } iface;
    struct OutputDriverABI {        // output/monitor + visualization
        KokkosGlobalElementFn KokkosOutput     = nullptr;
        KokkosGlobalElementFn KokkosMonitor    = nullptr;
        KokkosElementFn       KokkosVisScalars = nullptr;
        KokkosElementFn       KokkosVisVectors = nullptr;
        KokkosElementFn       KokkosVisTensors = nullptr;
    } output;
    struct QoIDriverABI {           // quantities of interest
        KokkosElementFn  KokkosQoIvolume   = nullptr;
        KokkosBoundaryFn KokkosQoIboundary = nullptr;
    } qoi;
    struct InitDriverABI {          // initial conditions (device + host)
        KokkosInitFn KokkosInitu   = nullptr;
        KokkosInitFn KokkosInitq   = nullptr;
        KokkosInitFn KokkosInitudg = nullptr;
        KokkosInitFn KokkosInitwdg = nullptr;
        KokkosInitFn KokkosInitodg = nullptr;
        CpuInitFn    cpuInitu      = nullptr;
        CpuInitFn    cpuInitq      = nullptr;
        CpuInitFn    cpuInitudg    = nullptr;
        CpuInitFn    cpuInitwdg    = nullptr;
        CpuInitFn    cpuInitodg    = nullptr;
    } init;
    struct HdgJacDriverABI {        // HDG pointwise jacobians
        HdgElementJacFn            HdgFlux        = nullptr;
        HdgElementJacFn            HdgSource      = nullptr;
        HdgElementJacFn            HdgSourcew     = nullptr;
        HdgSourcewOnlyFn           HdgSourcewonly = nullptr;
        HdgElementJacFn            HdgEoS         = nullptr;
        HdgBoundaryJacFn           HdgFbou        = nullptr;
        HdgBoundaryStateFn         HdgFbouonly    = nullptr;
        HdgBoundaryJacFn           HdgFint        = nullptr;
        HdgBoundaryStateFn         HdgFintonly    = nullptr;
        HdgBoundaryExternalJacFn   HdgFext        = nullptr;
        HdgBoundaryExternalStateFn HdgFextonly    = nullptr;
    } hdgjac;
    // Optional per-model size query (PR #33): if non-null the solver calls this with the
    // builtinmodelID to get model dimension constants instead of pdeapp.txt (per-model
    // providers set ncu/nco/... directly and leave this null).
    ModelSizes (*GetModelSizes)(int modelnumber) = nullptr;
};

#endif
