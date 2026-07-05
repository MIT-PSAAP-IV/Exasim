//#include "../ModelDispatch/driver_abi.h"
#include "libbuiltinmodel.hpp"

#include <stdexcept>

// Provided by the auto-generated builtin_model_sizes.cpp (part of the builtin
// model library).  Returns compile-time model dimension constants.
extern ModelSizes builtinGetModelSizes(int modelnumber);

namespace {

bool IsValidBuiltInLibraryABI(const ExasimDriverABI& abi)
{
    return abi.abi_version == kExasimDriverABIVersion &&
           abi.struct_size == sizeof(ExasimDriverABI) &&
           abi.KokkosFlux &&
           abi.KokkosSource &&
           abi.KokkosSourcew &&
           abi.KokkosTdfunc &&
           abi.KokkosAvfield &&
           abi.KokkosEoS &&
           abi.KokkosEoSdu &&
           abi.KokkosEoSdw &&
           abi.KokkosFbou &&
           abi.KokkosUbou &&
           abi.KokkosFbouJac &&
           abi.KokkosUbouJac &&
           abi.KokkosFhat &&
           abi.KokkosUhat &&
           abi.KokkosStab &&
           abi.KokkosOutput &&
           abi.KokkosMonitor &&
           abi.KokkosVisScalars &&
           abi.KokkosVisVectors &&
           abi.KokkosVisTensors &&
           abi.KokkosQoIvolume &&
           abi.KokkosQoIboundary &&
           abi.KokkosInitu &&
           abi.KokkosInitq &&
           abi.KokkosInitudg &&
           abi.KokkosInitwdg &&
           abi.KokkosInitodg &&
           abi.cpuInitu &&
           abi.cpuInitq &&
           abi.cpuInitudg &&
           abi.cpuInitwdg &&
           abi.cpuInitodg &&
           abi.HdgFlux &&
           abi.HdgSource &&
           abi.HdgSourcew &&
           abi.HdgSourcewonly &&
           abi.HdgEoS &&
           abi.HdgFbou &&
           abi.HdgFbouonly &&
           abi.HdgFint &&
           abi.HdgFintonly &&
           abi.HdgFext &&
           abi.HdgFextonly;
}

} // namespace

const ExasimDriverABI& getBuiltInLibraryExasimDriverABI()
{
    static const ExasimDriverABI abi = [] {
        ExasimDriverABI value;
        value.abi_version = kExasimDriverABIVersion;
        value.struct_size = sizeof(ExasimDriverABI);

        value.KokkosFlux = &::builtinKokkosFlux;
        value.KokkosSource = &::builtinKokkosSource;
        value.KokkosSourcew = &::builtinKokkosSourcew;
        value.KokkosTdfunc = &::builtinKokkosTdfunc;
        value.KokkosAvfield = &::builtinKokkosAvfield;
        value.KokkosEoS = &::builtinKokkosEoS;
        value.KokkosEoSdu = &::builtinKokkosEoSdu;
        value.KokkosEoSdw = &::builtinKokkosEoSdw;
        value.KokkosFbou = &::builtinKokkosFbou;
        value.KokkosUbou = &::builtinKokkosUbou;
        value.KokkosFbouJac = &::builtinKokkosFbouJac;
        value.KokkosUbouJac = &::builtinKokkosUbouJac;
        value.KokkosFhat = &::builtinKokkosFhat;
        value.KokkosUhat = &::builtinKokkosUhat;
        value.KokkosStab = &::builtinKokkosStab;
        value.KokkosOutput = &::builtinKokkosOutput;
        value.KokkosMonitor = &::builtinKokkosMonitor;
        value.KokkosVisScalars = &::builtinKokkosVisScalars;
        value.KokkosVisVectors = &::builtinKokkosVisVectors;
        value.KokkosVisTensors = &::builtinKokkosVisTensors;
        value.KokkosQoIvolume = &::builtinKokkosQoIvolume;
        value.KokkosQoIboundary = &::builtinKokkosQoIboundary;

        value.KokkosInitu = &::builtinKokkosInitu;
        value.KokkosInitq = &::builtinKokkosInitq;
        value.KokkosInitudg = &::builtinKokkosInitudg;
        value.KokkosInitwdg = &::builtinKokkosInitwdg;
        value.KokkosInitodg = &::builtinKokkosInitodg;
        value.cpuInitu = &::builtincpuInitu;
        value.cpuInitq = &::builtincpuInitq;
        value.cpuInitudg = &::builtincpuInitudg;
        value.cpuInitwdg = &::builtincpuInitwdg;
        value.cpuInitodg = &::builtincpuInitodg;

        value.HdgFlux = &::builtinHdgFlux;
        value.HdgSource = &::builtinHdgSource;
        value.HdgSourcew = &::builtinHdgSourcew;
        value.HdgSourcewonly = &::builtinHdgSourcewonly;
        value.HdgEoS = &::builtinHdgEoS;
        value.HdgFbou = &::builtinHdgFbou;
        value.HdgFbouonly = &::builtinHdgFbouonly;
        value.HdgFint = &::builtinHdgFint;
        value.HdgFintonly = &::builtinHdgFintonly;
        value.HdgFext = &::builtinHdgFext;
        value.HdgFextonly = &::builtinHdgFextonly;

        value.GetModelSizes = &builtinGetModelSizes;

        return value;
    }();

    if (!IsValidBuiltInLibraryABI(abi))
        throw std::runtime_error("BuiltIn model library ABI table is incomplete or incompatible with Exasim");

    return abi;
}
