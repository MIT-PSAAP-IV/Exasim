//#include "../ModelDispatch/driver_abi.h"
#include "libbuiltinmodel.hpp"

#include <stdexcept>

namespace {

bool IsValidBuiltInLibraryABI(const ExasimDriverABI& abi)
{
    return abi.abi_version == kExasimDriverABIVersion &&
           abi.struct_size == sizeof(ExasimDriverABI) &&
           abi.volume.KokkosFlux &&
           abi.volume.KokkosSource &&
           abi.volume.KokkosSourcew &&
           abi.volume.KokkosTdfunc &&
           abi.volume.KokkosAvfield &&
           abi.eos.KokkosEoS &&
           abi.eos.KokkosEoSdu &&
           abi.eos.KokkosEoSdw &&
           abi.boundary.KokkosFbou &&
           abi.boundary.KokkosUbou &&
           abi.boundary.KokkosFbouJac &&
           abi.boundary.KokkosUbouJac &&
           abi.iface.KokkosFhat &&
           abi.iface.KokkosUhat &&
           abi.iface.KokkosStab &&
           abi.output.KokkosOutput &&
           abi.output.KokkosMonitor &&
           abi.output.KokkosVisScalars &&
           abi.output.KokkosVisVectors &&
           abi.output.KokkosVisTensors &&
           abi.qoi.KokkosQoIvolume &&
           abi.qoi.KokkosQoIboundary &&
           abi.init.KokkosInitu &&
           abi.init.KokkosInitq &&
           abi.init.KokkosInitudg &&
           abi.init.KokkosInitwdg &&
           abi.init.KokkosInitodg &&
           abi.init.cpuInitu &&
           abi.init.cpuInitq &&
           abi.init.cpuInitudg &&
           abi.init.cpuInitwdg &&
           abi.init.cpuInitodg &&
           abi.hdgjac.HdgFlux &&
           abi.hdgjac.HdgSource &&
           abi.hdgjac.HdgSourcew &&
           abi.hdgjac.HdgSourcewonly &&
           abi.hdgjac.HdgEoS &&
           abi.hdgjac.HdgFbou &&
           abi.hdgjac.HdgFbouonly &&
           abi.hdgjac.HdgFint &&
           abi.hdgjac.HdgFintonly &&
           abi.hdgjac.HdgFext &&
           abi.hdgjac.HdgFextonly;
}

} // namespace

const ExasimDriverABI& getBuiltInLibraryExasimDriverABI()
{
    static const ExasimDriverABI abi = [] {
        ExasimDriverABI value;
        value.abi_version = kExasimDriverABIVersion;
        value.struct_size = sizeof(ExasimDriverABI);

        value.volume.KokkosFlux = &::builtinKokkosFlux;
        value.volume.KokkosSource = &::builtinKokkosSource;
        value.volume.KokkosSourcew = &::builtinKokkosSourcew;
        value.volume.KokkosTdfunc = &::builtinKokkosTdfunc;
        value.volume.KokkosAvfield = &::builtinKokkosAvfield;
        value.eos.KokkosEoS = &::builtinKokkosEoS;
        value.eos.KokkosEoSdu = &::builtinKokkosEoSdu;
        value.eos.KokkosEoSdw = &::builtinKokkosEoSdw;
        value.boundary.KokkosFbou = &::builtinKokkosFbou;
        value.boundary.KokkosUbou = &::builtinKokkosUbou;
        value.boundary.KokkosFbouJac = &::builtinKokkosFbouJac;
        value.boundary.KokkosUbouJac = &::builtinKokkosUbouJac;
        value.iface.KokkosFhat = &::builtinKokkosFhat;
        value.iface.KokkosUhat = &::builtinKokkosUhat;
        value.iface.KokkosStab = &::builtinKokkosStab;
        value.output.KokkosOutput = &::builtinKokkosOutput;
        value.output.KokkosMonitor = &::builtinKokkosMonitor;
        value.output.KokkosVisScalars = &::builtinKokkosVisScalars;
        value.output.KokkosVisVectors = &::builtinKokkosVisVectors;
        value.output.KokkosVisTensors = &::builtinKokkosVisTensors;
        value.qoi.KokkosQoIvolume = &::builtinKokkosQoIvolume;
        value.qoi.KokkosQoIboundary = &::builtinKokkosQoIboundary;

        value.init.KokkosInitu = &::builtinKokkosInitu;
        value.init.KokkosInitq = &::builtinKokkosInitq;
        value.init.KokkosInitudg = &::builtinKokkosInitudg;
        value.init.KokkosInitwdg = &::builtinKokkosInitwdg;
        value.init.KokkosInitodg = &::builtinKokkosInitodg;
        value.init.cpuInitu = &::builtincpuInitu;
        value.init.cpuInitq = &::builtincpuInitq;
        value.init.cpuInitudg = &::builtincpuInitudg;
        value.init.cpuInitwdg = &::builtincpuInitwdg;
        value.init.cpuInitodg = &::builtincpuInitodg;

        value.hdgjac.HdgFlux = &::builtinHdgFlux;
        value.hdgjac.HdgSource = &::builtinHdgSource;
        value.hdgjac.HdgSourcew = &::builtinHdgSourcew;
        value.hdgjac.HdgSourcewonly = &::builtinHdgSourcewonly;
        value.hdgjac.HdgEoS = &::builtinHdgEoS;
        value.hdgjac.HdgFbou = &::builtinHdgFbou;
        value.hdgjac.HdgFbouonly = &::builtinHdgFbouonly;
        value.hdgjac.HdgFint = &::builtinHdgFint;
        value.hdgjac.HdgFintonly = &::builtinHdgFintonly;
        value.hdgjac.HdgFext = &::builtinHdgFext;
        value.hdgjac.HdgFextonly = &::builtinHdgFextonly;

        return value;
    }();

    if (!IsValidBuiltInLibraryABI(abi))
        throw std::runtime_error("BuiltIn model library ABI table is incomplete or incompatible with Exasim");

    return abi;
}
