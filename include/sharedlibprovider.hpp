#pragma once
#include <stdexcept>

extern "C" const ExasimDriverABI* GetText2CodeExasimDriverABI();

namespace {

bool IsValidSharedLibraryABI(const ExasimDriverABI& abi)
{
    return abi.abi_version == kExasimDriverABIVersion &&
           abi.struct_size == sizeof(ExasimDriverABI) &&
           abi.volume.KokkosFlux &&
           abi.volume.KokkosSource &&
           abi.volume.KokkosSourcew &&
           abi.volume.KokkosMaterialstate &&
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
           abi.hdgjac.HdgMaterialstate &&
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

const ExasimDriverABI& getSharedLibraryExasimDriverABI()
{
    const ExasimDriverABI* abi = GetText2CodeExasimDriverABI();

    if (abi == nullptr || !IsValidSharedLibraryABI(*abi))
        throw std::runtime_error("Shared model library ABI table is incomplete or incompatible with Exasim");

    return *abi;
}
