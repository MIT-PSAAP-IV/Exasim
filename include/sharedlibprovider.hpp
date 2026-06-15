#pragma once
#include <stdexcept>

extern "C" const ExasimDriverABI* GetText2CodeExasimDriverABI();

namespace {

bool IsValidSharedLibraryABI(const ExasimDriverABI& abi)
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

const ExasimDriverABI& getSharedLibraryExasimDriverABI()
{
    const ExasimDriverABI* abi = GetText2CodeExasimDriverABI();

    if (abi == nullptr || !IsValidSharedLibraryABI(*abi))
        throw std::runtime_error("Shared model library ABI table is incomplete or incompatible with Exasim");

    return *abi;
}
