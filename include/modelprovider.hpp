/**
 * @file provider.hpp
 * @brief Low-level kernel provider ABI export for Exasim.
 *
 * This translation unit intentionally exposes only model metadata and
 * low-level kernel entry points. It does not include backend-facing driver
 * wrappers such as FluxDriver/SourceDriver, so it stays independent of
 * mesh/master/sol/temp/common runtime state. Higher-level adapters in the
 * reusable core library can dispatch through this ABI later.
 */

#pragma once

#include "driver_abi.hpp"
#include "modeldefaults.hpp"

namespace kokkos_kernel_source {

using ::PdeModel;

#include "kernels/KokkosFlux.hpp"
#include "kernels/KokkosFhat.hpp"
#include "kernels/KokkosFbou.hpp"
#include "kernels/KokkosUbou.hpp"
#include "kernels/KokkosFbouJac.hpp"
#include "kernels/KokkosUbouJac.hpp"
#include "kernels/KokkosUhat.hpp"
#include "kernels/KokkosStab.hpp"
#include "kernels/KokkosSource.hpp"
#include "kernels/KokkosVisScalars.hpp"
#include "kernels/KokkosVisVectors.hpp"
#include "kernels/KokkosVisTensors.hpp"
#include "kernels/KokkosQoIvolume.hpp"
#include "kernels/KokkosQoIboundary.hpp"
#include "kernels/KokkosSourcew.hpp"
#include "kernels/KokkosOutput.hpp"
#include "kernels/KokkosMonitor.hpp"
#include "kernels/KokkosInitu.hpp"
#include "kernels/KokkosInitq.hpp"
#include "kernels/KokkosInitwdg.hpp"
#include "kernels/KokkosInitudg.hpp"
#include "kernels/KokkosInitodg.hpp"
#include "kernels/KokkosEoS.hpp"
#include "kernels/KokkosEoSdu.hpp"
#include "kernels/KokkosEoSdw.hpp"
#include "kernels/KokkosAvfield.hpp"
#include "kernels/KokkosTdfunc.hpp"

#include "kernels/cpuInitu.hpp"
#include "kernels/cpuInitq.hpp"
#include "kernels/cpuInitwdg.hpp"
#include "kernels/cpuInitudg.hpp"
#include "kernels/cpuInitodg.hpp"

#include "kernels/HdgFlux.hpp"
#include "kernels/HdgSource.hpp"
#include "kernels/HdgSourcew.hpp"
#include "kernels/HdgSourcewonly.hpp"
#include "kernels/HdgFbou.hpp"
#include "kernels/HdgFbouonly.hpp"
#include "kernels/HdgFint.hpp"
#include "kernels/HdgFintonly.hpp"
#include "kernels/HdgFext.hpp"
#include "kernels/HdgFextonly.hpp"
#include "kernels/HdgEoS.hpp"

} 

inline const ExasimDriverABI& getKokkosKernelExasimDriverABI()
{
    static const ExasimDriverABI abi = [] {
        ExasimDriverABI value;
        value.abi_version = kExasimDriverABIVersion;
        value.struct_size = sizeof(ExasimDriverABI);

        value.volume.KokkosFlux = &kokkos_kernel_source::KokkosFlux;
        value.volume.KokkosSource = &kokkos_kernel_source::KokkosSource;
        value.volume.KokkosSourcew = &kokkos_kernel_source::KokkosSourcew;
        value.volume.KokkosTdfunc = &kokkos_kernel_source::KokkosTdfunc;
        value.volume.KokkosAvfield = &kokkos_kernel_source::KokkosAvfield;
        value.eos.KokkosEoS = &kokkos_kernel_source::KokkosEoS;
        value.eos.KokkosEoSdu = &kokkos_kernel_source::KokkosEoSdu;
        value.eos.KokkosEoSdw = &kokkos_kernel_source::KokkosEoSdw;
        value.boundary.KokkosFbou = &kokkos_kernel_source::KokkosFbou;
        value.boundary.KokkosUbou = &kokkos_kernel_source::KokkosUbou;
        value.boundary.KokkosFbouJac = &kokkos_kernel_source::KokkosFbouJac;
        value.boundary.KokkosUbouJac = &kokkos_kernel_source::KokkosUbouJac;
        value.iface.KokkosFhat = &kokkos_kernel_source::KokkosFhat;
        value.iface.KokkosUhat = &kokkos_kernel_source::KokkosUhat;
        value.iface.KokkosStab = &kokkos_kernel_source::KokkosStab;
        value.output.KokkosOutput = &kokkos_kernel_source::KokkosOutput;
        value.output.KokkosMonitor = &kokkos_kernel_source::KokkosMonitor;
        value.output.KokkosVisScalars = &kokkos_kernel_source::KokkosVisScalars;
        value.output.KokkosVisVectors = &kokkos_kernel_source::KokkosVisVectors;
        value.output.KokkosVisTensors = &kokkos_kernel_source::KokkosVisTensors;
        value.qoi.KokkosQoIvolume = &kokkos_kernel_source::KokkosQoIvolume;
        value.qoi.KokkosQoIboundary = &kokkos_kernel_source::KokkosQoIboundary;

        value.init.KokkosInitu = &kokkos_kernel_source::KokkosInitu;
        value.init.KokkosInitq = &kokkos_kernel_source::KokkosInitq;
        value.init.KokkosInitudg = &kokkos_kernel_source::KokkosInitudg;
        value.init.KokkosInitwdg = &kokkos_kernel_source::KokkosInitwdg;
        value.init.KokkosInitodg = &kokkos_kernel_source::KokkosInitodg;
        value.init.cpuInitu = &kokkos_kernel_source::cpuInitu;
        value.init.cpuInitq = &kokkos_kernel_source::cpuInitq;
        value.init.cpuInitudg = &kokkos_kernel_source::cpuInitudg;
        value.init.cpuInitwdg = &kokkos_kernel_source::cpuInitwdg;
        value.init.cpuInitodg = &kokkos_kernel_source::cpuInitodg;

        value.hdgjac.HdgFlux = &kokkos_kernel_source::HdgFlux;
        value.hdgjac.HdgSource = &kokkos_kernel_source::HdgSource;
        value.hdgjac.HdgSourcew = &kokkos_kernel_source::HdgSourcew;
        value.hdgjac.HdgSourcewonly = &kokkos_kernel_source::HdgSourcewonly;
        value.hdgjac.HdgEoS = &kokkos_kernel_source::HdgEoS;
        value.hdgjac.HdgFbou = &kokkos_kernel_source::HdgFbou;
        value.hdgjac.HdgFbouonly = &kokkos_kernel_source::HdgFbouonly;
        value.hdgjac.HdgFint = &kokkos_kernel_source::HdgFint;
        value.hdgjac.HdgFintonly = &kokkos_kernel_source::HdgFintonly;
        value.hdgjac.HdgFext = &kokkos_kernel_source::HdgFext;
        value.hdgjac.HdgFextonly = &kokkos_kernel_source::HdgFextonly;

        return value;
    }();

    return abi;
}
