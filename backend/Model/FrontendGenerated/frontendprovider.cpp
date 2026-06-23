/**
 * @file provider.cpp
 * @brief Low-level FrontendGenerated provider ABI export for Exasim.
 *
 * This translation unit exposes only generated low-level model kernels through
 * GetExasimDriverABI(). It intentionally avoids backend-facing driver wrappers and
 * therefore stays independent of mesh/master/sol/temp/common runtime state.
 */

//#include "../ModelDispatch/driver_abi.h"

namespace frontend_generated_source {

#include "KokkosFlux.cpp"
#include "KokkosFhat.cpp"
#include "KokkosFbou.cpp"
#include "KokkosUbou.cpp"
#include "KokkosUhat.cpp"
#include "KokkosStab.cpp"
#include "KokkosSource.cpp"
#include "KokkosVisScalars.cpp"
#include "KokkosVisVectors.cpp"
#include "KokkosVisTensors.cpp"
#include "KokkosQoIvolume.cpp"
#include "KokkosQoIboundary.cpp"
#include "KokkosSourcew.cpp"
#include "KokkosOutput.cpp"
#include "KokkosMonitor.cpp"
#include "KokkosInitu.cpp"
#include "KokkosInitq.cpp"
#include "KokkosInitwdg.cpp"
#include "KokkosInitudg.cpp"
#include "KokkosInitodg.cpp"
#include "KokkosEoS.cpp"
#include "KokkosEoSdu.cpp"
#include "KokkosEoSdw.cpp"
#include "KokkosAvfield.cpp"
#include "KokkosTdfunc.cpp"

#include "cpuInitu.cpp"
#include "cpuInitq.cpp"
#include "cpuInitwdg.cpp"
#include "cpuInitudg.cpp"
#include "cpuInitodg.cpp"

#include "HdgFlux.cpp"
#include "HdgSource.cpp"
#include "HdgSourcew.cpp"
#include "HdgSourcewonly.cpp"
#include "HdgFbou.cpp"
#include "HdgFbouonly.cpp"
#include "HdgFint.cpp"
#include "HdgFintonly.cpp"
#include "HdgFext.cpp"
#include "HdgFextonly.cpp"
#include "HdgEoS.cpp"

} // namespace frontend_generated_source

const ExasimDriverABI& getFrontendGeneratedExasimDriverABI()
{
    static const ExasimDriverABI abi = [] {
        ExasimDriverABI value;
        value.abi_version = kExasimDriverABIVersion;
        value.struct_size = sizeof(ExasimDriverABI);

        value.volume.KokkosFlux = &frontend_generated_source::KokkosFlux;
        value.volume.KokkosSource = &frontend_generated_source::KokkosSource;
        value.volume.KokkosSourcew = &frontend_generated_source::KokkosSourcew;
        value.volume.KokkosTdfunc = &frontend_generated_source::KokkosTdfunc;
        value.volume.KokkosAvfield = &frontend_generated_source::KokkosAvfield;
        value.eos.KokkosEoS = &frontend_generated_source::KokkosEoS;
        value.eos.KokkosEoSdu = &frontend_generated_source::KokkosEoSdu;
        value.eos.KokkosEoSdw = &frontend_generated_source::KokkosEoSdw;
        value.boundary.KokkosFbou = &frontend_generated_source::KokkosFbou;
        value.boundary.KokkosUbou = &frontend_generated_source::KokkosUbou;
        value.boundary.KokkosFbouJac = &frontend_generated_source::KokkosFbouJac;
        value.boundary.KokkosUbouJac = &frontend_generated_source::KokkosUbouJac;
        value.iface.KokkosFhat = &frontend_generated_source::KokkosFhat;
        value.iface.KokkosUhat = &frontend_generated_source::KokkosUhat;
        value.iface.KokkosStab = &frontend_generated_source::KokkosStab;
        value.output.KokkosOutput = &frontend_generated_source::KokkosOutput;
        value.output.KokkosMonitor = &frontend_generated_source::KokkosMonitor;
        value.output.KokkosVisScalars = &frontend_generated_source::KokkosVisScalars;
        value.output.KokkosVisVectors = &frontend_generated_source::KokkosVisVectors;
        value.output.KokkosVisTensors = &frontend_generated_source::KokkosVisTensors;
        value.qoi.KokkosQoIvolume = &frontend_generated_source::KokkosQoIvolume;
        value.qoi.KokkosQoIboundary = &frontend_generated_source::KokkosQoIboundary;

        value.init.KokkosInitu = &frontend_generated_source::KokkosInitu;
        value.init.KokkosInitq = &frontend_generated_source::KokkosInitq;
        value.init.KokkosInitudg = &frontend_generated_source::KokkosInitudg;
        value.init.KokkosInitwdg = &frontend_generated_source::KokkosInitwdg;
        value.init.KokkosInitodg = &frontend_generated_source::KokkosInitodg;
        value.init.cpuInitu = &frontend_generated_source::cpuInitu;
        value.init.cpuInitq = &frontend_generated_source::cpuInitq;
        value.init.cpuInitudg = &frontend_generated_source::cpuInitudg;
        value.init.cpuInitwdg = &frontend_generated_source::cpuInitwdg;
        value.init.cpuInitodg = &frontend_generated_source::cpuInitodg;

        value.hdgjac.HdgFlux = &frontend_generated_source::HdgFlux;
        value.hdgjac.HdgSource = &frontend_generated_source::HdgSource;
        value.hdgjac.HdgSourcew = &frontend_generated_source::HdgSourcew;
        value.hdgjac.HdgSourcewonly = &frontend_generated_source::HdgSourcewonly;
        value.hdgjac.HdgEoS = &frontend_generated_source::HdgEoS;
        value.hdgjac.HdgFbou = &frontend_generated_source::HdgFbou;
        value.hdgjac.HdgFbouonly = &frontend_generated_source::HdgFbouonly;
        value.hdgjac.HdgFint = &frontend_generated_source::HdgFint;
        value.hdgjac.HdgFintonly = &frontend_generated_source::HdgFintonly;
        value.hdgjac.HdgFext = &frontend_generated_source::HdgFext;
        value.hdgjac.HdgFextonly = &frontend_generated_source::HdgFextonly;

        return value;
    }();

    return abi;
}
