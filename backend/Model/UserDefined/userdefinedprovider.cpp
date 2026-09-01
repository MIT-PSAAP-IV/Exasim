/**
 * @file provider.cpp
 * @brief Low-level UserDefined provider ABI export for Exasim.
 *
 * This translation unit intentionally exposes only model metadata and
 * low-level kernel entry points. It does not include backend-facing driver
 * wrappers such as FluxDriver/SourceDriver, so it stays independent of
 * mesh/master/sol/temp/common runtime state. Higher-level adapters in the
 * reusable core library can dispatch through this ABI later.
 */

#include "my_model.hpp"

namespace user_defined_source {

using ::PdeModel;

#include "KokkosFlux.cpp"
#include "KokkosFhat.cpp"
#include "KokkosFbou.cpp"
#include "KokkosUbou.cpp"
#include "KokkosFbouJac.cpp"
#include "KokkosUbouJac.cpp"
#include "KokkosUhat.cpp"
#include "KokkosStab.cpp"
#include "KokkosSource.cpp"
#include "KokkosMaterialstate.cpp"
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
#include "HdgMaterialstate.cpp"
#include "HdgSourcew.cpp"
#include "HdgSourcewonly.cpp"
#include "HdgFbou.cpp"
#include "HdgFbouonly.cpp"
#include "HdgFint.cpp"
#include "HdgFintonly.cpp"
#include "HdgFext.cpp"
#include "HdgFextonly.cpp"
#include "HdgEoS.cpp"

}

using ::PdeModel;

const ExasimDriverABI& getUserDefinedExasimDriverABI()
{
    static const ExasimDriverABI abi = [] {
        ExasimDriverABI value;
        value.abi_version = kExasimDriverABIVersion;
        value.struct_size = sizeof(ExasimDriverABI);

        value.volume.KokkosFlux = &user_defined_source::KokkosFlux;
        value.volume.KokkosSource = &user_defined_source::KokkosSource;
        value.volume.KokkosSourcew = &user_defined_source::KokkosSourcew;
        value.volume.KokkosMaterialstate = &user_defined_source::KokkosMaterialstate;
        value.volume.KokkosTdfunc = &user_defined_source::KokkosTdfunc;
        value.volume.KokkosAvfield = &user_defined_source::KokkosAvfield;
        value.eos.KokkosEoS = &user_defined_source::KokkosEoS;
        value.eos.KokkosEoSdu = &user_defined_source::KokkosEoSdu;
        value.eos.KokkosEoSdw = &user_defined_source::KokkosEoSdw;
        value.boundary.KokkosFbou = &user_defined_source::KokkosFbou;
        value.boundary.KokkosUbou = &user_defined_source::KokkosUbou;
        value.boundary.KokkosFbouJac = &user_defined_source::KokkosFbouJac;
        value.boundary.KokkosUbouJac = &user_defined_source::KokkosUbouJac;
        value.iface.KokkosFhat = &user_defined_source::KokkosFhat;
        value.iface.KokkosUhat = &user_defined_source::KokkosUhat;
        value.iface.KokkosStab = &user_defined_source::KokkosStab;
        value.output.KokkosOutput = &user_defined_source::KokkosOutput;
        value.output.KokkosMonitor = &user_defined_source::KokkosMonitor;
        value.output.KokkosVisScalars = &user_defined_source::KokkosVisScalars;
        value.output.KokkosVisVectors = &user_defined_source::KokkosVisVectors;
        value.output.KokkosVisTensors = &user_defined_source::KokkosVisTensors;
        value.qoi.KokkosQoIvolume = &user_defined_source::KokkosQoIvolume;
        value.qoi.KokkosQoIboundary = &user_defined_source::KokkosQoIboundary;

        value.init.KokkosInitu = &user_defined_source::KokkosInitu;
        value.init.KokkosInitq = &user_defined_source::KokkosInitq;
        value.init.KokkosInitudg = &user_defined_source::KokkosInitudg;
        value.init.KokkosInitwdg = &user_defined_source::KokkosInitwdg;
        value.init.KokkosInitodg = &user_defined_source::KokkosInitodg;
        value.init.cpuInitu = &user_defined_source::cpuInitu;
        value.init.cpuInitq = &user_defined_source::cpuInitq;
        value.init.cpuInitudg = &user_defined_source::cpuInitudg;
        value.init.cpuInitwdg = &user_defined_source::cpuInitwdg;
        value.init.cpuInitodg = &user_defined_source::cpuInitodg;

        value.hdgjac.HdgFlux = &user_defined_source::HdgFlux;
        value.hdgjac.HdgSource = &user_defined_source::HdgSource;
        value.hdgjac.HdgMaterialstate = &user_defined_source::HdgMaterialstate;
        value.hdgjac.HdgSourcew = &user_defined_source::HdgSourcew;
        value.hdgjac.HdgSourcewonly = &user_defined_source::HdgSourcewonly;
        value.hdgjac.HdgEoS = &user_defined_source::HdgEoS;
        value.hdgjac.HdgFbou = &user_defined_source::HdgFbou;
        value.hdgjac.HdgFbouonly = &user_defined_source::HdgFbouonly;
        value.hdgjac.HdgFint = &user_defined_source::HdgFint;
        value.hdgjac.HdgFintonly = &user_defined_source::HdgFintonly;
        value.hdgjac.HdgFext = &user_defined_source::HdgFext;
        value.hdgjac.HdgFextonly = &user_defined_source::HdgFextonly;

        value.ncu  = PdeModel::ncu;
        value.nco  = PdeModel::nco;
        value.ncw  = PdeModel::ncw;
        value.nsca = PdeModel::nsca;
        value.nvec = PdeModel::nvec;
        value.nten = PdeModel::nten;
        value.nsurf = PdeModel::nsurf;
        value.nvqoi = PdeModel::nvqoi;
        value.nmaterialstate = PdeModel::nmaterialstate;
        value.GetModelSizes = [](int) -> ModelSizes {
            return {PdeModel::ncu, PdeModel::nco, PdeModel::ncw,
                    PdeModel::nsca, PdeModel::nvec, PdeModel::nten,
                    PdeModel::nsurf, PdeModel::nvqoi,
                    PdeModel::nmaterialstate};
        };

        return value;
    }();

    return abi;
}
