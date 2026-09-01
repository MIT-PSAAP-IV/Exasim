/**
 * @file provider.cpp
 * @brief Low-level Text2codeGenerated provider ABI export for Exasim.
 *
 * This translation unit exposes only generated low-level model kernels through
 * the provider ABI table. It intentionally avoids backend-facing driver
 * wrappers and therefore stays independent of mesh/master/sol/temp/common
 * runtime state.
 */

//#include "../ModelDispatch/driver_abi.h"

#include "my_model.hpp"
#include <exasim/kernels/materialstate.hpp>

namespace text2code_generated_source {
using ::PdeModel;

#include <cmath>
#include <Kokkos_Core.hpp>

#ifdef USE_FLOAT
typedef float dstype;
#else
typedef double dstype;
#endif

using namespace std;

#include "KokkosFlux.cpp"
#include "KokkosFhat.cpp"
#include "KokkosFbou.cpp"
#include "KokkosUbou.cpp"
#include "KokkosUhat.cpp"
#include "KokkosStab.cpp"
#include "KokkosSource.cpp"
#include "KokkosMaterialstate.cpp"
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
#include "HdgFbou.cpp"
#include "HdgFbouonly.cpp"
#include "HdgFint.cpp"
#include "HdgFintonly.cpp"
#include "HdgFext.cpp"
#include "HdgFextonly.cpp"
#include "HdgSource.cpp"
#include "HdgMaterialstate.cpp"
#include "HdgSourcew.cpp"
#include "HdgSourcewonly.cpp"
#include "HdgEoS.cpp"
#include "KokkosVisScalars.cpp"
#include "KokkosVisVectors.cpp"
#include "KokkosVisTensors.cpp"
#include "KokkosQoIvolume.cpp"
#include "KokkosQoIboundary.cpp"

} 

const ExasimDriverABI& getText2codeGeneratedExasimDriverABI()
{
    static const ExasimDriverABI abi = [] {
        ExasimDriverABI value;
        value.abi_version = kExasimDriverABIVersion;
        value.struct_size = sizeof(ExasimDriverABI);

        value.volume.KokkosFlux = &text2code_generated_source::KokkosFlux;
        value.volume.KokkosSource = &text2code_generated_source::KokkosSource;
        value.volume.KokkosSourcew = &text2code_generated_source::KokkosSourcew;
        value.volume.KokkosMaterialstate = &text2code_generated_source::KokkosMaterialstate;
        value.volume.KokkosTdfunc = &text2code_generated_source::KokkosTdfunc;
        value.volume.KokkosAvfield = &text2code_generated_source::KokkosAvfield;
        value.eos.KokkosEoS = &text2code_generated_source::KokkosEoS;
        value.eos.KokkosEoSdu = &text2code_generated_source::KokkosEoSdu;
        value.eos.KokkosEoSdw = &text2code_generated_source::KokkosEoSdw;
        value.boundary.KokkosFbou = &text2code_generated_source::KokkosFbou;
        value.boundary.KokkosUbou = &text2code_generated_source::KokkosUbou;
        value.boundary.KokkosFbouJac = &text2code_generated_source::KokkosFbouJac;
        value.boundary.KokkosUbouJac = &text2code_generated_source::KokkosUbouJac;
        value.iface.KokkosFhat = &text2code_generated_source::KokkosFhat;
        value.iface.KokkosUhat = &text2code_generated_source::KokkosUhat;
        value.iface.KokkosStab = &text2code_generated_source::KokkosStab;
        value.output.KokkosOutput = &text2code_generated_source::KokkosOutput;
        value.output.KokkosMonitor = &text2code_generated_source::KokkosMonitor;
        value.output.KokkosVisScalars = &text2code_generated_source::KokkosVisScalars;
        value.output.KokkosVisVectors = &text2code_generated_source::KokkosVisVectors;
        value.output.KokkosVisTensors = &text2code_generated_source::KokkosVisTensors;
        value.qoi.KokkosQoIvolume = &text2code_generated_source::KokkosQoIvolume;
        value.qoi.KokkosQoIboundary = &text2code_generated_source::KokkosQoIboundary;

        value.init.KokkosInitu = &text2code_generated_source::KokkosInitu;
        value.init.KokkosInitq = &text2code_generated_source::KokkosInitq;
        value.init.KokkosInitudg = &text2code_generated_source::KokkosInitudg;
        value.init.KokkosInitwdg = &text2code_generated_source::KokkosInitwdg;
        value.init.KokkosInitodg = &text2code_generated_source::KokkosInitodg;
        value.init.cpuInitu = &text2code_generated_source::cpuInitu;
        value.init.cpuInitq = &text2code_generated_source::cpuInitq;
        value.init.cpuInitudg = &text2code_generated_source::cpuInitudg;
        value.init.cpuInitwdg = &text2code_generated_source::cpuInitwdg;
        value.init.cpuInitodg = &text2code_generated_source::cpuInitodg;

        value.hdgjac.HdgFlux = &text2code_generated_source::HdgFlux;
        value.hdgjac.HdgSource = &text2code_generated_source::HdgSource;
        value.hdgjac.HdgMaterialstate = &text2code_generated_source::HdgMaterialstate;
        value.hdgjac.HdgSourcew = &text2code_generated_source::HdgSourcew;
        value.hdgjac.HdgSourcewonly = &text2code_generated_source::HdgSourcewonly;
        value.hdgjac.HdgEoS = &text2code_generated_source::HdgEoS;
        value.hdgjac.HdgFbou = &text2code_generated_source::HdgFbou;
        value.hdgjac.HdgFbouonly = &text2code_generated_source::HdgFbouonly;
        value.hdgjac.HdgFint = &text2code_generated_source::HdgFint;
        value.hdgjac.HdgFintonly = &text2code_generated_source::HdgFintonly;
        value.hdgjac.HdgFext = &text2code_generated_source::HdgFext;
        value.hdgjac.HdgFextonly = &text2code_generated_source::HdgFextonly;

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
