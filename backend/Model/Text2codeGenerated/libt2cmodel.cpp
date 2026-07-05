#include <cmath>
#include <Kokkos_Core.hpp>
#include <driver_abi.hpp>
#include <modeldefaults.hpp>
#include "my_model.hpp"

#ifdef USE_FLOAT
typedef float dstype;
#else
typedef double dstype; //  double is default precision 
#endif

using namespace std;

namespace text2code_shared_lib {
using ::PdeModel;

#include "KokkosFlux.cpp"
#include "KokkosFhat.cpp"
#include "KokkosFbou.cpp"
#include "KokkosUbou.cpp"
#include "KokkosUhat.cpp"
#include "KokkosStab.cpp"
#include "KokkosSource.cpp"
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
#include "HdgSourcew.cpp"
#include "HdgSourcewonly.cpp"
#include "HdgEoS.cpp"
#include "KokkosVisScalars.cpp"
#include "KokkosVisVectors.cpp"
#include "KokkosVisTensors.cpp"
#include "KokkosQoIvolume.cpp"
#include "KokkosQoIboundary.cpp"

} // namespace text2code_shared_lib

extern "C" const ExasimDriverABI* GetText2CodeExasimDriverABI()
{
    static const ExasimDriverABI abi = [] {
        ExasimDriverABI value;
        value.abi_version = kExasimDriverABIVersion;
        value.struct_size = sizeof(ExasimDriverABI);

        value.volume.KokkosFlux = &text2code_shared_lib::KokkosFlux;
        value.volume.KokkosSource = &text2code_shared_lib::KokkosSource;
        value.volume.KokkosSourcew = &text2code_shared_lib::KokkosSourcew;
        value.volume.KokkosTdfunc = &text2code_shared_lib::KokkosTdfunc;
        value.volume.KokkosAvfield = &text2code_shared_lib::KokkosAvfield;
        value.eos.KokkosEoS = &text2code_shared_lib::KokkosEoS;
        value.eos.KokkosEoSdu = &text2code_shared_lib::KokkosEoSdu;
        value.eos.KokkosEoSdw = &text2code_shared_lib::KokkosEoSdw;
        value.boundary.KokkosFbou = &text2code_shared_lib::KokkosFbou;
        value.boundary.KokkosUbou = &text2code_shared_lib::KokkosUbou;
        value.iface.KokkosFhat = &text2code_shared_lib::KokkosFhat;
        value.iface.KokkosUhat = &text2code_shared_lib::KokkosUhat;
        value.boundary.KokkosFbouJac = &text2code_shared_lib::KokkosFbouJac;
        value.boundary.KokkosUbouJac = &text2code_shared_lib::KokkosUbouJac;
        value.iface.KokkosStab = &text2code_shared_lib::KokkosStab;
        value.output.KokkosOutput = &text2code_shared_lib::KokkosOutput;
        value.output.KokkosMonitor = &text2code_shared_lib::KokkosMonitor;
        value.output.KokkosVisScalars = &text2code_shared_lib::KokkosVisScalars;
        value.output.KokkosVisVectors = &text2code_shared_lib::KokkosVisVectors;
        value.output.KokkosVisTensors = &text2code_shared_lib::KokkosVisTensors;
        value.qoi.KokkosQoIvolume = &text2code_shared_lib::KokkosQoIvolume;
        value.qoi.KokkosQoIboundary = &text2code_shared_lib::KokkosQoIboundary;

        value.init.KokkosInitu = &text2code_shared_lib::KokkosInitu;
        value.init.KokkosInitq = &text2code_shared_lib::KokkosInitq;
        value.init.KokkosInitudg = &text2code_shared_lib::KokkosInitudg;
        value.init.KokkosInitwdg = &text2code_shared_lib::KokkosInitwdg;
        value.init.KokkosInitodg = &text2code_shared_lib::KokkosInitodg;
        value.init.cpuInitu = &text2code_shared_lib::cpuInitu;
        value.init.cpuInitq = &text2code_shared_lib::cpuInitq;
        value.init.cpuInitudg = &text2code_shared_lib::cpuInitudg;
        value.init.cpuInitwdg = &text2code_shared_lib::cpuInitwdg;
        value.init.cpuInitodg = &text2code_shared_lib::cpuInitodg;

        value.hdgjac.HdgFlux = &text2code_shared_lib::HdgFlux;
        value.hdgjac.HdgSource = &text2code_shared_lib::HdgSource;
        value.hdgjac.HdgSourcew = &text2code_shared_lib::HdgSourcew;
        value.hdgjac.HdgSourcewonly = &text2code_shared_lib::HdgSourcewonly;
        value.hdgjac.HdgEoS = &text2code_shared_lib::HdgEoS;
        value.hdgjac.HdgFbou = &text2code_shared_lib::HdgFbou;
        value.hdgjac.HdgFbouonly = &text2code_shared_lib::HdgFbouonly;
        value.hdgjac.HdgFint = &text2code_shared_lib::HdgFint;
        value.hdgjac.HdgFintonly = &text2code_shared_lib::HdgFintonly;
        value.hdgjac.HdgFext = &text2code_shared_lib::HdgFext;
        value.hdgjac.HdgFextonly = &text2code_shared_lib::HdgFextonly;

        value.ncu  = PdeModel::ncu;
        value.nco  = PdeModel::nco;
        value.ncw  = PdeModel::ncw;
        value.nsca = PdeModel::nsca;
        value.nvec = PdeModel::nvec;
        value.nten = PdeModel::nten;
        value.nsurf = PdeModel::nsurf;
        value.nvqoi = PdeModel::nvqoi;
        value.GetModelSizes = [](int) -> ModelSizes {
            return {PdeModel::ncu, PdeModel::nco, PdeModel::ncw,
                    PdeModel::nsca, PdeModel::nvec, PdeModel::nten,
                    PdeModel::nsurf, PdeModel::nvqoi};
        };

        return value;
    }();

    return &abi;
}
