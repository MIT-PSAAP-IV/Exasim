#pragma once

#include "driver_abi.hpp"
#include <stdexcept>

namespace exasim::composite {

// This separate contract leaves DriverABI and existing model libraries unchanged.
// Layout 1 asserts CNS5air3D: N,O,NO,N2,O2,mx,my,mz,E;
// layout 2 asserts CNS5air2D: N,O,NO,N2,O2,mx,my,E;
// w is temperature and N2 is the dependent species in the composite basis.
// HdgSourcew is the local temperature residual. Its thermodynamic derivatives
// must remain valid for zero x, q and other-data inputs used by that basis.
struct ModelV1 {
    unsigned version = 1;
    unsigned size = sizeof(ModelV1);
    unsigned layout = 1;
    int modelnumber = 0;
    ExasimDriverABI::HdgElementJacFn sourcew = nullptr;
};

inline bool validDriver(const ExasimDriverABI& abi) {
    return abi.abi_version == kExasimDriverABIVersion &&
           abi.struct_size == sizeof(ExasimDriverABI) &&
           abi.hdgjac.HdgFlux && abi.hdgjac.HdgSource && abi.hdgjac.HdgSourcew;
}

// Bind the declaration to the actual model implementation, including callbacks
// used for residual-only evaluation, time terms, boundary terms and EOS updates.
// Output/initialization callbacks may differ without changing this physics.
inline bool samePhysics(const ExasimDriverABI& a, const ExasimDriverABI& b) {
    return validDriver(a) && validDriver(b) &&
           a.volume.KokkosFlux == b.volume.KokkosFlux &&
           a.volume.KokkosSource == b.volume.KokkosSource &&
           a.volume.KokkosSourcew == b.volume.KokkosSourcew &&
           a.volume.KokkosTdfunc == b.volume.KokkosTdfunc &&
           a.volume.KokkosAvfield == b.volume.KokkosAvfield &&
           a.volume.KokkosMaterialstate == b.volume.KokkosMaterialstate &&
           a.eos.KokkosEoS == b.eos.KokkosEoS &&
           a.eos.KokkosEoSdu == b.eos.KokkosEoSdu &&
           a.eos.KokkosEoSdw == b.eos.KokkosEoSdw &&
           a.hdgjac.HdgFlux == b.hdgjac.HdgFlux &&
           a.hdgjac.HdgSource == b.hdgjac.HdgSource &&
           a.hdgjac.HdgSourcew == b.hdgjac.HdgSourcew &&
           a.hdgjac.HdgSourcewonly == b.hdgjac.HdgSourcewonly &&
           a.hdgjac.HdgEoS == b.hdgjac.HdgEoS &&
           a.hdgjac.HdgMaterialstate == b.hdgjac.HdgMaterialstate &&
           a.hdgjac.HdgFbou == b.hdgjac.HdgFbou &&
           a.hdgjac.HdgFbouonly == b.hdgjac.HdgFbouonly &&
           a.hdgjac.HdgFint == b.hdgjac.HdgFint &&
           a.hdgjac.HdgFintonly == b.hdgjac.HdgFintonly &&
           a.hdgjac.HdgFext == b.hdgjac.HdgFext &&
           a.hdgjac.HdgFextonly == b.hdgjac.HdgFextonly;
}

inline bool compatibleSizes(const ExasimDriverABI& abi, int modelnumber, int nd=3) {
    if(nd!=2 && nd!=3)return false;
    const auto sizes = abi.GetModelSizes ? abi.GetModelSizes(modelnumber) : ModelSizes{};
    // Zero means unspecified in DriverABI. Runtime component checks still apply.
    return (!abi.ncu || abi.ncu == 6+nd) && (!abi.ncw || abi.ncw == 1) &&
           (!sizes.ncu || sizes.ncu == 6+nd) && (!sizes.ncw || sizes.ncw == 1);
}

// Use only after the model author has checked the semantic contract above.
// The explicit declaration, never the dimension test, grants this capability.
template<int ND> class Cns5airProvider {
    static_assert(ND==2 || ND==3);
    ExasimDriverABI expected_;
    ModelV1 model_;
public:
    Cns5airProvider(int modelnumber, const ExasimDriverABI& expected)
        : expected_(expected), model_{1, sizeof(ModelV1), ND==3?1u:2u, modelnumber,
                                     expected.hdgjac.HdgSourcew} {
        if (!validDriver(expected_) || !compatibleSizes(expected_, modelnumber, ND))
            throw std::invalid_argument("compositehdg: incompatible dimension-matched CNS5air provider ABI");
    }

    const ModelV1* query(const ExasimDriverABI* actual, int modelnumber) const {
        return actual && modelnumber == model_.modelnumber &&
               samePhysics(expected_, *actual) && compatibleSizes(*actual, modelnumber, ND)
            ? &model_ : nullptr;
    }
};
using Cns5air3dProvider=Cns5airProvider<3>;
using Cns5air2dProvider=Cns5airProvider<2>;
} // namespace exasim::composite

#if defined(__APPLE__)
// Mach-O requires weak_import and -U,_GetExasimCompositeHdgModelV1 at link time.
// Exasim's CMake targets propagate that symbol-specific option to consumers.
extern "C" const exasim::composite::ModelV1*
GetExasimCompositeHdgModelV1(const ExasimDriverABI*, int modelnumber) __attribute__((weak_import));
#elif defined(__GNUC__) || defined(__clang__)
extern "C" const exasim::composite::ModelV1*
GetExasimCompositeHdgModelV1(const ExasimDriverABI*, int modelnumber) __attribute__((weak));
#endif

namespace exasim::composite {
inline ModelV1 requireModel(const ExasimDriverABI& abi, int modelnumber, int nd=3) {
#if defined(__GNUC__) || defined(__clang__)
    if (validDriver(abi) && compatibleSizes(abi, modelnumber, nd) && GetExasimCompositeHdgModelV1) {
        const auto* model = GetExasimCompositeHdgModelV1(&abi, modelnumber);
        if (model && model->version == 1 && model->size == sizeof(ModelV1) &&
            model->layout == (nd==3?1u:2u) && model->modelnumber == modelnumber &&
            model->sourcew && model->sourcew == abi.hdgjac.HdgSourcew)
            return *model;
    }
#endif
    throw std::invalid_argument(
        "compositehdg needs an explicitly registered dimension-matched CNS5air ModelV1 provider; "
        "see backend/Model/CompositeHDG.txt");
}
} // namespace exasim::composite

// Place this in the provider translation unit, alongside its strongly referenced
// ABI getter. A separate adapter must be linked as an object/direct source, not
// an otherwise unreferenced member of a static archive: weak references alone
// do not portably extract archive members. Exactly one query export per program.
#define EXASIM_DEFINE_COMPOSITE_CNS5AIR3D_PROVIDER(MODEL_NUMBER, ABI_EXPRESSION) \
    extern "C" const exasim::composite::ModelV1* \
    GetExasimCompositeHdgModelV1(const ExasimDriverABI* abi, int modelnumber) { \
        static const exasim::composite::Cns5air3dProvider provider( \
            (MODEL_NUMBER), (ABI_EXPRESSION)); \
        return provider.query(abi, modelnumber); \
    }

#define EXASIM_DEFINE_COMPOSITE_CNS5AIR2D_PROVIDER(MODEL_NUMBER, ABI_EXPRESSION) \
    extern "C" const exasim::composite::ModelV1* \
    GetExasimCompositeHdgModelV1(const ExasimDriverABI* abi, int modelnumber) { \
        static const exasim::composite::Cns5air2dProvider provider( \
            (MODEL_NUMBER), (ABI_EXPRESSION)); \
        return provider.query(abi, modelnumber); \
    }
