#pragma once

#include "driver_abi.hpp"
#include <stdexcept>

namespace exasim::composite {

// This separate contract leaves DriverABI and existing model libraries unchanged.
// Layout 1 asserts CNS5air in three dimensions: N,O,NO,N2,O2,mx,my,mz,E;
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

inline bool compatibleSizes(const ExasimDriverABI& abi, int modelnumber) {
    const auto sizes = abi.GetModelSizes ? abi.GetModelSizes(modelnumber) : ModelSizes{};
    // Zero means unspecified in DriverABI. Runtime component checks still apply.
    return (!abi.ncu || abi.ncu == 9) && (!abi.ncw || abi.ncw == 1) &&
           (!sizes.ncu || sizes.ncu == 9) && (!sizes.ncw || sizes.ncw == 1);
}

// Use only after the model author has checked the semantic contract above.
// The explicit declaration, never the dimension test, grants this capability.
class Cns5air3dProvider {
    ExasimDriverABI expected_;
    ModelV1 model_;
public:
    Cns5air3dProvider(int modelnumber, const ExasimDriverABI& expected)
        : expected_(expected), model_{1, sizeof(ModelV1), 1, modelnumber,
                                     expected.hdgjac.HdgSourcew} {
        if (!validDriver(expected_) || !compatibleSizes(expected_, modelnumber))
            throw std::invalid_argument("compositehdg: incompatible CNS5air3D provider ABI");
    }

    const ModelV1* query(const ExasimDriverABI* actual, int modelnumber) const {
        return actual && modelnumber == model_.modelnumber &&
               samePhysics(expected_, *actual) && compatibleSizes(*actual, modelnumber)
            ? &model_ : nullptr;
    }
};
} // namespace exasim::composite

#if defined(__GNUC__) || defined(__clang__)
extern "C" const exasim::composite::ModelV1*
GetExasimCompositeHdgModelV1(const ExasimDriverABI*, int modelnumber) __attribute__((weak));
#endif

namespace exasim::composite {
inline ModelV1 requireModel(const ExasimDriverABI& abi, int modelnumber) {
#if defined(__GNUC__) || defined(__clang__)
    if (validDriver(abi) && compatibleSizes(abi, modelnumber) && GetExasimCompositeHdgModelV1) {
        const auto* model = GetExasimCompositeHdgModelV1(&abi, modelnumber);
        if (model && model->version == 1 && model->size == sizeof(ModelV1) &&
            model->layout == 1 && model->modelnumber == modelnumber &&
            model->sourcew && model->sourcew == abi.hdgjac.HdgSourcew)
            return *model;
    }
#endif
    throw std::invalid_argument(
        "compositehdg needs an explicitly registered CNS5air3D ModelV1 provider; "
        "see backend/Model/CompositeHDG.txt");
}
} // namespace exasim::composite

// Place this in the provider translation unit, alongside its strongly referenced
// ABI getter. A separate adapter must be linked as an object/direct source, not
// an otherwise unreferenced member of a static archive: weak references alone
// do not extract archive members. Exactly one query export is allowed per program.
#define EXASIM_DEFINE_COMPOSITE_CNS5AIR3D_PROVIDER(MODEL_NUMBER, ABI_EXPRESSION) \
    extern "C" const exasim::composite::ModelV1* \
    GetExasimCompositeHdgModelV1(const ExasimDriverABI* abi, int modelnumber) { \
        static const exasim::composite::Cns5air3dProvider provider( \
            (MODEL_NUMBER), (ABI_EXPRESSION)); \
        return provider.query(abi, modelnumber); \
    }
