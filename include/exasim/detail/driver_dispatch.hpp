// SPDX-License-Identifier: see LICENSE
//
// <exasim/detail/driver_dispatch.hpp>
//
// EXASIM_DRIVER_CALL(Name, args...) — used inside the templated FEM
// classes (CDiscretization<M> & friends) and the templated chain
// free functions (residual.hpp, uequation.hpp, …) to dispatch driver
// calls based on the user Model type M:
//
//   - If `M == exasim::detail::AbiAdapter` (legacy build): call the
//     global non-templated `::Name(args...)`, which is defined by
//     backend/Model/{KokkosDrivers,ModelDrivers}.cpp and ultimately
//     dispatches through the libpdemodel.hpp ABI symbols.
//
//   - Otherwise (user code instantiating CSolution<MyModel>): call
//     `::exasim::Name<M>(args...)`, which is defined in
//     <exasim/drivers.hpp> and routes through the templated kernels
//     in <exasim/kernels/*.hpp>, ultimately invoking the user's
//     pointwise math (`M::flux`, `M::flux_jac_uq`, …).
//
// The macro must be expanded inside a context where `M` is in scope
// (a `template <class M>` class member or free function). `if constexpr`
// requires C++17.
//
// We use a macro rather than a wrapper function template because the
// driver argument lists are heterogeneous (mesh/master/app refs +
// scalars + variable counts) and there are many overload sets where
// argument-dependent template deduction would be brittle. The macro
// is a thin pass-through.

#pragma once

#include <type_traits>

#include "abi_adapter.hpp"

// Precision cut (Phase 3, stance "A"): the AbiAdapter branch dispatches to the frontend-generated
// kernels, which are hard-typed `dstype` behind the ExasimDriverABI function pointers. So that path
// requires the caller's scalar type == the global dstype. `dstype` below resolves to whatever it
// means at the expansion site: today (kernels not yet precision-templated) it is the global ::dstype,
// so the assert is trivially true and the build is byte-identical. Once a kernel is templated with a
// `using dstype = T;` shadow (rest of Phase 3), the SAME assert automatically becomes the T==dstype
// guard -- a clear diagnostic instead of a raw `T* -> dstype*` type error, and the documented hook to
// later swap in a conversion shim / Phase-4 templated dispatch (see docs/internals/precision-threading.md).
#define EXASIM_DRIVER_CALL(Name, ...)                                      \
    do {                                                                   \
        if constexpr (std::is_same_v<M, exasim::detail::AbiAdapter>) {     \
            static_assert(std::is_same_v<dstype, ::dstype>,                \
                "frontend-generated (AbiAdapter) kernels are dstype-only; " \
                "instantiate a concrete Model M for T != dstype");         \
            Name(__VA_ARGS__);                                             \
        } else {                                                           \
            exasim::Name<M>(__VA_ARGS__);                                  \
        }                                                                  \
    } while (0)

// HDG w-equation source dispatch (backend/Discretization/wequation.hpp).
// `Name` is the ABI symbol under `driver_abi->hdgjac` (HdgSourcew /
// HdgSourcewonly). For the AbiAdapter build we call the frontend-generated
// ABI function pointer; for a real Model `M` we call the templated
// `exasim::Name<M>` forwarder in <exasim/kernels/sourcew.hpp>, which routes
// through `hdg_sourcew_kernel<M>` / `hdg_sourcewonly_kernel<M>` and compiles
// to a no-op when `M::ncw == 0` (the whole HDG w-equation is only reached for
// ncw>0 models). The discarded branch must still parse: `driver_abi->hdgjac`
// exists on commonstruct, and `exasim::Name<M>` is a dependent call not
// instantiated for AbiAdapter.
#define EXASIM_LEGACY_W_CALL(Name, ...)                                    \
    do {                                                                   \
        if constexpr (std::is_same_v<M, exasim::detail::AbiAdapter>) {     \
            common.driver_abi->hdgjac.Name(__VA_ARGS__);                   \
        } else {                                                           \
            exasim::Name<M>(__VA_ARGS__);                                  \
        }                                                                  \
    } while (0)
