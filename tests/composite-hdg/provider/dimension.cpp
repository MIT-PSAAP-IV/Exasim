#include "fixture.hpp"
#include <cassert>
#include <iostream>

#ifndef COMPOSITE_TEST_DIM
#error "Set COMPOSITE_TEST_DIM to the dimension of the registered model"
#endif

static ExasimDriverABI modelABI() {
    auto abi = fixtureABI();
    abi.ncu = 6 + COMPOSITE_TEST_DIM;
    return abi;
}

#if COMPOSITE_TEST_DIM == 2
EXASIM_DEFINE_COMPOSITE_CNS5AIR2D_PROVIDER(100, modelABI())
#elif COMPOSITE_TEST_DIM == 3
EXASIM_DEFINE_COMPOSITE_CNS5AIR3D_PROVIDER(100, modelABI())
#else
#error "Unsupported test dimension"
#endif

template<class F> static void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

int main() {
    using exasim::composite::requireModel;
    const auto abi = modelABI();
    const auto contract = requireModel(abi, 100, COMPOSITE_TEST_DIM);
    assert(contract.layout == (COMPOSITE_TEST_DIM == 2 ? 2u : 1u));
    assert(contract.sourcew == abi.hdgjac.HdgSourcew);
    rejects([&] { requireModel(abi, 101, COMPOSITE_TEST_DIM); });
    rejects([&] { requireModel(abi, 100, 5 - COMPOSITE_TEST_DIM); });
    auto changed = abi;
    changed.hdgjac.HdgSourcew = otherJac;
    rejects([&] { requireModel(changed, 100, COMPOSITE_TEST_DIM); });
    std::cout << "COMPOSITE_PROVIDER_DIMENSION_PASS nd=" << COMPOSITE_TEST_DIM << '\n';
}
