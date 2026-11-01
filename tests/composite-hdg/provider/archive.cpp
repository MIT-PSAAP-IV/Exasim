#include "fixture.hpp"
const ExasimDriverABI& fixtureStrongGetter() {
    static const auto abi = fixtureABI();
    return abi;
}
EXASIM_DEFINE_COMPOSITE_CNS5AIR3D_PROVIDER(100, fixtureStrongGetter())
