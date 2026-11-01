#include "fixture.hpp"
#include <iostream>
const ExasimDriverABI& fixtureStrongGetter();
int main() {
    const auto& abi = fixtureStrongGetter(); // Forces the provider archive member in.
    auto model = exasim::composite::requireModel(abi, 100);
    if (model.modelnumber != 100) return 1;
    std::cout << "COMPOSITE_PROVIDER_ARCHIVE_PASS\n";
}
