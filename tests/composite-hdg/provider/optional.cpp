#include "fixture.hpp"
#include <iostream>
#ifndef EXPECT_REGISTERED
#define EXPECT_REGISTERED 0
#endif
int main() {
    bool found = false;
    try {
        auto model = exasim::composite::requireModel(fixtureABI(), 100);
        found = model.modelnumber == 100;
    } catch (const std::invalid_argument&) {}
    if (found != bool(EXPECT_REGISTERED)) {
        std::cerr << "Provider registration mismatch: expected=" << bool(EXPECT_REGISTERED)
                  << " actual=" << found << '\n';
        return 1;
    }
    std::cout << "COMPOSITE_PROVIDER_OPTIONAL_PASS registered=" << found << '\n';
}
