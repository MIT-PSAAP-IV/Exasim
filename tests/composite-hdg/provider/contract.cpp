#include "fixture.hpp"
#include <cassert>
#include <iostream>

static const auto expected = fixtureABI();
static const exasim::composite::Cns5air3dProvider provider(100, expected);
static exasim::composite::ModelV1 overridden;
static bool overrideQuery = false;
extern "C" const exasim::composite::ModelV1*
GetExasimCompositeHdgModelV1(const ExasimDriverABI* actual, int id) {
    return overrideQuery ? &overridden : provider.query(actual, id);
}

template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}

int main() {
    using namespace exasim::composite;
    auto good = requireModel(expected, 100);
    assert(good.modelnumber == 100 && good.sourcew == fixtureJac);
    assert(!provider.query(nullptr, 100));
    rejects([&] { requireModel(expected, 101); });
    for (int field = 0; field < 6; ++field) {
        auto changed = expected;
        if (field == 0) changed.hdgjac.HdgFlux = otherJac;
        if (field == 1) changed.hdgjac.HdgSource = otherJac;
        if (field == 2) changed.hdgjac.HdgSourcew = otherJac;
        if (field == 3) changed.hdgjac.HdgEoS = otherJac;
        if (field == 4) changed.volume.KokkosTdfunc = fixtureElement;
        if (field == 5) changed.volume.KokkosAvfield =
            [](dstype*, const dstype*, const dstype*, const dstype*, const dstype*,
               const dstype*, const dstype*, dstype, int, int, int, int, int,
               int, int, int, int, int, int) {};
        rejects([&] { requireModel(changed, 100); });
    }
    for (int fault = 0; fault < 6; ++fault) {
        auto bad = expected;
        if (fault == 0) --bad.abi_version;
        if (fault == 1) --bad.struct_size;
        if (fault == 2) bad.ncu = 8;
        if (fault == 3) bad.ncw = 2;
        if (fault == 4) bad.hdgjac.HdgSourcew = nullptr;
        if (fault == 5) bad.GetModelSizes = [](int) { ModelSizes s; s.ncu = 8; return s; };
        rejects([&] { Cns5air3dProvider invalid(100, bad); });
        rejects([&] { requireModel(bad, 100); });
    }
    overrideQuery = true;
    for (int fault = 0; fault < 6; ++fault) {
        overridden = good;
        if (fault == 0) ++overridden.version;
        if (fault == 1) --overridden.size;
        if (fault == 2) ++overridden.layout;
        if (fault == 3) ++overridden.modelnumber;
        if (fault == 4) overridden.sourcew = nullptr;
        if (fault == 5) overridden.sourcew = otherJac;
        rejects([&] { requireModel(expected, 100); });
    }
    std::cout << "COMPOSITE_PROVIDER_CONTRACT_PASS\n";
}
