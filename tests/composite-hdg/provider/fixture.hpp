#pragma once
#include <composite_hdg_model.hpp>

inline void fixtureJac(dstype*, dstype*, dstype*, const dstype*, const dstype*,
                       const dstype*, const dstype*, const dstype*, const dstype*,
                       dstype, int, int, int, int, int, int, int, int) {}
inline void otherJac(dstype*, dstype*, dstype*, const dstype*, const dstype*,
                     const dstype*, const dstype*, const dstype*, const dstype*,
                     dstype, int, int, int, int, int, int, int, int) {}
inline void fixtureElement(dstype*, const dstype*, const dstype*, const dstype*,
                           const dstype*, const dstype*, const dstype*, dstype,
                           int, int, int, int, int, int, int, int) {}
inline ExasimDriverABI fixtureABI() {
    ExasimDriverABI abi;
    abi.abi_version = kExasimDriverABIVersion;
    abi.struct_size = sizeof(ExasimDriverABI);
    abi.ncu = 9;
    abi.ncw = 1;
    abi.hdgjac.HdgFlux = fixtureJac;
    abi.hdgjac.HdgSource = fixtureJac;
    abi.hdgjac.HdgSourcew = fixtureJac;
    return abi;
}
