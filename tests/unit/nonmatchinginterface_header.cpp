// Phase 0 of the non-matching interface solver: the frozen data-structure header
// (backend/Discretization/nonmatchinginterface.hpp) compiles when included after
// common.h, and every field of a default-constructed instance is 0 / nullptr
// (direction is -1 = unset), for both directions. No solver code is involved.
//
// Registered in tests/CMakeLists.txt as unit_nonmatchinginterface_header, with the
// same include dirs, defines and libraries as unit_struct_defaults.

#ifdef HAVE_MPI
#include <mpi.h>
MPI_Comm EXASIM_COMM_WORLD = MPI_COMM_NULL;
MPI_Comm EXASIM_COMM_LOCAL = MPI_COMM_NULL;
#endif
#include <Kokkos_Core.hpp>
#include "../../backend/Common/common.h"
#include "../../backend/Discretization/nonmatchinginterface.hpp"

#include <cstdio>
#include <type_traits>

static int failures = 0;

static void expect_eq(const char* field, long long got, long long want)
{
    if (got != want) {
        std::printf("  FAIL  %-28s expected %lld, got %lld\n", field, want, got);
        ++failures;
    }
}

static void expect_null(const char* field, const void* got)
{
    if (got != nullptr) {
        std::printf("  FAIL  %-28s expected nullptr, got %p\n", field, got);
        ++failures;
    }
}

template <class S>
static void check_defaults(const S& s)
{
#define EQ(f, v) expect_eq(#f, static_cast<long long>(s.f), v)
#define NUL(f) expect_null(#f, s.f)
    EQ(direction, -1); EQ(isowner, 0); EQ(isdonor, 0);
    EQ(nd, 0); EQ(npe, 0); EQ(npf, 0); EQ(nfe, 0); EQ(ngf, 0);
    EQ(ncu, 0); EQ(ncq, 0); EQ(ncu12, 0);
    EQ(nownfaces, 0); NUL(ownface); NUL(ownfaceglobal);
    EQ(nrecvnb, 0); NUL(recvnb); NUL(recvcount); NUL(recvface);
    EQ(npts, 0); NUL(pe); NUL(pa); NUL(pf); NUL(pslot); NUL(pg);
    NUL(px); NUL(pnl); NUL(pwJ);
    NUL(pxi); NUL(pzeta); NUL(pphi); NUL(pphibar); NUL(ppsi); NUL(pd); NUL(pdphibar);
    EQ(nslots, 0); NUL(slotrank); NUL(slotface); NUL(slotptr);
    EQ(npairs, 0); NUL(pairslot); NUL(pairelem); NUL(pairptr);
    EQ(nsendnb, 0); NUL(sendnb); NUL(sendcount);
    NUL(Ri); NUL(Hi); NUL(sendbuf); NUL(recvbuf);
    EQ(szRi, 0); EQ(szHi, 0); EQ(szsendbuf, 0); EQ(szrecvbuf, 0);
#undef EQ
#undef NUL
}

int main()
{
    static_assert(std::is_same<nonmatchinginterface::dstype, dstype>::value, "dstype");
    static_assert(std::is_same<nonmatchinginterface::Int, Int>::value, "Int");
    static_assert(std::is_same<decltype(nonmatchingdata{}.dir[0]), nonmatchinginterface&&>::value ||
                  std::is_same<std::remove_reference<decltype(nonmatchingdata{}.dir[0])>::type,
                               nonmatchinginterface>::value, "dir element type");
    static_assert(std::is_standard_layout<nonmatchinginterface>::value, "plain data");

    nonmatchingdata nm{};     // how the discretization will hold it (one per rank)
    expect_eq("interfacetype", nm.interfacetype, 0);
    for (int k = 0; k < 2; k++) {
        std::printf("direction slot %d\n", k);
        check_defaults(nm.dir[k]);
    }
    nonmatchinginterfaceT<float, int> nf{};   // other instantiation compiles too
    check_defaults(nf);

    if (failures == 0) std::printf("PASS nonmatchinginterface_header\n");
    else std::printf("FAIL nonmatchinginterface_header: %d field(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
