// [nonmatching] Phase 9 Part B: direct test of text2code initializeDMD (single rank).
//
// mesh.inte holds one entry per coupled-interface face (interface_elements), so an
// element with two coupled faces appears twice. initializeDMD must use the unique
// interface elements and build elempart as an exact permutation of 0..ne-1: interior
// elements first, then interface elements, both increasing, with elempartpts =
// {ne-n, n, 0} and intepartpts = {ne-n, n, 0, 0} (n unique interface elements).
// Uncoupled models keep the identity with elempartpts = {ne}.
//
// Build: the same include chain as text2code.cpp up to domaindecomposition.cpp.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <string>
#include <vector>

using namespace std;

#ifndef EXASIM_INSTALL_PREFIX
#  define EXASIM_INSTALL_PREFIX ""  // as in text2code.cpp
#endif

#include "TextParser.hpp"
#include "tinyexpr.cpp"
#include "helpers.cpp"
#include "readpdeapp.cpp"
#include "readmesh.cpp"
#include "makemesh.cpp"
#include "makemaster.cpp"
#include "domaindecomposition.cpp"

static int failures = 0;

static void check(bool ok, const std::string& what)
{
    if (!ok) { failures++; std::printf("FAIL %s\n", what.c_str()); }
}

static void run_case(const std::string& name, int ne, int coupled, const std::vector<int>& inte)
{
    PDE pde;
    pde.mpiprocs = 1;
    pde.coupledinterface = coupled;
    Mesh mesh;
    mesh.ne = ne;
    mesh.inte = inte;
    DMD dmd = initializeDMD(pde, mesh);

    std::vector<int> unique = inte;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    const int n = coupled > 0 ? static_cast<int>(unique.size()) : 0;

    // Exact permutation of 0..ne-1.
    check(static_cast<int>(dmd.elempart.size()) == ne, name + ": elempart has ne entries");
    std::vector<int> sorted = dmd.elempart;
    std::sort(sorted.begin(), sorted.end());
    bool perm = static_cast<int>(sorted.size()) == ne;
    for (int i = 0; perm && i < ne; i++) perm = sorted[i] == i;
    check(perm, name + ": elempart is a permutation of 0..ne-1");

    // Expected order: interior ascending, then unique interface ascending.
    std::vector<int> expected;
    for (int e = 0; e < ne; e++)
        if (coupled == 0 || !std::binary_search(unique.begin(), unique.end(), e)) expected.push_back(e);
    if (coupled > 0) expected.insert(expected.end(), unique.begin(), unique.end());
    check(dmd.elempart == expected, name + ": interior first, then interface, both ascending");

    if (coupled > 0) {
        check(dmd.elempartpts == std::vector<int>({ne - n, n, 0}), name + ": elempartpts uses the unique count");
        check(dmd.intepartpts == std::vector<int>({ne - n, n, 0, 0}), name + ": intepartpts uses the unique count");
    }
    else {
        check(dmd.elempartpts == std::vector<int>({ne}), name + ": uncoupled elempartpts");
        check(dmd.intepartpts == std::vector<int>({ne, 0, 0, 0}), name + ": uncoupled intepartpts");
    }
    std::printf("%s: ne=%d, %zu inte entries, %d unique interface elements\n", name.c_str(), ne, inte.size(), n);
}

int main()
{
    run_case("uncoupled", 6, 0, {});
    run_case("coupled, no interface elements", 6, 1, {});
    run_case("coupled, one interface face per element", 8, 1, {1, 3, 5, 7});
    run_case("coupled, unsorted interface elements", 8, 1, {7, 1, 5, 3});
    run_case("coupled, element with two coupled faces", 8, 1, {3, 5, 7, 7});
    run_case("coupled, every element on the interface", 4, 1, {2, 0, 3, 1, 1});
    std::printf("%s text2code_initializedmd\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
