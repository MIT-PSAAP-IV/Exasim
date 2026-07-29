#ifndef __EXASIM_BOUNDSCHECK
#define __EXASIM_BOUNDSCHECK

// Rank-aware bounds checking for indices derived from the LOCAL MESH PARTITION.
//
// WHY THIS EXISTS
// ---------------
// A recurring and expensive class of bug in this code is an index taken from
// partition-dependent mesh data (face codes in `f`, boundary ids, element/face block
// indices) used to subscript an array sized from GLOBAL configuration (the boundary
// list, the QoI vectors, ...). The index is in range for some decompositions and out of
// range for others, so the code is correct at one rank count and reads out of bounds at
// another.
//
// Two properties make these very costly to diagnose:
//
//   1. The out-of-range read usually does not fault. It returns adjacent memory, which
//      then steers control flow (e.g. a zeroed "is this boundary curved" flag reads as
//      non-zero) and the program crashes LATER, somewhere unrelated -- typically as a
//      null dereference several call levels away, in code that is not at fault.
//
//   2. They are rank-count dependent, so they survive every test at the rank counts CI
//      happens to use, then appear the first time somebody runs at a different size.
//
// Real instances found in this codebase:
//   - project_dgnodes_onto_curved_boundaries: `f[k] > -1` bounded the index below but
//     never above, so `curvedboundary[f[k]]` read past a 4-element array on a mesh with
//     NO curved boundaries at all; the resulting bogus flag led to a NULL expression
//     string being handed to tinyexpr, and the crash appeared inside the expression
//     parser.
//   - a further null dereference in a generated boundary-flux kernel at high rank
//     counts, whose faulting frame likewise names innocent code.
//
// WHAT THIS GIVES YOU
// -------------------
// A check that names the RANK, the index, the bound, the array, and the source location
// -- because "index 7 >= bound 4" is only actionable if you also know it happened on
// rank 5 of 12 and nowhere else. It aborts at the point of the bad index instead of
// letting the program corrupt its own control flow and die somewhere innocent.
//
// Zero cost unless EXASIM_BOUNDS_CHECK is defined (cmake -DEXASIM_BOUNDS_CHECK=ON).
// Intended for CI and for debugging a rank-count-dependent failure, not for production.

#ifdef EXASIM_BOUNDS_CHECK

#include <cstdio>
#include <cstdlib>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

inline void exasim_index_fail(long idx, long n, const char* what,
                              const char* file, int line)
{
    int rank = -1, size = -1;
#ifdef HAVE_MPI
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
#endif
    std::fprintf(stderr,
        "\n[exasim] FATAL out-of-range index on rank %d of %d\n"
        "  array      : %s\n"
        "  index      : %ld\n"
        "  valid range: [0, %ld)\n"
        "  at         : %s:%d\n"
        "  This index came from the local mesh partition while the array is sized from\n"
        "  global configuration, so it is decomposition dependent: it is very likely in\n"
        "  range at other rank counts. Re-run at the SAME rank count to reproduce.\n\n",
        rank, size, what, idx, n, file, line);
    std::fflush(stderr);
#ifdef HAVE_MPI
    if (inited) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    std::abort();
}

// Check that 0 <= idx < n before subscripting.
#define EXASIM_CHECK_INDEX(idx, n, what)                                        \
    do {                                                                        \
        const long _ei = (long)(idx);                                           \
        const long _en = (long)(n);                                             \
        if (_ei < 0 || _ei >= _en)                                              \
            exasim_index_fail(_ei, _en, (what), __FILE__, __LINE__);            \
    } while (0)

// Check a pointer that is about to be dereferenced after an index lookup.
#define EXASIM_CHECK_PTR(p, what)                                               \
    do {                                                                        \
        if ((p) == nullptr)                                                     \
            exasim_index_fail(-1, -1, (what), __FILE__, __LINE__);              \
    } while (0)

#else   // EXASIM_BOUNDS_CHECK not defined -- compiles to nothing

#define EXASIM_CHECK_INDEX(idx, n, what) ((void)0)
#define EXASIM_CHECK_PTR(p, what)        ((void)0)

#endif  // EXASIM_BOUNDS_CHECK

#endif  // __EXASIM_BOUNDSCHECK
