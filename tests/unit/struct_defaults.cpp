// Every field of commonstruct that the SERIAL init path does not assign must default to a
// defined value.
//
// Why this test exists
// --------------------
// commonstruct is a plain default-initialized member (discretization.h:76). A field with no
// default member initializer is INDETERMINATE, not zero -- and several are assigned only on
// the `mpiProcs > 1` path in setstructs.cpp while being read unconditionally:
//
//   meshsizes.nbf0            a loop bound in residual.hpp (RuFace(..., 0, nbf0, ...))
//   nnbintf / nfacesend       gate the coupling exchange in matvec.hpp
//   couplingparams.ncie       sizes res.szRi/szKi/szHi (discretization.cpp:558-560)
//                             and res.szGi (residualeval.cpp:274)
//
// Note `mpiProcs` is the size of the EXASIM group, not of MPI_COMM_WORLD, so the serial path
// is reached in MPI builds: a coupled app that splits COMM_WORLD and runs at np=2 hands
// Exasim a one-rank group and lands there with coupledinterface > 0.
//
// Reading as zero on a fresh heap page is allocator luck, not an invariant. `ncie` in
// particular was missed when the other fields were fixed, and was caught in review rather
// than by a test -- which is why this exists. A new field added without an initializer, or
// an initializer dropped in a refactor, fails here rather than three releases later as a
// wrong allocation size.
//
// This deliberately checks the DEFAULT-CONSTRUCTED value only. What the serial branch of
// setstructs.cpp computes (ncie must be COUNTED from eblks, not merely zeroed -- a one-rank
// coupled group can still own -1-tagged interface blocks) is covered by the app-level runs.

// mpi.h FIRST when HAVE_MPI: common.h uses MPI_Datatype/MPI_DOUBLE under that guard but
// does not include <mpi.h> itself -- it relies on the including TU having done so.
#ifdef HAVE_MPI
#include <mpi.h>
// common.h USES EXASIM_COMM_WORLD (PrintErrorAndExit) but declares it nowhere -- it relies
// on the including TU having defined it first. Every in-tree consumer happens to, so the
// dependency is invisible until something includes common.h directly, as this test does.
// Defining them here is exactly what a direct CSolution consumer must do; see the guard
// this PR adds in setcommonstruct. Left as MPI_COMM_NULL because nothing here communicates.
MPI_Comm EXASIM_COMM_WORLD = MPI_COMM_NULL;
MPI_Comm EXASIM_COMM_LOCAL = MPI_COMM_NULL;
#endif
#include <Kokkos_Core.hpp>
#include "../../backend/Common/common.h"

#include <cstdio>

static int failures = 0;

static void expect_zero(const char* field, long long got)
{
    if (got != 0) {
        std::printf("  FAIL  %-42s expected 0, got %lld\n", field, got);
        ++failures;
    }
}

static void expect_eq(const char* field, long long got, long long want)
{
    if (got != want) {
        std::printf("  FAIL  %-42s expected %lld, got %lld\n", field, want, got);
        ++failures;
    }
}

static void expect_null(const char* field, const void* got)
{
    if (got != nullptr) {
        std::printf("  FAIL  %-42s expected nullptr, got %p\n", field, got);
        ++failures;
    }
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        commonstruct c{};   // exactly how discretization.h holds it

        std::printf("commonstruct default-initialization\n");

        // --- fields the serial branch of setstructs.cpp relies on -------------------
        expect_zero("meshsizes.nbf0",            (long long) c.meshsizes.nbf0);
        expect_zero("meshsizes.nbf1",            (long long) c.meshsizes.nbf1);
        expect_zero("nnbintf",                   (long long) c.nnbintf);
        expect_zero("nfacesend",                 (long long) c.nfacesend);
        expect_zero("nfacerecv",                 (long long) c.nfacerecv);
        expect_zero("couplingparams.ncie",       (long long) c.couplingparams.ncie);

        // --- the rest of meshsizesstruct -------------------------------------------
        expect_zero("meshsizes.maxnbc",          (long long) c.meshsizes.maxnbc);
        expect_zero("meshsizes.ne",              (long long) c.meshsizes.ne);
        expect_zero("meshsizes.nf",              (long long) c.meshsizes.nf);
        expect_zero("meshsizes.nv",              (long long) c.meshsizes.nv);
        expect_zero("meshsizes.nfe",             (long long) c.meshsizes.nfe);
        expect_zero("meshsizes.nbe",             (long long) c.meshsizes.nbe);
        expect_zero("meshsizes.neb",             (long long) c.meshsizes.neb);
        expect_zero("meshsizes.nbf",             (long long) c.meshsizes.nbf);
        expect_zero("meshsizes.nfb",             (long long) c.meshsizes.nfb);
        expect_zero("meshsizes.nbe0",            (long long) c.meshsizes.nbe0);
        expect_zero("meshsizes.nbe1",            (long long) c.meshsizes.nbe1);
        expect_zero("meshsizes.nbe2",            (long long) c.meshsizes.nbe2);
        expect_zero("meshsizes.ne0",             (long long) c.meshsizes.ne0);
        expect_zero("meshsizes.ne1",             (long long) c.meshsizes.ne1);
        expect_zero("meshsizes.ne2",             (long long) c.meshsizes.ne2);
        expect_zero("meshsizes.nf0",             (long long) c.meshsizes.nf0);

        // --- commonstruct scalars ---------------------------------------------------
        expect_zero("backend",                   (long long) c.backend);
        expect_zero("mpiRank",                   (long long) c.mpiRank);
        expect_zero("nomodels",                  (long long) c.nomodels);
        expect_zero("nnbsd",                     (long long) c.nnbsd);
        expect_zero("nelemsend",                 (long long) c.nelemsend);
        expect_zero("nelemrecv",                 (long long) c.nelemrecv);
        expect_zero("szinterfacefluxmap",        (long long) c.szinterfacefluxmap);
        expect_zero("szcartgridpart",            (long long) c.szcartgridpart);

        // mpiProcs defaults to 1, not 0: "one rank" is the meaningful serial default, and a
        // zero here would make `mpiProcs > 1` false for the wrong reason.
        expect_eq  ("mpiProcs",                  (long long) c.mpiProcs, 1);

        // --- couplingparamsstruct ---------------------------------------------------
        expect_zero("couplingparams.ncuext",     (long long) c.couplingparams.ncuext);
        expect_zero("couplingparams.coupledinterface",
                                                 (long long) c.couplingparams.coupledinterface);
        expect_zero("couplingparams.coupledcondition",
                                                 (long long) c.couplingparams.coupledcondition);
        expect_zero("couplingparams.coupledboundarycondition",
                                                 (long long) c.couplingparams.coupledboundarycondition);
        expect_zero("couplingparams.nintfaces",  (long long) c.couplingparams.nintfaces);
        expect_zero("couplingparams.nvindx",     (long long) c.couplingparams.nvindx);
        expect_zero("couplingparams.ndofuhatinterface",
                                                 (long long) c.couplingparams.ndofuhatinterface);

        // --- pointers the guards short-circuit on -----------------------------------
        // hasFaceBlocks() is `fblks != nullptr && meshsizes.nbf > 0`. It is only safe
        // because fblks is null here -- assert that rather than rely on it.
        expect_null("fblks",                     (const void*) c.fblks);
        expect_null("eblks",                     (const void*) c.eblks);
        expect_null("nbsd",                      (const void*) c.nbsd);
        expect_null("elemsend",                  (const void*) c.elemsend);
        expect_null("nbintf",                    (const void*) c.nbintf);
#ifdef HAVE_MPI
        expect_null("requests",                  (const void*) c.requests);
        expect_null("statuses",                  (const void*) c.statuses);
#endif

        // --- the partition-aware accessors must agree with an empty partition -------
        if (c.hasFaceBlocks()) { std::printf("  FAIL  hasFaceBlocks() true on a default commonstruct\n"); ++failures; }
        if (c.hasElemBlocks()) { std::printf("  FAIL  hasElemBlocks() true on a default commonstruct\n"); ++failures; }
        if (c.hasInteriorElemBlocks()) { std::printf("  FAIL  hasInteriorElemBlocks() true on a default commonstruct\n"); ++failures; }

        std::printf(failures ? "FAILED (%d)\n" : "ok (%d failures)\n", failures);
    }
    Kokkos::finalize();
    return failures ? 1 : 0;
}
