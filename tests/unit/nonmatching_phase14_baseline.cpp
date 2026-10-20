// [nonmatching] Phase 14 T0 byte-exact runtime baseline emitter.
//
// This file deliberately observes the production runtime overloads without
// changing them.  It includes the accepted Phase 13 fixture implementation so
// that the same source can be copied into the immutable reference tree and the
// candidate tree.  Every record is written after a Kokkos fence and hashes the
// raw host bytes; T0 is CPU/Serial only.

#ifdef HAVE_MPI
#include <mpi.h>
extern MPI_Comm EXASIM_COMM_WORLD;
extern MPI_Comm EXASIM_COMM_LOCAL;
#endif

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../../backend/Common/common.h"
#include "../../backend/Common/cpuimpl.h"
#include "../../backend/Common/kokkosimpl.h"
#include "../../backend/Common/pblas.h"
#include "../../backend/Discretization/residual.hpp"
void FintDriver(...);
#include "../../backend/Discretization/nonmatchingcoupling.hpp"

namespace phase14_baseline {

static std::string context;
static std::string dumpDirectory;
static std::ofstream manifest;
static std::uint64_t sequence = 0;
static int rank = 0;

inline std::string safe_name(std::string value)
{
    for (char& c : value)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') c = '_';
    return value;
}

inline std::uint64_t fnv1a64(const void* data, std::size_t bytes)
{
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= static_cast<std::uint64_t>(p[i]);
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

inline void emit(const char* stage, int direction, const char* name,
        const void* data, std::size_t count, std::size_t elementBytes)
{
    const std::size_t bytes = count * elementBytes;
    const std::uint64_t hash = fnv1a64(data, bytes);
    std::ostringstream line;
    line << "record=" << sequence << " fixture=" << safe_name(context)
         << " stage=" << stage << " rank=" << rank << " direction=" << direction
         << " array=" << name << " count=" << count << " bytes=" << bytes
         << " fnv1a64=" << std::hex << std::setw(16) << std::setfill('0') << hash;
    if (manifest.is_open()) manifest << line.str() << '\n';
    else std::printf("PHASE14_BASELINE %s\n", line.str().c_str());

    if (!dumpDirectory.empty() && bytes > 0) {
        std::ostringstream path;
        path << dumpDirectory << "/rank" << rank << "_" << std::setw(6)
             << std::setfill('0') << sequence << "_" << safe_name(context) << "_"
             << stage << "_d" << direction << "_" << name << ".bin";
        std::ofstream output(path.str(), std::ios::binary);
        if (!output) throw std::runtime_error("cannot open Phase 14 raw-array dump");
        output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!output) throw std::runtime_error("cannot write Phase 14 raw-array dump");
    }
    ++sequence;
}

template <class T, class I>
inline void capture_runtime(const exasim::nonmatching::NonmatchingRuntime<T, I>& runtime,
        const char* stage)
{
    Kokkos::fence();
    for (int direction = 0; direction < 2; ++direction) {
        const auto& d = runtime.dir[direction];
        if (!d.view.isdonor) continue;
        emit(stage, direction, "Ri", d.Ri.data, static_cast<std::size_t>(d.Ri.size), sizeof(T));
        if (std::strcmp(stage, "coupling_jacobian") == 0) {
            emit(stage, direction, "K", d.K.data, static_cast<std::size_t>(d.K.size), sizeof(T));
            emit(stage, direction, "Hi", d.Hi.data, static_cast<std::size_t>(d.Hi.size), sizeof(T));
            emit(stage, direction, "pairResidual", d.pairResidual.data,
                 static_cast<std::size_t>(d.pairResidual.size), sizeof(T));
        }
        else if (std::strcmp(stage, "matvec") == 0) {
            emit(stage, direction, "Hi", d.Hi.data, static_cast<std::size_t>(d.Hi.size), sizeof(T));
            emit(stage, direction, "rpair", d.rpair.data,
                 static_cast<std::size_t>(d.rpair.size), sizeof(T));
        }
    }
}

template <class T, class I>
inline void capture_owner(const T* values, const commonstructT<T, I>& common,
        const char* stage)
{
    Kokkos::fence();
    const std::size_t count = static_cast<std::size_t>(common.components.ncu) *
        common.grid.npf * common.meshsizes.nfe * common.meshsizes.ne;
    emit(stage, -1, "owner", values, count, sizeof(T));
}

} // namespace phase14_baseline

namespace exasim {
namespace nonmatching {

template <class M, class T, class I>
inline void phase14_observed_compute_nonmatching_coupling(NonmatchingRuntime<T, I>& runtime,
        solstructT<T, I>& sol, resstructT<T, I>& res, appstructT<T, I>& app,
        masterstructT<T, I>& master, meshstructT<T, I>& mesh,
        tempstructT<T, I>& tmp, commonstructT<T, I>& common,
        cublasHandle_t handle, bool withJacobian, I backend)
{
    compute_nonmatching_coupling<M>(runtime, sol, res, app, master, mesh, tmp,
                                    common, handle, withJacobian, backend);
    phase14_baseline::capture_runtime(runtime,
        withJacobian ? "coupling_jacobian" : "coupling_residual");
}

template <class T, class I>
inline void phase14_observed_nonmatching_matvec_products(NonmatchingRuntime<T, I>& runtime,
        const T* ve, I m, I ne1, cublasHandle_t handle, I backend)
{
    nonmatching_matvec_products(runtime, ve, m, ne1, handle, backend);
    phase14_baseline::capture_runtime(runtime, "matvec");
}

#ifdef HAVE_MPI
template <class T, class I>
inline void phase14_observed_assemble_coupling_slot_owner(T* elementResidual,
        const NonmatchingRuntime<T, I>& runtime, const meshstructT<T, I>& mesh,
        const appstructT<T, I>& app, const commonstructT<T, I>& common)
{
    assemble_coupling_slot_owner(elementResidual, runtime, mesh, app, common);
    phase14_baseline::capture_owner(elementResidual, common, "owner_assembly");
}
#endif

} // namespace nonmatching
} // namespace exasim

// Observe calls made directly by the fixture and calls compiled in matvec.hpp.
#define compute_nonmatching_coupling phase14_observed_compute_nonmatching_coupling
#define nonmatching_matvec_products phase14_observed_nonmatching_matvec_products
#define assemble_coupling_slot_owner phase14_observed_assemble_coupling_slot_owner
#define main phase14_embedded_nonmatching_3d_main
#include "nonmatching_3d.cpp"
#undef main
#undef assemble_coupling_slot_owner
#undef nonmatching_matvec_products
#undef compute_nonmatching_coupling

namespace {

struct Phase14Options {
    int split = -1;
    std::string dump;
    std::string manifest;
};

Phase14Options phase14_options(int argc, char** argv, int size)
{
    Phase14Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--split") == 0) {
            if (++i >= argc) throw std::runtime_error("--split requires a rank count");
            char* end = nullptr;
            const long value = std::strtol(argv[i], &end, 10);
            if (!end || *end || value <= 0 || value >= size)
                throw std::runtime_error("--split must satisfy 0 < split < world size");
            options.split = static_cast<int>(value);
        }
        else if (std::strcmp(argv[i], "--dump") == 0) {
            if (++i >= argc) throw std::runtime_error("--dump requires a directory");
            options.dump = argv[i];
        }
        else if (std::strcmp(argv[i], "--manifest") == 0) {
            if (++i >= argc) throw std::runtime_error("--manifest requires a directory");
            options.manifest = argv[i];
        }
        else throw std::runtime_error(std::string("unexpected argument: ") + argv[i]);
    }
    if (options.split < 0) throw std::runtime_error("--split is required");
    if (!options.dump.empty() && !options.manifest.empty())
        throw std::runtime_error("use only one of --dump and --manifest");
    return options;
}

template <class Model>
void run_fixture(const char* label, int elemtype, int p, int rank, int size,
        int split, bool mixed, bool reverse)
{
    std::ostringstream name;
    name << label << "_p" << p << "_split" << split;
    phase14_baseline::context = name.str();
    model_operator_case<Model>(label, elemtype, p, rank, size, 0.0,
                               split, mixed, reverse);
}

} // namespace

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    EXASIM_COMM_WORLD = MPI_COMM_WORLD;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &phase14_baseline::rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#else
    const int size = 1;
#endif
    int failed = 0;
    try {
        const Phase14Options options = phase14_options(argc, argv, size);
        if (!options.dump.empty() || !options.manifest.empty()) {
            phase14_baseline::dumpDirectory = options.dump;
            const std::string outputDirectory = options.dump.empty() ? options.manifest : options.dump;
            std::ostringstream path;
            path << outputDirectory << "/manifest_rank" << phase14_baseline::rank << ".txt";
            phase14_baseline::manifest.open(path.str());
            if (!phase14_baseline::manifest)
                throw std::runtime_error("cannot open Phase 14 manifest");
        }
        Kokkos::initialize(argc, argv);
        for (int p = 1; p <= 4; ++p) {
            run_fixture<NLCoupled3D>("tet-tet-q", 0, p, phase14_baseline::rank,
                                    size, options.split, false, false);
            run_fixture<NLCoupled3D>("hex-hex-q", 1, p, phase14_baseline::rank,
                                    size, options.split, false, false);
            run_fixture<WCoupled3D>("tet-owner-hex-donor-w", 0, p,
                                    phase14_baseline::rank, size, options.split, true, false);
            run_fixture<WCoupled3D>("hex-owner-tet-donor-w", 0, p,
                                    phase14_baseline::rank, size, options.split, true, true);
            run_fixture<QFreeCoupled3D>("tet-owner-hex-donor-qfree", 0, p,
                                    phase14_baseline::rank, size, options.split, true, false);
            run_fixture<QFreeCoupled3D>("hex-owner-tet-donor-qfree", 0, p,
                                    phase14_baseline::rank, size, options.split, true, true);
        }
        Kokkos::finalize();
    }
    catch (const std::exception& exception) {
        std::fprintf(stderr, "[Rank %d] ERROR: %s\n", phase14_baseline::rank,
                     exception.what());
        failed = 1;
        if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
    }
#ifdef HAVE_MPI
    int globalFailed = 0;
    MPI_Allreduce(&failed, &globalFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    failed = globalFailed;
    MPI_Finalize();
#endif
    return failed ? 1 : 0;
}
