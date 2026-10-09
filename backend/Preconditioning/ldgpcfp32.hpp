/*
    ldgpcfp32.hpp -- optional fp32 storage of the LDG block-Jacobi inverses (EXASIM_LDG_PC_FP32=1).

    The LDG block-Jacobi preconditioner applies one dense (npe*ncu)^2 inverse block per element, every GMRES
    iteration. At p = 2 in 3D a block is (27*ncu)^2 doubles per element, so the apply streams the whole K arena
    (gigabytes per rank for multi-component models) each time and is memory-bandwidth bound. With EXASIM_LDG_PC_FP32=1 the inverses are
    copied to fp32 once per preconditioner build and the apply reads that copy, accumulating in fp64:

        x_e = sum_j (double) Kf_e(:, j) * Ru_e(j)

    This halves the bytes per apply. It changes the preconditioner, not the residual: GMRES still converges to the
    same tolerance (the products and sums stay fp64; only the stored inverse entries are rounded to fp32). Off by
    default. The fp32 copy is allocated only when the switch is set.
*/
#ifndef __LDGPCFP32
#define __LDGPCFP32

#include <cstdlib>

struct LDGPcFP32State {
    float* Kf = nullptr;            // fp32 copy of the block inverses, same layout as res.K
    std::size_t cap = 0;            // allocated entries
    const void* src = nullptr;      // the res.K the copy was made from
    bool valid = false;             // the copy matches the current preconditioner build
};

inline LDGPcFP32State& LDGPcFP32()
{
    static LDGPcFP32State s;
    static bool hooked = false;
    if (!hooked) {
        Kokkos::push_finalize_hook([]() {
            LDGPcFP32State& t = LDGPcFP32();
            if (t.Kf) Kokkos::kokkos_free<Kokkos::DefaultExecutionSpace::memory_space>(t.Kf);
            t = LDGPcFP32State();
        });
        hooked = true;
    }
    return s;
}

inline bool LDGPcFP32Enabled()
{
    static const bool on = [] { const char* e = std::getenv("EXASIM_LDG_PC_FP32"); return e && e[0] == '1'; }();
    return on;
}

// After a preconditioner build: refresh the fp32 copy of K (n*n*ne entries).
template <class Ty>
void LDGPcFP32Refresh(const Ty* K, Int n, Int ne)
{
    LDGPcFP32State& s = LDGPcFP32();
    s.valid = false;
    if (!LDGPcFP32Enabled() || K == nullptr || n <= 0 || ne <= 0) return;
    using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
    const std::size_t total = (std::size_t)n * (std::size_t)n * (std::size_t)ne;
    if (total > s.cap) {
        if (s.Kf) Kokkos::kokkos_free<MemSpace>(s.Kf);
        s.Kf = (float*) Kokkos::kokkos_malloc<MemSpace>("ldg_pc_fp32", total * sizeof(float));
        s.cap = total;
    }
    float* Kf = s.Kf;
    Kokkos::parallel_for("LDGPcFP32Refresh", Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, total),
        KOKKOS_LAMBDA(const std::size_t i) { Kf[i] = (float) K[i]; });
    s.src = (const void*) K;
    s.valid = true;
}

// x_e = K_e x_in_e for every element block, from the fp32 copy, accumulating in fp64. Returns false (and does
// nothing) when the copy is not current, so the caller falls back to the fp64 apply.
template <class Ty>
bool LDGPcFP32Apply(const Ty* K, const Ty* xin, Ty* x, Int n, Int ne)
{
    LDGPcFP32State& s = LDGPcFP32();
    if (!LDGPcFP32Enabled() || !s.valid || s.src != (const void*) K) return false;
    const float* Kf = s.Kf;
    using Policy = Kokkos::TeamPolicy<>;
    using Scratch = Kokkos::View<double*, Kokkos::DefaultExecutionSpace::scratch_memory_space, Kokkos::MemoryUnmanaged>;
    const int nn = (int) n;
    Kokkos::parallel_for("LDGPcFP32Apply",
        Policy((int) ne, Kokkos::AUTO).set_scratch_size(0, Kokkos::PerTeam(Scratch::shmem_size(nn))),
        KOKKOS_LAMBDA(const Policy::member_type& team) {
            const std::size_t e = (std::size_t) team.league_rank();
            Scratch xs(team.team_scratch(0), nn);
            const Ty* xe = xin + (std::size_t) nn * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nn), [&](const int j) { xs(j) = (double) xe[j]; });
            team.team_barrier();
            const float* Ke = Kf + (std::size_t) nn * (std::size_t) nn * e;
            Ty* ye = x + (std::size_t) nn * e;
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nn), [&](const int t) {
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
                int j = 0;
                for (; j + 3 < nn; j += 4) {   // column-major: lane t reads row t of each column, coalesced
                    a0 += (double) Ke[t + (std::size_t) nn * j]       * xs(j);
                    a1 += (double) Ke[t + (std::size_t) nn * (j + 1)] * xs(j + 1);
                    a2 += (double) Ke[t + (std::size_t) nn * (j + 2)] * xs(j + 2);
                    a3 += (double) Ke[t + (std::size_t) nn * (j + 3)] * xs(j + 3);
                }
                for (; j < nn; j++) a0 += (double) Ke[t + (std::size_t) nn * j] * xs(j);
                ye[t] = (Ty) ((a0 + a1) + (a2 + a3));
            });
        });
    return true;
}

#endif
