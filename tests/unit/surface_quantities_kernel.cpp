// Kernel test for the templated boundary-output kernels in <exasim/kernels/qoi.hpp>:
// exasim::surface_quantities_kernel<M> and exasim::qoi_boundary_kernel<M>.
//
// Locks the contract the model sees at each point:
//   - tau is the full stabilization array (here ntau = 4 != ncu = 2; the model reads tau[3]),
//     not a copy truncated/extended to ncu entries;
//   - uinf is forwarded (the model reads uinf[1]), not replaced by nullptr;
//   - the boundary id passed to the kernel reaches the model unchanged (ib = 3);
//   - surface_quantities_kernel refuses nsurfq > kSurfaceQuantitiesMax before launching
//     (checked in a forked child, which must terminate abnormally).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

#include <exasim/kernels/qoi.hpp>

// nd = 2, ncu = 2 => Nq = ncu*(1+nd) = 6; no v/w fields.
struct SProbe : exasim::ModelDefaults<SProbe> {
    static constexpr int nd = 2, ncu = 2, nparam = 1, nco = 0, ncw = 0;
    static constexpr auto disc = exasim::Discretization::HDG;

    KOKKOS_INLINE_FUNCTION static
    void flux(double f[], const double[], const double[], const double[],
              const double[], const double[], double) {
        for (int k = 0; k < ncu*nd; ++k) f[k] = 0.0;
    }
    KOKKOS_INLINE_FUNCTION static
    void initu(double ui[], const double[], const double[], const double[]) { ui[0] = 0.0; ui[1] = 0.0; }

    // f = [ib, tau3*(u0-uh0), uinf1*mu0 + x1, n0*uq2 + n1*uq4]
    KOKKOS_INLINE_FUNCTION static
    void surface_quantities(double f[], int ib, const double x[], const double uq[],
                            const double[], const double[], const double uh[],
                            const double n[], const double tau[], const double mu[],
                            const double uinf[], double) {
        f[0] = ib;
        f[1] = tau[3]*(uq[0] - uh[0]);
        f[2] = uinf[1]*mu[0] + x[1];
        f[3] = n[0]*uq[2] + n[1]*uq[4];
    }
    // qoi_boundary shares the boundary-trace contract: same tau/uinf requirements.
    KOKKOS_INLINE_FUNCTION static
    void qoi_boundary(double f[], int ib, const double[], const double uq[],
                      const double[], const double[], const double uh[],
                      const double[], const double tau[], const double[],
                      const double uinf[], double) {
        f[0] = ib*tau[3]*(uq[1] - uh[1]) + uinf[1];
    }
};

static int nfail = 0;
static void check(const char* what, double got, double want) {
    if (std::abs(got - want) > 1e-14*(1.0 + std::abs(want))) {
        std::printf("FAIL %s: got %.17g want %.17g\n", what, got, want);
        ++nfail;
    }
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        constexpr int ng = 5, nd = 2, Nq = 6, ncu = 2, ib = 3;
        const double tauh[4]  = {0.5, 0.25, 0.125, 7.0};    // ntau = 4, model reads tau[3]
        const double uinfh[2] = {11.0, 13.0};                // model reads uinf[1]
        const double mu = 2.0;

        Kokkos::View<double*> x("x", nd*ng), uq("uq", Nq*ng), uh("uh", ncu*ng), n("n", nd*ng),
                              tau("tau", 4), uinf("uinf", 2), param("param", 1), none("none", 1),
                              fs("fs", 4*ng), fq("fq", ng);
        auto xh = Kokkos::create_mirror_view(x);  auto uqh = Kokkos::create_mirror_view(uq);
        auto uhh = Kokkos::create_mirror_view(uh); auto nh = Kokkos::create_mirror_view(n);
        auto th = Kokkos::create_mirror_view(tau); auto ih = Kokkos::create_mirror_view(uinf);
        auto ph = Kokkos::create_mirror_view(param);
        for (int i = 0; i < ng; ++i) {
            xh(i) = 0.1*i; xh(ng+i) = 0.3 + 0.01*i;
            for (int k = 0; k < Nq; ++k) uqh(k*ng+i) = 1.0 + k + 0.1*i;
            uhh(i) = 0.2*i; uhh(ng+i) = -0.5 + i;
            nh(i) = 0.6; nh(ng+i) = 0.8;
        }
        for (int k = 0; k < 4; ++k) th(k) = tauh[k];
        ih(0) = uinfh[0]; ih(1) = uinfh[1]; ph(0) = mu;
        Kokkos::deep_copy(x, xh); Kokkos::deep_copy(uq, uqh); Kokkos::deep_copy(uh, uhh);
        Kokkos::deep_copy(n, nh); Kokkos::deep_copy(tau, th); Kokkos::deep_copy(uinf, ih);
        Kokkos::deep_copy(param, ph);

        exasim::surface_quantities_kernel<SProbe>(fs.data(), x.data(), uq.data(), none.data(), none.data(),
            uh.data(), n.data(), tau.data(), uinf.data(), param.data(), 0.0, ib, ng, /*nout=*/4);
        exasim::qoi_boundary_kernel<SProbe>(fq.data(), x.data(), uq.data(), none.data(), none.data(),
            uh.data(), n.data(), tau.data(), uinf.data(), param.data(), 0.0, /*modelnumber=*/1, ib, ng,
            /*nc_runtime=*/1, ncu, nd, nd, 0, 0);
        Kokkos::fence();
        auto fsh = Kokkos::create_mirror_view(fs); Kokkos::deep_copy(fsh, fs);
        auto fqh = Kokkos::create_mirror_view(fq); Kokkos::deep_copy(fqh, fq);

        for (int i = 0; i < ng; ++i) {
            check("surface ib",    fsh(0*ng+i), ib);
            check("surface tau",   fsh(1*ng+i), tauh[3]*(uqh(i) - uhh(i)));
            check("surface uinf",  fsh(2*ng+i), uinfh[1]*mu + xh(ng+i));
            check("surface n.q",   fsh(3*ng+i), nh(i)*uqh(2*ng+i) + nh(ng+i)*uqh(4*ng+i));
            check("qoi_boundary",  fqh(i),      ib*tauh[3]*(uqh(ng+i) - uhh(ng+i)) + uinfh[1]);
        }

        // nsurfq beyond the per-point buffer must be refused before the launch.
        std::fflush(stdout);
        pid_t pid = fork();
        if (pid == 0) {
            exasim::surface_quantities_kernel<SProbe>(fs.data(), x.data(), uq.data(), none.data(), none.data(),
                uh.data(), n.data(), tau.data(), uinf.data(), param.data(), 0.0, ib, ng,
                exasim::kSurfaceQuantitiesMax + 1);
            _exit(0);   // reaching here means the guard did not fire
        }
        int status = 0;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            std::printf("FAIL nsurfq guard: kernel accepted nsurfq=%d\n", exasim::kSurfaceQuantitiesMax + 1);
            ++nfail;
        }
    }
    Kokkos::finalize();
    std::printf(nfail ? "surface_quantities_kernel: %d FAILURES\n" : "surface_quantities_kernel: OK\n", nfail);
    return nfail ? 1 : 0;
}
