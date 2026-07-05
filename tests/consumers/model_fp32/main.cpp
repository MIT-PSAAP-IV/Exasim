// model_fp32 — a *consumer-side* single-precision (float32) test of the templated
// concrete-model kernel path (drivers/kernels/model), with NO Exasim rebuild.
//
// This is the payoff of the dstype->T template threading (objective ⑤): precision is a
// type-level choice a consumer makes, not a build-time macro. Here we instantiate the
// Poisson2D model + its flux/source/qoi Kokkos kernels at BOTH float and double against
// the SAME (double-default) installed Exasim, and check:
//   (1) the float32 instantiation compiles and runs (Kokkos, so on CPU *and* GPU);
//   (2) float32 results track the double reference to single-precision tolerance;
//   (3) both match the analytic Poisson2D physics (flux = mu*grad u, source = 2pi^2 u,
//       qoi = u for the manufactured solution u = sin(pi x) sin(pi y)).
//
// Runs on whatever backend Kokkos was built for: serial/OpenMP (CPU) or CUDA (GPU) — the
// "various targets" for the one global precision choice.

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
#include <vector>

#include "poisson2d.hpp"                    // Poisson2DT<T> (+ using Poisson2D = Poisson2DT<double>)
#include <exasim/kernels/flux.hpp>          // exasim::flux_kernel<M,T>
#include <exasim/kernels/source.hpp>        // exasim::source_kernel<M,T>
#include <exasim/kernels/qoi.hpp>           // exasim::qoi_volume_kernel<M,T>

static constexpr double PI = 3.141592653589793;

template <class T>
struct Fields { std::vector<double> fx, fy, src, qu; };  // stored back in double for comparison

// Run the three templated kernels at precision T over ng quadrature points.
template <class T>
static Fields<T> run(int ng, const std::vector<double>& X, const std::vector<double>& Y)
{
    using Model = Poisson2DT<T>;
    constexpr int nd = Model::nd, ncu = Model::ncu, Nq = ncu * (1 + nd);
    // Number of QoI outputs the qoi_volume kernel writes: Poisson2D::QoI::qoi_volume
    // writes s[0]=(u-uexact)^2 and s[1]=u -> 2. This is the kernel's write count
    // (nc_runtime); it is NOT Nq (the udg component count). Passing Nq here made the
    // kernel write a 3rd component past q's end -> heap overflow (benign on macOS,
    // fatal on glibc/Linux).
    constexpr int nqoi = 2;

    Kokkos::View<T*> xdg("xdg", ng * nd), udg("udg", ng * Nq), param("param", 1);
    Kokkos::View<T*> f("f", ng * ncu * nd), s("s", ng * ncu), q("q", ng * nqoi);

    auto hx = Kokkos::create_mirror_view(xdg);
    auto hu = Kokkos::create_mirror_view(udg);
    auto hp = Kokkos::create_mirror_view(param);
    for (int i = 0; i < ng; ++i) {
        const double x = X[i], y = Y[i];
        const double u  = std::sin(PI * x) * std::sin(PI * y);
        const double ux = PI * std::cos(PI * x) * std::sin(PI * y);
        const double uy = PI * std::sin(PI * x) * std::cos(PI * y);
        hx(0 * ng + i) = (T)x;  hx(1 * ng + i) = (T)y;      // SoA: [comp*ng + point]
        hu(0 * ng + i) = (T)u;  hu(1 * ng + i) = (T)ux; hu(2 * ng + i) = (T)uy;
    }
    hp(0) = (T)1.0;                                          // mu = 1
    Kokkos::deep_copy(xdg, hx); Kokkos::deep_copy(udg, hu); Kokkos::deep_copy(param, hp);

    const T t = (T)0;
    exasim::flux_kernel<Model, T>    (f.data(), xdg.data(), udg.data(), nullptr, nullptr,
                                      nullptr, param.data(), t, 0, ng, Nq, ncu, nd, nd, 0, 0);
    exasim::source_kernel<Model, T>  (s.data(), xdg.data(), udg.data(), nullptr, nullptr,
                                      nullptr, param.data(), t, 0, ng, Nq, ncu, nd, nd, 0, 0);
    exasim::qoi_volume_kernel<typename Model::QoI, T>
                                     (q.data(), xdg.data(), udg.data(), nullptr, nullptr,
                                      nullptr, param.data(), t, 0, ng, nqoi, ncu, nd, nd, 0, 0);
    Kokkos::fence();

    auto hf = Kokkos::create_mirror_view(f);
    auto hs = Kokkos::create_mirror_view(s);
    auto hq = Kokkos::create_mirror_view(q);
    Kokkos::deep_copy(hf, f); Kokkos::deep_copy(hs, s); Kokkos::deep_copy(hq, q);

    Fields<T> r;
    r.fx.resize(ng); r.fy.resize(ng); r.src.resize(ng); r.qu.resize(ng);
    for (int i = 0; i < ng; ++i) {
        r.fx[i]  = (double)hf(0 * ng + i);
        r.fy[i]  = (double)hf(1 * ng + i);
        r.src[i] = (double)hs(i);
        r.qu[i]  = (double)hq(1 * ng + i);   // qoi_volume s[1] = u
    }
    return r;
}

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        // 8x8 interior grid of the unit square (avoid the boundary zeros of sin(pi x)).
        const int nside = 8, ng = nside * nside;
        std::vector<double> X(ng), Y(ng);
        for (int iy = 0; iy < nside; ++iy)
            for (int ix = 0; ix < nside; ++ix) {
                const int k = iy * nside + ix;
                X[k] = (ix + 1.0) / (nside + 1.0);
                Y[k] = (iy + 1.0) / (nside + 1.0);
            }

        const auto d = run<double>(ng, X, Y);   // reference
        const auto s = run<float> (ng, X, Y);   // single precision (the point of the test)

        double max_fd = 0.0;   // float-vs-double relative
        double max_an = 0.0;   // double-vs-analytic relative (sanity that the physics is right)
        for (int i = 0; i < ng; ++i) {
            const double x = X[i], y = Y[i];
            const double u  = std::sin(PI * x) * std::sin(PI * y);
            const double ux = PI * std::cos(PI * x) * std::sin(PI * y);
            const double uy = PI * std::sin(PI * x) * std::cos(PI * y);
            const double src_an = 2.0 * PI * PI * u;

            auto rel = [](double a, double b){ return std::abs(a - b) / (std::abs(b) + 1e-30); };
            // float vs double
            max_fd = std::max(max_fd, rel(s.fx[i], d.fx[i]));
            max_fd = std::max(max_fd, rel(s.fy[i], d.fy[i]));
            max_fd = std::max(max_fd, rel(s.src[i], d.src[i]));
            max_fd = std::max(max_fd, rel(s.qu[i], d.qu[i]));
            // double vs analytic
            max_an = std::max(max_an, rel(d.fx[i], ux));
            max_an = std::max(max_an, rel(d.fy[i], uy));
            max_an = std::max(max_an, rel(d.src[i], src_an));
            max_an = std::max(max_an, rel(d.qu[i], u));
        }

        // float32 has ~7 significant digits; sin/mul over these magnitudes lands ~1e-6.
        const double TOL_FD = 1e-4;    // float tracks double
        const double TOL_AN = 1e-12;   // double reproduces the analytic physics (exact formula)
        const bool pass = std::isfinite(max_fd) && std::isfinite(max_an)
                        && max_fd < TOL_FD && max_an < TOL_AN;

        std::printf("[model_fp32] exec space: %s\n", Kokkos::DefaultExecutionSpace::name());
        std::printf("[model_fp32] ng=%d  flux/source/qoi kernels run at float32 AND double\n", ng);
        std::printf("[model_fp32] max |float - double| / |double|  = %.3e   (tol %.0e)\n", max_fd, TOL_FD);
        std::printf("[model_fp32] max |double - analytic| / |ana|   = %.3e   (tol %.0e)\n", max_an, TOL_AN);
        std::printf("[model_fp32] sample pt0: flux=(%.6f,%.6f) f32=(%.6f,%.6f)  src=%.4f qoi_u=%.6f\n",
                    d.fx[0], d.fy[0], s.fx[0], s.fy[0], d.src[0], d.qu[0]);
        std::printf("[model_fp32] %s: float32 Poisson2D kernels vs double, no Exasim rebuild\n",
                    pass ? "PASS" : "FAIL");
        rc = pass ? 0 : 1;
    }
    Kokkos::finalize();
    return rc;
}
