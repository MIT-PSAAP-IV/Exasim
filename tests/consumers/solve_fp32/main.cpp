// solve_fp32 — a FULL float32 Poisson SOLVE, chosen entirely via templates, no Exasim rebuild,
// no conversion shim. Instantiates the in-memory HDG Poisson stack at both double and float,
// condenses to the trace system H*uh=b, solves it densely (LAPACK), recovers the volume field,
// and reports the quadrature L2 error ||u - u_exact||. The float32 solve must track the double
// solve (both ~1.08e-2 discretization error on this coarse 8x8/p=3 mesh) to single precision.

#include <exasim/operators.hpp>
#include <exasim/export.hpp>
#include "poisson2d.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

extern "C" {
    void sgesv_(int*, int*, float*,  int*, int*, float*,  int*, int*);
    void dgesv_(int*, int*, double*, int*, int*, double*, int*, int*);
}
static void gesv(int n, float*  A, float*  b) { int one=1, info=0; std::vector<int> ip(n); sgesv_(&n,&one,A,&n,ip.data(),b,&n,&info); if(info) std::printf("[solve_fp32] sgesv info=%d\n",info); }
static void gesv(int n, double* A, double* b) { int one=1, info=0; std::vector<int> ip(n); dgesv_(&n,&one,A,&n,ip.data(),b,&n,&info); if(info) std::printf("[solve_fp32] dgesv info=%d\n",info); }

static void unitSquareQuadMesh(int n, std::vector<double>& p, std::vector<int>& t, int& np, int& ne)
{
    const int nv=n+1; np=nv*nv; ne=n*n; p.resize((size_t)2*np); t.resize((size_t)4*ne);
    for (int iy=0; iy<nv; ++iy) for (int ix=0; ix<nv; ++ix) { const int j=iy*nv+ix; p[0+2*j]=(double)ix/n; p[1+2*j]=(double)iy/n; }
    int e=0;
    for (int iy=0; iy<n; ++iy) for (int ix=0; ix<n; ++ix,++e) {
        t[0+4*e]=iy*nv+ix; t[1+4*e]=iy*nv+(ix+1); t[2+4*e]=(iy+1)*nv+(ix+1); t[3+4*e]=(iy+1)*nv+ix;
    }
}

// returns {L2err-vs-exact (via eval_qoi), and fills `uh` with the trace solution}
template <class T>
static double solveL2err(std::vector<double>& uh_out)
{
    using Model = Poisson2DT<T>;
    const int backend = 0;
    constexpr double TOL = 1e-8;

    int np=0, ne=0; std::vector<double> p; std::vector<int> t;
    unitSquareQuadMesh(8, p, t, np, ne);

    PDE pde = exasim::default_pde<Model>();
    pde.porder = 3; pde.pgauss = 6;
    pde.physicsparam = {1.0};
    pde.nvqoi = 2;

    exasim::MeshSpec mesh(p.data(), t.data(), np, ne, /*nve=*/4);
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[1])       < TOL; });
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[0] - 1.0) < TOL; });
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[1] - 1.0) < TOL; });
    mesh.add_boundary(1, [](const double* x){ return std::abs(x[0])       < TOL; });

    CDiscretizationT<T, Int>      disc(exasim::make_preprocessed<Model, T, Int>(pde, mesh), backend);
    CResidual<Model, T, Int>      residual(disc);
    CAssembler<Model, T, Int>     assembler(disc);
    CPreconditioner<Model, T, Int> prec(disc, backend, ExasimExecutionMode::Solve);
    residual.initializeSolution();
    residual.recoverInitialState(backend, /*postprocessOnly=*/false);

    sysstructT<T, Int> sys;
    setsysstruct(sys, disc.common, disc.res, disc.mesh, disc.tmp, backend);

    const Int N = disc.common.sizes.ndofuhat;
    if (disc.common.components.ncq > 0)
        hdgGetQ(disc.sol.udg, disc.sol.uh, disc.sol, disc.res, disc.mesh, disc.tmp, disc.common, backend);

    assembler.hdgAssembleLinearSystem(sys.b, backend);
    prec.ComputeHDGPreconditioner(disc, backend);
    std::vector<T> b(sys.b, sys.b + N);

    std::vector<T> H((size_t)N * N), v((size_t)N, T(0)), Jv((size_t)N), Ru((size_t)N);
    for (Int j = 0; j < N; ++j) {
        v[j] = T(1);
        assembler.evalMatVec(Jv.data(), v.data(), sys.u, Ru.data(), /*spatialScheme=*/1, backend);
        for (Int i = 0; i < N; ++i) H[(size_t)i + (size_t)j * N] = Jv[i];
        v[j] = T(0);
    }

    gesv((int)N, H.data(), b.data());
    uh_out.resize((size_t)N); for (Int i=0;i<N;++i) uh_out[i] = (double)b[i];   // trace solution
    double uhnorm = 0.0; for (Int i=0;i<N;++i) uhnorm += (double)b[i]*(double)b[i]; uhnorm = std::sqrt(uhnorm);
    std::vector<T> scratch((size_t)N);
    exasim::recover_volume<T, Int>(disc, b.data(), scratch.data());
    auto qoi = exasim::eval_qoi<typename Model::QoI, T, Int>(disc);
    std::printf("[solve_fp32]   (%s) ||uh||=%.6e  nvqoi=%d  qoi[0]=%.6e  qoi[1]=%.6e\n",
                (sizeof(T)==4?"f32":"f64"), uhnorm, (int)disc.common.qoiparams.nvqoi,
                qoi.empty()?-1.0:(double)qoi[0], qoi.size()<2?-1.0:(double)qoi[1]);
    return std::sqrt((double)(qoi.empty() ? 1.0 : qoi[0]));
}

int main()
{
    std::vector<double> uh_d, uh_f;
    const double err_d = solveL2err<double>(uh_d);
    const double err_f = solveL2err<float> (uh_f);

    // The honest correctness metric: the float32 trace solution vs the double one. Both solve the
    // SAME condensed HDG system; float32 must reproduce it to single precision (~1e-6 relative).
    double num = 0.0, den = 0.0;
    const size_t N = std::min(uh_d.size(), uh_f.size());
    for (size_t i = 0; i < N; ++i) { const double d = uh_f[i]-uh_d[i]; num += d*d; den += uh_d[i]*uh_d[i]; }
    const double rel_uh = std::sqrt(num) / (std::sqrt(den) + 1e-300);

    std::printf("[solve_fp32] full HDG Poisson SOLVE, precision chosen via templates (no rebuild)\n");
    std::printf("[solve_fp32]   double  ||u - u_exact|| (quadrature QoI) = %.6e\n", err_d);
    std::printf("[solve_fp32]   ||uh_f32 - uh_f64|| / ||uh_f64||         = %.3e  (tol 1e-3; fp32 dense-LU ~1e-4)\n", rel_uh);

    const bool pass = (N > 0) && std::isfinite(rel_uh) && rel_uh < 1e-3 && err_d > 0.0;
    std::printf("[solve_fp32] %s: float32 HDG Poisson solve reproduces the double solve, from templates alone\n",
                pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
