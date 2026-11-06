// Test ClangEnzyme derivatives through the public C++ model provider.
// All buffers use the configured Kokkos memory space. This includes GPU builds.
#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include "enzyme_model.hpp"
#include "modelprovider.hpp"

using View = Kokkos::View<dstype*>;
enum Field { U, V, W, U2, V2, W2, H, Count };
using State = std::array<View, Count>;
enum class Term { Flux, Source, Avfield, Tdfunc, Fbou, Ubou, Fhat, Stab };
constexpr int ng = 6, npe = 3, ne = 2, nc = 6, ncu = 2, nd = 2, nco = 2, ncw = 1;

static View data(const char* name, int size, dstype seed)
{
    View v(name, size);
    auto h = Kokkos::create_mirror_view(v);
    for (int i = 0; i < size; ++i) h(i) = seed + dstype(0.013)*(i+1);
    Kokkos::deep_copy(v, h);
    return v;
}

struct Fixture {
    State state, direction;
    View x = data("x", ng*nd, 0.2), n = data("normal", ng*nd, 0.4);
    View tau = data("tau", 2, 0.7), mu = data("mu", 1, 1.2), eta = data("eta", 1, 0.1);
    const ExasimDriverABI& abi = getKokkosKernelExasimDriverABI();

    Fixture() {
        const int widths[] = {nc, nco, ncw, nc, nco, ncw, ncu};
        for (int k = 0; k < Count; ++k) {
            state[k] = data("state", ng*widths[k], dstype(0.3 + k*0.1));
            direction[k] = data("direction", ng*widths[k], dstype(-0.2 + k*0.03));
        }
    }

    State perturbed(const State& d, const std::array<bool, Count>& active, dstype step) const {
        State result;
        for (int k = 0; k < Count; ++k) {
            result[k] = View("perturbed", state[k].extent(0));
            const auto out = result[k], in = state[k], delta = d[k];
            const dstype scale = active[k] ? step : dstype(0);
            Kokkos::parallel_for("perturb", out.extent(0), KOKKOS_LAMBDA(int i) {
                out(i) = in(i) + scale*delta(i);
            });
        }
        return result;
    }

    void evaluate(Term term, bool jvp, const State& s, const State& d,
                  View f, View df, int ib) const {
        auto element = [&](auto primal, auto tangent) {
            if (jvp)
                tangent(f.data(), df.data(), x.data(), s[U].data(), d[U].data(),
                    s[V].data(), d[V].data(), s[W].data(), d[W].data(), eta.data(),
                    mu.data(), 0.25, 0, ng, nc, ncu, nd, nd, nco, ncw);
            else
                primal(f.data(), x.data(), s[U].data(), s[V].data(), s[W].data(),
                    eta.data(), mu.data(), 0.25, 0, ng, nc, ncu, nd, nd, nco, ncw);
        };
        auto face = [&](auto primal, auto tangent) {
            if (jvp)
                tangent(f.data(), df.data(), x.data(), s[U].data(), d[U].data(),
                    s[U2].data(), d[U2].data(), s[V].data(), d[V].data(),
                    s[V2].data(), d[V2].data(), s[W].data(), d[W].data(),
                    s[W2].data(), d[W2].data(), s[H].data(), d[H].data(),
                    n.data(), tau.data(), eta.data(), mu.data(), 0.25, 0,
                    ng, nc, ncu, nd, nd, nco, ncw);
            else
                primal(f.data(), x.data(), s[U].data(), s[U2].data(),
                    s[V].data(), s[V2].data(), s[W].data(), s[W2].data(), s[H].data(),
                    n.data(), tau.data(), eta.data(), mu.data(), 0.25, 0,
                    ng, nc, ncu, nd, nd, nco, ncw);
        };
        switch (term) {
        case Term::Flux: element(abi.volume.KokkosFlux, abi.volume.KokkosFluxJvp); break;
        case Term::Source: element(abi.volume.KokkosSource, abi.volume.KokkosSourceJvp); break;
        case Term::Tdfunc: element(abi.volume.KokkosTdfunc, abi.volume.KokkosTdfuncJvp); break;
        case Term::Avfield:
            if (jvp)
                abi.volume.KokkosAvfieldJvp(f.data(), df.data(), x.data(),
                    s[U].data(), d[U].data(), s[V].data(), s[W].data(), d[W].data(),
                    eta.data(), mu.data(), 0.25, 0, ng, nc, ncu, nd, nd, nco, ncw,
                    1, npe, ne);
            else
                abi.volume.KokkosAvfield(f.data(), x.data(), s[U].data(),
                    s[V].data(), s[W].data(), eta.data(), mu.data(), 0.25, 0,
                    ng, nc, ncu, nd, nd, nco, ncw, 1, npe, ne);
            break;
        case Term::Fbou:
            if (jvp)
                abi.boundary.KokkosFbouJvp(f.data(), df.data(), x.data(),
                    s[U].data(), d[U].data(), s[V].data(), d[V].data(),
                    s[W].data(), d[W].data(), s[H].data(), d[H].data(),
                    n.data(), tau.data(), eta.data(), mu.data(), 0.25, 0, ib,
                    ng, nc, ncu, nd, nd, nco, ncw);
            else
                abi.boundary.KokkosFbou(f.data(), x.data(), s[U].data(), s[V].data(),
                    s[W].data(), s[H].data(), n.data(), tau.data(), eta.data(), mu.data(),
                    0.25, 0, ib, ng, nc, ncu, nd, nd, nco, ncw);
            break;
        case Term::Ubou:
            if (jvp)
                abi.boundary.KokkosUbouJvp(f.data(), df.data(), x.data(),
                    s[U].data(), d[U].data(), s[V].data(), s[W].data(), d[W].data(),
                    s[H].data(), d[H].data(), n.data(), tau.data(), eta.data(), mu.data(),
                    0.25, 0, ib, ng, nc, ncu, nd, nd, nco, ncw);
            else
                abi.boundary.KokkosUbou(f.data(), x.data(), s[U].data(), s[V].data(),
                    s[W].data(), s[H].data(), n.data(), tau.data(), eta.data(), mu.data(),
                    0.25, 0, ib, ng, nc, ncu, nd, nd, nco, ncw);
            break;
        case Term::Fhat: face(abi.iface.KokkosFhat, abi.iface.KokkosFhatJvp); break;
        case Term::Stab: face(abi.iface.KokkosStab, abi.iface.KokkosStabJvp); break;
        }
    }
};

static void run()
{
    Fixture a;
    const char* names[] = {"Flux", "Source", "Avfield", "Tdfunc", "Fbou", "Ubou", "Fhat", "Stab"};
    const dstype eps = sizeof(dstype) == 4 ? 0.01 : 1e-5;
    const dstype tolerance = sizeof(dstype) == 4 ? 2e-4 : 2e-9;
    for (int termIndex = 0; termIndex < 8; ++termIndex) {
        const Term term = static_cast<Term>(termIndex);
        std::array<bool, Count> active = {true, true, true, false, false, false, false};
        if (term == Term::Avfield || term == Term::Ubou) active[V] = false;
        if (term == Term::Fbou || term == Term::Ubou) active[H] = true;
        if (term == Term::Fhat || term == Term::Stab) active.fill(true);
        const int size = ng * (term == Term::Flux ? ncu*nd : term == Term::Avfield ? 1 : ncu);
        View f("f", size), df("df", size), primal("primal", size), plus("plus", size), minus("minus", size);
        for (int ib : {1, 2}) {
            for (bool zero : {false, true}) {
                State d = a.direction;
                if (zero) {
                    for (int k = 0; k < Count; ++k) {
                        d[k] = View("zero_direction", a.direction[k].extent(0));
                        Kokkos::deep_copy(d[k], dstype(0));
                    }
                }
                Kokkos::deep_copy(f, dstype(-999));
                Kokkos::deep_copy(df, dstype(-999));
                a.evaluate(term, true, a.state, d, f, df, ib);
                a.evaluate(term, false, a.state, d, primal, df, ib);
                const auto sp = a.perturbed(d, active, eps), sm = a.perturbed(d, active, -eps);
                a.evaluate(term, false, sp, d, plus, df, ib);
                a.evaluate(term, false, sm, d, minus, df, ib);
                auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), f);
                auto hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), df);
                auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), primal);
                auto hplus = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), plus);
                auto hminus = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), minus);
                dstype maxError = 0;
                for (int i = 0; i < size; ++i) {
                    const dstype ref = (hplus(i)-hminus(i))/(2*eps);
                    const dstype err = std::abs(hd(i)-ref)/(1+std::abs(ref));
                    maxError = std::max(maxError, err);
                    if (!std::isfinite(hf(i)) || !std::isfinite(hd(i)) ||
                        std::abs(hf(i)-hp(i)) > tolerance*(1+std::abs(hp(i))) || err > tolerance ||
                        (zero && std::abs(hd(i)) > tolerance))
                        throw std::runtime_error(std::string(names[termIndex])+" primal/JVP mismatch at "+std::to_string(i));
                }
                if (!zero) std::printf("%s ib=%d: relative JVP error %.3e\n", names[termIndex], ib, double(maxError));
            }
        }
    }
}

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard kokkos(argc, argv);
    try { run(); std::puts("PASS: actual Enzyme JVPs match finite differences"); }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
