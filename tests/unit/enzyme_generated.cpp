// Exercises the actual generated built-in kernels and their Enzyme JVPs.
// Device Views keep the fixture valid for Kokkos CPU/CUDA/HIP execution spaces.
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#ifdef EXASIM_EXTERNAL_PROVIDER
#include "driver_abi.hpp"
const ExasimDriverABI& getBuiltInLibraryExasimDriverABI();
#else
#include "builtinlibprovider.hpp"
#endif

using View = Kokkos::View<dstype*>;

static View device(const char* name, const std::vector<dstype>& values)
{
    View result(name, values.size());
    auto host = Kokkos::create_mirror_view(result);
    for (size_t i = 0; i < values.size(); ++i) host(i) = values[i];
    Kokkos::deep_copy(result, host);
    return result;
}

struct Field {
    View value, direction, seed, trial;
    Field(const char* name, const std::vector<dstype>& values, dstype seed_scale = 1) :
        value(device(name, values)), direction(name, values.size()),
        seed(name, values.size()), trial(name, values.size())
    {
        std::vector<dstype> d(values.size());
        for (size_t i = 0; i < d.size(); ++i) d[i] = seed_scale*0.003*(1+i%7);
        seed = device(name, d);
        Kokkos::deep_copy(direction, seed);
    }
    void perturb(dstype step) {
        auto v = value, d = seed, p = trial;
        Kokkos::parallel_for("perturb", v.extent(0), KOKKOS_LAMBDA(size_t i) {
            p(i) = v(i) + step*d(i);
        });
    }
};

static void run_model(int model)
{
    const auto& abi = getBuiltInLibraryExasimDriverABI();
    const auto sizes = abi.GetModelSizes(model);
    constexpr int ng = 3, nd = 2, npe = 3;
    const int ncu = sizes.ncu, nc = 3*ncu, nco = sizes.nco, ncw = sizes.ncw;
    if (ncu == 0) throw std::runtime_error("Generated test requires model dimension metadata");
    std::vector<dstype> state(ng*nc), trace(ng*ncu);
    const dstype mixture[] = {0.05,0.04,0.03,0.65,0.23,0.8,0.1,2.0};
    for (int k = 0; k < nc; ++k)
        for (int i = 0; i < ng; ++i)
            state[k*ng+i] = model == 4 && k < ncu ? mixture[k] + 0.001*i
                : 0.01*(k+1) + 0.002*i;
    std::copy(state.begin(), state.begin()+ng*ncu, trace.begin());
    Field u("u", state), o("o", std::vector<dstype>(ng*std::max(1,nco), 0.015));
    Field w("w", std::vector<dstype>(ng*std::max(1,ncw), 4.2)), h("h", trace);
    for (auto& value : state) value *= 1.1;
    Field u2("u2", state, -0.7), o2("o2", std::vector<dstype>(ng*std::max(1,nco), 0.02), -0.3);
    Field w2("w2", std::vector<dstype>(ng*std::max(1,ncw), 4.3), -0.5);
    View x = device("x", {0.3,0.4,0.5, 1.2,1.3,1.4});
    View normal = device("normal", {0.6,0.6,0.6,0.8,0.8,0.8});
    View tau = device("tau", {5});
    View mu = model == 1 ? device("mu", {2}) : device("mu", {
        0.0023402724,6921,112099.61,260.6,1.801932e-05,0.024602882,
        1009.9704,1,181.99298,0.7397093,898869.96,1400});
    View eta = device("eta", {0,0,0,0.76708249,0.23291751,1.0041566,0,0.49763987,
                             0,0,0,0.76708249,0.23291751,0,0,0,0,0});
    const dstype time = 0.25;
    const dstype eps = 1e-6, tol = 2e-6;
    auto verify = [&](const char* name, int count, auto primal, auto jvp,
                      bool active_o = true, bool active_h = true, bool inout = false) {
        View f("f",count), df("df",count), reference("ref",count);
        View plus("plus",count), minus("minus",count);
        Kokkos::deep_copy(f, inout ? 2 : 0); Kokkos::deep_copy(reference, inout ? 2 : 0);
        Kokkos::deep_copy(df, inout ? 0.4 : 123);
        primal(reference.data(), u.value.data(), o.value.data(), w.value.data(), h.value.data(),
               u2.value.data(), o2.value.data(), w2.value.data());
        jvp(f.data(), df.data());
        for (int sign : {-1,1}) {
            u.perturb(sign*eps); o.perturb(active_o ? sign*eps : 0);
            w.perturb(sign*eps); h.perturb(active_h ? sign*eps : 0);
            u2.perturb(sign*eps); o2.perturb(sign*eps); w2.perturb(sign*eps);
            auto target = sign == 1 ? plus : minus;
            Kokkos::deep_copy(target, inout ? 2 + sign*eps*0.4 : 0);
            primal(target.data(),u.trial.data(),o.trial.data(),w.trial.data(),h.trial.data(),
                   u2.trial.data(),o2.trial.data(),w2.trial.data());
        }
        auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),f);
        auto hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),df);
        auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),reference);
        auto hp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),plus);
        auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),minus);
        dstype max_error = 0;
        for (int i = 0; i < count; ++i) {
            const dstype fd = (hp(i)-hm(i))/(2*eps);
            const dstype error = std::abs(hd(i)-fd)/(1+std::abs(fd));
            max_error = std::max(max_error,error);
            if (!std::isfinite(hf(i)) || !std::isfinite(hd(i)) || !std::isfinite(fd) ||
                std::abs(hf(i)-hr(i)) > tol*(1+std::abs(hr(i))) || error > tol) {
                std::fprintf(stderr,"model %d %s[%d]: f=%g ref=%g jvp=%g fd=%g\n",
                             model,name,i,double(hf(i)),double(hr(i)),double(hd(i)),double(fd));
                throw std::runtime_error("Generated Enzyme JVP mismatch");
            }
        }
        for (auto* field : {&u,&o,&w,&h,&u2,&o2,&w2}) Kokkos::deep_copy(field->direction,0);
        Kokkos::deep_copy(df,inout ? 0 : 123);
        jvp(f.data(),df.data());
        hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),df);
        for (int i = 0; i < count; ++i)
            if (!std::isfinite(hd(i)) || std::abs(hd(i)) > 1e-12)
                throw std::runtime_error("Zero direction retained a stale JVP");
        for (auto* field : {&u,&o,&w,&h,&u2,&o2,&w2}) Kokkos::deep_copy(field->direction,field->seed);
        std::printf("model %d %-7s relative FD error %.3e\n",model,name,double(max_error));
    };

    auto element = [&](const char* name, int count, auto primal, auto jvp) {
        verify(name,count,
            [&](dstype* f,const dstype* u0,const dstype* o0,const dstype* w0,const dstype*,
                const dstype*,const dstype*,const dstype*) {
                primal(f,x.data(),u0,o0,w0,eta.data(),mu.data(),time,model,ng,nc,ncu,nd,nd,nco,ncw);
            }, [&](dstype* f,dstype* df) {
                jvp(f,df,x.data(),u.value.data(),u.direction.data(),o.value.data(),o.direction.data(),
                    w.value.data(),w.direction.data(),eta.data(),mu.data(),time,model,ng,nc,ncu,nd,nd,nco,ncw);
            });
    };
    element("Flux",ng*ncu*nd,abi.volume.KokkosFlux,abi.volume.KokkosFluxJvp);
    element("Source",ng*ncu,abi.volume.KokkosSource,abi.volume.KokkosSourceJvp);
    element("Tdfunc",ng*ncu,abi.volume.KokkosTdfunc,abi.volume.KokkosTdfuncJvp);
    verify("Fbou",ng*ncu,
        [&](dstype* f,const dstype* u0,const dstype* o0,const dstype* w0,const dstype* h0,
            const dstype*,const dstype*,const dstype*) {
            abi.boundary.KokkosFbou(f,x.data(),u0,o0,w0,h0,normal.data(),tau.data(),eta.data(),mu.data(),
                                   time,model,1,ng,nc,ncu,nd,nd,nco,ncw);
        }, [&](dstype* f,dstype* df) {
            abi.boundary.KokkosFbouJvp(f,df,x.data(),u.value.data(),u.direction.data(),o.value.data(),o.direction.data(),
                w.value.data(),w.direction.data(),h.value.data(),h.direction.data(),normal.data(),tau.data(),
                eta.data(),mu.data(),time,model,1,ng,nc,ncu,nd,nd,nco,ncw);
        });
    verify("Ubou",ng*ncu,
        [&](dstype* f,const dstype* u0,const dstype* o0,const dstype* w0,const dstype* h0,
            const dstype*,const dstype*,const dstype*) {
            abi.boundary.KokkosUbou(f,x.data(),u0,o0,w0,h0,normal.data(),tau.data(),eta.data(),mu.data(),
                                   time,model,1,ng,nc,ncu,nd,nd,nco,ncw);
        }, [&](dstype* f,dstype* df) {
            abi.boundary.KokkosUbouJvp(f,df,x.data(),u.value.data(),u.direction.data(),o.value.data(),
                w.value.data(),w.direction.data(),h.value.data(),h.direction.data(),normal.data(),tau.data(),
                eta.data(),mu.data(),time,model,1,ng,nc,ncu,nd,nd,nco,ncw);
        },false,true);
    verify("Avfield",ng*std::max(1,nco),
        [&](dstype* f,const dstype* u0,const dstype* o0,const dstype* w0,const dstype*,
            const dstype*,const dstype*,const dstype*) {
            abi.volume.KokkosAvfield(f,x.data(),u0,o0,w0,eta.data(),mu.data(),time,model,
                                    ng,nc,ncu,nd,nd,nco,ncw,std::max(1,nco),npe,1);
        }, [&](dstype* f,dstype* df) {
            abi.volume.KokkosAvfieldJvp(f,df,x.data(),u.value.data(),u.direction.data(),o.value.data(),
                w.value.data(),w.direction.data(),eta.data(),mu.data(),time,model,
                ng,nc,ncu,nd,nd,nco,ncw,std::max(1,nco),npe,1);
        },false,false);
    auto face = [&](const char* name, auto primal, auto jvp, bool inout) {
        verify(name,ng*ncu,
            [&](dstype* f,const dstype* u0,const dstype* o0,const dstype* w0,const dstype* h0,
                const dstype* u1,const dstype* o1,const dstype* w1) {
                primal(f,x.data(),u0,u1,o0,o1,w0,w1,h0,normal.data(),tau.data(),eta.data(),mu.data(),
                       time,model,ng,nc,ncu,nd,nd,nco,ncw);
            }, [&](dstype* f,dstype* df) {
                jvp(f,df,x.data(),u.value.data(),u.direction.data(),u2.value.data(),u2.direction.data(),
                    o.value.data(),o.direction.data(),o2.value.data(),o2.direction.data(),
                    w.value.data(),w.direction.data(),w2.value.data(),w2.direction.data(),
                    h.value.data(),h.direction.data(),normal.data(),tau.data(),eta.data(),mu.data(),
                    time,model,ng,nc,ncu,nd,nd,nco,ncw);
            },true,true,inout);
    };
    face("Fhat",abi.iface.KokkosFhat,abi.iface.KokkosFhatJvp,false);
    face("Stab",abi.iface.KokkosStab,abi.iface.KokkosStabJvp,true);
}

int main(int argc,char** argv)
{
    Kokkos::ScopeGuard guard(argc,argv);
    try {
        if (argc > 1) run_model(std::atoi(argv[1]));
        else { run_model(1); run_model(4); }
    }
    catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
    return 0;
}
