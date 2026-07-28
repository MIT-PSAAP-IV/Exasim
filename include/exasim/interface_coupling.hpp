// interface_coupling.hpp — buffer-owning interface geometry + flux exchange for one coupled
// boundary, model-templated so BOTH the runtime-ABI path (ExasimSolver, M = AbiAdapter) and
// the concrete-template path (a consumer's own CSolution<PdeModel>) share one implementation.
//
// WHY THIS EXISTS
// ---------------
// The setup is ~50 lines of allocation whose sizes are derived from Exasim internals
// (common.components.ncx, grid.npf / ngf / nd), plus ownership of sol.uext, plus a four-call
// sampler sequence that must happen in order, plus the couplingparams.FextCall = ibc+1
// "arm the BC" convention. None of that is a consumer's business, and getting a size wrong is
// a silent out-of-bounds rather than an error.
//
// It previously lived inside ExasimSolver::IntializeMeshInterface, reachable only from the ABI
// path because ExasimSolver is bound to CSolution<AbiAdapter>. A consumer on the no-ABI
// concrete-model path could not call it, so the CHEFSI isoq2d_cht-petsc-fluid app copied it
// verbatim — that copy's own comment reads "lifted from ExasimSolver::IntializeMeshInterface".
// This class is that code, extracted once, with both callers routed through it.
//
// ONE TEMPLATE COVERS BOTH PATHS because the sampler's non-templated flux entry point is
// itself the templated one at AbiAdapter:
//     CInterfaceSampler::getInterfaceFluxesAtGaussPoints(...)
//         -> getInterfaceFluxesAtGaussPointsFor<exasim::detail::AbiAdapter>(...)
//   (backend/Discretization/interfacesampler.cpp:144)
// so InterfaceCoupling<AbiAdapter> reproduces the ABI behaviour byte-for-byte, while
// InterfaceCoupling<ConcreteModel> routes through the templated FintDriver<M> with no ABI.
// Everything else — buffers, sizes, sampler sequence, uext, FextCall — was already identical
// between the two copies and is shared verbatim here.
//
// Include AFTER the solution/sampler types are visible (e.g. from operators.hpp or
// ExasimSolver's include chain); only CSolution<M>'s `disc` and `sampler` members are used.
#pragma once

#include <vector>

// Declared in solution.h / solution.hpp; only ever used through a pointer here.
template <class M> class CSolution;

namespace exasim {

// A point on the coupled interface, in physical coordinates. Kept here so a consumer of this
// header needs nothing from ExasimSolver.hpp.
struct InterfacePoint {
    dstype x = 0.0;
    dstype y = 0.0;
    dstype z = 0.0;
};

// Owns the interface geometry buffers and the flux exchange for ONE coupled boundary
// condition of ONE model.
//
// M is the model the FLUX EXTRACTION is threaded on: exasim::detail::AbiAdapter for the
// runtime-ABI path, a concrete text2code model for the no-ABI path.
//
// Lifetime: RAII for the six interface buffers it allocates. It does NOT own sol.uext —
// that lives in the solution's solstruct, which frees it itself (common.h:1343); this class
// only replaces it on re-initialise, exactly as the original ExasimSolver body did.
template <class M>
class InterfaceCoupling {
public:
    InterfaceCoupling() = default;
    ~InterfaceCoupling() { release(); }

    // Non-copyable: owns raw device allocations and writes a borrowed pointer into sol.uext.
    InterfaceCoupling(const InterfaceCoupling&) = delete;
    InterfaceCoupling& operator=(const InterfaceCoupling&) = delete;

    // Geometry, valid after initialize(). Coupled drivers size their arrays as nb = ngf*nfaces.
    int ngf = 0, nfaces = 0, npf = 0, ncx = 0;

    // Allocate the interface buffers and sample the geometry for boundary condition `ibc`.
    //   ncuext — components of the EXTERNAL state pushed in (sizes sol.uext)
    //   ncuint — components of the INTERNAL flux pulled out
    // Safe to call again: any previous allocation is released first.
    int initialize(CSolution<M>& model, int ncuext, int ncuint, int ibc)
    {
        release();
        // Re-init only: drop any previous uext before allocating a new one. This mirrors
        // the original ExasimSolver body; the solstruct frees whatever is left at the end.
        if (model.disc.sol.uext) {
            TemplateFree(model.disc.sol.uext, static_cast<int>(model.disc.common.backend));
            model.disc.sol.uext   = nullptr;
            model.disc.sol.szuext = 0;
        }
        model_   = &model;
        backend_ = static_cast<int>(model.disc.common.backend);
        ncuext_  = ncuext;
        ncuint_  = ncuint;
        ibc_     = ibc;

        auto& disc = model.disc;
        ncx    = static_cast<int>(disc.common.components.ncx);
        npf    = static_cast<int>(disc.common.grid.npf);
        ngf    = static_cast<int>(disc.common.grid.ngf);
        // The sampler allocates faces_; we own it from here.
        nfaces = static_cast<int>(model.sampler.getFacesOnInterface(&faces_, ibc + 1));

        const Int nd = disc.common.grid.nd;

        disc.common.couplingparams.ncuext = ncuext;
        TemplateMalloc(&disc.sol.uext, ngf * nfaces * ncuext, backend_);
        disc.sol.szuext = ngf * nfaces * ncuext;

        TemplateMalloc(&xdgint_,   npf * nfaces * ncx,    backend_);
        TemplateMalloc(&nlint_,    npf * nfaces * nd,     backend_);
        TemplateMalloc(&xdggint_,  ngf * nfaces * ncx,    backend_);
        TemplateMalloc(&nlgint_,   ngf * nfaces * nd,     backend_);
        TemplateMalloc(&flux_dev_, ngf * nfaces * ncuint, backend_);

        // Order matters: nodes -> normals (needs nodes) -> Gauss interpolation of each.
        model.sampler.getDGNodesOnInterface(xdgint_, faces_, nfaces);
        model.sampler.getNormalVectorOnInterface(nlint_, xdgint_, nfaces);
        model.sampler.getFieldsAtGaussPointsOnInterface(xdggint_, xdgint_, nfaces, ncx);
        model.sampler.getFieldsAtGaussPointsOnInterface(nlgint_,  nlint_,  nfaces, nd);
        return 0;
    }

    // Physical coordinates of the interface Gauss points, host-side.
    std::vector<InterfacePoint> points() const
    {
        std::vector<InterfacePoint> pts;
        const int ngb = nfaces * ngf;
        if (ngb <= 0 || xdggint_ == nullptr) return pts;
        std::vector<dstype> xh(static_cast<size_t>(ncx) * static_cast<size_t>(ngb));
        TemplateCopytoHost(xh.data(), xdggint_, ncx * ngb, backend_);
        pts.reserve(ngb);
        for (int i = 0; i < ngb; ++i) {
            InterfacePoint p{};
            p.x = xh[0 * ngb + i];
            p.y = (ncx > 1) ? xh[1 * ngb + i] : 0.0;
            p.z = (ncx > 2) ? xh[2 * ngb + i] : 0.0;
            pts.push_back(p);
        }
        return pts;
    }

    // Extract the interface flux this model produces (the "send" half of the exchange).
    // Threaded on M: AbiAdapter -> loaded ABI, concrete model -> templated FintDriver<M>.
    void fluxes_out(std::vector<dstype>& send_flux) const
    {
        const Int sz = ngf * nfaces * ncuint_;
        send_flux.resize(static_cast<size_t>(sz));
        if (backend_ > 1) {
            model_->sampler.template getInterfaceFluxesAtGaussPointsFor<M>(
                flux_dev_, xdggint_, nlgint_, faces_, nfaces);
            TemplateCopytoHost(send_flux.data(), flux_dev_, sz, backend_);
        } else {
            model_->sampler.template getInterfaceFluxesAtGaussPointsFor<M>(
                send_flux.data(), xdggint_, nlgint_, faces_, nfaces);
        }
    }

    // Push the partner's state in (the "receive" half) and ARM the external-flux boundary
    // term. FextCall is 1-based on purpose — 0 means "no coupling BC active" — which is why
    // it is ibc+1, a convention that used to be an unexplained magic number at each call site.
    void fluxes_in(const std::vector<dstype>& recv_flux)
    {
        auto& disc = model_->disc;
        TemplateCopytoDevice(disc.sol.uext, recv_flux.data(),
                             ngf * nfaces * disc.common.couplingparams.ncuext, backend_);
        disc.common.couplingparams.FextCall = ibc_ + 1;
    }

    // Raw buffer accessors, for callers that must reach the buffers directly.
    const dstype* raw_xdggint() const { return xdggint_; }
    const dstype* raw_nlgint()  const { return nlgint_;  }
    const Int*    raw_faces()   const { return faces_;   }
    dstype*       raw_fluxdev() const { return flux_dev_; }

    // Release early (the destructor does this too). Idempotent.
    void release()
    {
        if (faces_)    { TemplateFree(faces_, backend_);    faces_    = nullptr; }
        if (xdgint_)   { TemplateFree(xdgint_, backend_);   xdgint_   = nullptr; }
        if (nlint_)    { TemplateFree(nlint_, backend_);    nlint_    = nullptr; }
        if (xdggint_)  { TemplateFree(xdggint_, backend_);  xdggint_  = nullptr; }
        if (nlgint_)   { TemplateFree(nlgint_, backend_);   nlgint_   = nullptr; }
        if (flux_dev_) { TemplateFree(flux_dev_, backend_); flux_dev_ = nullptr; }
        // NOTE: sol.uext is deliberately NOT freed here. It lives in the solution's own
        // solstruct, which frees it in its cleanup path (backend/Common/common.h:1343).
        // Freeing it here as well is a DOUBLE FREE that crashes at teardown — which is
        // exactly what happened when this helper first took ownership of it: poisson2d
        // segfaulted on exit while isoq2d happened to survive. The original
        // ExasimSolver::IntializeMeshInterface freed uext only when RE-initialising (to
        // drop the previous allocation before making a new one), never at destruction;
        // free_uext_for_reinit() below reproduces exactly that.
        model_ = nullptr;
        ngf = nfaces = npf = ncx = 0;
    }

private:
    CSolution<M>* model_ = nullptr;

    Int*    faces_    = nullptr;
    dstype* xdgint_   = nullptr;
    dstype* nlint_    = nullptr;
    dstype* xdggint_  = nullptr;
    dstype* nlgint_   = nullptr;
    dstype* flux_dev_ = nullptr;

    int  backend_   = 0;
    int  ncuext_    = 1;
    int  ncuint_    = 1;
    int  ibc_       = 0;
};

} // namespace exasim
