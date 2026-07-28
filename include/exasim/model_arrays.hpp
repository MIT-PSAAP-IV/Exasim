// model_arrays.hpp — owns the per-model bookkeeping arrays (common.ncarray / sol.udgarray)
// that the multi-model kernels index, for a consumer driving a SINGLE CSolution.
//
// WHY THIS EXISTS
// ---------------
// The kernels reach other models' component counts and volume states through
//     common.nomodels, common.ncarray[j], sol.udgarray[j]
// which are RAW POINTERS the owner has to allocate, populate and keep alive for as long as
// the solution is used. Exasim populates them in ExasimSolver::InitializeModels (new[] +
// delete[] in DestroyModels) and again in postprocess.cpp; a consumer with one model has to
// reproduce the single-model case by hand. The CHEFSI isoq2d_cht-petsc-fluid app did:
//
//     model.disc.common.nomodels = 1;
//     ncarray_  = { model.disc.common.components.nc };   // std::vector members the app
//     udgarray_ = { &model.disc.sol.udg[0] };            //   had to keep alive itself
//     model.disc.common.ncarray = ncarray_.data();
//     model.disc.sol.udgarray   = udgarray_.data();
//
// That is a lifetime coupling with no compiler enforcement: if those vectors are moved,
// reallocated or destroyed before the solution stops being used, the kernels read freed
// memory and nothing diagnoses it.
//
// This class puts the storage and the pointers into ONE object, so they cannot get out of
// step, and nulls the borrowed pointers on destruction so a stale read fails loudly (null)
// instead of silently (freed).
#pragma once

#include <vector>

template <class M> class CSolution;

namespace exasim {

// Single-model bookkeeping arrays, owned.
//
// Construct after the solution exists and before it is solved; keep it alive as long as the
// solution is used. Non-copyable, because it publishes pointers into its own storage.
template <class M>
class ModelArrays {
public:
    ModelArrays() = default;
    explicit ModelArrays(CSolution<M>& model) { attach(model); }
    ~ModelArrays() { detach(); }

    ModelArrays(const ModelArrays&) = delete;
    ModelArrays& operator=(const ModelArrays&) = delete;

    // Point the solution's bookkeeping arrays at storage this object owns.
    void attach(CSolution<M>& model)
    {
        detach();
        model_ = &model;
        model.disc.common.nomodels = 1;
        nc_  = { model.disc.common.components.nc };
        udg_ = { &model.disc.sol.udg[0] };
        model.disc.common.ncarray = nc_.data();
        model.disc.sol.udgarray   = udg_.data();
    }

    // Null the borrowed pointers before the storage dies. Without this the solution would
    // keep pointers into freed vectors — the exact hazard this class exists to remove.
    void detach()
    {
        if (!model_) return;
        model_->disc.common.ncarray = nullptr;
        model_->disc.sol.udgarray   = nullptr;
        model_ = nullptr;
        nc_.clear();
        udg_.clear();
    }

private:
    CSolution<M>*        model_ = nullptr;
    std::vector<Int>     nc_;
    std::vector<dstype*> udg_;
};

} // namespace exasim
