#ifndef EXASIM_AUXILIARY_SNAPSHOT_HPP
#define EXASIM_AUXILIARY_SNAPSHOT_HPP

// A trial residual overwrites the auxiliary Newton initial guess in sol.wdg.
// Own a separate device-compatible copy, including halo elements. Never borrow
// tmp/res scratch: residual evaluation and GMRES overwrite those arenas.
// common.h and kokkosimpl.h must have been included by the backend caller.
template <class T> class AuxiliaryStateSnapshot {
    T* state_;
    int size_;
    bool rollback_ = true;
    Kokkos::View<T*> saved_;
public:
    AuxiliaryStateSnapshot(T* state, int size)
        : state_(state), size_(size),
          saved_(Kokkos::view_alloc(Kokkos::WithoutInitializing, "auxiliary_snapshot"), size)
    {
        if (size_) ArrayCopy(saved_.data(), state_, size_);
    }
    AuxiliaryStateSnapshot(const AuxiliaryStateSnapshot&) = delete;
    AuxiliaryStateSnapshot& operator=(const AuxiliaryStateSnapshot&) = delete;
    void restore() const {
        if (size_) ArrayCopy(state_, saved_.data(), size_);
    }
    void commit() { rollback_ = false; }
    ~AuxiliaryStateSnapshot() {
        if (rollback_) restore();
        // The saved allocation must outlive any asynchronous restore kernel.
        if (size_) Kokkos::fence();
    }
};

#endif
