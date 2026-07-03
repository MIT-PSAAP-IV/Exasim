# Precision threading: `dstype`/`Int` → `floatTy`/`intTy` template parameters

Exasim's scalar and index precision are **global typedefs** chosen at build time:

```cpp
// backend/Common/common.h
#ifdef USE_FLOAT      // -DEXASIM_FLOAT
typedef float  dstype;
#else
typedef double dstype;   // default
#endif
#ifdef USE_LONG       // -DEXASIM_INT64
typedef long Int;
#else
typedef int  Int;        // default
#endif
```

`dstype` appears in **~692 files** and `Int` in **~103**; the data structs hold raw `dstype*`/`Int*`
and every kernel takes them. So a build is single **or** double, 32- **or** 64-bit indices — never
both at once. The goal of this work is to make precision a **template parameter** so a single build
can serve multiple precisions (mixed-precision solvers; a PETSc consumer choosing `PetscScalar` at
the type level) without a rebuild.

This is a large refactor. It is being done in phases, **each phase keeping the tree compiling and
the golden app-regression at `rel_L2 = 0`** (a precision-preserving refactor: with the default
template arguments the generated code is byte-identical).

## Phase 0 — Boundary (DONE)
Establish the named types and the safety guard at the **exported interface** only; no backend change.
- `exasim::floatTy` / `exasim::intTy` aliases (`include/exasim/export.hpp`) — the canonical names
  consumers spell precision with, instead of the internal `dstype`/`Int`.
- `static_assert(sizeof(PetscScalar)==sizeof(floatTy) && sizeof(PetscInt)==sizeof(intTy))` in
  `petsc.hpp` — a wrong-precision PETSc build is now a compile error, not a silent ABI reinterpret.
- `scalar_type` / `index_type` member aliases on `Operator` / `ShellMat` so a consumer can query the
  exported precision generically.

## Phase 1 — Struct templating (IN PROGRESS)
Template the core data structs on `<class T = ::dstype, class I = ::Int>`, keeping the current names
as defaulted aliases so all existing code compiles unchanged.

**Proven low-risk pattern — shadow the typedefs, leave the body untouched.** Instead of editing every
`dstype`/`Int` in a struct body, add two member `using` aliases that shadow the globals; the struct
body is then *byte-for-byte unchanged* and, under the default args, the type is identical to before:
```cpp
template <class T = ::dstype, class I = ::Int>
struct solstructT {
    using dstype = T; using Int = I;   // <-- shadow; body below unchanged, resolves to T*/I*
    dstype* udg = nullptr;  Int* elemcon = nullptr;  Int szudg = 0;  ...  // (verbatim)
};
using solstruct = solstructT<::dstype, ::Int>;   // unchanged for every current user
```
Each struct is a 3-line edit (template line + `using` line + alias line). Verified byte-identical two
ways: the robustness harness compiles it header-inline and PASSes, and a full library rebuild +
`run-app-regression.sh` gives the *same* rel_L2 values as before (poisson3d 9.99e-11, isoq3d 1.95e-9,
…). Nested struct members (e.g. `commonstruct` holding `sizesstruct`) stay the default alias for now —
threading `T,I` into nested types is the tail of this phase / Phase 2.

- **Done + verified:** `sysstruct`, `scratcharenastruct`, `resstruct`, `tempstruct`, `appstruct`,
  `masterstruct`, `meshstruct`, `solstruct`, `precondstruct`, `commonstruct` (10 core structs).
- **Remaining:** the mostly-int param structs (`sizesstruct`, `gridstruct`, `componentsstruct`, the
  `*paramsstruct` family — low value, few/no `dstype*`), and the `Kokkos::View<dstype*>` / `view_1d`
  aliases → `view_1d<T>`. Then thread `T,I` into nested struct members (currently default aliases).

## Phase 2 — Class templating (IN PROGRESS)
Extend the FEM classes to carry `T=dstype, I=Int` so they hold the *Phase-1 `<T,I>` structs* rather
than the fixed-precision aliases.

**`CDiscretization` — DONE + verified.** This was the hard one: unlike `CResidual`/`CAssembler`/
`CPreconditioner`/`CSolver` (already `<M>` templates that only hold a *reference* to it), it is the
struct *owner* (holds `solstruct`/`resstruct`/`commonstruct`/… **by value**) and was a plain
non-template class with out-of-line methods split across two places. It is now
`template<class T=::dstype,class I=::Int> class CDiscretizationT`, with `using CDiscretization =
CDiscretizationT<::dstype,::Int>;`. Two mechanisms made it precision-preserving:
- **Shadow aliases inside the class** (`using dstype=T; using Int=I;` **and** `using solstruct =
  solstructT<T,I>;` … for every Phase-1 struct) so every member declaration and every out-of-line
  method body is *byte-for-byte unchanged* yet now resolves to the `<T,I>` struct instantiations.
- **`extern template` (in `discretization.h`) + explicit instantiation (in `discretization.cpp`)**
  for the backend-defined members (file ctor, dtor, `finalizeConstruction`, `compGeometry`,
  `compMassInverse`, `DG2CG{,2,3}`). This preserves the *compile-once* TU split: `main.cpp` builds
  `CSolution<M>`, whose inline ctor/dtor construct/destroy a `CDiscretization disc` **by value**, but
  it never sees those out-of-line bodies (they live only in the unity `ExasimSolver.cpp`). The
  in-memory ctors (`discretization_inmemory.hpp`, consumer-only) are deliberately left to implicit
  instantiation — consumer TUs that include that header see their definitions. The 8 `class
  CDiscretization;` forward declarations across the backend became `template<class,class> class
  CDiscretizationT; using CDiscretization = …;`.
- Verified: consumer path (in-memory ctor, pure implicit) robustness `ALL PASS`; **backend path**
  (file ctor via `extern template`→ library) full rebuild + `main.cpp` link + app-regression at the
  *same* rel_L2 (poisson3d 9.99e-11, isoq3d 1.95e-9, poisson2d 2.29e-12). Landed on the first full
  build.

**FEM classes — DONE + verified.** `CResidual<M>`, `CAssembler<M>`, `CPreconditioner<M>`, `CSolver<M>`
are now `<M, T=::dstype, I=::Int>`. These were *much* easier than `CDiscretization`: they are already
`<M>` templates (header-inline via the `operators.hpp` aggregation, so consumers instantiate them
locally — no `extern template`/explicit-instantiation TU split) that only hold a *reference* to the
owner. Same shadow-alias pattern (`using dstype=T; using Int=I; using CDiscretization =
CDiscretizationT<T,I>;` + `sysstruct`/`precondstruct` where used) keeps member decls + out-of-line
bodies byte-identical; the only explicit edits were the cross-references between them
(`CAssembler<M>`→`CAssembler<M,T,I>` etc. in the solve-surface signatures) and the 15 forward
declarations across the backend. One gotcha fixed: `CSolver::gmres` returns `Int` — the out-of-line
return type is looked up in *namespace* scope (not the class), so it had to be spelled `I` (the
template param) to match the in-class `Int`(=I); parameter types after the `CSolver<M,T,I>::` are in
class scope and needed no change. `ptcsolver.cpp`/`gmres.cpp` also define `CSolver` methods; the
`ApplyPoly<M>` free helpers in `gmres.cpp` stay `<M>` (their default-precision refs match under
`T=dstype`). Verified: consumer robustness `ALL PASS` + full backend rebuild + app-regression through
the real solver (gmres/PTC) at the same rel_L2 (naca0012 2.94e-13, poisson3d 9.99e-11, isoq3d 1.95e-9).

**Shim — DONE + verified.** The `include/exasim` PETSc shim now carries the exported precision as
template parameters: `Operator<M, Scalar=floatTy, Idx=intTy>`, `ShellMat<Scalar=floatTy>`, and
`make_mass_inverse<M,Scalar,Idx>` / `assemble_matrix<M,Scalar,Idx>`. `Operator` holds
`CDiscretizationT<Scalar,Idx>&` + `CAssembler<M,Scalar,Idx>&` + `CPreconditioner<M,Scalar,Idx>&` +
`sysstructT<Scalar,Idx>&`, exposes `scalar_type = Scalar` / `index_type = Idx`, and the zero-copy
Vec/Mat wrapping reinterprets PETSc's `PetscScalar`/`PetscInt` arrays as `Scalar`/`Idx`. The old
*global* ABI `static_assert(sizeof(PetscScalar)==sizeof(floatTy))` is gone — each wrapper carries its
own `static_assert(sizeof(PetscScalar)==sizeof(Scalar))`, so it fires only for the precision you
actually instantiate and including the header no longer forces a TU-wide precision match. (`floatTy`
*is* `dstype`, so the default `Operator<M>` = `Operator<M,dstype,Int>` stays byte-identical; the
threading makes precision an explicit, queryable property of the exported type.) Consumers spell bare
`ShellMat` as `ShellMat<>` (it is now a template; CTAD can't deduce `Scalar` from the apply lambda).
Verified: robustness `ALL PASS` + petsc_poisson (SNES resid 9.08e-14, ShellMat vs matrix-free 0.0,
MATAIJ vs matrix-free 2.65e-16) + petsc_heat (PETSc TS drives the heat loop) — all PASS, values
identical to pre-shim.

**Phase 2 is complete:** every layer from the data structs (Phase 1) up through the struct owner
(`CDiscretization`), the FEM classes (`CResidual`/`CAssembler`/`CPreconditioner`/`CSolver`), and the
PETSc export shim now threads `<T,I>` with `dstype`/`Int` as default arguments — the whole stack is
precision-parameterized and byte-identical under the defaults.

## Phase 3 — Kernels + BLAS + Kokkos dispatch
- Compute kernels (`backend/Discretization/*.hpp`, `Common/{cpuimpl,kokkosimpl}.h`) take `T*`.
- BLAS dispatched by `T`: `dgemm`/`dgetrf` for `double`, `sgemm`/`sgetrf` for `float` (a small
  `blas<T>` trait). `pblas.h` is the choke point.
- `Kokkos::View<T*>` throughout; parallel kernels are already templated on the functor.

## Phase 4 — Codegen kernels
`text2code` / the model codegen (`backend/Model/**`, `frontends/*/Gencode`) emit `dstype`-typed
kernels. Either template the generated signatures on `T` or generate a per-precision set. This is the
last mile (generated code is the bulk of the 692 files) and the highest-churn.

## Phase 5 — Cutover
Once everything is templated with `dstype`/`Int` as *default* args, mixed precision is available:
instantiate a `float` solve inside a `double` build, or let the PETSc consumer pick. The global
`dstype`/`Int` typedefs remain only as the default arguments.

## Strategy & verification
- **Bottom-up**: structs → classes → kernels → codegen. Each phase leaves `dstype`/`Int` as defaulted
  aliases so the whole tree keeps compiling; land it incrementally, file group by file group.
- **Verify precision-preserving**: after each phase, `tests/run-app-regression.sh` must stay at
  `rel_L2 = 0` (default args ⇒ identical code) — the golden proof the refactor changed no numerics.
- **Then** add a genuinely mixed-precision test (single-precision inner solve) to exercise the new
  degree of freedom.
