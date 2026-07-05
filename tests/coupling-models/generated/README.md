# text2code-GENERATED concrete coupled models (Fint/Fext)

These `*_my_model.hpp` are the `PdeModel` structs that **text2code emits**
into `my_model.hpp` for a coupled PDE (one that defines `Fint`/`Fext`), on the
concrete (templated) model path — the analog of the checked-in ABI kernels in
`../reference/`, but for the pure-template side. They exercise the codegen half
of the interface-coupling surface: text2code must set
`static constexpr bool has_external_coupling = true;` + `nfint`/`nfext`/`ncuext`
and emit concrete `fint`/`fext` (+ their `_jac_uq/_w/_uh` companions) with the
SAME trace/input-index-outer Jacobian SoA layout the `fint_kernel<M>` /
`fext_kernel<M>` templates consume.

`tests/coupling-models/compare_generated_fint_fext.cpp` drives the GENERATED
`PdeModel` through those kernels and asserts byte-identical residual + Jacobian
buffers against the ABI `HdgFint`/`HdgFext` references in `../reference/` — the
generated-model analog of `coupling_fint_fext_equivalence` (which used
hand-written coupled models).

- `pde2_my_model.hpp`  — generated from `apps/poisson/poisson2d/pdemodel2.txt`
  verbatim (`Fint` has 2 output components with ncu==1; `Fext` reads `uext`).
  Real app coverage; matches `../reference/pde2_*`.
- `probe_my_model.hpp` — generated from `probe_model.txt` (pdemodel2 with the
  trivial `Fint`/`Fext` bodies replaced by ones with **non-zero** `uq`- and
  `uhat`-derivatives), so the test locks the input-index-outer Jacobian SoA
  layout that the all-zero pdemodel2 `uq`-Jacobians cannot distinguish. Matches
  `../reference/probe_*`.

Regenerate (only if the codegen changes) with the coupling-aware text2code:

    text2code <pdeapp referencing pdemodel2.txt>  --out-dir <dir> --gen-only   # -> pde2_my_model.hpp
    text2code <pdeapp referencing probe_model.txt> --out-dir <dir> --gen-only  # -> probe_my_model.hpp

They are checked in (not regenerated at test-build time) so the equivalence test
needs no text2code / SymEngine toolchain in the ctest environment.
