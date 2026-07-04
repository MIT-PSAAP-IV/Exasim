# ABI reference kernels for the Fint/Fext equivalence test

These `*_HdgFint.cpp` / `*_HdgFext.cpp` files are the **libpdemodel ABI**
interface-coupling kernels, emitted by `text2code` (the SymEngine symbolic
codegen path) from two model files. They are the ground-truth reference that
`tests/coupling-models/compare_fint_fext.cpp` compares the templated
`exasim::fint_kernel<M>` / `exasim::fext_kernel<M>` against, byte-for-byte —
the same relationship `compare_model4.cpp` has with `backend/Model/BuiltIn/model4/`.

- `pde2_*`  — generated from `apps/poisson/poisson2d/pdemodel2.txt` verbatim
  (the one canonical app that defines `Fint`/`Fext`). `Fint` has 2 output
  components (with ncu==1), `Fext` has 1 and reads `uext`.
- `probe_*` — generated from the same model with the trivial `Fint`/`Fext`
  bodies replaced by ones with **non-zero** `uq`- and `uhat`-derivatives, so
  the test actually exercises the trace/input-index-outer Jacobian SoA layout
  (`J[(j*nf+o)*ng+i] = ∂f[o]/∂input[j]`), which the trivial pdemodel2 bodies
  (all-zero `uq` Jacobians) cannot distinguish.

Regenerate (if the codegen changes) with:

    text2code <pdeapp.txt referencing the model> --out-dir <dir>

They are checked in (not regenerated at test-build time) so the equivalence
test needs no text2code / SymEngine toolchain in the ctest environment.
