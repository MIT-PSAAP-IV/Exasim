# State-dependent time coefficients

`pdemodel.mass(u,q,w,v,x,t,mu,eta)` defines the coefficient of `du/dt`.
It does **not** define a conserved time functional `H(u,q,w)`.
For example, `mass = C(theta)` represents `C(theta)*dtheta/dt`, not
`d(C(theta)*theta)/dt`.

## Residual and Jacobian

For fixed stage history `S` and time coefficient `a = dtfactor`, the element
source contains

```
T_i = m_i(U,w(U)) * (S_i - a*u_i),  U = (u,q).
```

Its complete derivative is

```
dT_i/dU_j = -a*m_i*delta_ij
          + (S_i-a*u_i) * (dm_i/dU_j + dm_i/dw * dw/dU_j).
```

The Kronecker term applies only to primary-unknown columns. Gradient columns,
cross-component dependencies and locally eliminated auxiliary/material fields
also require the product-rule term.

MATLAB, Python, Julia and Text2Code generate both the existing value-only
`KokkosTdfunc` and an overload returning values, partials with respect to `uq`,
and partials with respect to `w`. The latter is stored in `HdgTdfunc.cpp`.
Text2Code generates these mass partials even without a `jacobian` declaration.
Both use Exasim's existing component-major arrays, with partial index
`point + ng*(output + ncu*input)`.

The provider exposes the overload through `hdgjac.HdgTdfunc`. The overloaded
`TdfuncDriver` supplies it to HDG element assembly. `ApplyMassTime` adds the
direct partials and the `w` partials before the existing `sg_w*wg_uq` product,
so the local elimination chain is applied once. Values and partials borrow
dead flux scratch, which is reset before flux evaluation. No new numerical
allocation, communication, host staging or global unknown is introduced.

Hand-written C++ models with state-dependent `tdfunc` must supply
`tdfunc_jac_uq` and `tdfunc_jac_w`. Their zero defaults apply only to
unknown-independent coefficients. Native symbolic frontends generate these
partials automatically; existing `mass` model-function signatures are unchanged.

## Compatibility

- Residual-only evaluation and finite-difference matrix-vector products retain
  the value-only kernel.
- The identity-mass path (`tdfunc=0`) is unchanged.
- Unknown-independent mass has zero additional partials; focused tests check
  the old arithmetic bitwise on CPU Serial.
- Provider ABI version is **6**. Rebuild Exasim libraries and regenerate/rebuild
  application providers together; old binary providers are not compatible.
- Checked-in legacy built-ins 1--15 have audited unknown-independent mass
  coefficients (model 5 depends on geometry/time/parameters). The CMake
  fallback adapter is restricted to those directories, not custom models.

## Focused verification

After configuring/building the test targets:

```sh
ctest --test-dir <build> --output-on-failure \
  -R '^(mass_jacobian|frontend_python_mass_derivatives|text2code_tdfunc_jacobian|wequation_material_split|hdg_sourcew_equivalence)$'
```

The Text2Code test requires a temporary or ordinary installation of the
configured build and its generator; no existing installation needs overwriting.

`mass_jacobian` compares all assembled Q1 element `D/B` columns with centered
finite differences at two perturbation sizes. It includes nonzero source and
flux terms, constant mass, own/cross-primary dependence, gradient dependence,
and ordinary local-`w` elimination. It also checks residual parity, scratch
canaries, constant-coefficient arithmetic and identity-mass preservation.

## Original source-branch verification

The following records the original fix's evidence, not a rerun on the target
branch. Target-branch transfer checks are recorded separately below.

Source: `cuong-hexascale`, base commit `33716f0d46761a38a3dda24ad07344a0f8ff0794`.
Changes are uncommitted. AppleClang 21, Release, FP64, Kokkos 4.7 Serial:

- PASS: CPU solver/preprocessing static libraries and all regenerated built-in
  providers compile.
- PASS: all six focused tests above. Maximum scaled assembled-Jacobian error
  `5.53e-10`.
- PASS: focused ASan/UBSan Debug test, with leak detection disabled and external
  Kokkos/BLAS dependencies uninstrumented.
- PASS: existing Helmholtz and built-in model 4 kernel-equivalence checks.
- PASS: Julia/SymPy generation for nonlinear and absent steady mass.
- PASS: ordinary Python `exasim(pde,mesh)` transient manufactured solve with
  `m(u)=1+u^2`, source `1+u^2`, and exact spatially constant `u=t` after two
  backward-Euler steps. Maximum final errors: HDG `1.6e-15`, LDG `5.4e-12`.
- PASS: all 15 legacy built-in mass adapters preserve values and return zero
  partials in their focused probe.
- NOT RUN: MATLAB execution (batch process exits before diagnostics), CUDA,
  HIP, MPI and the full CTest suite.
- Existing limitations: the separate Python generator-inventory suite has two
  unrelated baseline failures; a complete no-generator legacy built-in build
  encounters missing SurfaceQuantities functions in models 7--15. The normal
  regenerated build above passes; neither unrelated issue was modified.

Build and launch evidence is under `/tmp/exasim-mass-fix.4oscwB86/`;
sanitizer evidence under `/tmp/exasim-mass-regression.Soh3lR/`;
Text2Code and legacy-adapter probes under `/tmp/mass-text2code.6IWPZx/`.
Existing installed prefixes were not modified; the validation installation is
under the first temporary directory.

## Transfer to cuong-nonmatching-interface

Transferred from the uncommitted fix on `cuong-hexascale` at
`33716f0d46761a38a3dda24ad07344a0f8ff0794` onto
`cuong-nonmatching-interface` at
`62b14a239c67ea6b5d079a604a54483caeb6617f`.

The target uses packaged Python/Julia generators only. The transfer updates
those live paths without restoring removed legacy generator directories.
Target-specific optional material-state behavior and unrelated CTest entries
are preserved.

Verified on CPU Serial, Release, FP64:

- PASS: fresh Text2Code build and complete CPU solver static-library build,
  including its built-in, Helmholtz and linear-elasticity provider dependencies.
- PASS: five registered focused tests: `mass_jacobian`,
  `wequation_material_split`, `helmholtz_model`,
  `frontend_python_mass_derivatives`, and `hdg_sourcew_equivalence`.
- PASS: assembled constant/state/gradient/local-w Jacobian checks; maximum
  scaled error `5.53e-10` at the unchanged test gates.
- PASS: independent Text2Code mass test, including generation without an
  explicit Jacobian declaration and typed/ABI value/partial parity.
- PASS: Julia/SymPy generation of constant, state-dependent and absent-steady
  mass derivative overloads.
- PASS: existing `auxiliary_eos_trial` and `ldg_matvec_consistency` regressions.
- PASS: `mass_jacobian`, `auxiliary_eos_trial` and `ldg_matvec_consistency`
  under ASan/UBSan, with leak checks disabled and external Kokkos/BLAS
  dependencies uninstrumented.
- NOT RUN: MATLAB execution, CUDA, HIP, MPI, full CTest and end-to-end
  application launch against a rebuilt installed package.

Evidence: `/tmp/exasim-mass-transfer.qE0awUvY/`. No installed prefix was
modified, and no changes were committed or pushed. Existing `.gitignore` and
ignored thermoviscoplasticity documents were checksum-verified unchanged.
Before ordinary application use, rebuild the installed libraries and regenerate
providers together for ABI version 6.

