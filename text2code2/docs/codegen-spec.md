# pdemodel.txt → my_model.hpp codegen spec

Blueprint for the Python (`pyt2c`) reimplementation of the model codegen. The C++
`text2code` is a *metaprogram*: it emits a SymEngine C++ program
(`SymbolicFunctions.cpp`, `SymbolicScalarsVectors.cpp`, `Code2Cpp.cpp`), compiles
it, and runs it to produce `generated/my_model.hpp`. `pyt2c` collapses those three
stages into one Python program using the `symengine` pip package.

Reference copies of the emitted intermediates live in `docs/reference_intermediates/`.

## 1. Parse (mirror `TextParser.hpp`)

Global keys: `scalars`, `vectors name(size),...`, `jacobian`, `hessian`, `batch`,
`outputs`, `datatype`, `framework`, `codeformat`. Then `function Name(args)` blocks
whose body lines include `output_size(out) = N;`, `matrix X(r,c);` decls, and
statements. `namevectors` = declared vector order.

## 2. Symbols

Scalar `s` → `Symbol("s")`. Vector `name` size k → `[Symbol(f"{name}{j}")]` — **no
separator** (`uq[0]`→`uq0`, `mu[4]`→`mu4`). Size-0 vector → `[]`.

## 3. DSL body → list of expressions (interpreter)

Statement forms (first match wins), `;`/comments stripped:
- `output_size(out)=N` → output list of len N.
- `matrix K(r,c)` → DenseMatrix; `vector v(n)` → list.
- `for i in a:b ... endfor` → **inclusive** `range(a, b+1)`.
- `K[i][j] = rhs` → matrix set; `v[i] = rhs` → list set; `a = rhs` → bind name.
- RHS namespace: model funcs (return lists), `inv/transpose/det/trace`, `mul`,
  math (`sin cos tan asin acos atan sinh cosh tanh exp log log10 sqrt pow fabs abs
  atan2 ...`), `Expression(...)`, `pi`. Matrix elem access `K[i][j]` via a `Mat`
  wrapper. Float literals (`2.0/3.0`) stay Python floats → decimal in output.

## 4. Per-function metadata

`inputvectors[f]` = ordered `(argname, symvec)` for vector args. `jacobianInputs[f]`
= those vector args that also appear in the global `jacobian` list, in arg order
(the `_w` slot is empty when `ncw=0` and is skipped). `outputfunctions[f]` = name in
`outputs`.

## 5. Emit `my_model.hpp` (mirror `generateModelHeader`)

Preamble + `struct PdeModel : ModelDefaults<PdeModel> {` + sizes:
`nd=|x|`, `ncu=|uhat|`, `ncw=|w|`, `nco=|v|`, `nparam=|mu|`, `ntau=|tau|`,
`Nq=ncu*(1+nd)`. Coupling (only if Fint/Fext output): `has_external_coupling=true`,
`nfint=|Fint|`, `nfext=|Fext|`, `ncuext=|uext|`.

Method tables:
- Volume value (shared sig): Flux→flux, Source→source, Tdfunc→tdfunc,
  VisScalars→vis_scalars, VisVectors→vis_vectors, QoIvolume→qoi_volume.
- Initu→initu (special sig: f,x,uinf,mu).
- Boundary per-ib (shared sig, `int ib`): Fbou→fbou, Ubou→ubou, FbouHdg→fbou_hdg,
  QoIboundary→qoi_boundary. `nbc = len(f)//szuhat`; `if(ib==k){...}`.
- Volume jac (volume_sig): flux_jac_uq/_w, source_jac_uq/_w.
- Per-ib jac: fbou_hdg_jac_{uq,w,uh}, fbou_jac_*, ubou_jac_* — widened block
  `jblock = szuhat*|input|`.
- Fint/Fext value+jac plain (Fext sig inserts `uext[]` after `n[]`).

## 6. Pointwise emitter (`emit_pointwise_value` / `_per_ib`)

- `cse(exprs)` → (replacements, reduced). Temp names come from symengine (`x0,x1,...`).
- `used` = union of free_symbols of the output list. Only load inputs in `used`.
- Load line: `const dstype {name}{j} = {rename(name)}[{j}];`
  with `rename`: `eta→uinf`, `uhat→uh`, else verbatim.
- CSE temps: `const dstype {sym} = {kokkosify(ccode(rhs))};`
- Outputs: `f[{n}] = {kokkosify(ccode(reduced[n]))};`
- **kokkosify**: regex — `\b(pow|sqrt|exp|log|sin|cos|tan|asin|acos|atan|sinh|cosh|
  tanh|fabs|atan2)\b(?=\s*\()` → `Kokkos::$1`.

## 7. Jacobian layout

Column-major: `J[j*nf + i] = ∂f[i]/∂input[j]` (input/trace index outer, output inner).
CSE over f **and** all jac blocks together, then slice back.

## Validation bar

Byte-identical to the C++ golden is a stretch (CSE temp ordering can differ across
SymEngine versions). Real bar: same sizes + method set, and the generated header
compiles into the app and produces numerically-equivalent kernels (compile+run).
Goldens: `tests/goldens/{poisson2d,isoq2d_model100}/`.
