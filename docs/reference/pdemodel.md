# pdemodel.txt syntax reference

`pdemodel.txt` is a small line-oriented DSL that text2code parses
(`TextParser.hpp`) and turns into SymEngine-based C++ kernels
(`CodeGenerator.cpp`). This page documents the **syntax**; the **semantics** of
each function — what `flux`, `source`, etc. mean — live in the
[model contract](model-contract.md) and the [theory](../theory/index.md) pages,
which this page cross-links.

A file has two parts, in order: **header declarations** (`keyword value` lines),
then **function blocks** (`function NAME(args) … end`). Blank lines are ignored.

!!! note "Conventions vs. enforcement"
    The parser does not validate the *names* in `vectors` or their order — the
    conventional meanings below (`x`, `uq`, `w`, …) are enforced by the Exasim
    model contract and the C++ drivers, not by the parser. The only hard error is
    using a function argument that was not declared as a scalar or vector.

## Header declarations

| Declaration | Form | Meaning |
|---|---|---|
| `scalars` | `scalars t` | Scalar symbolic inputs; each becomes a `SymEngine::Expression`. |
| `vectors` | `vectors x(2), uq(3), …` | Vector symbolic inputs and their lengths; each becomes a `std::vector<Expression>` with components `name0, name1, …`. Size `0` = declared but empty. |
| `jacobian` | `jacobian uq, w, uhat` | Input vectors w.r.t. which first derivatives of each output are generated (`.diff(...)`). May be empty. |
| `hessian` | `hessian` | Input vectors w.r.t. which second derivatives are generated. Often empty. |
| `batch` | `batch x, uq, …` | Input vectors batched over the per-point (quadrature/element) loop in the kernels. |
| `outputs` | `outputs Flux, Source, …` | The function blocks to actually emit. A `function` not listed here is parsed but not emitted. |
| `datatype` | `datatype dstype` | (optional) C++ scalar type for kernel signatures. Default `dstype`. |
| `framework` | `framework kokkos` | (optional) Backend target. Default `kokkos`. |
| `codeformat` | `codeformat exasim` | (optional) Default `exasim`; enables the Exasim kernel path and requires the six core functions (below). |

### Conventional vector symbols

| Symbol | Meaning |
|---|---|
| `x` | spatial coordinates (size = dimension) |
| `uq` | solution + gradients packed: `uq[0..ncu-1]` is $u$, the rest are gradients $q$ |
| `w` | auxiliary scalar (`wdg`) field(s) |
| `v` | auxiliary "other DG" (`odg`) field(s) |
| `uhat` | trace unknown on faces (HDG hybrid variable $\hat u$) |
| `uext` | externally supplied face data (used by `Fext`) |
| `n` | outward face normal |
| `tau` | stabilization parameter(s) |
| `mu` | physics parameters |
| `eta` | additional parameter vector |

```text
scalars t
vectors x(2), uq(3), v(0), w(0), uhat(1), uext(1), n(2), tau(1), mu(1), eta(0)
jacobian uq, w, uhat
hessian
batch x, uq, v, w, uhat, n, uext
outputs Flux, Source, Tdfunc, Ubou, Fbou, FbouHdg, Initu, QoIvolume
```

## Function blocks

```text
function NAME(arg1, arg2, ...)
  output_size(RESULT) = N;
  <body statements>
end
```

- `function NAME(args)` — each arg must be a declared scalar (emitted
  `const Expression&`) or vector (`const std::vector<Expression>&`).
- `output_size(RESULT) = N;` — names the result vector and its length; the
  function returns a `std::vector<Expression> RESULT` of size `N`.
- The block runs to the next `function` or end of file; the literal `end` is
  written for readability.

### Body sublanguage

Each body line is matched against an ordered set of regexes. `pi` is rewritten to
`SymEngine::pi`. Recognized statement forms:

| Form | Example | Emits |
|---|---|---|
| Result element | `f[0] = kappa*uq[1]` | `f[0] = kappa*uq[1];` |
| Scalar assignment | `kappa = mu[0]` | `Expression kappa = mu[0];` |
| Function call | `f = Flux(x, uq, ...)` | `auto f = Flux(...);` |
| `vector NAME(N)` | `vector tmp(3)` | `std::vector<Expression> tmp(3);` |
| `matrix NAME(R,C)` | `matrix A(2,2)` | `SymEngine::DenseMatrix A(2,2);` |
| `A[i][j] = expr` / `x = A[i][j]` | | matrix element set / get |
| `v[i] = det(A)` / `trace(A)` | | `A.det()` / `A.trace()` |
| `B = inv(A)` / `transpose(A)` | | `A.inv(B)` / `A.transpose(B)` |
| `C = A + B` / `A * B` | | matrix or scalar add/mul |
| `ones(NAME)` / `zeros(NAME)` / `fill(NAME, val)` | `ones(m)` | fill result/vector with constant |
| `for VAR in A:B` … `endfor` | | `for (int VAR=A; VAR<=B; ++VAR) { … }` |

Expressions transliterate to C++/SymEngine. Available: `+ - * /`, parentheses,
indexing `name[i]`; the symbolic product `mul(a, b)`; `pi`; `Expression(...)` and
`SymEngine::integer(N)` literals; the math wrappers `sin cos tan asin acos atan
atan2 sinh cosh tanh exp log log10 pow sqrt abs floor ceiling erf erfc gamma
lgamma` (from `backend/Model/SymEngineFunctionWrappers.hpp`); and calls to other
functions defined in the same file.

```text
function Flux(x, uq, v, w, eta, mu, t)
  output_size(f) = 2;
  kappa = mu[0];
  f[0] = kappa*uq[1];
  f[1] = kappa*uq[2];
end

function Fbou(x, uq, v, w, uhat, n, tau, eta, mu, t)
  output_size(fb) = 1;
  f = Flux(x, uq, v, w, eta, mu, t);
  fb[0] = f[0]*n[0] + f[1]*n[1] + tau[0]*(uq[0]-uhat[0]);
end
```

## Output functions → model contract

The recognized function names are fixed (`ParsedSpec::exasimfunctions`). Listing
one in `outputs` emits its kernel(s). Each corresponds to a
[model contract](model-contract.md) method (the SEMANTICS):

| DSL function | Role | Contract method |
|---|---|---|
| `Flux` | volume flux $F$ | `flux` |
| `Source` | volume source $S$ | `source` |
| `Tdfunc` | time-derivative mass coefficient | `tdfunc` |
| `Ubou` | Dirichlet boundary value of $u$ | `ubou` |
| `Fbou` | boundary flux (LDG) | `fbou` |
| `FbouHdg` | HDG boundary flux | `fbou_hdg` |
| `Sourcew` | source for the `w` field | `sourcew` |
| `EoS` | equation of state for `w` | `eos` |
| `Initu` / `Initq` / `Inituq` / `Initw` / `Initv` | initial conditions | `initu` / `initq` / `initudg` / `initwdg` / `initodg` |
| `Avfield` | artificial-viscosity field | `avfield` |
| `VisScalars` / `VisVectors` / `VisTensors` | visualization outputs | `vis_scalars` / `vis_vectors` / `vis_tensors` |
| `QoIvolume` / `QoIboundary` | quantity-of-interest integrands | `qoi_volume` / `qoi_boundary` |
| `Output` / `Monitor` | output / solver-monitor hooks | `output` / `monitor` |
| `Fint` / `Fext` | interior / external interface flux | used by the [solver coupling interface](../driving-the-solver.md) |

**Required in `exasim` mode (the default):** `Flux`, `Source`, `Tdfunc`, `Ubou`,
`Fbou`, `FbouHdg` must appear in `outputs`; all others are optional.

Derivatives are generated w.r.t. the vectors named in the `jacobian` (and
`hessian`) declarations — these become the HDG Jacobian methods of the
[model contract](model-contract.md), the per-element blocks of the
[block-diagonal Jacobian](../theory/block-diagonal-jacobian.md).
