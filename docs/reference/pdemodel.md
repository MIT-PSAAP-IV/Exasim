# pdemodel Reference

`pdemodel.txt` is the Text2Code model-definition file. It declares symbolic
inputs and function blocks, then Text2Code emits C++ kernels for the requested
outputs.

This page documents the exact text format. For mathematical meaning, see
[Physics Models](../physics-models/index.md). For the C++ runtime contract, see
[Model contract](model-contract.md).

## File Structure

A `pdemodel.txt` file has two sections:

1. Header declarations.
2. Function blocks.

```text
scalars t
vectors x(2), uq(3), v(0), w(0), uhat(1), uext(1), n(2), tau(1), mu(1), eta(0)
jacobian uq, w, uhat
hessian
batch x, uq, v, w, uhat, n, uext
outputs Flux, Source, Tdfunc, Ubou, Fbou, FbouHdg, Initu

function Flux(x, uq, v, w, eta, mu, t)
  output_size(f) = 2;
  f[0] = mu[0]*uq[1];
  f[1] = mu[0]*uq[2];
end
```

## Header Declarations

| Declaration | Form | Required | Meaning |
| --- | --- | --- | --- |
| `scalars` | `scalars t` | Usually yes | Scalar symbolic inputs. |
| `vectors` | `vectors x(2), uq(3)` | Yes | Vector symbolic inputs and sizes. |
| `jacobian` | `jacobian uq, w, uhat` | Optional | Vectors with respect to which first derivatives are generated. |
| `hessian` | `hessian` | Optional | Vectors for second derivatives; often empty. |
| `batch` | `batch x, uq, ...` | Optional | Inputs batched over generated point loops. |
| `outputs` | `outputs Flux, Source, ...` | Yes | Function blocks to emit. |
| `datatype` | `datatype dstype` | Optional | C++ scalar type; default is `dstype`. |
| `framework` | `framework kokkos` | Optional | Generation framework; default is `kokkos`. |
| `codeformat` | `codeformat exasim` | Optional | Default `exasim`; enforces core Exasim outputs. |

## Conventional Vector Names

Text2Code allows arbitrary vector names, but Exasim-generated model files use
the following conventions:

| Name | Meaning |
| --- | --- |
| `x` | Spatial coordinate vector. |
| `uq` | Packed solution vector; `u` plus gradients/auxiliary `q` entries. |
| `v` | External or other-DG field, stored as `odg` at runtime. |
| `w` | Auxiliary state field, stored as `wdg`. |
| `uhat` | HDG trace state on faces. |
| `uext` | External face data for coupled/interface functions. |
| `n` | Outward unit normal on a face. |
| `tau` | Stabilization parameters. |
| `mu` | Physics parameters, from `physicsparam`. |
| `eta` | Extra/reference parameters, mapped to `uinf` in the ABI. |
| `t` | Current time. |

## Required Outputs in `exasim` Code Format

When `codeformat` is `exasim` or omitted, the parser requires the first six
outputs:

| Function | Purpose |
| --- | --- |
| `Flux` | Volume flux. |
| `Source` | Volume source. |
| `Tdfunc` | Time-derivative coefficient/mass function. |
| `Ubou` | Boundary state/value function. |
| `Fbou` | LDG boundary flux. |
| `FbouHdg` | HDG boundary residual/flux. |

Optional outputs are emitted when listed:

| Function | Purpose |
| --- | --- |
| `Sourcew` | Source/evolution term for auxiliary `w`. |
| `Output` | Output field callback. |
| `Monitor` | Monitor callback. |
| `Initu` | Initial primary state. |
| `Initq` | Initial gradient/auxiliary `q`. |
| `Inituq` | Initial packed `u,q` state. |
| `Initw` | Initial auxiliary `w`. |
| `Initv` | Initial external/other-DG field. |
| `Avfield` | Artificial-viscosity field. |
| `Fint` | Internal interface flux. |
| `EoS` | Equation-of-state/closure output. |
| `VisScalars` | Scalar visualization fields. |
| `VisVectors` | Vector visualization fields. |
| `VisTensors` | Tensor visualization fields. |
| `QoIvolume` | Volume QoI integrand. |
| `QoIboundary` | Boundary QoI integrand. |
| `Fext` | External interface flux. |

## Function Block Syntax

```text
function Name(arg1, arg2, ...)
  output_size(out) = N;
  statement;
  out[0] = expression;
end
```

Rules:

- Arguments must be declared in `scalars` or `vectors`.
- `output_size(name) = N;` declares the output vector and size.
- The function body is parsed line-by-line until the next `function` or EOF.
- `end` is conventional and improves readability.

## Body Sublanguage

The body sublanguage is intentionally small. It supports scalar/vector
assignments, simple matrix operations, loops, and calls to other functions
declared in the same `pdemodel.txt` file.

## Supported Body Statements

| Form | Example |
| --- | --- |
| Output assignment | `f[0] = uq[1];` |
| Scalar assignment | `p = mu[0]*uq[0];` |
| Function call | `f = Flux(x, uq, v, w, eta, mu, t);` |
| Vector declaration | `vector tmp(3);` |
| Matrix declaration | `matrix A(2,2);` |
| Matrix element assignment | `A[0][1] = x[0];` |
| Matrix operations | `B = inv(A);`, `C = transpose(A);` |
| Fill operations | `zeros(f);`, `ones(f);`, `fill(f, 1.0);` |
| Loop | `for i in 0:2` ... `endfor` |

Supported mathematical functions include common functions such as `sin`,
`cos`, `tan`, `exp`, `log`, `pow`, `sqrt`, `abs`, `atan2`, hyperbolic
functions, and error/gamma functions as wrapped by the generated C++ support.

## Model-Type Mapping

| Model type | Typical `uq` contents | Required model meaning |
| --- | --- | --- |
| `ModelC` | `u` only | Convection/source systems without a `q + grad(u)` equation. |
| `ModelD` | `u` and gradient variables `q` | Diffusion/mixed systems with auxiliary gradient variables. |
| `ModelW` | `u` and time-dependent auxiliary wave variables | Wave-type first-order systems. |

The parser does not prove that the mathematics matches the selected model
type; the user/model author is responsible for consistency.

## Minimal Poisson-Style Fragment

```text
scalars t
vectors x(2), uq(3), v(0), w(0), uhat(1), n(2), tau(1), mu(1), eta(0)
jacobian uq, w, uhat
hessian
batch x, uq, v, w, uhat, n
outputs Flux, Source, Tdfunc, Ubou, Fbou, FbouHdg, Initu

function Flux(x, uq, v, w, eta, mu, t)
  output_size(f) = 2;
  f[0] = mu[0]*uq[1];
  f[1] = mu[0]*uq[2];
end

function Source(x, uq, v, w, eta, mu, t)
  output_size(s) = 1;
  s[0] = 0.0;
end

function Tdfunc(x, uq, v, w, eta, mu, t)
  output_size(m) = 1;
  m[0] = 1.0;
end
```

## Cross-References

- [Model functions](model-functions.md)
- [Boundary conditions](boundary-conditions.md)
- [Equations of state](equations-of-state.md)
- [Auxiliary equations](auxiliary-equations.md)
- [Text2Code](text2code.md)
