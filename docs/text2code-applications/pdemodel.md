# `pdemodel.txt` Guide

`pdemodel.txt` is the symbolic PDE model file consumed by Text2Code. It defines
the mathematical terms that become generated Exasim model kernels.

For the complete grammar reference, see
[pdemodel.txt syntax](../reference/pdemodel.md).

## File Structure

A `pdemodel.txt` file has two parts:

1. Header declarations for symbolic inputs, derivative targets, batching, and
   output functions.
2. Function blocks that define PDE terms and optional derived outputs.

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
```

## Header Declarations

| Declaration | Purpose |
| --- | --- |
| `scalars` | Declares scalar symbolic inputs such as time `t`. |
| `vectors` | Declares vector symbolic inputs and their sizes. |
| `jacobian` | Selects vectors with respect to which first derivatives are generated. |
| `hessian` | Selects vectors for second derivatives when needed. |
| `batch` | Selects inputs that are evaluated over quadrature/element batches. |
| `outputs` | Lists functions that should emit generated kernels. |
| `datatype` | Optional generated scalar type, default `dstype`. |
| `framework` | Optional backend target, default `kokkos`. |
| `codeformat` | Optional format, default `exasim`. |

In default `exasim` mode, `Flux`, `Source`, `Tdfunc`, `Ubou`, `Fbou`, and
`FbouHdg` are required outputs.

## Conventional Variables

| Symbol | Meaning |
| --- | --- |
| `x` | Coordinates. |
| `uq` | Packed solution and gradient fields; `uq[0:ncu-1]` is `u`. |
| `v` | `odg` auxiliary variables. |
| `w` | `wdg` auxiliary variables. |
| `uhat` | HDG trace state. |
| `uext` | External face data for interface/coupling functions. |
| `n` | Face normal. |
| `tau` | Stabilization parameters. |
| `mu` | `physicsparam`. |
| `eta` | `externalparam` / additional parameters. |
| `t` | Time. |

## Function Blocks

Every emitted function declares its output vector size:

```text
function Source(x, uq, v, w, eta, mu, t)
  output_size(s) = 1;
  s[0] = 0.0;
end
```

Supported body features include scalar assignments, vector assignments,
function calls, matrices, loops, matrix inverse/transpose/determinant/trace,
and common math functions such as `sin`, `cos`, `sqrt`, `exp`, `log`, `pow`,
and `tanh`.

## Core PDE Components

| Function | Meaning | Typical output size |
| --- | --- | --- |
| `Flux` | Volume flux tensor. | `ncu * nd` for first-order systems, or model-specific packed size. |
| `Source` | Volume source term. | `ncu`. |
| `Tdfunc` | Time-derivative/mass factor. | `ncu`. |
| `Ubou` | Boundary state value. | Boundary/model dependent. |
| `Fbou` | LDG boundary flux. | Boundary/model dependent. |
| `FbouHdg` | HDG boundary flux. | Boundary/model dependent. |
| `Initu` | Initial primary state. | `ncu`. |
| `Initq`, `Inituq` | Initial gradient or packed state. | Model dependent. |
| `Initw`, `Initv` | Initial auxiliary state. | `ncw`, `ncv`. |

## Postprocessing Components

| Function | Purpose |
| --- | --- |
| `VisScalars` | Scalar fields written to visualization output. |
| `VisVectors` | Vector fields written to visualization output. |
| `VisTensors` | Tensor fields written to visualization output. |
| `QoIvolume` | Volume quantity-of-interest integrands. |
| `QoIboundary` | Boundary quantity-of-interest integrands. |
| `Output` | Additional output field. |
| `Monitor` | Solver-monitor hook. |

## Boundary Conditions

Text2Code does not infer physical boundary types from names. The app file maps
geometry to integer boundary-condition IDs, and the model functions implement
the behavior.

Example HDG homogeneous Dirichlet boundary:

```text
function FbouHdg(x, uq, v, w, uhat, n, tau, eta, mu, t)
  output_size(fb) = 1;
  fb[0] = tau[0]*(0.0 - uhat[0]);
end
```

Example LDG boundary flux:

```text
function Fbou(x, uq, v, w, uhat, n, tau, eta, mu, t)
  output_size(fb) = 1;
  f = Flux(x, uq, v, w, eta, mu, t);
  fb[0] = f[0]*n[0] + f[1]*n[1] + tau[0]*(uq[0]-uhat[0]);
end
```

## Complete Poisson Model Example

```text
scalars t
vectors x(2), uq(3), v(0), w(0), uhat(1), n(2), tau(1), mu(1), eta(0)
jacobian uq, w, uhat
hessian
batch x, uq, v, w, uhat, n
outputs Flux, Source, Tdfunc, Ubou, Fbou, FbouHdg, Initu, VisScalars

function Flux(x, uq, v, w, eta, mu, t)
  output_size(f) = 2;
  f[0] = mu[0]*uq[1];
  f[1] = mu[0]*uq[2];
end

function Source(x, uq, v, w, eta, mu, t)
  output_size(s) = 1;
  s[0] = 2*pi*pi*sin(pi*x[0])*sin(pi*x[1]);
end

function Tdfunc(x, uq, v, w, eta, mu, t)
  output_size(m) = 1;
  ones(m);
end

function Ubou(x, uq, v, w, uhat, n, tau, eta, mu, t)
  output_size(ub) = 1;
  ub[0] = 0.0;
end

function Fbou(x, uq, v, w, uhat, n, tau, eta, mu, t)
  output_size(fb) = 1;
  f = Flux(x, uq, v, w, eta, mu, t);
  fb[0] = f[0]*n[0] + f[1]*n[1] + tau[0]*(uq[0]-uhat[0]);
end

function FbouHdg(x, uq, v, w, uhat, n, tau, eta, mu, t)
  output_size(fb) = 1;
  fb[0] = tau[0]*(0.0 - uhat[0]);
end

function Initu(x, eta, mu)
  output_size(ui) = 1;
  ui[0] = 0.0;
end

function VisScalars(x, uq, v, w, eta, mu, t)
  output_size(s) = 1;
  s[0] = uq[0];
end
```

## Translation To Exasim Code

Text2Code parses each function into SymEngine expressions, computes requested
Jacobians/Hessians, and emits C++/Kokkos kernels matching Exasim's model
contract. The backend solver calls those kernels during residual assembly,
linearization, initialization, and postprocessing.

```mermaid
flowchart LR
  D["pdemodel.txt declarations"] --> P["TextParser"]
  P --> S["SymEngine expressions"]
  S --> J["Jacobians and Hessians"]
  J --> K["Generated C++ kernels"]
  K --> B["Backend residual and postprocessing drivers"]
```

## See Also

- [Complete pdemodel.txt syntax reference](../reference/pdemodel.md)
- [Model contract](../reference/model-contract.md)
- [PDE modeling guide](modeling-guide.md)
- [Postprocessing](../usage-modes/postprocessing.md)
