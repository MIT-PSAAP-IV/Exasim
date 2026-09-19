# Artificial Viscosity

Artificial viscosity (AV) is Exasim's shock-capturing and stabilization
extension. It is configured by app flags and, when needed, model callbacks.

Conceptually, AV modifies the conservation law as

$$
\frac{\partial u}{\partial t}
+ \nabla \cdot F
= S + \nabla \cdot F_{\mathrm{AV}}.
$$

In the current implementation, AV is not an automatic universal flux. The
application enables AV-related flags and the model author uses `avfield`,
`v`/`odg`, and flux definitions to include artificial-viscosity effects.

## Relevant User-Facing Fields

The `pdeapp.txt` reference documents these AV-related fields:

| Field | Meaning |
| --- | --- |
| `AV` | Artificial-viscosity flag. |
| `AVdistfunction` | Distance-function option used by AV workflows. |
| `AVsmoothingIter` | Number of repeated DG-to-CG passes for smoothing method `0`. |
| `AVsmoothingMethod` | `0` selects repeated DG-to-CG averaging; `1` selects the internal HDG Helmholtz filter. |
| `AVHelmholtzCoeff` | Positive multiplier for the Helmholtz filter length scale. |
| `AVcontinuationIter` | Number of continuation solves; values below `2` disable generated continuation. |
| `AVcontinuationLogScale` | Exponential continuation-shape parameter; a value near zero gives linear interpolation. |
| `AVcoeffStart`, `AVcoeffEnd` | Initial first coefficient and final second coefficient for generated continuation. |
| `frozenAVflag` | Freeze/update behavior for AV fields. |
| `avparam1`, `avparam2` | AV parameter vectors. |

Frontend models may define `avfield(u,q,w,v,x,t,mu,eta)`. Text2Code uses the
corresponding `Avfield` function.

## How AV Is Used In Examples

Examples such as `examples/Euler/mach8cylinder`,
`examples/NavierStokes/flaredplate2d`, and
`examples/NavierStokes/flaredplatecut2d` compute AV fields and use those fields
inside flux routines. Several examples store AV data in `mesh.vdg` / `v`.

Typical AV workflows include:

- Build a sensor from solution gradients or flow variables.
- Smooth or freeze the AV field according to app flags.
- Pass the AV field through `v`/`odg` or `avfield`.
- Add AV contributions to the physical flux.

## Smoothing Methods

With `AVsmoothingMethod = 0`, Exasim applies `AVsmoothingIter` DG-to-CG
averaging passes to the frozen AV field. This is the legacy behavior and the
default, so applications that omit the new fields retain their previous
results.

With `AVsmoothingMethod = 1`, Exasim solves one scalar HDG Helmholtz problem
for each AV component,

$$
v - \nabla \cdot (\ell^2 \nabla v) = v_{\mathrm{raw}},
$$

with homogeneous natural flux at physical boundaries. The element-local
length-scale coefficient is `AVHelmholtzCoeff` times the square root of the
twice-smoothed nodal mapping Jacobian. The internal problem uses the same mesh,
partitioning, polynomial order, and backend as the flow problem. This method
requires a frozen AV field (`frozenAVflag > 0`); `AVsmoothingIter` does not
repeat the Helmholtz solve.

## AV Continuation

When `AVcontinuationIter = N >= 2`, preprocessing replaces explicit
`avparam1` and `avparam2` arrays with `N` coefficient pairs. For zero-based
continuation index $i$, let $t=i/(N-1)$ and
$\alpha=\mathtt{AVcontinuationLogScale}$. For $|\alpha|>10^{-14}$,

$$
c_1(i) = c_{\mathrm{start}}
\frac{\exp(\alpha(1-t))-1}{\exp(\alpha)-1}, \qquad
c_2(i) = c_{\mathrm{end}}
\frac{\exp(\alpha t)-1}{\exp(\alpha)-1}.
$$

For a near-zero $\alpha$, the weights are $1-t$ and $t$. The first and last
pairs are forced to `(AVcoeffStart, 0)` and `(0, AVcoeffEnd)`. Each pair is
copied into the final two physics parameters before one steady nonlinear solve;
the final pair remains active afterward. Values of `AVcontinuationIter` below
`2` disable generated continuation and preserve explicitly supplied
`avparam1`/`avparam2` arrays.

## Practical Guidance

- Use AV for shocks, contact discontinuities, or under-resolved steep fronts.
- Keep physical diffusion and artificial diffusion conceptually separate.
- Document which component of `v` or `w` stores the AV field.
- Verify conservation and shock thickness when changing AV parameters.
