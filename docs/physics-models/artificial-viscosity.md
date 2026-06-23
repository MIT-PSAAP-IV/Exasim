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
| `AVsmoothingIter` | Number of AV smoothing iterations. |
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

## Practical Guidance

- Use AV for shocks, contact discontinuities, or under-resolved steep fronts.
- Keep physical diffusion and artificial diffusion conceptually separate.
- Document which component of `v` or `w` stores the AV field.
- Verify conservation and shock thickness when changing AV parameters.
