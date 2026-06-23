# PDE Modeling Guide

This guide shows how to translate common PDE terms into `pdemodel.txt`.
Examples are intentionally small; production examples in `apps/` and
`examples/` contain fuller models.

## General Modeling Steps

1. Choose the physical variables and their ordering in `uq`.
2. Decide which entries of `uq` represent primary unknowns and gradients.
3. Declare symbolic vectors with matching sizes.
4. Implement `Flux`, `Source`, `Tdfunc`, `Ubou`, `Fbou`, `FbouHdg`, and
   `Initu`.
5. Add optional visualization and QoI functions.
6. Set consistent field counts in `pdeapp.txt`.
7. Run `/path/to/exasim-prefix/local/bin/text2code pdeapp.txt`.

## Poisson Equation

Governing form:

\[
-\nabla \cdot (\kappa \nabla u) = f.
\]

For a 2D LDG/HDG representation, use `uq = [u, q_x, q_y]` and
`mu[0] = kappa`.

```text
function Flux(x, uq, v, w, eta, mu, t)
  output_size(f) = 2;
  f[0] = mu[0]*uq[1];
  f[1] = mu[0]*uq[2];
end

function Source(x, uq, v, w, eta, mu, t)
  output_size(s) = 1;
  s[0] = 2*pi*pi*sin(pi*x[0])*sin(pi*x[1]);
end
```

Run:

```bash
cd apps/poisson/poisson2d
/path/to/exasim-prefix/local/bin/text2code pdeapp.txt
```

## Advection Equation

For scalar advection with velocity \(a=(a_x,a_y)\):

\[
u_t + \nabla \cdot (a u) = 0.
\]

Use `mu[0] = ax` and `mu[1] = ay`.

```text
function Flux(x, uq, v, w, eta, mu, t)
  output_size(f) = 2;
  f[0] = mu[0]*uq[0];
  f[1] = mu[1]*uq[0];
end

function Source(x, uq, v, w, eta, mu, t)
  output_size(s) = 1;
  s[0] = 0.0;
end
```

Set `dt` to positive values for a transient run.

## Convection-Diffusion Equation

For

\[
u_t + \nabla \cdot (a u - \kappa \nabla u) = s,
\]

pack `uq = [u, q_x, q_y]` and combine advective and diffusive fluxes.

```text
function Flux(x, uq, v, w, eta, mu, t)
  output_size(f) = 2;
  ax = mu[0];
  ay = mu[1];
  kappa = mu[2];
  f[0] = ax*uq[0] + kappa*uq[1];
  f[1] = ay*uq[0] + kappa*uq[2];
end
```

## Burgers Equation

For scalar inviscid Burgers in 2D:

\[
u_t + \partial_x(u^2/2) + \partial_y(u^2/2) = 0.
\]

```text
function Flux(x, uq, v, w, eta, mu, t)
  output_size(f) = 2;
  f[0] = 0.5*uq[0]*uq[0];
  f[1] = 0.5*uq[0]*uq[0];
end
```

For viscous Burgers, include gradient entries in `uq` and add diffusive terms.

## Euler Equations

For compressible Euler in 2D, use primary state
`[rho, rho*u, rho*v, rho*E]`. The flux has eight entries: four in the
x-direction and four in the y-direction.

```text
function Flux(x, uq, v, w, eta, mu, t)
  output_size(f) = 8;
  gam = mu[0];
  gam1 = gam - 1.0;
  r = uq[0];
  ru = uq[1];
  rv = uq[2];
  rE = uq[3];
  r1 = 1/r;
  ux = ru*r1;
  uy = rv*r1;
  p = gam1*(rE - 0.5*(ru*ux + rv*uy));
  h = (rE + p)*r1;
  f[0] = ru;
  f[1] = ru*ux + p;
  f[2] = rv*ux;
  f[3] = ru*h;
  f[4] = rv;
  f[5] = ru*uy;
  f[6] = rv*uy + p;
  f[7] = rv*h;
end
```

Boundary functions must handle inflow, outflow, wall, and far-field behavior
consistently with `boundaryconditions`.

## Navier-Stokes Equations

Navier-Stokes models extend Euler with gradient entries in `uq`, viscosity,
heat conduction, and boundary-state logic. See
`apps/navierstokes/naca0012steady/pdemodel.txt` for a full working example with
primitive-variable reconstruction, viscous stresses, heat flux, and HDG
boundary treatment.

Typical physical parameters:

```text
physicsparam = [gamma, Re, Pr, Minf, rho_inf, rhou_inf, rhov_inf, rhoE_inf];
```

## Reaction-Diffusion Systems

For a multi-component reaction-diffusion system, set `ncu` to the number of
species and return vector-valued source terms:

```text
function Source(x, uq, v, w, eta, mu, t)
  output_size(s) = 2;
  k = mu[0];
  s[0] = -k*uq[0]*uq[1];
  s[1] =  k*uq[0]*uq[1];
end
```

Ensure `Flux` and all boundary functions return components in the same ordering
as `uq`.

## Best Practices

- Document the ordering of every component in `uq`, `mu`, and `eta`.
- Keep `physicsparam` for values that should be swept or changed at runtime.
- Keep output sizes consistent with `pdeapp.txt` counts.
- Start with CPU and `mpiprocs = 1` before moving to MPI or GPU.
- Use manufactured solutions for new models when possible.

## See Also

- [pdemodel.txt guide](pdemodel.md)
- [Application setup guide](application-setup.md)
- [Theory overview](../theory/index.md)
