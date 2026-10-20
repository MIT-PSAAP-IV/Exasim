# Mach-8 conjugate heat transfer with a non-matching interface

This example extends `../cylindermach8_matching` by independently meshing the
fluid and solid sides of the circular fluid-solid interface.

- The Navier-Stokes mesh has 21 interface faces.
- The solid heat-transfer mesh has 28 interface faces.
- Both interfaces lie on the same radius-one semicircle, giving a 4:3
  non-matching face ratio without a physical gap or overlap.
- Both models use quadrilateral elements, polynomial order 4, and the same
  face quadrature.

The coupled driver selects the distributed non-matching interface path with

```matlab
pde{1}.interfacetype = 1;
pde{2}.interfacetype = 1;
```

## Run

Start MATLAB in this directory and run:

```matlab
pdeapp
```

The fluid setup first performs the same artificial-viscosity continuation as
the matching example. The final call couples the Navier-Stokes energy flux to
the solid heat equation. During coupled setup, both donor-search diagnostics
(`Gamma^1` and `Gamma^2`) should be printed.

## Current requirements

- An MPI-enabled Exasim build; this example uses two ranks per model by
  default.
- CPU execution. The non-matching GPU runtime remains disabled pending CUDA
  and HIP validation.
- Two-dimensional non-matching interfaces with polynomial order 1 through 4.

Generated `.exasim`, `datain*`, and `dataout*` directories are run artifacts
and are not part of the source example.
