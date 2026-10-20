# Mach-8 conjugate heat transfer with quad-triangle meshes

This example extends `../cylindermach8_nonmatching` with different volume
element families on the two sides of the fluid-solid interface.

- The Navier-Stokes domain uses quadrilateral elements and 28 interface faces.
- The solid heat-transfer domain uses triangular elements and 21 interface
  faces.
- Both interfaces lie on the same radius-one semicircle, giving a 4:3
  non-matching face ratio.
- Both models use polynomial order 4 and compatible one-dimensional face
  quadrature.

The coupled driver enables the distributed non-matching interface path:

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
the matching and quad-quad examples. The final solve couples the Navier-Stokes
energy flux to the solid heat equation. During setup, donor-search diagnostics
for both `Gamma^1` and `Gamma^2` should appear.

## Current requirements

- An MPI-enabled Exasim build; the default is two ranks per model.
- CPU execution. CUDA and HIP validation of the non-matching runtime remains
  open.
- Two-dimensional mixed-family interfaces with the same polynomial order and
  compatible face quadrature on both models.

Generated `.exasim`, `datain*`, and `dataout*` directories are run artifacts
and are not part of the source example.
