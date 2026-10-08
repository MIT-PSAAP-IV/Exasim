# GPU-aware coupling interface

Track device-pointer interface for coupling surface, eliminating the
device→host→device roundtrip when GPU-aware MPI is available.

## What needs to change

- `ExasimSolver::getInterfaceFluxes()` — add device-pointer overload that
  returns a raw `dstype*` on device, skipping `TemplateCopytoHost`.
- `ExasimSolver::setInterfaceFluxes()` — add device-pointer overload that
  accepts a raw `dstype*` on device, skipping `TemplateCopytoDevice`.
- Existing host-vector API remains the fallback for non-GPU-aware MPI.

## Companion PRs

- Kitesurf: GPU-aware MPI exchange path + host fallback
- CHEFSI-apps: wiring, topology-aware rank placement, benchmarking

## Target

tuolumne (MI300A / gfx942, Cray MPICH — GPU-aware MPI).
