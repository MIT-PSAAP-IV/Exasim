# MATLAB-frontend golden baselines

Volume solutions (`outudg_np*.bin`, one per MPI rank) produced by running the examples in
[`../matlab-examples.txt`](../matlab-examples.txt) through the **MATLAB frontend** `exasim(pde,mesh)`
on **main**. Stored as git-lfs blobs.

These are the reference for [`../run-matlab-regression.sh`](../run-matlab-regression.sh), which reruns
each example through *this repo's* frontend and reports the relative L2 of the udg vector against the
baseline. A frontend that codegens / meshes / preprocesses / solves identically to main gives
`rel_L2 = 0`.

## Relation to `../app-baselines/`
- `app-baselines/` tests the compiled **C++ app** (`exasimapp`): preprocessing + kernels + HDG/LDG
  assembly + Newton/GMRES + QoI. No MATLAB.
- `matlab-baselines/` (this dir) additionally exercises the **MATLAB codegen + mesh generation +
  preprocessing** path that feeds that solver.

## Regenerate
```sh
# on a main checkout (so pdeapp.m picks up main's frontends/Matlab):
EXASIM_REPO=/path/to/Exasim-main tests/gen-matlab-baselines.sh tests/matlab-baselines
```
`Euler_EulerVortex` (~156M transient) is git-ignored — uncomment it in `matlab-examples.txt` to
generate locally.
