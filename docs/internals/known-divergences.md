# Known Divergences

Known divergences are documented cases where different execution paths produce
different outputs. The goal is to distinguish expected floating-point drift
from real implementation bugs.

## Why Track Divergences

Exasim supports CPU, CUDA, HIP, MPI, serial execution, frontend-generated
apps, Text2Code apps, built-in models, shared-library providers, and
postprocessing-only execution. Differences can be legitimate or indicate a
bug. Tracking them prevents the same issue from being rediscovered without
context.

## Divergence Categories

| Category | Meaning | Action |
| --- | --- | --- |
| Platform numerics | Small differences from compiler, BLAS, GPU backend, or reduction ordering. | Accept with documented tolerance. |
| Partition effects | Output ordering or local rank ownership differs, but global solution agrees. | Use partition-invariant comparison. |
| Configuration mismatch | Inputs differ across runs. | Fix test setup before analyzing numerics. |
| Real regression | Physics, residuals, QoI, or convergence differ beyond expected tolerance. | Fix before updating baselines. |

## How to Document a Divergence

Record:

- Example name and path.
- Backend and rank count.
- Exact command if practical.
- Expected value and observed value.
- Magnitude of difference.
- Whether codegen/frontend/built-in paths agree with each other.
- Suspected root cause.
- Current status.

## Common Sources of Real Divergence

| Symptom | Likely issue |
| --- | --- |
| MPI QoI larger than serial | Ghost elements included in integration. |
| GPU first case correct, later sweep cases wrong | Device parameters not refreshed. |
| Postprocess output empty | Saved solution files truncated or wrong execution mode. |
| Frontend and standalone sweep differ | Runtime sweep metadata or output path mismatch. |
| Built-in and Text2Code providers differ | ABI callback mismatch or generated provider inconsistency. |

## Current Policy

Do not add a divergence here as a permanent workaround. Use this page to
track unresolved or intentionally tolerated differences, then remove or move
entries when the issue is fixed.

When a divergence is fixed, keep a short note in the relevant pull request or
commit message explaining the root cause and validation evidence.

## Related Pages

- [Testing](testing.md)
- [Baselines](baselines.md)
- [MPI implementation](mpi-implementation.md)
- [GPU implementation](gpu-implementation.md)
