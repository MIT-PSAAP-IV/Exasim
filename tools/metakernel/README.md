# metakernel: generated LDG residual, preconditioner and GMRES stages

`metakernel` generates case-specific GPU kernels for the LDG residual, the LDG block-Jacobian preconditioner build and
the GMRES inner loop. It plugs them into Exasim through the opt-in stage tables in
`backend/Discretization/residualstages.hpp` (`ResidualStageTable`) and `backend/Discretization/precondstages.hpp`
(`PrecondStageTable`). Nothing in Exasim changes unless a table is registered; with no table every path is the
production code.

The generator takes the case's own generated physics kernels (`KokkosFlux`, `KokkosSource`, `KokkosFbou*`,
`KokkosUbou*`, `KokkosEoS*`), rewrites them into include-able bodies with read/write macros, and emits fused HIP
kernels with every size fixed at compile time. A harness runs the real case through Exasim, evaluates the production
residual and the generated one on the same device arrays, and reports the difference and per-stage times.

Target: AMD MI300A (gfx942), HIP, GPU-aware MPI. Other backends fall back to production.

## What the generated stages do

One residual evaluation, in order (pass a runs on interior elements while the halo exchange is in flight, pass b on
the elements next to ghosts):

| stage | kernels | notes |
|---|---|---|
| uhat | interior average + one switch-merged kernel for all boundary types | pass a / pass b select the faces each pass can change |
| q | `rqface` (pass a) + element kernel on the fp64 matrix cores | R stays in the MFMA accumulators between the two products; pass b computes its own face integrals |
| w | Newton step + fused block-norm/decide + masked update | the convergence check is deferred to the tail: one host sync per residual, with an exact re-run if a range had not converged |
| elem | one fused element kernel | interpolation and integration on the matrix cores, source + flux from one body; a prologue writes this element's side of its faces' Gauss-point values |
| face | ghost-side gather, side-1 flux, side-2 flux + integrate, boundary interp + flux | interior face interpolation is produced by the element kernel |
| tail | ordered gather of face contributions + scaling | production's contribution order, no atomics |

Every per-block kernel is launched once per stage pass over all blocks (a device table maps workgroups to blocks).
All face-to-element sums walk production's ordered contribution lists. The generated residual matches production to
rounding (relative L2 ~1e-16 on the benchmark).

Preconditioner and GMRES (`e2e/mk_pcstages.hpp`, `e2e/mk_jacstages.hpp`, `e2e/mk_gmstages.hpp`): fused element /
element-face Jacobian blocks, a batched Gauss-Jordan inverse, the block apply, and a classical Gram-Schmidt step with
the coefficients kept on the device. `MK_PC_F32=1` stores the block inverses in fp32 (fp64 accumulation in the apply);
this changes the preconditioner, not the residual.

## Layout

| file | role |
|---|---|
| `rewrite.py` | case kernel `.cpp` -> `body_*.inc` (read/write macros) |
| `gen.py`, `gen_face.py`, `gen_q.py`, `gen_rest.py` | schedule -> `schedule.hpp` (element, face, q, uhat, w, tail kernels; batching transforms) |
| `schedules.py`, `e2e/schedules_e2e.py` | schedule definitions (`S(name, kernels, face=..., q=..., ...)`) |
| `mk_common.hpp` | argument blocks shared by the generated kernels |
| `e2e/mk_stages.hpp` | the residual stage table: setup, block tables, dispatch, deferred w check |
| `e2e/mk_pcstages.hpp`, `e2e/mk_jacstages.hpp`, `e2e/mk_gmstages.hpp` | preconditioner build, apply, CGS |
| `e2e/mk_e2e.cpp` | harness: production vs generated residual on the real case (`MKREF` / `MKRES` / `MKKER` lines) |
| `e2e/mk_app.cpp` | the normal Exasim solve with the tables registered |
| `e2e/mk_pc.cpp` | preconditioner-build harness |
| `e2e/analyze_counters.py`, `e2e/analyze_stalls.py`, `report.py` | profiling helpers (rocprofv3 output) |
| `DESIGN.md` | the stage graph and the schedule axes |

## Using it with a case

1. **Case sizes.** Write `mk_case_sizes.hpp` and put it in the bodies directory:

   ```cpp
   #pragma once
   constexpr int MK_NC = /*nc*/, MK_NCW = /*ncw*/, MK_NCO = /*nco*/, MK_NCU = /*ncu*/, MK_ND = /*nd*/,
                 MK_NCX = /*ncx*/, MK_NPE = /*npe*/, MK_NGE = /*nge*/, MK_NPR = /*property-stage width*/,
                 MK_NPF = /*npf*/, MK_NGF = /*ngf*/;
   ```

   The harness checks these against the loaded case and falls back to production on any mismatch.
   The face kernels assume `npf == ngf`.

2. **Bodies.** For each generated kernel of the case:
   `python3 rewrite.py <case>/kernels/KokkosFlux.cpp bodies/body_Flux.inc --nco <nco>` (same for `Source`,
   `Fbou<ib>`, `Ubou<ib>`, `EoS`). `--func` picks one function from a multi-function file.

3. **Schedules and generation.** Add or pick schedules in `e2e/schedules_e2e.py`, then
   `python3 gen.py e2e/schedules_e2e.py <gen dir>`. This writes `<gen dir>/<schedule>/schedule.hpp` and
   `<gen dir>/schedules.txt`; copy the bodies to `<gen dir>/bodies`.

4. **Build.** Configure `e2e/` against an Exasim install that has the stage tables (`residualstages.hpp`,
   `precondstages.hpp` with the v4 `apply` / `cgs` entries):

   ```sh
   cmake -S tools/metakernel/e2e -B build-mk -DCMAKE_CXX_COMPILER=hipcc \
         -DExasim_DIR=<install>/lib/cmake/Exasim -DKokkos_DIR=<kokkos build> \
         -DCASE_KERNELS=<case>/kernels -DGEN=<gen dir> -DMKDIR=$PWD/tools/metakernel \
         -DEXASIM_VARIANT=gpumpi -DMK_NOROCTHRUST=ON -DMK_CASE_KIND=matlab
   cmake --build build-mk --target mk_e2e_<schedule> mk_app_<schedule>
   ```

5. **Run.** Both take the `exasimapp` command line: `mpirun -n P mk_e2e_<schedule> 1 <datain>/ <dataout prefix>`.
   `MKRES` reports the generated residual time, the production time, the speedup and the relative difference.
   `mk_app_<schedule>` runs the real solve (`MK_STAGES=0` for production residual stages, `MK_PC_STAGES=1` to add the generated preconditioner and GMRES stages).

## Runtime switches

| variable | effect |
|---|---|
| `MK_STAGES` | mk_app: generated residual stages unless `0` |
| `MK_PC_STAGES` | mk_app: generated preconditioner / GMRES stages when `1` (default off) |
| `MK_PC_ELEM`, `MK_PC_FACE`, `MK_PC_CROSS`, `MK_PC_APPLY`, `MK_PC_CGS`, `MK_PC_FUSED`, `MK_PC_SLAYOUT` | 0: production for that preconditioner stage / layout |
| `MK_PC_F32` | 1: fp32 storage of the block inverses |
| `MK_W_DEFER`, `MK_W_DEVICE`, `MK_W_FASTNORM` | 0: the earlier synchronous / host-decided / production-norm w paths |
| `MK_W_FIRST` | cap on speculative w iterations (testing the deferred-check fallback) |
| `MK_CONN_CHECK` | 0: skip the setup check that the face connectivity matches the gather indices |
| `MK_PERTURB_W`, `MK_REPEAT_CHECK`, `MK_DIAG` | harness correctness modes |
| `MK_SAVE_FINAL` | 1: write the final udg/wdg per rank (for solution comparisons) |

## Status

Measured on a 3D p = 2 LDG benchmark with 4x MI300A: one residual evaluation 2.9x faster than the production stages,
and the full Newton-GMRES solve about 2x faster (2.3x with `MK_PC_F32=1`). The solution is accepted with the same
statistical test as rounding-only changes (median relative-L2 distance over 16 runs).
