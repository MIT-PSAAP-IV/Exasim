# Meta-kernel: one generator for the LDG residual design space

A *schedule* is a short spec; the generator turns it into Kokkos/HIP kernels; one driver times it, reports
per-kernel resources, and checks it against the production staged path. Every point we have measured by hand
(staged, props split, full element fusion, ping-pong, L2 hand-off) is one schedule, and so is everything we have
not tried.

## 1. The pipeline as a stage graph

Element residual (v0 scope):

| stage | reads | writes (comps/pt) | notes |
|---|---|---|---|
| `interp` | nodal udg, wdg | `u` (nc), `w` (ncw) | gemm form = gather + MFMA GEMM (production); lane form = per-lane dot over npe |
| `source` | u, w, o, x | `s` (ncu) | generated body |
| `time` | u, s, sdg | `sg` (ncu) | `sdg - dtfactor*u + s` |
| `flux` | u, w, o, x | `f` (ncu*nd) | monolithic generated body |
| `props` | u, w, o, x | `pr` (npr) | model property values FluxP takes |
| `fluxp` | u, w, o, x, pr | `f` (ncu*nd) | flux given pr |
| `scale` | sg, f, jac, Xx | `rg` (ncu*(nd+1)) | ApplyXx4: `sg*jac`, `f.Xx` |
| `integrate` | rg | `R` | per element reduction; gemm form (MFMA) or team form (LDS, element map only) |

Physics form = `flux` (mono) or `props` + `fluxp` (split). Face residual, Rq and uhat are phase 2 (section 6).

## 2. Axes of a schedule

| axis | values | where |
|---|---|---|
| kernel boundaries | any ordered partition of the stage chain | schedule |
| work mapping | `pt` (64-lane tile of points), `el(E, W)` (E elements on W waves, packed), `gemm` | kernel |
| hand-off inside a kernel | `reg` (live value), `lds` (tile array), `glb` (per-point global buffer, L2) | per value class, per kernel |
| hand-off between kernels | global buffer (always) | implied |
| occupancy target | LaunchBounds waves/SIMD | kernel |
| register control | `launder` (opaque base pointers per stage), `phased` (stages as branches of a nounroll loop) | kernel |
| compiler scheduler | default / iterative-maxocc / ... | schedule (per executable) |
| backend | `hip` (plain `__global__`, default) / `kokkos` (TeamPolicy functor) | schedule |
| roles (ping-pong) | producer/consumer wave groups with ratio + pipeline depth | kernel (v1) |

Legality (the generator enforces these): `integrate` in-kernel needs an `el` map and `rg` in LDS; `phased`
forbids `reg` hand-offs; LDS use is summed and reported; a value read before it is produced is an error.

## 3. What gets generated

- `rewrite.py`: turns a generated Exasim kernel into a body include with every array access abstracted:
  `udg[k*ng+i]` -> `MK_RD_u(k)`, `odg` -> `MK_RD_o(k)` (or `MK_RD_pr(k - nco)` for FluxP), `wdg`, `xdg`, and every
  output store -> `MK_WR(k)`. The same body then compiles against any binding (register array, LDS tile,
  global buffer) by defining the macros around the include.
- `gen.py <schedule>`: one header per schedule with one functor per kernel (`MK_<sched>_K<j>`, so compiler
  remarks map back to kernels), `mk_run()`, `mk_run_kernel(j)`, LDS sizes and a kernel table.
- `mk_bench.cpp`: data setup from datain, the production staged path as reference, timing (total and
  per-kernel medians), correctness (bitwise flag + max rel diff), CSV output.
- `CMakeLists.txt` / `run_tuo.sh`: one executable per schedule (so the compiler flags can differ), parallel
  build, run on one MI300A, then `report.py` joins timings with the `-Rpass-analysis` resources.

## 4. Validation

`ref_staged` expresses the production path in the DSL; it must match the reference and time like it. The
props split must come out ~1.10x and the old one-element-per-team fusion ~0.1-0.3x. Only then are new
schedules believed.

## 5. First sweep (v0)

1. `ref_staged`, `props_split` (baselines)
2. light fusion: `source+time`, `fluxp+scale` / `flux+scale` merged, GEMMs kept
3. lane interpolation feeding physics (drops gather + interp GEMM + the u round trip)
4. full pointwise fusion (everything but integrate) with reg / phased-glb hand-offs
5. element maps: E=1 (old fused, 27/64 lanes busy), E=2 (54/64), E=7 on 3 waves (189/192), integrate in LDS

## 6. Phase 2 (reserved in the spec, not generated yet)

Face stages (`fgather`, `finterp`, `uhat`, `fflux`, `fintegrate` + scatter), the Rq/q lifting, and roles with
ratio (2 producers : 1 consumer for the measured 2:1 props:fluxp cost). Face fusion adds a scatter axis
(atomic / colored / owner-computes).
