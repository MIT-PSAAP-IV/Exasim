# MATLAB utilities → C++ port tracker

The MATLAB frontend (`frontends/Matlab/Utilities/`) carries a large collection
of solution-manipulation routines — projecting a field between polynomial
orders, interpolating a solution from one mesh onto another, locating points,
computing gradients, extracting surface data, extruding 2D solutions to 3D.
These are the *"maneuvering a solution"* operations: taking an existing DG
solution and re-representing it somewhere else.

Much of this belongs in the C++ backend, where it is reusable from every
frontend and from standalone consumers rather than living only in MATLAB. This
document tracks that migration: what each MATLAB utility does, whether a C++
equivalent already exists, and what is still worth porting.

## The porting pattern

Follow the existing utility ports (`backend/Postprocessing/eulereval.*`,
`reynolds_averages_3d.*`): a **self-contained, host-side** function on flat
column-major arrays, independent of the Kokkos solver structs, paired with a
standalone `*_test.cpp` that has its own `main()` and hand-verified assertions.
These compile and run on their own —

```sh
g++ -std=c++17 -O2 backend/Discretization/dgprojection.cpp \
                   backend/Discretization/dgprojection_test.cpp -o /tmp/t && /tmp/t
```

— so correctness is pinned before any wiring into `CDiscretization`. Once a
routine is proven, a thin Kokkos/`CDiscretizationT` entry point can call it (or
mirror it as a device kernel) without re-litigating the math.

For a routine that runs over the whole mesh (not a one-off), the scalar
reference is the **oracle**, not the shipping implementation. The performant
version is written in the backend idiom (`massinv.hpp` / `getuhat.hpp`): a
templated `<T=dstype,I=Int>` function that block-iterates `common.eblks` and
expresses the math as batched calls to the existing primitives
(`ShapJac`, `Node2Gauss`, `Gauss2Node`, `Inverse`, `ArrayGemmBatch1`, …), which
already dispatch CPU / CUDA / HIP by the `backend` argument. Because these
operations are element-local, running them on this rank's blocks is inherently
MPI-ready — no halo exchange. `dgprojection` ships both: the scalar reference
(`dgprojection.*`) and the batched backend path (`dgprojection_backend.hpp`),
cross-checked against each other by `dgprojection_backend_test.cpp`.

**Do `volgeom` early.** The Jacobian/determinant kernel is copy-pasted as a
nested function inside `dgprojection.m`, `l2eprojection.m`, `massinv.m`,
`averagevector.m`, `calerror.m`, and `gradu.m`. `dgprojection.cpp` exposes it as
`volgeom_det()`; a full standalone `volgeom` (also returning the inverse-Jacobian
`Xx`) would let every future port share one kernel.

## Status legend

| status | meaning |
|---|---|
| ✅ **done (C++)** | a C++ equivalent already exists in the backend — reuse it, do not re-port |
| 🟡 **partial** | building blocks exist in C++ but the specific routine/glue does not |
| 🔴 **missing** | no C++ equivalent; a port is wanted |
| ✳️ **this PR** | ported here |
| ⏭️ **skip** | not a maneuvering op, or superseded — do not port as-is |

## Already in C++ — reuse, do not re-port

The backend already covers a large part of the maneuvering surface. Confirmed
equivalents (so a "port" would just duplicate them):

| MATLAB | C++ equivalent | location |
|---|---|---|
| `mkshape`, `masternodes`, `koornwinder`, `tensorproduct` | `mkshape`, `koornwinder*`, `tensorproduct`, `masternodes`, `gaussnodes` | `backend/Preprocessing/makemaster.cpp` |
| `findxi`, `findex`, `findinvalidxi`, `locatexinmesh`, `newtonx` | `ComputeReferenceCoordinates`, `FindPointInElement[Iterative]`, `CPointLocator` — **already nd-generic (2D & 3D)** | `backend/PointLocator/pointlocation.hpp`, `pointlocator.h` |
| `evalfield`, `fieldatx`, `interpsol` (eval side) | `InterpolateFieldAtReferencePoint/Batch`, `FindPointShapeAndField*` | `backend/PointLocator/pointinterpolation.hpp` |
| `volgeom` (jac, `Xx`) — vectorized | `ElemGeom{1,2,3}D`, `ElemGeomBlock` | `backend/Common/kokkosimpl.h`, `backend/Discretization/geometry.cpp` |
| `massinv` | `ComputeMinv`, `ApplyMinv` | `backend/Discretization/massinv.hpp` |
| `dg2cg`, `dg2cg2` | `DG2CG`/`DG2CG2`/`DG2CG3`, `ArrayDG2CG` | `backend/Discretization/discretization.cpp`, `Common/kokkosimpl.h` |
| `getuhat` | `GetUhat`, `UhatBlock` | `backend/Discretization/getuhat.hpp` |
| `eulereval`, `eulereval3d`, `nseval*`, `FavreAverages` | `eulereval`, `reynolds_averages_3d` | `backend/Postprocessing/` |
| `tensorproductquadrature` (test script) | `gaussnodes`, `tensorproduct` | `backend/Preprocessing/makemaster.cpp` |

Note the MATLAB locate chain (`newtonx`, `findinvalidxi`) is **2D-locked**,
while the C++ `PointLocator` is already full nd — the C++ side is the better
implementation here, not a gap.

## Genuinely missing in C++ — port targets

Ranked by value × tractability for maneuvering solutions.

| # | MATLAB | category | status | notes / plan |
|---|---|---|---|---|
| **1** | `dgprojection` | projection | ✳️ **this PR** | Order-to-order L2 projection on a fixed mesh: `M\(C·U)` with cross-mass `C=∫φ_tgt φ_src`. The C++ backend had `M`, `M⁻¹`, `ApplyMinv`, `mkshape`, `volgeom` — but **no cross-mass and no `M\(C·U)` glue**. Shipped as a scalar reference (`dgprojection.{hpp,cpp}`) **and** a batched, GPU/MPI-ready backend path (`dgprojection_backend.hpp`, `DGProjection<T,I>`). |
| 2 | `volgeom` (standalone, with `Xx`) | geometry | 🟡 partial | `volgeom_det` landed in this PR (determinant path). A full standalone returning inverse-Jacobian `Xx` would retire the 6 nested copies. Vectorized C++ exists as `ElemGeomBlock` but not as a self-contained utility. |
| 3 | `l2eprojection` | projection | ✳️ **this PR** | L2-project a function onto the DG space (init / MMS / error setup). Same `M` machinery as #1 plus a load vector `F=∫φ f jac`. Shipped as scalar reference (`l2eprojection.{hpp,cpp}`) + backend Kokkos kernel (`l2eprojection_backend.hpp`, `L2eProjection`); the func→`fg` (f at Gauss points) step is the caller's, so the device path is pure linear algebra. |
| 4 | `gradu` | gradient | 🔴 missing | Physical gradient of a DG field at nodes (chain rule + inverse Jacobian). Only the LDG auxiliary `q=∇u` exists in C++; a direct nodal-gradient utility is absent. |
| 5 | `graduface` | gradient | 🔴 missing | HDG face-lifted gradient `∫(u−uhat)·n·jac`. Intricate (nested `facegeom`, HDG face indexing) but self-contained. |
| 6 | `extrudesol`, `extrudecoord`, `extrudevelocity` | extrusion | ✳️ **this PR** | 2D→3D solution/coordinate extrusion (z-layers × slabs) and radial→Cartesian velocity rotation. Was entirely absent in C++. Shipped as a scalar reference (`extrudesol.{hpp,cpp}`) **and** backend-portable Kokkos kernels (`extrudesol_backend.hpp`, `ExtrudeSolution`/`ExtrudeCoord`/`ExtrudeVelocity`) — one `parallel_for` each, GPU/CPU, element-local (MPI-ready). |
| 7 | `fieldatdgnodes`, `fieldatuniquedgnodes` | interpolation | 🟡 partial | Exact mesh-to-mesh transfer (locate + eval). The C++ pieces exist (`CPointLocator` + `InterpolateFieldBatch`); the missing part is a turnkey "field on mesh A → field on mesh B nodes" driver. ⚠ `fieldatuniquedgnodes.m` has a bug: nd==3 sets `z = dgnodes1(:,2,:)` (should be `(:,3,:)`) — do not carry it over. |
| 8 | `surfacefield`, `surfacenormal`, `getsolonsurface` | surface-extraction | 🔴 missing | Interpolate solution to boundary-face quadrature points (+ coords, normals) and nodal surface extraction. Clean, modern, 2D/3D in MATLAB; directly useful for coupling/BC sampling. `interfacesampler.*` is adjacent but not the same op. |
| 9 | `averagevector` | mass-matrix | 🔴 missing | Lumped integral vector `L=∫φ jac` (mass-weighted node volumes). LOW; reuses `volgeom`. |
| 10 | `computeElemMeasure` | geometry | 🔴 missing | Element measure `∫jac dξ`. Trivial once `volgeom` weights are available. |
| 11 | `meshdist`, `meshdist2`, `meshdist3` | geometry | 🔴 missing | Wall-distance field (min distance to boundary nodes). Consolidate the three MATLAB variants into one nd-generic routine; consider a KD-tree over brute force. |
| 12 | `calerror` | field-eval / error | 🔴 missing | L2 error vs an exact `func` at Gauss points, per component. Needs the same callback mechanism as #3. |
| 13 | `smoothing` | utility | 🔴 missing | 1D `[0.25 0.5 0.25]` filter, k passes. Trivial, no deps. |
| 14 | `fixdgmesh` | geometry (repair) | 🔴 missing | Average `dgnodes` on shared faces until conforming. Indexing only. |
| — | `interpsol`, `fieldatx` | interpolation | ⏭️ skip | Superseded: `interpsol` mixes in `scaplot` plotting and calls `fieldatx` with an outdated signature; `fieldatx` is 2D-only and nearest-neighbor *approximate*. The clean replacement is a `fieldatdgnodes`-style path (#7) on the C++ `PointLocator`. |
| — | `matvecExasim` | I/O | ⏭️ skip | Binary readback of `outhdgMatVec.bin`; not a maneuvering op. |
| — | colormaps, `scaplot*`, `*plot*`, `mkmovie`, `mkpng`, `MarchingCubes`, etc. | plotting | ⏭️ skip | Visualization; out of scope for the backend. |

## Notes / traps carried out of the audit

- **`volgeom` is silently duplicated** as a nested function in ≥6 utilities.
  Port it once and reuse.
- **The MATLAB locate chain is 2D-locked** (`newtonx` hardcodes a 2×2 Jacobian,
  `findinvalidxi` only implements nd==2). Do **not** port these — the C++
  `PointLocator` already does nd-generic point location correctly.
- **`fieldatuniquedgnodes.m` nd==3 bug**: `z = dgnodes1(:,2,:)` should be
  `(:,3,:)`.
- **`massinv.m` dev cruft**: a stray `Mi(:,:,k)` print + `pause` inside the
  element loop — drop it in any port.
- The tensor-product factorization documented in `tensorproductquadrature.m`
  (Kronecker node↔Gauss apply) is worth baking into the C++ projection/eval
  kernels for speed once the straightforward versions are in.

## Done in this PR

- `backend/Discretization/dgprojection.{hpp,cpp}` — portable **scalar reference**
  (`dgprojection`) and `volgeom_det`.
- `backend/Discretization/dgprojection_backend.hpp` — **performant backend path**
  `DGProjection<T,I>`: batched, CPU/CUDA/HIP-portable, MPI-ready (element-local
  over `common.eblks`), reusing `ShapJac` / `Gauss2Node` / `Inverse` /
  `ArrayGemmBatch1` exactly as `ComputeMinv` does. Straight meshes use a single
  shared master operator `P0 = M0⁻¹C0` (the Jacobian cancels); curved meshes take
  the per-element path. Included via `residual.hpp` alongside `massinv.hpp`.
- `backend/Discretization/dgprojection_test.cpp` — oracle test: `volgeom_det`
  determinants (1D/2D/3D), identity round-trip on curved multi-element geometry,
  cross-basis p1→p2 exactness + p2→p1 round trip.
- `backend/Discretization/dgprojection_backend_test.cpp` — reimplements the
  backend primitives with their exact semantics/layouts and replays the
  `DGProjection` straight and curved algorithms, asserting they match the scalar
  oracle (including a curved element where the Jacobian genuinely varies). Both
  tests compile and pass under `-Wall -Wextra`.
- `CDiscretizationT::projectField(...)` and `projectionSelfTest(...)`
  (`discretization.{h,cpp}`) — expose the batched projection as a discretization
  method and run an on-device identity self-test (gated by
  `EXASIM_TEST_PROJECTION`) during construction, per MPI rank.
- `backend/Discretization/extrudesol.{hpp,cpp}` + `extrudesol_test.cpp` — scalar
  reference + standalone oracle for the 2D→3D extrusion family (gather map,
  coordinate field, velocity rotation), hand-verified against `extrudesol.m`'s
  permute/reshape semantics.
- `backend/Discretization/extrudesol_backend.hpp` — Kokkos kernels
  `ExtrudeSolution`/`ExtrudeCoord`/`ExtrudeVelocity` (GPU/CPU, element-local),
  included via `residual.hpp`. `CDiscretizationT::extrusionSelfTest(...)` runs a
  mesh-free on-device check (synthetic 2D field: exact gather + `vx²+vy²==1`
  rotation) per MPI rank, gated by `EXASIM_TEST_EXTRUDE`.
- `backend/Discretization/l2eprojection.{hpp,cpp}` + `l2eprojection_test.cpp` +
  `l2eprojection_backend.hpp` (`L2eProjection`) — L2 projection of a Gauss-sampled
  load onto the DG space (`M⁻¹F`, `F=∫φ f jac`), scalar reference + Kokkos kernel
  (straight fast path + curved). `CDiscretizationT::l2eProjectionSelfTest(...)`
  projects the coordinate field `f=x` and checks reproduction at the nodes per
  rank, gated by `EXASIM_TEST_L2EPROJ`.
- `backend/Discretization/refinemesh.{hpp,cpp}` + `refinemesh_test.cpp` +
  `refinemesh_backend.hpp` (`RefineMeshHighOrder`) — **high-order uniform mesh
  refinement** (tensor elements): each child's geometry is the parent's
  isoparametric map at the child node positions, so curvature is preserved
  exactly. A *new* util (not a direct MATLAB port — closest are the Python
  `refine_dg_hexmesh.py` p2-hex refiner and MATLAB `uniref`), built as a
  shared-operator batched apply `refined_c = P_c · dgnodes` (`P_c` = parent basis
  at the child nodes) — the same fast path as the straight-mesh projection, so
  it is GPU/CPU + MPI-ready, and the *same* `P_c` prolongs a DG field onto the
  children. `CDiscretizationT::refineSelfTest(...)` (tensor elements) builds `P_c`
  from `master.xpe` as a tensor-product Lagrange basis (the unique nodal basis for
  the node set, so it equals Exasim's basis without needing `mkshape`), refines
  the mesh on-device, and checks operator partition-of-unity + device-vs-host-
  reference apply per rank (`EXASIM_TEST_REFINE`); the host oracle pins the
  geometric high-order exactness.

### On-device validation (done)

`projectionSelfTest` runs an **identity** projection (source basis == target
basis) of a real field during `finalizeConstruction`, on whatever backend/device
the solver is on, independently per MPI rank. For identity `C == M` exactly, so
`U1 = M⁻¹(M U)` and the residual is bounded by the mass-matrix conditioning
`κ(M)·ε` — machine precision when `jac` is constant (straight), larger on curved
high-order elements (the projection is exact; this is inversion conditioning, and
the `C≠M` math is pinned to machine precision by the host oracle). The gate is
tight on the straight path (`1e-9`) and conditioning-tolerant on curved (`1e-3`).

Measured on the `dgprojection` pass (builtin consumer, 256 elements):

| target | backend | mesh | identity relerr |
|---|---|---|---|
| laptop CPU (np=1) | 1 | straight | 4.4e-16 |
| laptop CPU (np=1) | 1 | curved p3 | 4.8e-7 |
| laptop CPU (np=2, MPI) | 1 | both | PASS (each rank) |
| CSAIL dgx-b, **NVIDIA V100** | 2 (CUDA) | straight | **1.332e-15** |
| LLNL tuolumne, **AMD MI300A** | 3 (HIP) | straight | **1.221e-15** |

The GPU runs are real Exasim GPU solves (`CUDA Device: 0` / `HIP Device: 0`, then
Newton converging) with the projection self-test firing at construction. The
batched path therefore runs correctly on both GPU vendors and, being
element-local, on every MPI rank without communication.

The **extrusion** kernels (`ExtrudeSolution`/`ExtrudeVelocity`) are validated the
same way by `extrusionSelfTest` (`EXASIM_TEST_EXTRUDE`): a synthetic 2D field
gives an exact gather (`gather_maxmiss=0`) and a `vx²+vy²==1` rotation identity
within precision, per rank:

| target | backend | gather miss | rotation err |
|---|---|---|---|
| laptop CPU (np=1) double model | 1 | 0 | 2.2e-16 |
| laptop CPU (np=1) single model | 1 | 0 | 6.0e-8 (float; prec-scaled PASS) |
| laptop CPU (np=2, MPI) | 1 | 0 | PASS (each rank) |
| CSAIL dgx-b, **NVIDIA V100** | 2 (CUDA) | 0 | 0.0 |
| LLNL tuolumne, **AMD MI300A** | 3 (HIP) | 0 | 1.1e-16 |

The gather is an integer-exact index copy on every backend; the rotation check's
tolerance scales with `sizeof(dstype)` because the builtin consumer runs both a
double and a single-precision model.

**`l2eprojection`** is validated by `l2eProjectionSelfTest` (`EXASIM_TEST_L2EPROJ`),
which projects the coordinate field `f=x` and checks reproduction at the nodes
(`κ(M)`-limited, like the projection identity):

| target | backend | identity relerr |
|---|---|---|
| laptop CPU (np=1, np=2 MPI) | 1 | PASS (100%) |
| CSAIL dgx-b, **NVIDIA V100** | 2 (CUDA) | **1.887e-15** |
| LLNL tuolumne, **AMD MI300A** | 3 (HIP) | **1.776e-15** |

All three ports (dgprojection, extrusion, l2eprojection) are therefore validated
on-device on both GPU vendors (NVIDIA V100 / AMD MI300A) and under CPU-MPI.

### Next step

A registered ctest (rather than an env-gated hook) and a `C≠M` cross-order
on-device check (needs a second-order master, i.e. `mkmasternodes` + `mkshape` in
the backend) would tighten the on-device coverage from "identity + host-oracle
math" to "device cross-order vs reference".
