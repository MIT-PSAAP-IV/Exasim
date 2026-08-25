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
| **1** | `dgprojection` | projection | ✳️ **this PR** | Order-to-order L2 projection on a fixed mesh: `M\(C·U)` with cross-mass `C=∫φ_tgt φ_src`. The C++ backend had `M`, `M⁻¹`, `ApplyMinv`, `mkshape`, `volgeom` — but **no cross-mass and no `M\(C·U)` glue**. Ported as `backend/Discretization/dgprojection.{hpp,cpp}` + `dgprojection_test.cpp`. |
| 2 | `volgeom` (standalone, with `Xx`) | geometry | 🟡 partial | `volgeom_det` landed in this PR (determinant path). A full standalone returning inverse-Jacobian `Xx` would retire the 6 nested copies. Vectorized C++ exists as `ElemGeomBlock` but not as a self-contained utility. |
| 3 | `l2eprojection` | projection | 🔴 missing | L2-project an **analytic** `func(x,param,t)` onto the DG space (init / MMS / error setup). Same `M` machinery as #1 plus a load vector `F=∫φ f`; needs a callback mechanism. |
| 4 | `gradu` | gradient | 🔴 missing | Physical gradient of a DG field at nodes (chain rule + inverse Jacobian). Only the LDG auxiliary `q=∇u` exists in C++; a direct nodal-gradient utility is absent. |
| 5 | `graduface` | gradient | 🔴 missing | HDG face-lifted gradient `∫(u−uhat)·n·jac`. Intricate (nested `facegeom`, HDG face indexing) but self-contained. |
| 6 | `extrudesol`, `extrudecoord`, `extrudevelocity` | extrusion | 🔴 missing | 2D→3D solution/coordinate extrusion (z-layers × slabs) and radial→Cartesian velocity rotation. **Entirely absent** in C++ (grep of the whole backend finds nothing). Pure indexing, LOW difficulty. |
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

- `backend/Discretization/dgprojection.hpp` — clean host-side declarations.
- `backend/Discretization/dgprojection.cpp` — `dgprojection` (order-to-order L2
  projection) and `volgeom_det`.
- `backend/Discretization/dgprojection_test.cpp` — standalone test:
  `volgeom_det` determinants (1D/2D/3D), an identity round-trip on curved,
  multi-element geometry, and cross-basis p1→p2 linear exactness with a p2→p1
  round trip. Compiles and passes with `-Wall -Wextra`.
