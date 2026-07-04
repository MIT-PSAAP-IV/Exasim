# Codebase review & cleanup pass (2026-07-03)

Branch `teoc-properly-separate-out`, at `2c348b72`. A "declare the precision/PETSc
work done, then review and clean the codebase" pass. Six review dimensions + a
coupling regression test + a cleanup inventory + a DMPlex scoping question.

**One-line verdict:** the architecture is in good shape to declare the current
milestone done — one true core (no per-backend math duplication), a clean
consumer/template surface for the linear-PETSc target, a real pure-C++/binary
library path, and the CHEFSI-apps coupling apps still build and run correctly
against a fresh install from this branch. The remaining work is a concrete,
mostly-low-risk backlog captured below.

---

## Test result — CHEFSI-apps coupling still works ✅

Rebuilt + installed Exasim from this branch **out of tree** into
`/Users/teoc/projects/psaap4/Exasim-build` (install prefix
`Exasim-build/install`), then ran
`~/projects/psaap4/remote/test-coupling.sh` with `EXASIM_INSTALL` pointed at it.

| Variant | Model path | Result |
|---|---|---|
| `poisson2d-new-architecture-with-kitesurf` | `Exasim::builtinmodel` (model ID 8) | ✅ build + `mpirun -np 2` run, coupled Newton/GMRES converged, QoI check passed |
| `poisson2d-external-model-with-kitesurf` | `exasim_add_external_builtin_model` (model ID 100) | ✅ text2code regen → build → run, Newton residual `7.3e-3 → 1.6e-9`, QoI check passed |

Final harness line: `==> DONE: all coupling Poisson tests passed.` (rc=0). The
refactor did **not** break the coupling apps' C++/CMake consumer API or their
numerics.

**Caveat (a CHEFSI-apps data defect, not Exasim):** the first run crashed with
`YAML::BadSubscript` because
`poisson2d-new-architecture-with-kitesurf/input.yaml` was committed as a literal
**Git-LFS pointer stub** yet is **not** `.gitattributes`-tracked, so
`git lfs pull` will not smudge it. Fix for a fresh checkout:
`git lfs fetch --all`, then copy `.git/lfs/objects/<oid-prefix>/<oid>` over each
stub file. Worth fixing in `CHEFSI-apps@9a64e2c` so a clean clone runs.

Build/environment notes:
- In-source build directories are now **rejected** (`CMakeLists.txt:31` guard) —
  configure out of tree (`Exasim-build`), not `build_local`.
- `runjob`'s `taskpolicy` wrapper hit `posix_spawn: Permission denied` on the
  coupling driver script; ran it via plain `nohup bash … &` instead.

---

## Review 1 — Do all backends thread through one core? ✅ Yes

There is **one core codebase**. Two orthogonal dispatch axes, neither duplicates
the math:

- **Model dispatch** — `EXASIM_DRIVER_CALL` (`include/exasim/detail/driver_dispatch.hpp:45`)
  is an `if constexpr` on the model *type* (frontend-generated ABI kernels vs
  templated `exasim::Name<M>`). About *which PDE*, not which hardware.
- **Backend dispatch** — a runtime `Int backend` (0/1 CPU, 2 CUDA, 3 HIP)
  threaded as a trailing arg. It steers only **allocation, BLAS vendor, and
  sync** (`backend/Common/common.h:452`, `backend/Common/pblas.h:59`), never the
  compute logic. The pointwise math is written **once** as
  `KOKKOS_INLINE_FUNCTION` kernels over the *default* execution space, so
  CPU-vs-GPU is a compile-time Kokkos configuration, not a second source path
  (no `.cu`, no `cpuFlux`/`gpuFlux` pairs).
- **MPI is a pure halo-exchange wrapper** (`backend/Discretization/residual.hpp:600`):
  `RuResidualMPI` adds `MPI_Isend/Irecv` and interior/interface range-splitting,
  then calls the *same* `GetUhat/GetQ/GetW/RuElem/RuFace` compute. GPU-MPI = the
  same wrapper over a CUDA-configured Kokkos build.

**Duplication/threading smells worth cleaning:**
1. `include/kernels/cpuInitu.hpp` vs `KokkosInitu.hpp` — the one genuine
   per-backend kernel duplication (identical IC math; the host copy exists only
   to stage the GPU initial solution on the host). Could be a `Kokkos::Serial`
   instantiation of the same kernel.
2. The `backend` int is largely redundant with the fixed compile-time Kokkos
   space on the compute side — threading noise in every compute signature.
3. `ioutilities.hpp`/`.cpp` duplicated (stale include-vs-TU leftover).
4. The `include/exasim/{kokkosimpl,cpuimpl,pblas}.h` re-include shims are
   documented Phase-1.2 transitional debt.

---

## Review 2 — Class/data organization 🔶 improved, real smells remain

The C1–C5 refactor already split the old god `commonstruct` into named
per-concern sub-structs and gave ownership to `CDiscretization` (owns
`sol/res/app/master/mesh/tmp/common/scratch/wallmodel`);
`CResidual`/`CAssembler`/`CPreconditioner` are stateless operators over a
`CDiscretization&`. That part is genuinely good.

Remaining smells (ranked by cleanup value):
1. **`appstruct` magic-index decode** — raw `ndims/nsize/lsize/flag` int arrays
   decoded by literal offset (`backend/Discretization/setstructs.cpp:68`), ~341
   brittle reads. It is a serialization blob, not a domain type. *(highest
   value, highest risk)*
2. **Every struct is a hand-rolled malloc bag** — parallel `sz*` fields +
   `printinfo/sizeof*/freememory` boilerplate, no RAII. Correctness depends on
   hand-matched alloc/free lists and alias guards (`res.fhAliasesK`,
   `K=nullptr`). An owning `DeviceArray<T>` (size + backend + RAII) would kill
   these hazards structurally.
3. **`commonstruct` still large** (`backend/Common/common.h:2032`) — sub-structs
   *plus* ~40 loose top-level fields (`ind_*`, comm arrays, DIRK/BDF coeffs,
   `timing[128]`). **Partially drained 2026-07-04:** the LDG block-Jacobian CRS
   index cluster (`ind_ii/ind_ji/ind_il/ind_jl/num_ji/num_jl/Lnum_ji/Lind_ji/
   Unum_ji/Uind_ji`) moved into `blockjacindexstruct bjindex` — a cohesive,
   common-only concern contained in 4 files (96 uniform `common.X` sites),
   verified byte-identical (app-regression 12/12). **Remaining, deferred:** the
   halo/MPI-exchange cluster (`nbsd/elemsend/…`, ~450 sites) and the
   time-integrator coeffs (`dt/dae_dt/DIRK*/BDF*`, ~125 sites) — larger and
   collision-prone (`dt`/`dae_dt` also live on `appstruct`), each wants its own
   careful pass. Note: `bjindex` arrays are allocated in `crs_init` but never
   freed (pre-existing leak, preserved as-is).
4. ~~**Duplicated state**~~ — **investigated 2026-07-04: false positive, no
   change.** `PDEStateSnapshot` (`backend/Solution/solution.h:123`) is a
   load-bearing checkpoint buffer (`SaveState`/`RestoreState`, coupling rollback
   via the `ExasimSolver` ABI), not a redundant mirror. `solutionlayoutstruct`
   (`common.h:1959`) is an intentional model-decoupling scaffold (G4/Phase D —
   will carry model-supplied field names). The wall-model/STG `app`↔`common`
   mirroring is the standard serialized-input→runtime-copy pattern, entangled
   with the `appstruct` item (#1) rather than an isolated dedup.
5. ~~**Naming**~~ — **addressed 2026-07-04 by documenting, not renaming.** The
   `res.C/E/D/B/F/G/H/K` single letters are meaningful HDG block-Jacobian
   notation (the `[D F; K H]` local static-condensation system, LDG `q` Schur-
   eliminated via `D += B·Minv·C` etc.), used as the working local-variable names
   across ~300 `res.X` sites and matching `block-diagonal-jacobian.md`. A rename
   is high-risk/low-value; "split residual vs matrix store" fights the
   intentional shared-`K`-arena (`fhAliasesK`; `K` doubles as `sys.v` scratch).
   Fixed the real pain instead — 5 of 8 members carried an identical misleading
   "store the diffusion matrix" comment — with accurate per-block role docs.

---

## Review 3 — Core exposed to consumers? ✅ Clean for the linear-PETSc goal

`find_package(Exasim [COMPONENTS petsc])` installs the templated internals under
`include/backend/` (`install/CMakeLists.txt:737`), so a consumer compiles
`CDiscretizationT<T,I>`, `make_preprocessed`, `recover_volume`, `eval_qoi`,
`Operator`, `ShellMat` directly. Precision is a real template arg. `export.hpp`
is the curated free-function surface; `petsc.hpp` gives MatShell MatMult,
PCShell PCApply, zero-copy condensed RHS, `assemble_matrix` (real MATAIJ), and
`make_mass_inverse`. The `tests/consumers/{operators,solve_fp32}` and
`tests/petsc/*` drivers exercise exactly this.

**Gaps a PETSc-driving consumer hits:**
- **Nonlinear Newton isn't wrapped** — `Operator::formjacobian` is a no-op
  (`include/exasim/petsc.hpp:164`) and `formfunction` hard-codes affine
  `F = H·U − b0`. A nonlinear PDE consumer must re-call
  `hdgAssembleLinearSystem` + `ComputeHDGPreconditioner` itself each SNES
  iterate (those are public, but the "PETSc owns the loop" facade is turnkey
  only for linear/affine problems).
- **`sysstruct`/`setsysstruct` leak** as raw backend types every driver must
  construct.
- **DG↔CG trace scatter map is internal only** — no accessor; a PCFieldSplit or
  custom-scatter consumer must reach into `disc.mesh.elemcon`.
- **GPU in-memory ctor contract is under-documented** (header says CPU-only;
  code and tests carry GPU branches).

---

## Review 4 — Exasim as a pure C++ library (no text config) 🔶 real, with gaps

The pure-C++/binary path is end-to-end (`tests/consumers/solve_fp32/main.cpp`):
raw C++ mesh arrays → `exasim::default_pde<M>()` → `MeshSpec::add_boundary`
coordinate predicates → `make_preprocessed<M,T,I>` → `CDiscretizationT<T,I>` →
solve → `eval_qoi`. No `pdeapp.txt`/`pdemodel.txt`, no frontend, no datain files
touched. `ExasimSolver<M>` is the same path with a fluent wrapper.

**Gaps (what still forces files / is missing):**
1. The model must be a **compile-time C++ struct** satisfying `is_model_v<M>`.
   "No pdemodel.txt" means you hand-write the physics in C++; the only
   runtime-loadable alternative is the text2code/ABI path, which needs
   `pdeapp*.txt`.
2. Two binary runtime assets are still read from disk: `masternodes.bin` +
   `gaussnodes.bin` (nodal/quadrature tables) via `$EXASIM_DATA_DIR`
   (`backend/Preprocessing/makemasterexasim.hpp:1024`).
3. `readMeshFromFile`/binary/Gmsh/VTU readers exist
   (`backend/Preprocessing/readmesh.hpp`) but there is **no helper bridging them
   to `MeshSpec`** — a consumer reads arrays itself, then hands `p`/`t` to
   `MeshSpec`.
4. Solution I/O exists (`writesolstruct`/`readsolstruct`,
   `backend/Postprocessing/solution_io_lib.hpp`) but there is **no clean
   `save_solution/load_solution` helper** in `export.hpp`; reload is coupled to
   the datain/dataout layout.

---

## Scope — DMPlex → Exasim mesh converter ✅ feasible & cheap

### Serial: minimal input, local rebuild

`meshFromArrays` (`backend/Preprocessing/makemeshexasim.hpp:1318`) consumes
**only** `p` (coords) + `t` (cell→vertex, 0-based) + `nve` + `nd` + boundary
specs. Everything heavy — `facecon`, `elemcon`, `f2e`, `bf`, perms — is *rebuilt*
inside `connectivity()` from cell-vertex alone (`mkf2e_hash`). Exasim **ignores**
DMPlex's cone/support/face graph.

- **Input footprint is minimal:** pass `DMGetCoordinatesLocal` + cell closures
  (→ vertex ids). No connectivity-graph duplication at the interface — both
  sides build faces, but you only ever materialize Exasim's. That is the best
  achievable.
- **Natural seam:** `MeshSpec` / `meshFromArrays`. The one real code addition is
  boundary tagging: Exasim tags by *coordinate predicate*, DMPlex by *face-label
  id* ("Face Sets"). Cleanest fix — extend `MeshSpec` to accept explicit
  per-boundary-face tags (write `bf`/`boundaryConditions` directly, bypass the
  predicate scan).
- **Constraint:** one homogeneous element type per mesh (single `nve`; no hybrid
  DMPlex without splitting).

Rebuilding this connectivity locally is cheap and safe: it is per-rank compute
with **no communication**, so re-deriving faces from cell-vertex is not a concern.

### Distributed goal: don't recompute anything that needs *communication*

The real design intent is stronger than "don't duplicate data": **if Exasim is
handed an already-partitioned DMPlex mesh — with its cell partition and its point
star-forest (`PetscSF`) describing shared/ghost points — it should not re-run any
step that requires MPI communication.** DMPlex has *already* done the global
discovery (who owns what, which points are shared); redoing it is wasted
collectives, and worse, a second partitioner can disagree with PETSc's, breaking
the correspondence the consumer expects.

The current distributed path is `make_preprocessed_distributed`
(`include/exasim/export.hpp:175`) → `meshFromArraysDistributed`
(`makemeshexasim.cpp:1356`) → `CPreprocessing::takeParallel`
(`backend/Preprocessing/preprocessing.cpp:204`). Its communication-requiring
steps, and whether a DMPlex partition + point-SF makes them elidable:

| Preprocessing step | Comm | Elidable with DMPlex partition + SF? |
|---|---|---|
| `callParMetis` → `ParMETIS_V3_PartMeshKway` (`parmetisexasim.cpp:581,129`) | `Allgather` + collective repartition | **Yes** — the partition is given. (Already partly supported via `pde.partitionfile`, :589.) |
| `migrateMeshWithParMETIS` (`:159`) — `Alltoall(v)` moving cells + a node-request protocol (`:338`) | `Alltoall(v)` | **Yes, becomes a no-op** if the consumer feeds each rank exactly its owned cells (DMPlex already has). |
| `mke2e_fill_first_neighbors` (`:637`) — cross-rank face-neighbor discovery → `dmd.nbinfo` | `Alltoall(v)` (`:721`) | **Yes** — the point-SF *is* this shared-interface adjacency. |
| `buildElemsend` (`:2018`) — negotiate which ghosts each neighbor must send | `Isend/Irecv` (`:2081`) | **Yes** — the send/recv pattern *is* the SF (leaves↔roots). |
| Global node-id resolution (`mesh.nodeGlobalID`/`tg`) | (folded into the above) | **Yes** — DMPlex global point numbering supplies it directly. |
| 3× `sendrecvdata` ghost fills of `bf`/`tg`/`xdg` (`preprocessing.cpp:262,286,295`) | `Isend/Irecv` | **Payload only** — the *pattern* is free from the SF, but ghost values must physically arrive once. Collapses to a local copy *only* if the consumer also hands over ghost-cell data. |
| `mergePeriodicNodeIDs` / `setperiodicfaces` (`:1361,1051`) | `Allgather(v)` | Only if periodic BCs; DMPlex periodicity would supply the same. |

**Verdict:** with a supplied partition + point-SF, **every collective/discovery
step is elidable** — the ParMETIS repartition, the cross-rank neighbor discovery,
the send/recv negotiation, and global-node numbering all come straight from
DMPlex. The one irreducible communication is a **single one-shot ghost-payload
exchange** (coords/global-ids/boundary-flags for ghost cells), and even that
disappears if the converter passes ghost-cell data alongside owned cells. So the
goal — *no communication-bearing recompute* — is achievable.

**Injection seam:** branch in `CPreprocessing::takeParallel`
(`preprocessing.cpp:204`) *before* `callParMetis` (`:237`) and `initializeDMD`
(`:243`): if a prebuilt decomposition is supplied, skip both and populate the
`DMD` struct directly. Carry it on an extended **`MeshSpecDistributed`**
(`export.hpp:151`). The DMD fields the consumer must fill (all derivable from
PetscSF leaves/roots + the DMPlex cell partition + global point numbering) are:
`dmd.elempart`, `elempart_local`, `elem2cpu`, `nbsd`, `elemsend`/`elemsendpts`,
`elemrecv`/`elemrecvpts`, `localelemsend`/`localelemrecv`, plus
`mesh.nodeGlobalID`, `mesh.elemGlobalID`, `mesh.tg`. These are exactly what the
runtime halo exchange (`RuResidualMPI`, `residual.hpp:378`) already consumes, so
once populated the solve path is unchanged.

Note: a lighter-weight but *not* communication-free fallback already exists —
`pde.partitionfile` (`parmetisexasim.cpp:589`) skips the ParMETIS *partitioning*
call but still runs `migrate` + neighbor-discovery + send/recv negotiation. To
hit the "zero comm recompute" goal you want the full DMD-injection seam above,
not just the partition-file path.

---

## Cleanup inventory

`apps/` and `examples/` are **not duplicates** — `apps/` are C++/text2code
standalone apps (12 are live regression fixtures for `run-app-regression.sh`);
`examples/` are MATLAB-frontend examples (740 files, only 5 tested). Overlapping
names (poisson2d, cone, isoq3d…) exercise different toolchains. The annoyance is
curation, not literal duplication; neither should be deleted wholesale.

The real waste (ranked):

| Path | Why | Recommended action | Risk |
|---|---|---|---|
| `apps/navierstokes/{reactingsharpb2,orion}` (~52M raw `.bin`) | built/tested by nothing (not in `tests/app-baselines/`) | delete or move to LFS | low (verify not a demo ref) |
| `apps/**/*.bin` (~84M, not LFS) | raw mesh/solution blobs in git history | LFS-ify the *tested* ones | medium (regression fixtures) |
| `old/` (13M; `baseline/` = 11M) | self-documented "nothing references these" | delete (git history preserves) | low |
| `tests/archive/` | superseded runners, not wired into `tests/CMakeLists.txt` | delete | low |
| root `heat_petsc_final.vtu`, `poisson_petsc.vtu` | stray local run outputs (already gitignored) | `rm` locally | none |
| `examples/` untested tail (~735 of 740) | only 5 wired to CI | keep for docs; prune heaviest unused | low |

No committed `.o`/`.a`/`.so` build outputs found in the index.

---

## Test surfaces & CI gaps

| Surface | Where it runs |
|---|---|
| consumer builtin CPU / MPI (np=2) | in-CI |
| `model_fp32`, `solve_fp32`, `operators`, `facade` (self-checking binaries) | in-CI (CPU-serial Kokkos only) |
| python frontend + exports/coexistence/combined/postprocess/modelcache | in-CI |
| `builtin_model4_kernel_equivalence`, `sharedlibrary_app` | in-CI |
| `petsc_operator_export` + `petsc/{heat,poisson_mpi,robustness}` | **CI-registered but SKIP** (no PETSc in apt) |
| app golden regression (`run-app-regression.sh`, 12 baselines) | **local-only** (not a ctest) |
| MATLAB / Julia frontends + regression | local-only (SKIP 77 on CI) |
| GPU (`tests/remote/consumers/builtin-gpu*`) | remote-only (dgx-b), never automatic |
| LDG (hybrid=0) | **none** (known broken on non-MATLAB paths) |

**Most valuable missing surfaces (ranked):**
1. **Wire `run-app-regression.sh` into CI** — biggest single guard for the
   precision/consumer-API work (full native stack, 12 golden L2 baselines,
   tol 1e-8). Effort: medium (git-lfs pull + build apps on the runner).
2. **Install PETSc on the runner** so `petsc_operator_export` + `robustness`
   RUN instead of SKIP — they exercise the zero-copy `PetscScalar` path the
   precision work touched. Effort: low.
3. **Scheduled GPU lane** (dgx-b) running `builtin-gpu*` + `model_fp32`/
   `solve_fp32` to make the "CPU/GPU byte-identical" claim testable, not
   asserted. Effort: medium-high.
4. **LDG xfail marker** so the known gap is explicit. Effort: low.
5. **Fix the stale "Registered tests" comment** in
   `.github/workflows/smoke-cpu.yml:13` — it omits `sharedlibrary_app`,
   `petsc_operator_export`, and the `frontend_*_export`/`combined`/`postprocess`
   tests that unfiltered `ctest` actually runs.

---

## Cleanup backlog

**Done (2026-07-04):**
- ✅ Deleted `old/` (13M; verified no active script/CMake consumer — prose-only
  references in `tests/README.md` + `docs/theory/references.md`, both updated).
- ✅ Deleted `tests/archive/` (superseded runners; only a comment in
  `tests/CMakeLists.txt` + a `tests/README.md` "Archived" section referenced it,
  both updated).
- ✅ `rm`'d stray root `heat_petsc_final.vtu` / `poisson_petsc.vtu` (untracked,
  already gitignored).

**Held — NOT dead weight after verification (correction to the audit):** the
untested `apps/**/*.bin` flagged for deletion are **documented examples**, not
clutter — `apps/navierstokes/reactingsharpb2` is the reacting-flows demo in the
Frontier/Tuolumne install guides + `docs/install/frontier.md`, and
`apps/navierstokes/orion` is built-in-model 11 in `docs/usage-modes/builtin.md`.
`apps/poisson/orion` is additionally a live regression fixture (`poisson_orion`
baseline). So do **not** delete them. The only real lever left is a git-LFS
*migration* of the large `.bin` snapshots (history rewrite — a separate,
deliberate operation, not a routine cleanup); left for an explicit decision.

**Structural (larger, future):** `appstruct` named fields (kill the magic-index
decode); owning `DeviceArray<T>` to replace the malloc-bag pattern.

**CI (future):** add app-regression + PETSc lanes; scheduled GPU lane; fix the
stale registered-tests comment.
