# Verification overview (branch `teoc-properly-separate-out`)

How the precision-threading / operator-export / LDG work on this branch is
verified, what each harness checks, where it runs, and the honest gaps. Written
as merge-readiness context (see the "prep for merge with master" task).

## The core invariant: byte-identical under defaults

Every code change is held to **byte-identical-under-defaults**: with the default
`dstype` (double) and `Int`, each change must reproduce the golden solutions.
In practice every commit's app-regression `rel_L2` lands in **1e-12 … 1e-17**,
far under the **1e-8** tolerance — that residual is pre-existing round-off, and
"unchanged" is the proof the change altered no numerics. Two intentional
exceptions:
- **LDG** — a genuine *correctness fix*, so it produces new (correct) output;
  added as its own golden baseline rather than matched to the old (broken) one.
- **float32** — a consumer *type choice*, checked to a precision tolerance
  (~1e-4 for a full solve), not bit-identity.

## The harnesses

| Harness | What it verifies | Where it runs |
|---|---|---|
| **app-regression** (`tests/run-app-regression.sh`, 13 golden baselines, tol 1e-8) | full native solve → L2 vs golden `outudg`/`outuhat`/`qoi` across Poisson 2D/3D, curved (cone/isoq3d/orion), periodic, L-shape, NS (isoq/naca-steady/mach8/sharpb2), HDG **and** LDG, serial **and** MPI(np=2) | **local only** |
| **consumer guards** — `operators`, `solve_fp32`, `solio`, `meshload` | in-memory operator export + HDG residual/assembly; full float32 solve == double (rel ~1e-4); `save/load_solution` bit-exact; file-mesh == array-mesh (topology + `xdg`) | **CI** (self-checking binaries; swept by `run-consumer-tests.sh`) |
| **reproducibility** (run-twice `cmp`) | determinism across reruns and MPI ranks | local, ad-hoc |
| **`tests/remote/gpu-ldg-test.sh`** | LDG converges + reproduces the QoI on GPU and GPU+MPI | **dgx-b (V100), manual** |
| **coupling** (`remote/test-coupling.sh`) | CHEFSI apps build + run + QoI against a fresh install | local, manual |
| **smoke-cpu CI** (`.github/workflows/smoke-cpu.yml`) | builds the CPU stack + runs the WHOLE ctest suite; consumers, python frontend (+exports/combined/postprocess), model4-kernel-equivalence, sharedlibrary, and (now) PETSc drivers | **CI** — now triggers on this branch |

## Coverage matrix (verified this session)

| Axis | CPU | CPU-MPI | GPU | GPU+MPI |
|---|---|---|---|---|
| HDG solve (12 apps) | ✅ app-reg | ✅ np=2 cases | ~ manual | ~ manual |
| **LDG** (new fix) | ✅ | ✅ | ✅ dgx-b | ✅ dgx-b |
| float32 | ✅ solve_fp32 | — | ✅ model/solve_fp32 | — |
| precision-threading struct changes | ✅ 13/13 byte-identical after every change | ✅ | ✅ | ✅ |

Each of the code commits this session (bjindex drain, AppNdims enum, LDG,
save/load + mesh bridge) was gated on **app-regression 13/13 + the consumer
guards**; LDG and precision additionally on GPU via dgx-b. LDG (`∫u = 0.4052847`
identical on all four backends) is now the best-verified path in the tree.

## Gaps (honest)

1. **app-regression is local-only** — the strongest numerical guard does not run
   in CI. Biggest gap. Addressed by the Python-frontend-in-CI task (or by wiring
   `run-app-regression.sh` as a CI job: git-lfs pull + per-app builds).
2. **GPU is manual** — no scheduled dgx-b lane; the CPU/GPU-identical claim is
   re-established by hand each time.
3. **Untested capability axes** — transient (torder>1), reacting, 1D have no
   committed baseline: the `apps/{poisson1d,reactingsharpb2,navierstokes/orion}`
   dirs are incomplete (missing the retargeted-text2code `my_model.hpp`),
   `naca0012unsteady` is ~270M (intentionally gitignored), and 1D is
   Python-facenumbering-blocked.
4. **Coupling not re-verified at current head** — last confirmed at `2c348b72`,
   many commits ago.
5. **No verification against current master** — the branch has diverged; merge
   readiness needs a master reconcile.
6. **`frontend_python_modelcache` is broken** — the second app dir never hits the
   model cache; documented and left unwired in `tests/CMakeLists.txt`.
7. **MATLAB frontend** regression is unavoidably local-only (no MATLAB on
   GitHub-hosted runners).

**Net:** default-precision numerical correctness is strongly verified
(byte-identical, 13/13, after every change); the weak spots are about *where*
verification runs (local vs CI vs manual-GPU) and *coverage breadth*
(transient/reacting/coupling-at-head), not about the correctness of what has been
checked.
