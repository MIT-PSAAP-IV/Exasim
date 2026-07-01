# App golden baselines (git-lfs)

Volume-solution reference output (`outudg_np*.bin` = the `udg` DOF vector per MPI rank) for each
unique app, generated from **main**. `tests/run-app-regression.sh <install>` rebuilds each app
(from this repo's `apps/`) against a chosen Exasim install, runs it as-shipped, and diffs the
solution against these baselines via relative L2 (`tests/compare_app_l2.py`). A byte-identical
native solver gives `rel_L2 = 0`.

- Binaries are stored via **git-lfs** (`*.bin`, see top-level `.gitattributes`). `git lfs pull`
  to fetch them.
- Regenerate from a baseline install: `tests/gen-app-baselines.sh <baseline-install>`.
- Excluded from git (see `.gitignore`): the ~270M `naca0012unsteady` transient dump, and
  `nsmach8`/`sharpb2`/`isoq` (segfault on main as-shipped). Regenerate those locally if needed.

Committed baselines (8 apps): poisson2d, poisson3d, periodic, lshape, orion, isoq3d, cone,
naca0012steady.
