# Rank-separated HDG/LDG checks

These focused checks reuse native preprocessing, residuals, preconditioners,
GMRES and DIRK routines. The public fixture exercises the explicit coupled
solve; the preprocessing executable alone does not launch it. CPU checks
do not establish GPU support. Existing
`conservation_cases.cmake` and `gas_cases.cmake` are unrelated and unchanged.

Configure against an existing source-built Exasim package, outside the source:

```sh
cmake -S tests/hdgldg -B /tmp/hdgldg-checks \
  -DExasim_DIR=/absolute/source-build/package -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/hdgldg-checks --parallel 2
ctest --test-dir /tmp/hdgldg-checks -R '^hdgldg_(solver_np[234]|stages_dirk[23]_np[234])$' --output-on-failure
```

The CMake project uses the same configured METIS/GKlib as Exasim. The tests
cover periodic-aware selection, persistent IDs, induced-graph partitioning,
noncontiguous ranks, multiple HL faces, 3D, aligned fields and native regional
binary payloads. Generated files remain in the temporary build directory.

## Shared preprocessing entry

```sh
EXASIM_DATA_DIR=/absolute/Exasim/backend/Preprocessing \
  /tmp/hdgldg-checks/exasim_hdgldg_preprocess \
  /absolute/pdeapp.txt /absolute/selection.txt /tmp/new-hdgldg-output
```

Without `--native-metadata`, model dimensions are obtained from the existing
Text2Code parser. With that flag, the common configuration must explicitly
specify all native dimensions, including visualization and QoI dimensions;
no model export/generation is performed. Native MATLAB, Python and Julia
entry points are `hdgldgpreprocessing(config, manifest, outdir, executable)`.
They preserve native model generation and return the H/L input directories.
This standalone preprocessing API does not generate model kernels. The
native frontend must generate and compile its model separately. Four-language
physical example integration is the Phase 6 deliverable, not something
established by these preprocessing checks.

## Coupled public solve and checkpoint replay

Compile the ordinary model-provider application against the candidate runtime,
then launch both prepared regions together:

```sh
mpirun -np 2 /absolute/model-app --hdgldg /tmp/prepared-case /tmp/new-run
mpirun -np 2 /absolute/model-app --hdgldg /tmp/prepared-case /tmp/new-replay \
  --restart 2 /tmp/new-run
```

The output directory must be new, absolute and under `/tmp`. `H/` and `L/`
retain native file layouts and local rank numbers. Restart currently requires
the same prepared input, compiled model, partition and controls, and an interior
full-step checkpoint. DIRK22 and DIRK33 snapshots are supported; auxiliary w,
wave/AV, reduced-basis/polynomial acceleration and solution averaging are not
advertised. Unsupported coupled options do not restrict ordinary pure paths.

`stages.cpp` independently checks mass/source assembly, native stage recurrence,
history lifetimes, rollback and poisoned-ghost refresh. `lifecycle.py` checks
complete stage histories and compares actual public uninterrupted/restarted
files. `pure_lifecycle.cpp` uses the ordinary pure solver with test-only
checkpoint cadence, providing the native replay baseline first. Exact commands
and evidence are recorded in `docs/hdgldg/reports/`; do not run the full CTest
suite or unrelated Phase 14 baseline for this series.

Manifest version 1 accepts exactly `version`, `world_size`, `h_ranks`,
`l_ranks`, `selection`, and either `labels` or `seed_boundaries` plus `layers`.
Labels are 0=L/1=H in original input-element order. Rank sets cover the coupled
communicator, and boundary seeds refer to model boundary codes. Peer/world
translations and original IDs are written in versioned `hdgldg.meta` and
`hdgldg*.txt` sidecars. Native app/master/mesh/sol binary formats are unchanged.
