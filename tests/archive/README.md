# Archived test artifacts

These predate the consolidated regression suite (`tests/run-tests.sh` +
`tests/consumers/` + the `consumer_out_of_tree` ctest). They are kept for
reference only — nothing builds or runs them, and their paths may be stale.

| Archived | Replaced by | Why |
|---|---|---|
| `install_consumer/` | `tests/consumers/facade/` | The facade consumer was migrated from this project; it is the headers-resolve out-of-tree consumer the harness now drives. |
| `multi_tu_consumer/` | `tests/consumers/builtin/` | The in-tree multi-TU consumer reached into the source tree; the builtin consumer is a clean out-of-tree `find_package(Exasim)` consumer with a QoI gate. |
| `run-install-consumer.sh` | `tests/run-consumer-tests.sh` | A single bespoke runner for `install_consumer`; the harness now iterates every project under `tests/consumers/` (B3 source-tree guard, B2 out-of-tree build, B4 run + QoI gate). |
| `exasimtest.m` | `tests/run-tests.sh` (ctest) | MATLAB-driven smoke test; the regression is now build-system-native and runs without MATLAB. |

For the live regression: from the repo root run `bash tests/run-tests.sh`
(or configure with `-DEXASIM_BUILD_TESTS=ON` and run `ctest`).

The numerical baseline under `../../baseline/` (md5-of-binaries, macOS `md5 -r`,
legacy direct-`text2code` build) is likewise superseded by the portable B4 QoI
gate (`QOI_TOL`, default 1e-8); it remains in place only because the historical
`LIBRARY_PLAN.md` / `LIBRARY_PORT_INVENTORY.md` port notes reference it.
