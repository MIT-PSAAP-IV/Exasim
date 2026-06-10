# Archived / legacy

Moved out of the active tree. Nothing in the build, the test suite, or CI
references these — they are kept for history and may have stale internal paths.

| Dir | Was | Status |
|---|---|---|
| `baseline/` | macOS-only numerical regression (`verify.sh`, md5-of-binaries, legacy direct-`text2code` build) and 9 recorded cases | Superseded by the portable QoI gate in the ctest suite (`tests/`); only poisson2d is currently reproduced there. |
| `tutorial/` | Tutorial sections + `run-all.sh` smoke runner | Not wired into ctest/CI. |
| `docs/` | Older documentation | — |

The live build/install/test entry points are the superbuild (`./CMakeLists.txt`),
the package (`find_package(Exasim)`), and `tests/run-tests.sh`.
