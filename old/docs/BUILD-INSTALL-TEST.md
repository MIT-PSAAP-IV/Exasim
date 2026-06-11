# Exasim — build / install / test (TEMP overview)

> Scratch reference for the current state of the `teoc-Kitesurf-Coupling-build-and-test`
> branch (post-merge with `Kitesurf-Coupling`). Not committed. Delete when done.

## TL;DR

```sh
# build everything (one command; reuses already-built vendored deps)
cmake -B build && cmake --build build

# install a consumable CMake package
cmake --install build --prefix /path/to/prefix      # or set CMAKE_INSTALL_PREFIX at configure

# run the regression suite
bash tests/run-tests.sh
```

---

## Build

**One entry point: the superbuild** (`./CMakeLists.txt`). Orchestrates the whole
dependency chain with a **find-or-build** policy — prefer an installed dep, fall
back to the vendored source:

Vendored third-party deps live under **`deps/`** (`deps/{kokkos,metis,symengine}`);
`text2code/text2code` is Exasim's own tool. Legacy/unused trees (`baseline/`,
`tutorial/`, `docs/`) are parked under **`old/`**. `apps/` and `examples/` stay.

**Build out of the source tree.** The superbuild builds the vendored Kokkos and
SymEngine into the *build directory* (`<build>/deps/...`), not into `deps/`, and
forwards their locations to the solver build (`-DKokkos_DIR`, `EXASIM_SYMENGINE_DIR`).
A true in-source build is a fatal error; a build dir inside the repo warns. Prefer
a sibling dir: `cmake -B ../Exasim-build && cmake --build ../Exasim-build`.
(METIS is normally found via apt/conda; its rare vendored fallback still builds in
`deps/metis`, gitignored.)

| Dep | Found via | Vendored fallback |
|---|---|---|
| Kokkos | `find_package(Kokkos)` | `deps/kokkos/build{serial,cuda,hip}` (backend-aware; CUDA uses `nvcc_wrapper` + arch) |
| METIS/ParMETIS | inline `find_library` | `deps/metis/` (`make metis`) |
| SymEngine | `find_package(SymEngine)` | `deps/symengine/` (bundled boost) |

Then it builds `text2code` and finally the solver layer (`install/`). For
CUDA/HIP it forwards `nvcc_wrapper`/`hipcc` as the solver CXX.

**Configure options** (forwarded to the solver build):

```
-DEXASIM_MPI=ON/OFF        -DEXASIM_NOMPI=ON/OFF
-DEXASIM_CUDA=ON/OFF       -DEXASIM_HIP=ON/OFF
-DEXASIM_LIB=ON/OFF        (static libs only, skip executables)
-DWITH_PARMETIS=ON         -DWITH_TEXT2CODE=OFF
-DWITH_BUILTINMODEL=ON     -DWITH_KOKKOSKERNEL=OFF
-DEXASIM_BUILD_TESTS=OFF   (register the ctest suite)
-DCMAKE_INSTALL_PREFIX=...
```

**Advanced**: the solver layer alone, when deps already exist:
`cmake -S install -B build && cmake --build build`.

**Built-in models** are regenerated from text2code at build time via a
CMake-generated X-macro registry (`backend/Model/BuiltIn/CMakeLists.txt`):
`exasim_add_builtin_model(<id> SOURCE pdeapp<id>.txt)` → models 7–15; 1–6 reuse
checked-in kernels. External consumers can inject their own with the same call.
The library is a **static** archive (`libbuiltinmodel{serial,cuda,hip}.a`) to
avoid a second copy of Kokkos's globals in the consumer.

## Install / consume

Installs a relocatable CMake package. Consumers:

```cmake
find_package(Exasim REQUIRED COMPONENTS cpumpi)   # or cpu / gpu / gpumpi
target_link_libraries(app PRIVATE Exasim::headers Exasim::pre Exasim::builtinmodel)
```

`Exasim::pre` and `Exasim::builtinmodel` are stable **chooser** targets resolved
from the requested component; `find_package` fails loudly if that variant was
not built. `ExasimConfig.cmake` pulls in Kokkos itself (bakes `Kokkos_DIR`).

The three **provider paths** are selected by compile defines in the consumer:
`_BUILTINLIBRARY` (built-in model lib), `_TEXT2CODE` (codegen'd model lib),
`_KOKKOSKERNEL` (header-only Kokkos kernels). `_MPI` selects the MPI variant.

---

## Test

**One runner, one ctest.**

- `tests/run-tests.sh` — configures the superbuild with `-DEXASIM_BUILD_TESTS=ON`
  (+ a writable install prefix, bounded `JOBS`), builds, runs `ctest`. This is
  what CI runs. Knobs: `BUILD`, `JOBS`, `CMAKE_ARGS`; trailing args pass to ctest.
- ctest **`consumer_out_of_tree`** (`tests/CMakeLists.txt`, gated by
  `EXASIM_BUILD_TESTS`) → `tests/run-consumer-tests.sh`, which for each project
  under `tests/consumers/`:
  - **B3** asserts the consumer CMakeLists never reaches into the Exasim source tree;
  - **B2** configures it *only* via `find_package(Exasim)` against an install prefix, builds it;
  - **B4** runs it (if it ships `pdeapp.txt`) and checks the QoI gate
    (`QOI_TOL`, default `1e-8`).

**Consumers** (`tests/consumers/`):

| Consumer | Provider path | What it checks |
|---|---|---|
| `builtin/` | `_BUILTINLIBRARY` | builds + runs Poisson 2D (model 1), QoI[1] ≈ 5e-13 < 1e-8 |
| `facade/`  | headers only | the public headers resolve out-of-tree (compile, no run) |

Add a consumer = drop a `find_package(Exasim)` project under `tests/consumers/`;
the harness picks it up. Ship a `pdeapp.txt` to get the B4 run + QoI gate.

**CI** (`.github/workflows/smoke-cpu.yml`): `ubuntu-22.04`, externalized
METIS/BLAS/MPI via apt, vendored Kokkos + SymEngine built once and cached, then
`run-tests.sh`. Triggers on push to the teoc branch and PRs into
`Kitesurf-Coupling` (the protected merge target requires this check).

**Numerical gates of record**
- builtin consumer (poisson2d, model 1): `Domain_QoI1 = 4.992401e-13`
- coupled poisson2d (CHEFSI app, model 8): `4.839096e-02 / 1.994160e-01`

**Archived** under `tests/archive/` (superseded, nothing runs them): legacy
in-tree consumers `install_consumer` / `multi_tu_consumer`, the bespoke
`run-install-consumer.sh`, the MATLAB `exasimtest.m`. See its README.
