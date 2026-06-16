# Shared library

In the shared-library mode the model kernels are compiled into a **dynamic
library** that the solver loads at runtime. This is the path the Python / Julia /
MATLAB frontends drive: you author the PDE interactively, the frontend runs
`text2code` to generate the kernels, builds them into a model library, and runs a
pre-built solver executable against it.

Prerequisite: an [installed Exasim](../install/index.md). The generated model
satisfies the same [model contract](../reference/model-contract.md) as the other
modes; here it is delivered as a separate `.so`/`.dylib` rather than linked into
your binary.

## Build artifacts and reuse

The generated model is compiled into a dynamic library; per app, everything lives
in the hidden `pde.builddir` (default `<cwd>/.exasim/`):

```text
.exasim/
  kernels/                   # the generated model kernel .cpp set
  CMakeLists.txt, main.cpp   # rendered app project (from the installed templates)
  build/
    libfrontend_model.so     # the generated model as a dynamic provider library
    exasimapp                # the solver executable (loads the model at runtime)
    .exasim_model_hash       # SHA-256 of the model inputs from the last build
```

The heavy code — Kokkos, the solver libraries, the built-in model library — is
prebuilt in the install prefix and never recompiled. Reuse is **hash-based**:
when the kernel set and rendered app sources hash to the last successful build,
the build system is skipped and the solver runs directly. When the model
changes, only the provider translation unit recompiles and only
`libfrontend_model` relinks — `exasimapp` itself is never rebuilt. Mesh,
parameter, and solver-option changes never trigger compilation.

Built model libraries are additionally cached **per user** under
`~/.exasim/cache/<modelID>/<digest>/` (override the root with `EXASIM_CACHE_DIR`).
Any app directory whose model hashes to the same digest reuses it with zero
compilation. The digest covers the kernel set, the app templates, the
variant/model ID, and the identity of the Exasim install, so model changes and
Exasim upgrades invalidate cleanly.

!!! note "One Kokkos runtime"
    The model library deliberately does not embed Kokkos; it resolves Kokkos
    symbols from `exasimapp` at load time, so there is exactly one Kokkos runtime.

## Driving it from a frontend

From a frontend you author the model and call into the toolchain; the
shared-library build/run above happens automatically. See the Frontends section
for the Python / Julia / MATLAB entry points and the `pdemodel` authoring DSL.

## The C++ shared-library app

The raw C++ counterpart lives in `apps/sharedlibrary/`. It links a generated
model library (`t2cmodelserial`, `t2cmodelcuda`, or `t2cmodelhip`) found in the
install prefix and is compiled with `_SHAREDLIBRARY`:

```cpp
#include "ExasimSolverSetup.hpp"
#include "sharedlibprovider.hpp"

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    MPI_Comm comm = MPI_COMM_NULL;
#endif
    ExasimSolver solver;
    return RunExasimSolver(solver, argc, argv, comm);
}
```

Its `CMakeLists.txt` selects the model library by backend, locates it in
`${prefix}/lib`, and sets the RPATH so the solver finds it at load time:

```cmake
set(EXASIM_APP_MODEL_LIB t2cmodelserial)   # t2cmodelcuda / t2cmodelhip for GPU
find_library(EXASIM_APP_MODEL_LIBRARY NAMES ${EXASIM_APP_MODEL_LIB}
  HINTS "${EXASIM_APP_PREFIX}/lib" "${EXASIM_APP_PREFIX}/lib64" NO_DEFAULT_PATH)
target_compile_definitions(exasimapp PRIVATE _SHAREDLIBRARY)
target_link_libraries(exasimapp PRIVATE
  Exasim::headers Exasim::pre "${EXASIM_APP_MODEL_LIBRARY}")
```

Run it like any other app:

```bash
mpirun -np 4 build/exasimapp /path/to/pdeapp.txt
```

See `apps/sharedlibrary/` for the complete project.
