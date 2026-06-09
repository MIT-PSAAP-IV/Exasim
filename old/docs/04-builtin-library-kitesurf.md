# Built-in library coupling with Summit via kitesurf

This manual explains how to couple Exasim with Summit through
kitesurf when the Exasim side uses the built-in library path.
The reference application is
`apps/builtinlibrary/main.cpp`.

The focus here is not on authoring new PDE math inside a standalone
consumer application. Instead, the goal is to let kitesurf drive
Exasim as a reusable solver component while selecting an existing
registered built-in model at runtime through `builtinmodelID`.

## Scope and assumptions

- Summit is the external code that owns one side of the multiphysics
  workflow.
- kitesurf owns orchestration, data transfer, interpolation, and
  launch sequencing between Summit and Exasim.
- Exasim owns mesh preprocessing, DG assembly, linear and nonlinear
  solves, and output generation.
- The built-in model already exists under
  `backend/Model/BuiltIn/modelN/` and is registered in
  `backend/Model/BuiltIn/libbuiltinmodel.cpp`.
- The first production coupling should be file-based at runtime:
  kitesurf prepares Exasim input files, launches Exasim, and then
  reads Exasim outputs back into the coupled workflow.

This is the lowest-risk integration path because it avoids adding
new Exasim-side orchestration code while still using the new built-in
library provider ABI.

## Why use the built-in library path

The built-in library path gives you three useful properties:

1. The Exasim executable stays generic.
   The same app binary can run different built-in models by changing
   `builtinmodelID` in `pdeapp.txt`.

2. The PDE kernels live behind one provider ABI table.
   `backend/Model/BuiltIn/builtinlibprovider.cpp` exports the full
   `ExasimDriverABI` for all registered built-in models, and
   `RunExasimSolver(...)` attaches that ABI to each model definition.

3. The coupling app can be built out of tree.
   `apps/builtinlibrary/CMakeLists.txt` links against exported Exasim
   preprocessing libraries plus `libbuiltinmodelserial`,
   `libbuiltinmodelcuda`, or `libbuiltinmodelhip`.

For Summit coupling, this is preferable to a frontend-generated or
text2code-generated executable when the model set is known in advance
and should be selectable at runtime.

## Architecture

The runtime flow is:

`Summit -> kitesurf transfer/projection -> Exasim -> kitesurf transfer/projection -> Summit`

On the Exasim side, the important pieces are:

- `apps/builtinlibrary/main.cpp`
  Thin driver that includes `ExasimSolverSetup.hpp` and
  `builtinlibprovider.cpp`, then calls `RunExasimSolver(...)`.

- `backend/Model/BuiltIn/builtinlibprovider.cpp`
  Builds the `ExasimDriverABI` table from the exported built-in model
  entry points such as `builtinKokkosFlux`, `builtinHdgFlux`, and the
  `cpuInit*` kernels.

- `include/ExasimSolverSetup.hpp`
  Centralizes provider selection and applies the built-in library ABI
  to each model definition owned by `ExasimSolver`.

- `backend/Preprocessing/readpdeapp.cpp`
  Reads `builtinmodelID`, `meshfile`, `xdgfile`, `udgfile`,
  `vdgfile`, `wdgfile`, `exasimpath`, and the other runtime fields
  from `pdeapp.txt`.

- `backend/Model/BuiltIn/libbuiltinmodel.cpp`
  Dispatches every built-in kernel family by `builtinmodelID`.

## Build procedure

The built-in library app depends on two separate build products:

1. Exasim preprocessing and solver support libraries exported through
   CMake.
2. The built-in model shared library built from
   `backend/Model/BuiltIn/Makefile`.

The built-in model shared library is not
produced by the main `install/CMakeLists.txt`. The CMake package and
the built-in model Makefile must both be built.

### Step 1: build the Exasim static libraries

The app in `apps/builtinlibrary/` links against Exasim-exported static
libraries such as `Exasim::cpuprelib` and `Exasim::cpumpiprelib`.
Build those first from `Exasim/build` by configuring the install tree:

```bash
cd Exasim/build
cmake \
  -DEXASIM_LIB=ON \
  -DEXASIM_MPI=ON \
  -DEXASIM_NOMPI=ON \
  -DWITH_PARMETIS=ON \
  -DWITH_TEXT2CODE=ON \
  -DWITH_BUILTINMODEL=ON \
  ../install
```

Then build the configured targets:

```bash
cmake --build .
```

This produces the exported Exasim package metadata plus the static
libraries needed by `apps/builtinlibrary/CMakeLists.txt`.

### Step 2: build the built-in model shared library

After configuring and building Exasim, build the built-in model shared
library from `Exasim/backend/Model/BuiltIn` using one of the Makefile
targets below:

```bash
cd Exasim/backend/Model/BuiltIn
make serial   # builds the CPU shared library
make cuda     # builds the CUDA shared library
make hip      # builds the HIP shared library
```

These targets generate the shared libraries consumed by the app, for
example `libbuiltinmodelserial`, `libbuiltinmodelcuda`, and
`libbuiltinmodelhip` under `Exasim/lib/`.

### Step 3: build `apps/builtinlibrary`

After Steps 1 and 2 are complete, build the coupling app itself from
`Exasim/apps/builtinlibrary`:

```bash
cd Exasim/apps/builtinlibrary
cmake -B build -DExasim_DIR=/path/to/Exasim
cmake --build build
```

This produces the `exasimapp` executable that links against the Exasim
static libraries from Step 1 and the built-in model shared library
from Step 2.

`apps/builtinlibrary/CMakeLists.txt` is the glue layer between the
generic coupling app and the Exasim build products created in Steps 1
and 2.

Its logic is:

1. Resolve `Exasim_DIR`.
   The script accepts `Exasim_DIR` pointing to either:
   - the Exasim source root
   - the Exasim build directory
   - the installed package directory

   It then normalizes that path until it can find
   `ExasimConfig.cmake`, and from that path it derives
   `EXASIM_APP_PREFIX`, which is the root used to locate headers,
   exported CMake targets, and `Exasim/lib`.

2. Load the exported Exasim package.
   It calls:

   ```cmake
   find_package(Exasim REQUIRED)
   ```

   which makes the exported Exasim targets available to the app.

3. Select the Exasim preprocessing library target.
   The app does not link directly against the main solver executable.
   Instead, it links against one of the exported preprocessing-support
   static libraries:
   - `Exasim::cpuprelib` for serial CPU builds
   - `Exasim::cpumpiprelib` for MPI CPU builds
   - `Exasim::gpuprelib` for serial GPU builds
   - `Exasim::gpumpiprelib` for MPI+GPU builds

   The choice is driven by the CMake options:
   - `EXASIM_MPI`
   - `EXASIM_GPU`

   In other words, `apps/builtinlibrary/CMakeLists.txt` maps the app's
   requested execution mode onto the matching Exasim support library.

4. Select the built-in model shared library.
   The PDE kernels themselves come from the built-in shared library,
   not from the Exasim static preprocessing libraries. The CMake file
   therefore searches `EXASIM_APP_PREFIX/lib` for:
   - `builtinmodelserial` in CPU and MPI+CPU mode
   - `builtinmodelcuda` in CUDA mode

   This is why Step 2 is required: if `make serial` or `make cuda` has
   not been run in `backend/Model/BuiltIn`, configuration of the app
   will fail because the library cannot be found.

5. Add the required compile definitions.
   The app target is compiled with `_BUILTINLIBRARY`, and with `_MPI`
   when `EXASIM_MPI=ON`.

   `_BUILTINLIBRARY` is the key switch. It makes
   `ExasimSolverSetup.hpp` select:

   - `getBuiltInLibraryExasimDriverABI()`
   - provider name `BuiltInLibrary`

   rather than the frontend-generated, text2code-generated, or other
   provider modes.

6. Link the final app.
   The resulting `exasimapp` target links:
   - `Exasim::headers`
   - the selected Exasim preprocessing static library
   - the selected built-in model shared library

   It also sets `BUILD_RPATH` to the directory containing the
   built-in shared library so that `exasimapp` can locate
   `libbuiltinmodelserial`, `libbuiltinmodelcuda`, or
   `libbuiltinmodelhip` at runtime.

In short, `apps/builtinlibrary/CMakeLists.txt` does three distinct
jobs:

- import the Exasim static support libraries
- locate the built-in PDE shared library
- compile `main.cpp` in built-in-library mode so that
  `RunExasimSolver(...)` dispatches through the built-in ABI table

## Runtime contract for kitesurf

For this coupling mode, kitesurf should treat Exasim as a process
that consumes one working directory containing:

- `pdeapp.txt`
- `grid.bin`
- optional `xdg.bin`
- optional `udg.bin`
- optional `vdg.bin`
- optional `wdg.bin`

The most important runtime field is:

```text
builtinmodelID = N;
```

This selects which registered built-in model the app will use.


## Adding a new built-in model 

Adding a new built-in model means doing two things:

1. generate a new `backend/Model/BuiltIn/modelN/` source tree for the
   new model ID `N`
2. register that model ID in
   `backend/Model/BuiltIn/libbuiltinmodel.cpp` so the built-in library
   dispatch table can call it at runtime

Exasim supports two practical ways to do this:

- Matlab code generation, through `kkgenmodel` and related helpers
- `text2code`, through `pdeappN.txt` and the helper scripts in
  `backend/Model/BuiltIn`

Several working examples already exist in
`backend/Model/BuiltIn`, especially:

- `genmodel5.m`
- `genmodel6.m`
- `genmodel7.m`
- `genmodel8.m`
- `genmodel9.m`
- `genmodel10.m`
- `genmodel11.m`

These are the best starting points when adding the next model.

### What must exist for a built-in model

At the end of either workflow, the repository should contain:

- a model directory such as `backend/Model/BuiltIn/model12/`
- model wrapper files `model.hpp` and `model.cpp` in that directory
- generated kernel files in that directory, such as `KokkosFlux.cpp`,
  `KokkosFbou.cpp`, `HdgFlux.cpp`, `cpuInitu.cpp`, and related files
- registration of model 12 inside
  `backend/Model/BuiltIn/libbuiltinmodel.cpp`
- any source model inputs used to regenerate it, for example
  `pdemodel12.m`, `pdeapp12.txt`, or `pdemodel12.txt`

The registration step is essential. If the source tree exists but
`libbuiltinmodel.cpp` is not updated, the built-in library app cannot
dispatch the new model ID.

### Option A: add a model through the Matlab code generator

This path is appropriate when the model is authored as a Matlab PDE
definition and you want Exasim's Matlab-side generators to emit the
built-in kernels directly.

The current examples are `genmodel5.m` and `genmodel6.m`.

The typical workflow is:

1. Create or update the Matlab PDE model file.
   For a new built-in model ID `N`, create
   `backend/Model/BuiltIn/pdemodelN.m`.

2. Set the model dimensions and metadata.
   Your generator script must set at least:
   - `pde.builtinmodelID`
   - `pde.modelfile`
   - `pde.nd`
   - `pde.ncu`
   - `pde.ncq`
   - `pde.ncw`
   - `pde.ncv`
   - `pde.ntau`
   - `pde.nmu`
   - `pde.neta`

   `genmodel5.m` and `genmodel6.m` show the expected pattern.

3. Run `kkgenmodel(pde)`.
   This generates the built-in model sources under
   `backend/Model/BuiltIn/modelN/`.

4. Let the helper update the dispatch table.
   In this path, `kkgenmodel` clones the base
   `backend/Model/BuiltIn/model.hpp` and `model.cpp` into the new
   `modelN/` directory and calls
   `editlibbuiltinmodel(...)` to register the new model in
   `libbuiltinmodel.cpp`.

There is also a convenience helper:

- `frontends/Matlab/Gencode/addBuiltinModel.m`

`addBuiltinModel(app)` copies the current Matlab model into
`backend/Model/BuiltIn/pdemodelN.m`, creates a matching `genmodelN.m`,
and then runs that generator script.

Use this path when your source of truth is the Matlab PDE model and
you want the built-in library generated from that authoring workflow.

### Option B: add a model through `text2code`

This path is appropriate when the model is authored as a text-based
`pdeappN.txt` or `pdemodelN.txt` workflow and you want to generate the
built-in source tree from the `text2code` executable.

The current examples are `genmodel7.m` through `genmodel11.m`.

The typical workflow is:

1. Create the text2code inputs for the new model ID `N`.
   In practice this usually means adding:
   - `backend/Model/BuiltIn/pdeappN.txt`
   - any matching model description files it references

2. Make sure `text2code` has already been built.
   The helper scripts expect the executable at:
   `Exasim/build/text2code`

3. Create a generator script modeled after the existing examples.
   The scripts `genmodel7.m` through `genmodel11.m` all follow the
   same pattern:
   - set `modelID = N`
   - set `exasimpath`
   - compute `modelpath = .../backend/Model/BuiltIn/modelN`
   - call `editmodelhppcpp(...)`
   - call `editlibbuiltinmodel(...)`
   - run `text2code pdeappN.txt --out-dir modelN`

4. Run the generator script.
   This writes the generated built-in kernels into
   `backend/Model/BuiltIn/modelN/` and updates the built-in dispatch
   table.

The two helper functions used by this path are:

- `frontends/Matlab/Gencode/editmodelhppcpp.m`
  Creates `modelN/model.hpp` and `modelN/model.cpp` by cloning the
  base wrappers and rewriting the namespace from
  `exasim_model_1` to `exasim_model_N`.

- `frontends/Matlab/Gencode/editlibbuiltinmodel.m`
  Idempotently inserts the include lines, namespace alias, and
  dispatch `case N:` blocks into
  `backend/Model/BuiltIn/libbuiltinmodel.cpp`.

Use this path when your source of truth is the text2code input deck
and you want the built-in library to be regenerated from
`pdeappN.txt`.

### Manual checks after generation

Regardless of which path you use, inspect the following:

- `backend/Model/BuiltIn/modelN/`
- `backend/Model/BuiltIn/libbuiltinmodel.cpp`

At minimum, confirm that:

- `modelN/model.hpp` exists
- `modelN/model.cpp` exists
- the generated kernel files exist in `modelN/`
- `libbuiltinmodel.cpp` contains:
  - `#include "modelN/model.hpp"`
  - `#include "modelN/model.cpp"`
  - `namespace mN = exasim_model_N;`
  - `case N:` entries in the built-in dispatch switches

`editlibbuiltinmodel.m` is designed to be idempotent, so re-running it
should not duplicate registration lines.

### Rebuild the built-in library after adding the model

After the new model has been generated and registered, rebuild the
built-in shared library so that `apps/builtinlibrary/exasimapp` can
use it:

```bash
cd Exasim/backend/Model/BuiltIn
make serial   # CPU / MPI+CPU built-in library
make cuda     # CUDA built-in library, if needed
make hip      # HIP built-in library, if needed
```

Then rebuild the app if needed:

```bash
cd Exasim/apps/builtinlibrary
cmake --build build
```

Finally, test the new model by setting:

```text
builtinmodelID = N;
```

in the target `pdeapp.txt` and launching `exasimapp`. On rank 0, the
run should report `provider = BuiltInLibrary` and the expected model ID.


## Summary

For Summit coupling through kitesurf, the built-in library path should
be treated as a stable Exasim execution backend selected by
`builtinmodelID` and fed by file-based coupling inputs. The reference
pattern is:

1. build Exasim exported preprocessing libraries
2. build `libbuiltinmodel*`
3. build `apps/builtinlibrary/exasimapp`
4. let kitesurf prepare `pdeapp.txt` plus transferred field files
5. run Exasim and pull outputs back into the coupled workflow

That path is the best current match for production coupling because it
keeps Exasim generic, model selection runtime-configurable, and the
Summit integration boundary explicit.
