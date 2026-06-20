# Built-in model

The install ships a **built-in model library** with pre-generated models
(Poisson, advection, Navier–Stokes, …), built at Exasim build time by `text2code`
from `backend/Model/BuiltIn/pdeapp<N>.txt`. A pure out-of-tree consumer selects a
model by `builtinmodelID` in a `pdeapp.txt` and needs only `find_package` — there
is **no code generation at your build time**.

Prerequisite: an [installed Exasim](../install/index.md) with the built-in model
library (the default build). All three usage modes share the same
[model contract](../reference/model-contract.md).

## Consumer CMakeLists.txt

Link `Exasim::builtinmodel` (plus the headers and the preprocessing library):

```cmake
--8<-- "tests/consumers/builtin/CMakeLists.txt"
```

`Exasim::pre` includes the C++ preprocessing path, so the consumer reads
`pdeapp.txt` directly and generates its own input data. Select the variant with
`find_package(Exasim REQUIRED COMPONENTS <variant>)` — one of `cpu`, `cpumpi`,
`gpu`, `gpumpi`.

## Consumer main.cpp

The program is minimal — construct an `ExasimSolver` and hand it the command
line. Including `<exasim/builtinlibprovider.hpp>` is what wires in the built-in
model dispatcher:

```cpp
--8<-- "tests/consumers/builtin/main.cpp"
```

## pdeapp.txt

The model is chosen by `builtinmodelID`. A minimal Poisson 2D (`builtinmodelID = 1`)
setup:

```ini
--8<-- "tests/consumers/builtin/pdeapp.txt"
```

(The full set of keys is documented in the `pdeapp.txt` field reference.)

## Build and run

```bash
cmake -B build -DExasim_DIR=/path/to/prefix
cmake --build build
mpirun -np 2 build/consumer_builtin pdeapp.txt
```

## Available built-in models

Built-in models are registered in `backend/Model/BuiltIn/` (one `pdeapp<N>.txt`
per model ID) and compiled into `Exasim::builtinmodel` at Exasim build time. To
run a different physics, point your `pdeapp.txt` at the matching `builtinmodelID`
and provide a compatible mesh — the consumer binary does not change.

A complete, tested example (CMakeLists, `main.cpp`, `pdeapp.txt`, and a QoI gate)
lives in `tests/consumers/builtin/`. The `apps/` directory holds many
text2code-driven applications run the same way.
