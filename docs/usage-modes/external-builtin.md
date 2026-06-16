# External built-in model

To add a **new** model without touching the installed package, register it as an
*external built-in model*. The installed CMake helper
`exasim_add_external_builtin_model()` generates the kernels at build time and
produces a provider library that **intercepts your model ID and falls through to
the installed built-ins for all other IDs**.

Prerequisite: an [installed Exasim](../install/index.md). The model you register
satisfies the same [model contract](../reference/model-contract.md) as a built-in
one — it is just packaged out-of-tree.

## Three ways to supply the model

`exasim_add_external_builtin_model()` accepts the model in three forms:

| Variant | Argument | Source of kernels |
|---|---|---|
| **A — text2code** | `PDEMODEL <pdeapp.txt>` | generated at configure time from a `pdeapp`/`pdemodel` text pair |
| **B — hand-written** | `SOURCES model.hpp model.cpp` | a `model.hpp`/`model.cpp` pair in namespace `exasim_model_<ID>` |
| **C — pre-generated kernels** | `KERNELS <dir>` | a directory of already-generated kernel `.cpp` files (what the language frontends use under the hood) |

## Consumer CMakeLists.txt

This example registers model ID 100 from a text pair (variant A) and links the
generated `ext_model_100` target instead of `Exasim::builtinmodel`:

```cmake
--8<-- "tests/consumers/external-model/CMakeLists.txt"
```

The other two variants differ only in the `exasim_add_external_builtin_model()`
call:

```cmake
# variant B: hand-written model.hpp/model.cpp (namespace exasim_model_<ID>)
exasim_add_external_builtin_model(TARGET my_model_100 ID 100
  SOURCES model100.hpp model100.cpp)

# variant C: a directory of pre-generated kernel .cpp files
exasim_add_external_builtin_model(TARGET my_model_100 ID 100
  KERNELS ${CMAKE_CURRENT_BINARY_DIR}/kernels)
```

!!! warning "Do not also link `Exasim::builtinmodel`"
    The external target provides `getBuiltInLibraryExasimDriverABI()` and links
    `Exasim::builtinmodel` transitively. Linking both yields a duplicate symbol.

## Consumer main.cpp

The `main.cpp` is the same as the [built-in](builtin.md) one, with one
difference: it must **not** include `<exasim/builtinlibprovider.hpp>` — the
external provider library already defines `getBuiltInLibraryExasimDriverABI()`.
Select the model by setting `builtinmodelID = 100` in `pdeapp.txt`, or pre-seed
it in code:

```cpp
RunExasimSolver(solver, argc, argv, comm, {100});
```

## Build and run

```bash
cmake -B build -DExasim_DIR=/path/to/prefix
cmake --build build               # text2code runs here for variant A
mpirun -np 2 build/consumer_external_model pdeapp.txt
```

A complete, tested example is `tests/consumers/external-model/`. See
`cmake/ExasimExternalModel.cmake` for the full helper contract.
