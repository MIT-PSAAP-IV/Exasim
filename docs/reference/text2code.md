# text2code

`text2code` is the C++17 utility that turns a PDE application description into
solver inputs. It parses [`pdeapp.txt`](pdeapp.md) and its companion
[`pdemodel.txt`](pdemodel.md), builds mesh / master / connectivity data,
optionally generates the C++ model kernels, and writes the binary inputs the
Exasim backend consumes. It is self-contained (no external parser framework) and
uses a small expression engine (`tinyexpr`) for numeric formulas in the text
configuration.

It is the engine behind the codegen variant of the
[external built-in model](../usage-modes/external-builtin.md) and the
[shared-library](../usage-modes/shared-library.md) / [frontend](../frontends/index.md)
paths.

## Pipeline

Running `text2code pdeapp.txt` performs:

1. **Parse** `pdeapp.txt` and `pdemodel.txt` (`readpdeapp.cpp`, `TextParser.hpp`);
   numeric formulas are evaluated by `tinyexpr`.
2. **Build geometry** — mesh, master element, connectivity, and (for MPI)
   domain decomposition (`readmesh.cpp`, `makemesh.cpp`, `makemaster.cpp`,
   `connectivity.cpp`, `domaindecomposition.cpp`).
3. **Generate code** (when `gencode = 1`) — `CodeGenerator` emits the model
   kernel `.cpp` from the `pdemodel.txt` functions.
4. **Compile** the generated code (optional, the `USE_CMAKE` path via
   `CodeCompiler.cpp`).
5. **Write binaries** — `writebinaryfiles.cpp` emits `mesh.bin` and the rest of
   the `datain` bundle into the data path.

## Usage

```bash
text2code pdeapp.txt
```

Related files (e.g. `pdemodel.txt`) are discovered from the paths in
`pdeapp.txt`.

| Knob | Source | Effect |
|---|---|---|
| `gencode` | `pdeapp.txt` | when `1`, generate model C++ kernels |
| `gendatain` | `pdeapp.txt` | when `1`, write the `datain` binary bundle |
| `USE_CMAKE` | CMake option | select the `CodeCompiler` path that builds generated code |
| `datainpath` | `pdeapp.txt` / paths | directory where the binaries are written |

## Building text2code

```bash
cmake -S text2code/text2code -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Requires CMake ≥ 3.16 and a C++17 compiler. On HPC it is built as a dependency
of the [HPC build chain](../install/hpc.md) and the install ships a prebuilt
`text2code` binary; consumers reach it through
`exasim_add_external_builtin_model(... PDEMODEL ...)`.
