# Quickstart

Build and run a built-in Poisson 2D solve from an installed Exasim, without
writing any model code. This uses the [built-in model](../usage-modes/builtin.md)
mode.

## 1. Install Exasim

Configure, build, and install with the unified CMake superbuild (see
[Installation](../install/index.md) for your platform's prerequisites):

```bash
cmake -S Exasim -B Exasim-build -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --build Exasim-build -j8
cmake --install Exasim-build
```

## 2. Build the consumer

The `tests/consumers/builtin/` directory is a complete out-of-tree consumer
(CMakeLists, `main.cpp`, `pdeapp.txt`). Build it against the install:

```bash
cd tests/consumers/builtin
cmake -B build -DExasim_DIR=/path/to/prefix
cmake --build build
```

## 3. Run

`pdeapp.txt` selects Poisson 2D via `builtinmodelID = 1`:

```bash
mpirun -np 2 build/consumer_builtin pdeapp.txt
```

The solver reads `pdeapp.txt`, generates its input data, solves, and writes the
quantities of interest.

## Next steps

- Pick the right consumption path → [Choosing a usage mode](../usage-modes/index.md)
- Author your own PDE → [Frontends](../frontends/index.md) or the
  [pdemodel.txt syntax](../reference/pdemodel.md)
- Understand the math → [Theory](../theory/index.md)
- Embed Exasim in your own program → [Driving the solver](../driving-the-solver.md)
