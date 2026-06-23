# Verification And Testing

Use this page after installation to confirm that Exasim, dependencies, package
discovery, MPI, and optional GPU support are working.

## Verify Installed Files

```bash
export EXASIM_PREFIX=/path/to/exasim-prefix

find $EXASIM_PREFIX -name ExasimConfig.cmake
find $EXASIM_PREFIX -name "libcpuEXASIM*" -o -name "libcpumpiEXASIM*"
find $EXASIM_PREFIX -name "libbuiltinmodel*"
ls $EXASIM_PREFIX/bin/text2code
ls $EXASIM_PREFIX/share/exasim/masternodes.bin
ls $EXASIM_PREFIX/share/exasim/gaussnodes.bin
```

Expected:

- `ExasimConfig.cmake` is present;
- Exasim CPU/MPI libraries are present for CPU builds;
- built-in model libraries are present;
- runtime data exists under `share/exasim`;
- `text2code` exists if Text2Code was enabled.

## Verify CMake Package Discovery

From the Exasim source tree, build the built-in app:

```bash
cd /path/to/Exasim/apps/builtinlibrary
cmake -S . -B build -DExasim_DIR=$EXASIM_PREFIX
cmake --build build -j
```

Expected:

- CMake prints the resolved Exasim package path;
- the executable `build/exasimapp` is created.

## Verify Serial Or MPI Execution

Run a small Poisson case:

```bash
build/exasimapp ../poisson/poisson2d/pdeapp.txt
```

For MPI:

```bash
mpirun -np 2 build/exasimapp ../poisson/poisson2d/pdeapp.txt
```

Expected:

- preprocessing completes;
- Newton/GMRES output appears;
- output files are written under the app output directory.

## Verify Text2Code

```bash
cd /path/to/Exasim/apps/poisson/poisson2d
$EXASIM_PREFIX/bin/text2code pdeapp.txt
```

Expected:

- `datain/` is generated or updated;
- generated model/code files are created according to the app settings;
- no parser errors are reported.

## Verify GPU Builds

For CUDA/HIP installs:

```bash
find $EXASIM_PREFIX -name "libgpuEXASIM*" -o -name "libgpumpiEXASIM*"
find $EXASIM_PREFIX -name "builtinmodelcuda*" -o -name "builtinmodelhip*"
```

Build the app with the matching backend option:

```bash
cmake -S . -B build-gpu -DExasim_DIR=$EXASIM_PREFIX -DEXASIM_CUDA=ON
cmake --build build-gpu -j
```

or:

```bash
cmake -S . -B build-hip -DExasim_DIR=$EXASIM_PREFIX -DEXASIM_HIP=ON
cmake --build build-hip -j
```

Run using the scheduler or GPU-aware MPI setup required by your platform. See
[GPU Installation](gpu.md), [Frontier](frontier.md), and [Tuolumne](tuolumne.md).

## Next

- [Quick Start](../getting-started/quickstart.md)
- [Application Modes](../usage-modes/index.md)
- [Text2Code Applications](../text2code-applications/index.md)
