# Verification And Testing

Use this page after installation to confirm that Exasim, dependencies, package
discovery, MPI, and optional GPU support are working.

Run generated applications and Text2Code from a disposable copy outside the
source checkout. For the examples below, prepare:

```bash
export EXASIM_SOURCE=/path/to/Exasim
export EXASIM_VERIFY=$(mktemp -d)
if git -C "$EXASIM_SOURCE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$EXASIM_SOURCE" archive HEAD \
    apps/builtinlibrary apps/poisson/poisson2d | tar -x -C "$EXASIM_VERIFY"
else
  mkdir -p "$EXASIM_VERIFY/apps/poisson"
  cp -R "$EXASIM_SOURCE/apps/builtinlibrary" "$EXASIM_VERIFY/apps/"
  cp -R "$EXASIM_SOURCE/apps/poisson/poisson2d" \
    "$EXASIM_VERIFY/apps/poisson/"
fi
```

For a Git checkout, this copies tracked files only; use the same commit that was
installed. The fallback supports a verified source archive without Git metadata.
If an approved installation intentionally included uncommitted source changes,
create an equivalent clean verification copy that includes those changes and
record its provenance.

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
- `text2code` is installed.

## Verify CMake Package Discovery

Build the copied built-in app:

```bash
export EXASIM_APP_MPI=ON  # use OFF when Exasim was installed without MPI
cmake -S "$EXASIM_VERIFY/apps/builtinlibrary" \
  -B "$EXASIM_VERIFY/build" -DExasim_DIR="$EXASIM_PREFIX" \
  -DEXASIM_MPI="$EXASIM_APP_MPI"
cmake --build "$EXASIM_VERIFY/build" -j
```

Expected:

- CMake prints the resolved Exasim package path;
- the executable `build/exasimapp` is created.

## Verify Serial Or MPI Execution

Run a small Poisson case:

```bash
"$EXASIM_VERIFY/build/exasimapp" \
  "$EXASIM_VERIFY/apps/poisson/poisson2d/pdeapp.txt"
```

For MPI:

```bash
export EXASIM_MPI_LAUNCH="mpirun -np 2"  # use the approved site launcher
$EXASIM_MPI_LAUNCH "$EXASIM_VERIFY/build/exasimapp" \
  "$EXASIM_VERIFY/apps/poisson/poisson2d/pdeapp.txt"
```

Expected:

- preprocessing completes;
- Newton/GMRES output appears;
- output files are written under the app output directory.

## Verify Text2Code

```bash
cd "$EXASIM_VERIFY/apps/poisson/poisson2d"
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
export EXASIM_APP_MPI=ON  # use OFF when Exasim was installed without MPI
cmake -S "$EXASIM_VERIFY/apps/builtinlibrary" \
  -B "$EXASIM_VERIFY/build-gpu" \
  -DExasim_DIR="$EXASIM_PREFIX" -DEXASIM_CUDA=ON \
  -DEXASIM_MPI="$EXASIM_APP_MPI"
cmake --build "$EXASIM_VERIFY/build-gpu" -j
```

or:

```bash
cmake -S "$EXASIM_VERIFY/apps/builtinlibrary" \
  -B "$EXASIM_VERIFY/build-hip" \
  -DExasim_DIR="$EXASIM_PREFIX" -DEXASIM_HIP=ON \
  -DEXASIM_MPI="$EXASIM_APP_MPI"
cmake --build "$EXASIM_VERIFY/build-hip" -j
```

Run using the scheduler or GPU-aware MPI setup required by your platform. See
[GPU Installation](gpu.md), [Frontier](frontier.md), and [Tuolumne](tuolumne.md).

## Next

- [Quick Start](../getting-started/quickstart.md)
- [Application Modes](../usage-modes/index.md)
- [Text2Code Applications](../text2code-applications/index.md)
