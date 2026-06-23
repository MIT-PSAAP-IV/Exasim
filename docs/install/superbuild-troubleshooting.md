# Troubleshooting Superbuild Failures

Use this page before switching to manual dependency installation. Many failures
are caused by compiler, MPI, permission, or network issues that can be fixed
without manually building every dependency.

## Common Failure Modes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Install tries to write into `/usr/local` | No install prefix was provided at install time. | Use `cmake --install Exasim-build --prefix /path/to/exasim-prefix`. |
| CMake cannot find a compiler | Compiler modules or command-line tools are missing. | Install/load compilers and rerun CMake from a clean build directory. |
| CMake finds the wrong MPI | Multiple MPI installations are visible. | Set `MPI_C_COMPILER` and `MPI_CXX_COMPILER` explicitly. |
| ParMETIS or MPI link errors | MPI compiler/runtime mismatch. | Build and run with the same MPI stack. |
| BLAS/LAPACK not found | Missing system packages or unloaded vendor math module. | Install `libblas-dev liblapack-dev` or load site math libraries. |
| Download step fails | Network blocked or proxy required. | Use site mirrors, preload dependencies, or use manual install. |
| Kokkos fails to configure | Unsupported compiler, path with spaces, or wrong GPU toolchain. | Use a supported compiler and a whitespace-free path. |
| GPU compile fails | Wrong CUDA/HIP architecture or compiler. | Use [GPU Installation](gpu.md) and platform-specific GPU pages. |

## Diagnostic Commands

```bash
cmake --version
which cc c++ || true
which mpicc mpicxx || true
mpicc --version || true
mpicxx --version || true
```

For GPU builds:

```bash
nvcc --version || true
hipcc --version || true
```

Inspect the CMake logs:

```bash
find Exasim-build -name CMakeError.log -o -name CMakeOutput.log
find Exasim-build -name CMakeConfigureLog.yaml
```

## Clean Reconfigure

If CMake cached the wrong compiler or dependency path, use a fresh build
directory:

```bash
rm -rf Exasim-build
cmake -S Exasim -B Exasim-build
cmake --build Exasim-build -j8
cmake --install Exasim-build --prefix /path/to/exasim-prefix
```

## When To Use Manual Installation

Proceed to [Manual Dependency Installation](manual-dependencies.md) when:

- your site blocks Superbuild downloads;
- you must use preinstalled system or vendor dependencies;
- the cluster requires specific compiler/MPI/GPU module combinations;
- you need fine-grained control over Kokkos, SymEngine, METIS, or ParMETIS.

