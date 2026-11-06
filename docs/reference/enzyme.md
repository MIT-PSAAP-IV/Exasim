# Enzyme Jacobian-vector products

Enzyme computes Jacobian-vector products (JVPs) for pointwise physics functions.
The LDG solver uses these JVPs to differentiate the residual. Each Kokkos kernel
calls the differentiated function at one point. Enzyme does not differentiate
the Kokkos scheduler.

## Build and run

Build ClangEnzyme separately. Use the matching Clang compiler to build Exasim
and each model. Follow the [Enzyme CMake instructions](cmake.md#enzyme-compiler-plugin).
Set `EXASIM_ENZYME=ON`, `ENZYME_ROOT`, and `EXASIM_BUILD_TESTS=ON`.
CMake loads the plugin and defines `HAVE_ENZYME`. The compiled program does
not require an Enzyme runtime library.

For an existing build, run these commands from the source directory. Replace
`../Exasim-build` with your build directory if necessary.

```sh
cmake -S . -B ../Exasim-build -DEXASIM_BUILD_TESTS=ON
cmake --build ../Exasim-build --parallel 4
ctest --test-dir ../Exasim-build/exasim_build-prefix/src/exasim_build-build \
  -R '^enzyme_' --output-on-failure
cmake --install ../Exasim-build
```

Set `EXASIM_ENZYME=OFF` to use finite differences for LDG matrix-vector products.
Rebuild the solver and model libraries together. The new JVP callbacks require
driver ABI version 6.

For CUDA, use a separate build directory and installation prefix. Add
`-DEXASIM_CUDA=ON`, `-DEXASIM_GPU_ARCH=BLACKWELL120`, and
`-DCMAKE_CUDA_COMPILER=/path/to/llvm/bin/clang++` to the Enzyme configure command.
Use the same Clang compiler for C++ and CUDA. `BLACKWELL120` selects the RTX 5090;
select the correct architecture for other GPUs. To run the derivative tests on
GPU 1, use:

```sh
CUDA_VISIBLE_DEVICES=1 ctest \
  --test-dir ../Exasim-build-enzyme-cuda/exasim_build-prefix/src/exasim_build-build \
  -R '^enzyme_' --output-on-failure
```

## Define the physics

For C++, write pointwise functions such as `PdeModel::flux`, `source`, and
`fbou`. See `apps/poisson/poisson2d/my_model.hpp`. Include `modelprovider.hpp` in
the application. An Enzyme build registers the JVP callbacks automatically.

MATLAB, Python, Julia, and Text2Code generate Kokkos physics sources. The shared
CMake build extracts the pointwise function bodies and generates JVP wrappers
for `Flux`, `Source`, `Tdfunc`, `Avfield`, `Fbou`, `Ubou`, `Fhat`, and `Stab`.
ClangEnzyme compiles the `__enzyme_fwddiff` calls in these wrappers. The
installed Exasim package supplies the compiler flag to each model build.

The calculation holds geometry, normals, time, parameters, and time-history
data fixed. It includes derivatives of the state, reconstructed gradients,
auxiliary equations, and unfrozen artificial viscosity (AV). Each
matrix-vector product restores the input trace, AV, and auxiliary state.
HDG still uses its assembled Jacobians and condensation algorithm.

Use the standard residual stages. Custom stage callbacks require a derivative
implementation and cause an error in the Enzyme residual path.

For a Julia LDG case, set `pde.preconditioner = 0`, as in the C++ Poisson case.
The default block preconditioner requires full symbolic Jacobians. Julia LDG
code generation currently emits empty kernels for those Jacobians.
Set `ENV["CXX"]` to the matching Clang compiler before the first application
build; Julia's CMake configuration uses this environment variable.

## Verification

The tests compare JVPs with central differences and analytical values. They
cover zero directions, boundary tags, the AV nodal layout, absent auxiliary
fields, implicit auxiliary equations, and material-state Jacobians. Residual
tests check repeated calls and state restoration. Poisson tests check solver
convergence and solution error with handwritten and generated physics.
If MPI is available, a separate test checks residual JVPs with two processes.

CPU Serial and the two-process residual test have been run. Python and Julia
generated models have also been run. The three GPU derivative tests and a full
Poisson solve pass on an RTX 5090 with CUDA 12.9 and Clang 24. The GPU Poisson
solve reaches residual `2.03e-9`, with squared L2 error `1.22e-12`. Compute
Sanitizer reports no memory errors in these four GPU runs.
MATLAB, HIP, and GPU MPI execution have not been verified.

To check Python-generated physics, activate a Python environment with the
frontend dependencies. Run these commands from the source directory. Set
`CXX` to the matching Clang compiler and `Exasim_DIR` to the installed
`lib/cmake/Exasim` directory.

```sh
PYTHONPATH="$PWD/frontends/Python" python tests/enzyme/generate_python.py \
  --output ../Exasim-build/enzyme-python-kernels
cmake -S tests/enzyme -B ../Exasim-build/enzyme-python-test \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" \
  -DExasim_DIR="$Exasim_DIR" \
  -DKERNELS="$PWD/../Exasim-build/enzyme-python-kernels/kernels"
cmake --build ../Exasim-build/enzyme-python-test --parallel 4
ctest --test-dir ../Exasim-build/enzyme-python-test --output-on-failure
```
