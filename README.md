<p align="center">
<img src="docs/exasimlogosmall.png">
</p>

# Generating Discontinuous Galerkin Codes For Extreme Scalable Simulations
Exasim is an open-source software for generating high-order discontinuous Galerkin (DG) codes to numerically solve parametrized partial differential equations (PDEs) on different computing platforms with distributed memory.  It combines high-level languages and low-level languages to easily construct parametrized PDE models and automatically produce high-performance C++ codes. The construction of parametrized PDE models and the generation of the stand-alone C++ production code are handled by high-level languages, while the production code itself can run on various machines, from laptops to the largest supercomputers, with  AMD and Nvidia GPU processors. Exasim has the following capabilities:

   - Solve a wide variety of PDEs in fluid mechanics, solid mechanics, electromagnetism, and multi-physics models, in 1D, 2D, and 3D
   - Generate stand-alone C++ production code via the mathematical expressions of the PDEs
   - Implement local DG and hybridized DG methods for spatial discretization
   - Implement diagonally implicit Runge-Kutta methods for temporal discretization
   - Implement parallel Newton-GMRES solvers and scalable preconditioners using reduced basis method, additive Schwarz method, block ILU, and polynomial preconditioners
   - Implement monolithic multi-physics solvers for the HDG discretization    
   - Employ Kokkos to provide full GPU functionality for all code components from discretization schemes to iterative solvers
   - Provide auto-gen tools to calculate thermodynamic, transport, chemistry, and energy transfer properties for chemically-reacting flows 
   - Provide application interfaces to Julia, Python, and Matlab. 
   
After downloading the source code, please make sure that the name of the folder is `Exasim`. If it has a different name, please rename it to `Exasim`. Please make sure that the directory containing the folder Exasim does not have any white space, because Kokkos libraries can not be compiled properly in such case. See [the documentation](https://github.com/exapde/Exasim/blob/master/doc/Exasim.pdf) for more details. 

To deploy, compile, and run Exasim on **HPC systems**, please follow the intructions in [the hpc manual](https://github.com/exapde/Exasim/blob/master/install/hpc.txt).

## Installation: build and install everything

One command builds the whole stack — vendored dependencies (Kokkos, METIS/ParMETIS,
SymEngine) are found on the system or built from source automatically, then
text2code, the solver libraries, the built-in model library, and the language
frontends are built and installed:

```bash
cd Exasim
cmake -S . -B build                  # add -DEXASIM_CUDA=ON or -DEXASIM_HIP=ON for GPUs
cmake --build build -j
cmake --install build --prefix /path/to/prefix
```

Useful configure options: `EXASIM_MPI`/`EXASIM_NOMPI` (both ON by default),
`EXASIM_CUDA`, `EXASIM_HIP`, `WITH_PARMETIS`, `WITH_TEXT2CODE`,
`EXASIM_FRONTENDS` (ON: install the Python/Julia/MATLAB frontends),
`EXASIM_BUILD_TESTS` (register the ctest suite; see `tests/README.md`).

The install prefix then contains:

- `lib/` + `include/` + `lib/cmake/Exasim/` — the C++ package: solver libraries
  for every enabled variant (`Exasim::cpulib`, `Exasim::cpumpilib`,
  `Exasim::gpulib`, ...), the built-in model library, and the CMake config
  consumed by `find_package(Exasim)`
- `bin/text2code` — the text-to-code generator/preprocessor
- `lib/python3.X/site-packages/exasim` — the Python frontend
- `share/exasim/julia/Exasim` — the Julia frontend (Exasim.jl)
- `share/exasim/matlab` — the MATLAB frontend

To deploy on **HPC systems**, see [the hpc manual](install/hpc.txt).

## Using the frontends

Each frontend drives the same flow: define the PDE model symbolically in a
`pdeapp` script, and the frontend generates the model kernels, builds a small
solver app out of tree against the installed Exasim (via `find_package(Exasim)`
and the external-model mechanism, model ID `pde.modelid`, default 100), and
runs it. Runtime data (`datain/`, `dataout/`) is written under the working
directory (`pde.datapath`); generated code and the app build live in the hidden
`.exasim/` directory next to them (`pde.builddir`). The Exasim source tree is
never touched.

Migrated examples: `examples/Poisson/poisson2d/pdeapp.{py,jl,m}`. End-to-end
tests: `tests/frontends/` (`frontend_python` runs in CI; julia/matlab run
wherever those toolchains exist).

### Frontend dependencies

The frontends generate code symbolically and then compile it, so two kinds of
dependencies must be present at app-run time:

- the **C++ toolchain used for the install** — `cmake`, a C++17 compiler,
  BLAS/LAPACK, and (for the `*mpi` variants) the same MPI — on `PATH`, exactly
  as for the Exasim build itself;
- the **language's symbolic stack**:
  - Python ≥ 3.8 with `numpy`, `scipy`, `sympy` — install with
    `python3 -m pip install numpy scipy sympy` into whichever interpreter you
    use (system Python, a virtualenv, a module-loaded HPC Python, ...);
  - Julia ≥ 1.6 with SymPy.jl — `julia -e 'using Pkg; Pkg.add("SymPy")'`;
  - MATLAB with the Symbolic Math Toolbox.

### Python

```bash
# make the installed package importable (unnecessary only when the install
# prefix is already on your interpreter's sys.path, e.g. a virtualenv prefix):
export PYTHONPATH=/path/to/prefix/lib/python3.X/site-packages:$PYTHONPATH
```

```python
import exasim
pde, mesh = exasim.initializeexasim()
pde['model'] = "ModelD"; pde['modelfile'] = "pdemodel"   # pdemodel.py next to the script
# ... discretization/physics parameters, mesh ...
sol, pde, mesh = exasim.exasim(pde, mesh)[0:3]
```

The package is also pip-installable from `frontends/Python`
(`pip install ./frontends/Python`; set `EXASIM_PREFIX=/path/to/prefix` so it
can find the C++ install).

### Julia

```julia
push!(LOAD_PATH, "/path/to/prefix/share/exasim/julia")   # or Pkg.develop(path=...)
using Exasim
pde, mesh = Exasim.initializeexasim()
pde.model = "ModelD"
include("pdemodel.jl")          # defines flux/source/... in Main
# ... discretization/physics parameters, mesh ...
sol, pde, mesh = Exasim.exasim(pde, mesh)
```

### MATLAB

```matlab
run('/path/to/prefix/share/exasim/matlab/exasim_setup.m')
[pde, mesh] = initializeexasim();
pde.model = "ModelD"; pde.modelfile = "pdemodel";   % pdemodel.m on the path
% ... discretization/physics parameters, mesh ...
[sol, pde, mesh] = exasim(pde, mesh);
```

### Build artifacts and reuse

The generated model is compiled into a **dynamic library**; per app, everything
lives in the hidden `pde.builddir` (default `<cwd>/.exasim/`):

```
.exasim/
  kernels/                   # the generated model kernel .cpp set
  CMakeLists.txt, main.cpp   # rendered app project (from the installed templates)
  build/
    libfrontend_model.so     # the generated model as a dynamic provider library
    exasimapp                # the solver executable (loads the model at runtime)
    .exasim_model_hash       # SHA-256 of the model inputs from the last build
```

The heavy code — Kokkos, the solver libraries, the built-in model library — is
prebuilt in the install prefix and never recompiled. Reuse is hash-based:
`cmakecompile` hashes the kernel set and the rendered app sources, and when the
hash matches the last successful build it skips the build system entirely and
goes straight to the solver run. When the model **does** change, only the
provider translation unit recompiles and only `libfrontend_model` relinks —
`exasimapp` itself is never rebuilt. Mesh, parameter, and solver-option changes
never trigger compilation (they only affect `datain/`).

To reuse one build across runs, simply run from the same directory (or point
`pde.builddir` / `pde['builddir']` at a shared location — one model per
builddir). Delete `.exasim/` to force a clean rebuild.

(The model library deliberately does not embed Kokkos; it resolves Kokkos
symbols from `exasimapp` at load time, so there is exactly one Kokkos runtime
— see `cmake/ExasimExternalModel.cmake` and `backend/Model/BuiltIn/CMakeLists.txt`
for why this matters.)

## C++: running built-in models

The install ships a built-in model library with pregenerated models (Poisson,
advection, ...; built at Exasim build time by text2code from
`backend/Model/BuiltIn/pdeapp<N>.txt`). A pure out-of-tree consumer selects a
model by `builtinmodelID` in a `pdeapp.txt` and needs only `find_package`:

```cmake
find_package(Exasim REQUIRED COMPONENTS cpumpi)   # or: cpu, gpu, gpumpi
add_executable(consumer main.cpp)
target_compile_definitions(consumer PRIVATE _BUILTINLIBRARY _MPI)
target_link_libraries(consumer PRIVATE
    Exasim::headers Exasim::pre Exasim::builtinmodel Kokkos::kokkos MPI::MPI_CXX)
```

```cpp
#include <exasim/ExasimSolverSetup.hpp>
#include <exasim/builtinlibprovider.hpp>

int main(int argc, char** argv)
{
#ifdef HAVE_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    MPI_Comm comm = MPI_COMM_NULL;
#endif
    ExasimSolver solver;
    return RunExasimSolver(solver, argc, argv, comm);   // ./consumer pdeapp.txt
}
```

`Exasim::pre` includes the C++ preprocessing path, so the consumer reads
`pdeapp.txt` directly and generates its own input data. A complete working
example (CMakeLists, main.cpp, pdeapp.txt, QoI gate) is
`tests/consumers/builtin/`; the `apps/` directory holds many text2code-driven
applications run the same way.

## C++: external built-in models (out-of-tree model IDs)

To add a **new** model without touching the installed package, register it as
an external built-in model. The installed helper generates kernels at build
time and produces a provider library that intercepts your model ID and falls
through to the installed built-ins for all other IDs:

```cmake
find_package(Exasim REQUIRED COMPONENTS cpumpi)

# variant A: generate kernels from a pdeapp/pdemodel text pair via text2code
exasim_add_external_builtin_model(TARGET my_model_100
  ID 100
  PDEMODEL ${CMAKE_CURRENT_SOURCE_DIR}/pdeapp100.txt)

# variant B: hand-written model.hpp/model.cpp (namespace exasim_model_<ID>)
#   exasim_add_external_builtin_model(TARGET my_model_100 ID 100
#     SOURCES model100.hpp model100.cpp)
# variant C: a directory of pregenerated kernel .cpp files (what the language
#   frontends use under the hood)
#   exasim_add_external_builtin_model(TARGET my_model_100 ID 100
#     KERNELS ${CMAKE_CURRENT_BINARY_DIR}/kernels)

add_executable(my_solver main.cpp)
target_compile_definitions(my_solver PRIVATE _BUILTINLIBRARY _MPI)
target_link_libraries(my_solver PRIVATE
    Exasim::headers Exasim::pre my_model_100 Kokkos::kokkos MPI::MPI_CXX)
# Do NOT also link Exasim::builtinmodel — it comes in transitively.
```

The consumer's `main.cpp` is the same as the built-in one above, but must NOT
include `<exasim/builtinlibprovider.hpp>` (the provider library defines
`getBuiltInLibraryExasimDriverABI()`); declare nothing and set
`builtinmodelID = 100` in `pdeapp.txt`, or pre-seed it in code via
`RunExasimSolver(solver, argc, argv, comm, {100})`. See
`cmake/ExasimExternalModel.cmake` for the full contract.

## Examples

Exasim produces C++ code to solve a wide variety of parametrized partial differential equations from first-order, second-order elliptic, parabolic, hyperbolic PDEs, to higher-order PDEs. Many examples are provided in **Exasim/apps** and **Exasim/examples** to illustrate how to use Exasim for solving Poisson equation, wave equation, heat equation, advection, convection-diffusion, linear elasticity, nonlinear elasticity, Euler equations, Navier-Stokes equations, and MHD equations. The directory **Exasim/apps** include examples that use Text2Code to generate C++ source code from a text file (run them with the C++ built-in path above). The directory **Exasim/examples** include examples that use Matlab, Julia, or Python to generate C++ source code; run a `pdeapp.{py,jl,m}` from its directory with the frontend of your choice (`python3 pdeapp.py`, `julia pdeapp.jl`, or `pdeapp` in MATLAB). Note that most examples still use the legacy `setpath` preamble and are being migrated to the installed packages incrementally; the `examples/Poisson/poisson2d` example shows the new style.

## Header-only template-library authoring path (new)

In addition to the existing `pdemodel.txt` + `text2code` codegen
workflow, Exasim now supports a **header-only template-library** path
where you write the PDE math directly as a C++ struct with
`KOKKOS_INLINE_FUNCTION static` pointwise methods, and instantiate
the FEM internals on it as `exasim::CSolution<MyModel>`. No DSL, no
codegen, no autodiff — plain pointwise math + hand-written Jacobians.

Minimum consumer (after `cmake --install build --prefix /opt/exasim`):

```cmake
option(EXASIM_MPI "Link the MPI-enabled Exasim preprocessing library" ON)
option(EXASIM_GPU "Link the GPU-enabled Exasim preprocessing library" OFF)

find_package(Exasim REQUIRED)

set(EXASIM_APP_LIBRARY_TARGET Exasim::cpuprelib)
if(EXASIM_MPI)
  set(EXASIM_APP_LIBRARY_TARGET Exasim::cpumpiprelib)
endif()
if(EXASIM_GPU)
  if(EXASIM_MPI)
    set(EXASIM_APP_LIBRARY_TARGET Exasim::gpumpiprelib)
  else()
    set(EXASIM_APP_LIBRARY_TARGET Exasim::gpuprelib)
  endif()
endif()

add_executable(exasimapp main.cpp)
target_include_directories(exasimapp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(exasimapp PRIVATE _KOKKOSKERNEL)
if(EXASIM_MPI)
  target_compile_definitions(exasimapp PRIVATE _MPI)
endif()
target_link_libraries(exasimapp PRIVATE Exasim::headers ${EXASIM_APP_LIBRARY_TARGET})
```

```cpp
#include "ExasimSolverSetup.hpp"
#include "my_model.hpp"
#include "modelprovider.hpp"

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

Documentation lives under [`docs/`](docs/) — start at
[`docs/README.md`](docs/README.md) for the table of contents.

Quick pointers:

- [`tutorial/`](tutorial/) — solve Poisson 2D through every supported path; the worked walkthrough for picking among the PDE × Solver × Mesh combinations
- [`docs/00-introduction.md`](docs/00-introduction.md) — what Exasim is and how the pieces fit together
- [`docs/01-installation.md`](docs/01-installation.md) — per-platform install (macOS, Linux CPU, Linux + NVIDIA, Linux + AMD)
- [`docs/02-model-contract.md`](docs/02-model-contract.md) — full reference for the `Model` C++ struct that the templated FEM internals consume
- [`docs/03-internals/`](docs/03-internals/) — test harness, baseline format, architecture

The legacy `pdemodel.txt` + `text2code` + `cput2cEXASIM` workflow stays
fully supported. text2code now also emits a `my_model.hpp` for the
templated path; both authoring paths produce the same struct shape and
share one runtime body via `exasim::run<M>(argc, argv)`.

## Publications
[1] Vila-Pérez, J., Van Heyningen, R. L., Nguyen, N.-C., & Peraire, J. (2022). Exasim: Generating discontinuous Galerkin codes for numerical solutions of partial differential equations on graphics processors. SoftwareX, 20, 101212. https://doi.org/10.1016/j.softx.2022.101212

[2] Hoskin, D. S., Van Heyningen, R. L., Nguyen, N. C., Vila-Pérez, J., Harris, W. L., & Peraire, J. (2024). Discontinuous Galerkin methods for hypersonic flows. Progress in Aerospace Sciences, 146, 100999. https://doi.org/10.1016/j.paerosci.2024.100999

[3] Nguyen, N. C., Terrana, S., & Peraire, J. (2022). Large-Eddy Simulation of Transonic Buffet Using Matrix-Free Discontinuous Galerkin Method. AIAA Journal, 60(5), 3060–3077. https://doi.org/10.2514/1.j060459

[4] Nguyen, N. C., & Peraire, J. (2012). Hybridizable discontinuous Galerkin methods for partial differential equations in continuum mechanics. Journal of Computational Physics, 231(18), 5955–5988. https://doi.org/10.1016/j.jcp.2012.02.033
