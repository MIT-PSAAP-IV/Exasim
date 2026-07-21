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

## Documentation

Full documentation is published at **<https://mit-psaap-iv.github.io/Exasim>**
(source under [`docs/`](docs/), built with MkDocs —
`pip install -r requirements-docs.txt && mkdocs serve`):

- [Quickstart](docs/getting-started/quickstart.md) — build and run a built-in solve
- [Installation](docs/install/index.md) — local (macOS, Linux CPU/NVIDIA/AMD) and HPC (Frontier, Tuolumne, generic)
- [Usage modes](docs/usage-modes/index.md) — built-in, external built-in, shared library
- [Frontends](docs/frontends/index.md) — author models in Python / Julia / MATLAB
- [Model contract](docs/reference/model-contract.md), [`pdeapp.txt`](docs/reference/pdeapp.md), [`pdemodel.txt`](docs/reference/pdemodel.md), and [CMake](docs/reference/cmake.md) references
- [Driving the solver (C++ API)](docs/driving-the-solver.md) — embed Exasim in your own program
- [Theory](docs/theory/index.md) — the LDG formulation and Jacobian the contract maps to
- [Internals](docs/internals/architecture.md) — architecture, test harness, baselines

## Install

One command builds the whole stack — vendored dependencies (Kokkos,
METIS/ParMETIS, SymEngine) are found on the system or built from source, then
text2code, the solver libraries, the built-in model library, and the language
frontends are built and installed:

```bash
# build directories must be OUTSIDE the source tree (the repo stays pristine)
cmake -S Exasim -B Exasim-build -DCMAKE_INSTALL_PREFIX=/path/to/prefix  # add -DEXASIM_CUDA=ON or -DEXASIM_HIP=ON for GPUs
cmake --build Exasim-build -j
cmake --install Exasim-build 
```

After downloading the source code, make sure the folder is named `Exasim` and
that the path to it contains no whitespace (Kokkos will not compile otherwise).

See the [installation docs](docs/install/index.md) for prerequisites, all
configure options, and HPC instructions (Frontier, Tuolumne, generic clusters),
and [Testing](docs/internals/testing.md) / `tests/README.md` for the test suite.

## Publications
[1] Vila-Pérez, J., Van Heyningen, R. L., Nguyen, N.-C., & Peraire, J. (2022). Exasim: Generating discontinuous Galerkin codes for numerical solutions of partial differential equations on graphics processors. SoftwareX, 20, 101212. https://doi.org/10.1016/j.softx.2022.101212

[2] Hoskin, D. S., Van Heyningen, R. L., Nguyen, N. C., Vila-Pérez, J., Harris, W. L., & Peraire, J. (2024). Discontinuous Galerkin methods for hypersonic flows. Progress in Aerospace Sciences, 146, 100999. https://doi.org/10.1016/j.paerosci.2024.100999

[3] Nguyen, N. C., Terrana, S., & Peraire, J. (2022). Large-Eddy Simulation of Transonic Buffet Using Matrix-Free Discontinuous Galerkin Method. AIAA Journal, 60(5), 3060–3077. https://doi.org/10.2514/1.j060459

[4] Nguyen, N. C., & Peraire, J. (2012). Hybridizable discontinuous Galerkin methods for partial differential equations in continuum mechanics. Journal of Computational Physics, 231(18), 5955–5988. https://doi.org/10.1016/j.jcp.2012.02.033
