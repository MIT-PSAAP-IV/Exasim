# Exasim

Exasim is a framework for generating and running high-performance solvers for
parametrized partial differential equations (PDEs). It combines high-order DG,
HDG, and LDG discretizations with symbolic/code-generation workflows and a
portable C++/Kokkos backend for CPU, GPU, MPI, and MPI+GPU execution.

Exasim is designed for scientific computing workflows where users need to move
from a mathematical PDE model to a production solver that can run on laptops,
workstations, and large-scale HPC systems.

New to Exasim? Start with the [Quickstart](getting-started/quickstart.md), then
choose an [Application Mode](usage-modes/index.md).

## Why Exasim?

| Advantage | What it gives you |
| --- | --- |
| Code generation | Write models at a high level and generate optimized C++ kernels for the solver. |
| High-order accuracy | Use configurable polynomial order, quadrature order, DG/HDG/LDG formulations, and curved/high-order mesh support. |
| Performance portability | Build the same solver stack for CPU, NVIDIA GPU, AMD GPU, MPI, or MPI+GPU through Kokkos and Exasim package variants. |
| Scalable solver infrastructure | Use Newton-GMRES, preconditioners, domain decomposition, and distributed-memory execution. |
| Reusable application modes | Run installed built-in models, external built-ins, shared-library apps, or frontend-generated apps. |
| Reproducible studies | Organize parameter sweeps, warm starts, postprocessing, and output metadata consistently. |

## Key Capabilities

| Capability | Summary | Start here |
| --- | --- | --- |
| Text-to-code workflow | Generate solver kernels and input data from `pdeapp.txt` and `pdemodel.txt`. | [Text2Code Applications](text2code-applications/index.md) |
| Built-in PDE model library | Select installed Poisson, Navier-Stokes, reacting-flow, and related application models by `builtinmodelID`. | [Built-in library](usage-modes/builtin.md) |
| External built-in models | Register new built-in-style models out of tree without modifying the Exasim install. | [External built-in library](usage-modes/external-builtin.md) |
| Shared-library applications | Compile Text2Code-generated kernels into `libt2cmodel*` and link them into a standalone C++ app. | [Shared library](usage-modes/shared-library.md) |
| MATLAB, Python, and Julia frontends | Configure PDEs, meshes, sweeps, export apps, and postprocess results from high-level languages. | [Frontends](frontends/index.md) |
| Physics model formulation | Choose `ModelC`, `ModelD`, or `ModelW`; add `w`, `v`, EOS, AV, and coupling mechanisms consistently. | [Physics Models](physics-models/index.md) |
| Custom PDE development | Define fluxes, sources, boundary conditions, initial conditions, QoI, and visualization callbacks. | [pdemodel abstraction](frontends/pdemodel.md) |
| Parameter studies | Sweep `physicsparam` cases, write deterministic per-case outputs, and optionally warm-start continuation runs. | [Parameter sweeps](usage-modes/parameter-sweeps.md) |
| Postprocessing and visualization | Generate VTK/ParaView output, QoI, surface/volume data, output-CG fields, and frontend visualization products. | [Postprocessing](usage-modes/postprocessing.md) |
| HPC execution | Build and run on local CPU/GPU systems and HPC machines such as Frontier and Tuolumne. | [Installation](install/index.md) |
| Reduced-basis solver support | Configure reduced-basis dimensions and related solver/preconditioner parameters in the runtime input. | [pdeapp.txt fields](reference/pdeapp.md) |

## Getting Started Path

1. [Install Exasim](install/index.md) for your local or HPC platform.
2. Run the [Quickstart](getting-started/quickstart.md) to build and execute a
   built-in Poisson example.
3. Choose an [Application Mode](usage-modes/index.md): built-in, external
   built-in, shared library, parameter sweep, or postprocessing.
4. Use the [Frontends](frontends/index.md) if you prefer MATLAB, Python, or
   Julia workflows.
5. Use [Text2Code Applications](text2code-applications/index.md) if you want a
   text-file workflow based on `pdeapp.txt` and `pdemodel.txt`.
6. Learn how Exasim represents PDEs in
   [Physics Models](physics-models/index.md).
7. Learn custom model development through the
   [pdemodel abstraction](frontends/pdemodel.md) and
   [pdemodel.txt guide](text2code-applications/pdemodel.md).
8. Generate visualization and derived quantities with
   [Postprocessing](usage-modes/postprocessing.md).
9. Explore design or physics spaces with
   [Parameter sweeps](usage-modes/parameter-sweeps.md).

## Documentation Roadmap

| If you are... | Recommended path |
| --- | --- |
| A new user | [Quickstart](getting-started/quickstart.md) → [Installation](install/index.md) → [Application Modes](usage-modes/index.md) |
| Running checked-in applications | [Built-in library](usage-modes/builtin.md) → [pdeapp.txt fields](reference/pdeapp.md) → [Postprocessing](usage-modes/postprocessing.md) |
| Developing a custom PDE | [Physics Models](physics-models/index.md) → [Frontends](frontends/index.md) → [pdemodel abstraction](frontends/pdemodel.md) → [Text2Code `pdemodel.txt`](text2code-applications/pdemodel.md) |
| Building standalone text apps | [Text2Code overview](text2code-applications/index.md) → [pdeapp.txt guide](text2code-applications/pdeapp.md) → [Workflow](text2code-applications/workflow.md) |
| Running parameter studies | [Parameter sweeps](usage-modes/parameter-sweeps.md) → [Postprocessing](usage-modes/postprocessing.md) |
| Targeting HPC or GPUs | [HPC build chain](install/hpc.md) → [Frontier](install/frontier.md) or [Tuolumne](install/tuolumne.md) → [CMake reference](reference/cmake.md) |
| Embedding Exasim in C++ | [Driving the solver](driving-the-solver.md) → [Model contract](reference/model-contract.md) → [CMake targets](reference/cmake.md) |
| Contributing or debugging internals | [Architecture](internals/architecture.md) → [Testing](internals/testing.md) → [Known divergences](internals/known-divergences.md) |

## Core Workflows

### Run a Built-in Application

Use this path when an installed built-in model already matches your equations
and variable ordering.

```text
pdeapp.txt -> builtinmodelID -> Exasim builtin provider -> solver -> dataout/
```

Start with [Built-in library applications](usage-modes/builtin.md).

### Generate a Solver From Text Files

Use `pdeapp.txt` for application/runtime configuration and `pdemodel.txt` for
the PDE model definition.

```text
pdeapp.txt + pdemodel.txt -> Text2Code -> datain/ + kernels -> executable
```

Start with [Text2Code Applications](text2code-applications/index.md).

### Develop a Custom PDE Model

Use frontend `pdemodel` callbacks or Text2Code model files to define fluxes,
sources, boundary conditions, initial conditions, outputs, QoI, and
visualization fields.

Start with [Physics Models](physics-models/index.md),
[pdemodel abstraction](frontends/pdemodel.md), and the
[model contract](reference/model-contract.md).

### Perform Parameter Sweeps

Run the same generated solver over multiple `physicsparam` vectors, with
deterministic output directories and optional warm-start continuation.

```text
physicsparamcases -> paramcase_0001/, paramcase_0002/, ...
```

Start with [Parameter sweeps](usage-modes/parameter-sweeps.md).

### Postprocess Results

Use solve-time or standalone postprocessing to generate visualization files,
QoI, surface output, volume output, and frontend-readable results.

Start with [Postprocessing](usage-modes/postprocessing.md).

## Major Documentation Sections

| Section | Purpose |
| --- | --- |
| [Getting started](getting-started/quickstart.md) | First build/run path and application-mode selection. |
| [Installation](install/index.md) | Local and HPC builds, dependencies, CUDA/HIP, and platform notes. |
| [Text2Code Applications](text2code-applications/index.md) | Text-file application setup, model syntax, generation workflow, and references. |
| [Frontends](frontends/index.md) | MATLAB, Python, and Julia APIs, preprocessing, execution, postprocessing, and data structures. |
| [Application Modes](usage-modes/index.md) | Built-in, external built-in, shared-library, parameter-sweep, and postprocessing workflows. |
| [Physics Models](physics-models/index.md) | PDE formulations, auxiliary equations, EOS, external variables, AV, multiphysics, and coupling. |
| [Theory](theory/index.md) | DG, LDG, HDG, DIRK, Newton-GMRES, preconditioning, parallelism, GPU execution, and scalability. |
| [Reference](reference/model-contract.md) | Model contract, `pdeapp.txt`, `pdemodel.txt`, Text2Code, and CMake details. |
| [Driving the solver](driving-the-solver.md) | C++ API entry points for embedding or custom applications. |
| [Internals](internals/architecture.md) | Architecture, testing, baselines, and implementation notes. |

## Notes On Scope

Exasim emphasizes high-order PDE discretizations, generated model kernels, and
portable HPC execution. It supports configurable polynomial order and
stabilization/artificial-viscosity workflows; full automatic mesh adaptivity is
not documented as a primary built-in workflow on this page. When in doubt, use
the reference pages and checked-in examples as the implementation source of
truth.
