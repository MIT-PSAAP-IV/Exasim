# Exasim

Exasim is a C++ library for solving partial differential equations (PDEs) with
high-order discontinuous Galerkin (DG) methods. It combines a templated solver
core (HDG / LDG, GMRES, Newton, DIRK time stepping) with optional symbolic code
generation and Kokkos-based backend portability, so the same model definition
runs on CPU, GPU, MPI, and MPI+GPU without changing the math.

!!! note "Site under construction"
    This documentation site is being assembled. Sections land incrementally;
    the navigation reflects what is currently published.

## What you get

- **Spatial discretizations** — local DG and hybridized DG; hex / tet / quad /
  tri elements; arbitrary polynomial order.
- **Time stepping** — diagonally implicit Runge–Kutta (DIRK) with configurable
  stage count.
- **Solvers** — Newton outer + GMRES inner, multiple preconditioners.
- **Multi-physics** — monolithic HDG with multi-domain coupling.
- **Backend portability** — Kokkos picks CPU / NVIDIA GPU (CUDA) / AMD GPU (HIP)
  at compile time.
- **Distributed-memory** — MPI partitioning via ParMETIS.

## How you use it

To run a PDE through Exasim you specify three things — the PDE math, the solver
setup, and the mesh — then consume the solver in one of three modes:

| Usage mode | What it is |
|---|---|
| **Built-in model** | Pick a pre-generated model by ID; no code generation at your build time. |
| **External built-in model** | Register a *new* model out-of-tree; it falls through to the built-ins for other IDs. |
| **Shared library** | A frontend (Python / Julia / MATLAB) generates the model kernels into a shared library loaded at runtime. |
