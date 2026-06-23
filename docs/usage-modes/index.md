# Application Modes

Once Exasim is [installed](../install/index.md), you consume it from C++ in one
of three modes. All three ultimately satisfy the same
[model contract](../reference/model-contract.md); they differ in **where the
model kernels come from** and **how they are linked**.

| Mode | Where the model comes from | Codegen at *your* build? | Link target |
|---|---|---|---|
| [Built-in model](builtin.md) | A model pre-generated into the installed library, picked by `builtinmodelID` | No | `Exasim::builtinmodel` |
| [External built-in model](external-builtin.md) | A *new* model you register out-of-tree (text2code, hand-written, or pre-generated kernels) | Optional (text2code variant) | your `ext_model_<ID>` target |
| [Shared library](shared-library.md) | A frontend (Python / Julia / MATLAB) generates kernels into a model shared library loaded at runtime | Yes (by the frontend) | the generated `t2cmodel*` library |

## Choosing a mode

- **Just run an existing physics model** (Poisson, advection, Navier–Stokes, …)
  → [Built-in model](builtin.md). No code
  generation, only `find_package`.
- **Add your own model to a C++ application** without modifying the installed
  package → [External built-in model](external-builtin.md).
  Your provider intercepts your model ID and falls through to the built-ins for
  all other IDs.
- **Author a model interactively from Python / Julia / MATLAB** and let the
  toolchain build and run it →
  [Shared library](shared-library.md).

All three read the same `pdeapp.txt` solver-setup file and produce identical
results for the same physics.

## Common runtime workflows

Postprocessing is shared by all usage modes. After any solve, Exasim can write
VTK visualization files, integrated QoIs, residual histories, CG output fields,
and boundary extracts; generated standalone apps can also replay saved solution
records with a `postprocess` subcommand. See the
[Postprocessing](postprocessing.md) guide for the output formats, frontend
helpers, MPI behavior, and explicit replay syntax.

Parameter sweeps are also shared across workflows when only `physicsparam`
changes between cases. Frontend-driven sweeps loop in MATLAB/Python/Julia,
whereas exported apps and `pdeapp.txt`/text2code apps can run sweeps directly
inside the standalone C++ executable. See [Parameter sweeps](parameter-sweeps.md)
for supported input forms, output layout, warm-start continuation, and current
limitations.
