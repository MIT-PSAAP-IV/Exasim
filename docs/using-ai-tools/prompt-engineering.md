# Prompt Engineering For Exasim

AI tools perform best when the prompt is specific enough to constrain the
physics, numerics, files, and validation. A vague prompt such as "Create a
Navier-Stokes solver" leaves too many choices unspecified: formulation, mesh,
boundary conditions, nondimensional parameters, discretization, initial
conditions, outputs, and verification.

## Anatomy Of A Strong Prompt

Include these parts when asking an AI assistant to modify Exasim:

| Prompt part | Why it matters |
| --- | --- |
| Objective | Defines the concrete deliverable. |
| Location | Prevents files from being created in the wrong tree. |
| Reference example | Anchors style, APIs, dimensions, and solver settings. |
| Physics model | Identifies `ModelC`, `ModelD`, `ModelW`, or a built-in model. |
| Geometry and units | Prevents scale and boundary-tag mistakes. |
| Boundary conditions | Essential for well-posed PDEs. |
| Initial conditions | Required for transient runs and nonlinear convergence. |
| Parameters | Defines `physicsparam` ordering and dimensions. |
| Outputs | Specifies solution files, VTK, QoI, plots, or mesh images. |
| Verification | Tells the AI how success should be checked. |

## Weak Versus Strong Prompt

Weak:

```text
Create a Navier-Stokes solver.
```

Strong:

```text
Create a new Exasim MATLAB example under
Exasim/examples/NavierStokes/my_flat_plate.
Use examples/NavierStokes/flatplate2d as the reference.
Solve steady compressible Navier-Stokes in 2D with ModelD.
Geometry:
- rectangular flat-plate domain in meters: x in [0, 1.0], y in [0, 0.4]
- plate wall is the bottom boundary y = 0
- inlet is x = 0, outlet is x = 1.0, farfield is y = 0.4
- use a boundary-layer quadrilateral mesh refined toward y = 0
Use gamma = 1.4, Pr = 0.72, Re = 5000, Mach = 0.3.
Generate pdeapp.m and pdemodel.m only.
Keep MPI disabled, porder = 2, and saveParaview = 1.
Boundary conditions:
- inlet: uniform freestream
- wall: no-slip adiabatic wall
- outlet: pressure outflow
- farfield: freestream
Run the example if possible and report residual convergence and generated files.
```

The strong prompt supplies directory, reference example, model class, geometry,
parameters, solver settings, boundary conditions, output expectations, and
verification.

## Prompt Template

```text
Create or modify <thing> in <exact path>.

Reference examples:
- <path/to/similar/example>

Physical problem:
- PDE/model:
- Geometry and units:
- Boundary conditions:
- Initial conditions:
- Physical parameters and ordering:

Numerical requirements:
- ModelC/ModelD/ModelW or built-in model:
- porder/pgauss:
- steady or time-dependent:
- CPU/GPU/MPI assumptions:
- output and postprocessing:

Implementation requirements:
- files to create or edit:
- do not modify:
- preserve compatibility with:

Verification:
- commands to run:
- expected outputs:
- numerical checks:
```

## Useful Constraints To Add

- "Make minimal changes and explain affected files before editing."
- "Use the existing Exasim frontend APIs; do not invent new fields."
- "Use existing examples as style references."
- "If information is missing, state assumptions explicitly."
- "Run `mkdocs build --strict` for documentation changes."
- "Run the smallest representative example before scaling up."

## Avoid These Patterns

- Asking for multiple unrelated changes in one prompt.
- Omitting boundary conditions.
- Omitting units or coordinate ranges.
- Asking for "optimize the GPU code" without profiler output.
- Asking for "fix convergence" without residual history, mesh, parameters, and
  boundary information.
- Accepting generated code without compiling or validating it.
