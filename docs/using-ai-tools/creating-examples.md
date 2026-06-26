# AI-Assisted Example Development

AI tools are most effective at creating new Exasim examples when they can adapt
an existing example. Exasim examples already encode frontend conventions,
dimension fields, solver options, model callbacks, boundary tags, and plotting
style.

## Recommended Workflow

1. Find a similar existing example.
2. Give the AI the exact reference path.
3. Describe the physical differences.
4. Ask for a small set of files.
5. Review all assumptions.
6. Run the smallest case.
7. Validate outputs.
8. Iterate.

## What AI Can Generate

AI tools can draft:

- `pdeapp.m`, `pdeapp.py`, `pdeapp.jl`, or `pdeapp.txt`;
- `pdemodel.m` or `pdemodel.txt`;
- mesh-generation helper functions;
- boundary expressions and boundary-condition IDs;
- parameter-sweep variants;
- postprocessing scripts and plots;
- README-style notes for the example.

## Example Development Prompt

```text
Create a new Exasim example under
Exasim/examples/ConvectionDiffusion/rotating_gaussian.
Use Exasim/examples/Advection/GaussianRotating as the reference.

Physical problem:
- 2D convection-diffusion on a square domain [-1,1] x [-1,1]
- nondimensional coordinates
- use a structured quadrilateral mesh with boundary tags for left, right,
  bottom, and top sides
- velocity field beta = (-y, x)
- diffusivity kappa = 1e-3
- initial condition is a Gaussian centered at (0.4,0)
- periodic boundaries if the reference supports them; otherwise document the
  boundary assumption.

Generate:
- pdeapp.m
- pdemodel.m

Use porder = 3, mpiprocs = 1, saveParaview = 1.
Run the example if possible and report generated output files.
```

## Review Checklist

Before accepting the example, check:

- `pde.nd`, `pde.ncu`, `pde.nc`, `pde.ncx`, and `pde.physicsparam` are
  dimensionally consistent;
- boundary expressions match the geometry;
- boundary-condition IDs used in the model match `mesh.boundarycondition`;
- units and nondimensional parameters are documented;
- output path is not accidentally shared with another example;
- generated plots or VTK files visualize the intended fields;
- residuals converge or time-dependent behavior is physically plausible.

## Iterative Refinement

Good follow-up prompts are narrow:

```text
The example runs, but boundary 3 is using the inlet condition. Inspect
mesh.boundaryexpr and pdemodel.m and fix only the boundary ID mapping.
```

```text
The mesh is too coarse near the wall. Modify only mkmesh_*.m to add geometric
refinement toward y = 0 while preserving the same boundary tags.
```

```text
Add a parameter sweep over Reynolds number using the existing
physicsparamsweep workflow. Do not change the PDE model.
```
