# Example Prompts

This page provides copy-paste prompts for common Exasim development tasks. Adapt
paths, parameters, and validation criteria to your application.

## Case Study 1: Fluid Mesh Deformation

Prompt:

```text
Create a mesh deformation example in Exasim/examples/LinearElasticity.
This example involves the deformation of a 2D plate in a fluid domain.
The plate has a thickness of 0.05 m and a length of 1 m.
The fluid domain is a 10 m x 10 m square.
The plate is placed at the center of the fluid domain and fixed at both ends.
The left end is located at (0,0).
The displacement of the plate is
u_y = 0.1 sin(pi x).
Solve the linear elasticity equations to compute the deformed mesh induced by
the plate displacement.
Generate:
- the initial fluid mesh
- pdemodel.m
- pdeapp.m
Use existing LinearElasticity examples as references whenever possible.
```

Why it works:

- gives the application directory and physics area;
- provides physical dimensions and units;
- defines the displacement function;
- states the governing equation family;
- lists expected files;
- points to existing examples.

What to improve in a follow-up:

- specify exact boundary tags for the fluid-domain outer boundary and plate;
- request a mesh-quality plot;
- request a small verification case.

## Case Study 2: Boundary-Layer Mesh Generation

Prompt:

```text
Create a mesh deformation example in Exasim/examples/LinearElasticity.
The problem consists of a 2D plate occupying
(-0.5,0.5) x (0,0.2).
The left, right, and top boundaries are fixed.
The bottom boundary is displaced by
dx = 0
dy = a exp(-(x-b)^2/(2c^2))
where a, b, and c are user parameters.
Solve the linear elasticity equations to compute the resulting mesh deformation.
Generate:
- the initial mesh
- pdemodel.m
- pdeapp.m
The initial mesh must be a boundary-layer quadrilateral mesh refined toward the
bottom boundary.
Use existing LinearElasticity examples as references whenever possible.
```

Why it works:

- defines a precise rectangular domain;
- identifies fixed and displaced boundaries;
- makes displacement parameters user-adjustable;
- requests a boundary-layer quadrilateral mesh;
- lists required outputs.

How to generalize:

- replace the Gaussian with another displacement law;
- change refinement target from bottom wall to airfoil/cylinder/wake;
- add a parameter sweep over `a`, `b`, or `c`;
- request mesh-quality checks and plots.

## Creating A New PDE Application

```text
Create a new Exasim example under <path>.
Use <reference example path> as the style and API reference.

Problem:
- PDE:
- ModelC/ModelD/ModelW:
- geometry and units:
- coordinate ranges:
- boundary tags:
- boundary conditions:
- initial conditions:
- physicsparam ordering:

Generate:
- pdeapp.m
- pdemodel.m
- mesh helper if needed

Run a small serial CPU case if possible and report residual convergence and
generated output files.
```

## Modifying An Existing Application

```text
Modify <path>/pdeapp.m to add <feature>.
Do not change pdemodel.m.
Preserve current single-case behavior.
Add comments explaining the new parameters.
Run the example or explain why it could not be run.
```

## Generating Meshes

```text
Create a mesh helper named <function>.m for <geometry>.
Use quadrilateral elements.
Boundary tags:
- ...
Refinement:
- ...
Return an Exasim mesh structure compatible with <reference example>.
Add a plot command that verifies boundary tags.
```

## Implementing Boundary Conditions

```text
Update pdemodel.m to implement these boundary conditions:
- ID 1:
- ID 2:
- ID 3:

Use the existing callback names and argument order.
Do not change pdeapp.m except to update mesh.boundarycondition if needed.
Explain how each boundary ID maps to the mesh.
```

## Implementing Source Terms

```text
Add a source term to <path>/pdemodel.m.
Mathematical source:
S(u,x,t;mu) = ...
Parameter ordering:
physicsparam = [...]

Keep flux and boundary conditions unchanged.
Add one manufactured-solution check if practical.
```

## Implementing Equations Of State

```text
Implement an EOS helper for this Exasim model.
Inputs:
- u =
- w =
- physicsparam =
Outputs:
- pressure
- temperature
- sound speed

Before editing, list dependencies and assumptions. Then update only the flux,
source, and visualization callbacks that require the EOS.
```

## Creating Parameter Sweeps

```text
Add a physicsparam sweep to <path>/pdeapp.m.
Sweep:
- Re = [500, 1000, 1500, 2000]
Keep all other parameters fixed.
Use the existing Exasim physicsparamsweep workflow.
Write case outputs to paramcase_0001, paramcase_0002, ...
Enable warm start only if neighboring cases are physically close.
```

## Adding Postprocessing

```text
Update <path>/pdeapp.m and pdemodel.m to write ParaView scalar fields:
- pressure
- Mach number
- temperature

Use saveParaview = 1 and the existing visscalars callback pattern.
Run a small case and verify VTK files are written.
```

## Debugging Simulations

```text
This Exasim run fails with the error below:

<paste full output>

Command:
<command>

Relevant files:
- <pdeapp>
- <pdemodel>
- <mesh helper>

Diagnose likely root causes. Do not edit code yet. Suggest minimal diagnostic
runs and the first file to inspect.
```

## Extending Exasim

```text
Create an implementation plan for adding <feature> to Exasim.
Inspect current source first.
List affected backend, frontend, Text2Code, docs, and tests.
Preserve backward compatibility.
Explain CPU/GPU/MPI implications.
Do not modify code until the plan is reviewed.
```

## Advanced Templates

### ModelC

```text
Implement a ModelC application for <PDE>.
Geometry:
- domain =
- units =
- boundary tags =
State u = ...
Flux F(u,w,v,x,t;mu) = ...
Source S(u,w,v,x,t;mu) = ...
Boundary conditions = ...
Initial condition = ...
Use <reference ModelC example>.
```

### ModelD

```text
Implement a ModelD application for <PDE>.
Use q + grad(u) = 0.
Geometry:
- domain =
- units =
- boundary tags =
State u = ...
q layout = ...
Flux F(u,q,w,v,x,t;mu) = ...
Source S(u,q,w,v,x,t;mu) = ...
Boundary conditions = ...
Use <reference ModelD example>.
```

### ModelW

```text
Implement a ModelW wave application.
Geometry:
- domain =
- units =
- boundary tags =
State u = ...
Auxiliary q = ...
Equations:
- q_t + grad(u) = 0
- u_t + div(F(u,q)) = S(u,q)
Boundary and initial conditions = ...
Use <reference WaveEquation example>.
```

### Multiphysics

```text
Draft a multiphysics Exasim example coupling <physics A> and <physics B>.
Before coding, propose the state layout, coupling terms, auxiliary variables,
and validation strategy. Keep the first implementation monolithic unless there
is a clear reason for partitioned coupling.
```
