# Mesh Generation With AI

AI tools can draft mesh-generation routines, but mesh quality and boundary tags
must be checked carefully. A correct PDE model can fail if the mesh has
inverted elements, inconsistent boundary labels, or insufficient refinement.

## Mesh Types To Request

AI tools can help generate:

- structured rectangular meshes;
- boundary-layer quadrilateral meshes;
- multi-block meshes;
- curved-boundary meshes;
- imported mesh conversion helpers;
- refinement around walls, corners, shocks, or source regions.

Only request overset meshes if your target Exasim workflow supports the
required connectivity and interpolation. Otherwise ask for a conforming mesh.

## What To Specify

| Item | Example |
| --- | --- |
| Geometry | `(-0.5,0.5) x (0,0.2)` plate. |
| Units | meters, nondimensional units, or code units. |
| Element type | triangles, quadrilaterals, tetrahedra, hexahedra. |
| Resolution | `nx = 120`, `ny = 48`. |
| Refinement | geometric growth toward bottom wall. |
| Boundary tags | bottom = 1, right = 2, top = 3, left = 4. |
| Curvature | NACA airfoil, circular cylinder, spherical frustum. |
| Outputs | plot initial mesh and deformed mesh. |

## Boundary-Layer Mesh Prompt

```text
Create a MATLAB mesh helper named mkmesh_platebottomgaussian.m in
Exasim/examples/LinearElasticity/platebottomgaussian2d.

Geometry:
- rectangular plate domain (-0.5,0.5) x (0,0.2)
- quadrilateral elements
- nx = 120, ny = 48
- geometric refinement toward y = 0 with user parameter growth

Boundary tags:
- bottom boundary: displaced Gaussian boundary
- left, right, top: fixed boundaries

Return an Exasim mesh structure compatible with LinearElasticity examples.
Include a simple mesh plot in pdeapp.m.
```

## Mesh Review Checklist

- Plot the mesh before running the solver.
- Check element orientation and no inverted cells.
- Confirm boundary tags by plotting or printing boundary faces.
- Verify curved nodes if using high-order curved geometry.
- Run first with low polynomial order.
- Verify `mesh.boundarycondition` matches the model callback cases.
- For boundary-layer meshes, check growth ratio and minimum element size.

## Debugging Mesh Problems

Useful AI prompt:

```text
This Exasim example fails during preprocessing with the attached error.
Inspect mkmesh_*.m, boundaryexpr, boundarycondition, and element orientation.
Find the most likely mesh or boundary-tag issue and propose a minimal fix.
```

Provide the error message, the mesh helper, `pdeapp`, and any plot that shows
the problematic geometry.
