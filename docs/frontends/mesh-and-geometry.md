# Mesh And Geometry

The frontends support both generated meshes and imported meshes. The mesh must
provide element connectivity, node coordinates, and boundary information that
matches the PDE model boundary-condition IDs.

## Mesh Responsibilities

The mesh object supplies:

- physical coordinates and element connectivity;
- element type and geometric dimension;
- boundary group classification;
- optional high-order or curved geometry data;
- optional periodic boundary metadata;
- partitioning input for MPI preprocessing.

The frontend preprocessing step converts this information into backend `mesh`
and `dmd` binary data.

## Generated Meshes

Many examples use frontend mesh helper functions. Exact helper names differ by
language and example, but the resulting structure is the same.

=== "MATLAB"

    ```matlab
    mesh = mkmesh_square(16, 16, 1, 1, pde.porder, 1);
    mesh.boundarycondition = [1; 1; 1; 1];
    ```

=== "Python"

    ```python
    # Use the mesh helpers available in the Python frontend/examples.
    # The output must fill mesh["p"], mesh["t"], and boundary metadata.
    pde, mesh = initializeexasim()
    ```

=== "Julia"

    ```julia
    pde, mesh = Exasim.initializeexasim()
    # Use Exasim mesh helpers or assign mesh.p, mesh.t, and boundary metadata.
    ```

## Imported Meshes

Imported meshes should be normalized to the same frontend mesh fields before
calling `exasim(...)`. For Gmsh-based workflows, configure `pde.gmsh` if the
binary is not on `PATH`.

```matlab
pde.gmsh = "/path/to/gmsh";
% Generate or import mesh, then set mesh.boundarycondition consistently.
```

## Boundary Tagging

Boundary tagging is geometric in the frontend and semantic in the PDE model.
The mesh decides which face belongs to which boundary group; the PDE model
decides what that boundary group means physically.

Example:

```matlab
tol = 1e-8;
mesh.boundaryexpr = {
    @(p) abs(p(:,2)) < tol, ...
    @(p) abs(p(:,1)-1) < tol
};
mesh.boundarycondition = [1; 2];
```

For robust results, boundary predicates should include tolerances and should be
mutually consistent with the generated or imported mesh coordinates.

## Curved Meshes

Curved geometry is represented through high-order geometry nodes such as
`mesh.dgnodes` when used by an example. The polynomial order, geometry order,
and model dimension fields in `pde` must be consistent with this data.

Assumption: the mesh geometry is independent of `physicsparam` in the current
parameter-sweep workflow. If a parameter changes the geometry, regenerate and
preprocess the mesh explicitly for that case.

## Mesh Partitioning

`pde.mpiprocs` controls the number of MPI ranks used by preprocessing and
runtime execution. Parallel runs produce rank-local `datain` and `dataout`
files, and `dmd` records how to reconstruct global fields.

```python
pde["mpiprocs"] = 4
sol, pde, mesh = exasim(pde, mesh)
```

For a parameter sweep with fixed mesh and fixed `mpiprocs`, `master` and `dmd`
are reused across cases.

## See Also

- [Core data structures](data-structures.md)
- [Execution workflow](execution.md)
- [Postprocessing and visualization](postprocessing.md)
