# Isoq2D Non-Matching Heat Transfer

Run `pdeapp.m`, `pdeapp.py`, or `pdeapp.jl` to solve the axisymmetric
Navier-Stokes equations in the quadrilateral fluid mesh and steady heat
conduction in a triangular solid mesh.

The fluid-wall vertices define a spline for the top of the solid, but Gmsh
chooses the solid interface nodes and interior triangulation independently.
Those independently chosen interface nodes are projected onto the fluid's
high-order wall geometry. The two discrete interface polynomials consequently
track the same wall closely without sharing grid nodes, DG nodes, or face
partitions. The coupled solve uses `interfacetype = 1`.

The checked-in `canonical_mesh/` payload was generated from the MATLAB mesh
routines with Gmsh target size `0.0028`. It stores the fluid and solid grid
nodes, connectivity, boundary tags, curved DG nodes, and initial state in a
small frontend-neutral column-major format. All three frontends load this same
payload, so they solve the same discrete mesh rather than independently asking
Gmsh and three numerical libraries to reconstruct it. The retained
`mkmesh_isoq2d.m`, `mkmesh_isoqsolid.m`, and `isoq_mesh.py/.jl` routines remain
the documented source algorithms for regenerating or modifying the geometry.
The three drivers also compile the same checked-in generated model sources from
`canonical_kernels/`. Each frontend keeps ownership of its binary preprocessing
while the Kokkos physics kernels remain byte-identical. The canonical sources
were generated from the MATLAB model definitions used by the original case.
The payload also includes the quadrilateral and triangular `master.bin` files;
this avoids frontend-specific round-off in basis construction from steering
the shock-containing precursor into different nonlinear paths.
The precursor also uses a checked-in eight-step AV continuation schedule. This
removes the last libm-dependent `expm1` rounding differences between MATLAB,
NumPy, and Julia while retaining the original continuation path.

`pdeapp1.m` remains a stand-alone flow example. The coupled driver first runs
that complete flow solve, copies the converged conservative and gradient state
into the fluid mesh, and then performs the monolithic fluid-solid solve. The
coupled run uses four MPI ranks for the fluid and two for the solid; both models
use the fluid model's Krylov, preconditioner, and Newton controls, including
the ten-iteration Newton cap. These controls must agree because the coupled
problem is solved as one monolithic system. MATLAB opts into the same external
combined-model path used by Python and Julia, with model slots 0 and 1.
Both meshes use the shared C++ partition preprocessing path. In particular,
this makes the two-rank triangular solid partition identical across all three
frontends instead of relying on three independently maintained face
partitioners.

After running all three native drivers, verify the discrete inputs, generated
Kokkos sources, geometry, and solutions with:

```bash
python3 ../verify_frontends.py Isoq2D_nonmatching
```

For this case the verifier requires the input binaries and generated Kokkos
source sets to be byte-identical before it compares the computed fields.
