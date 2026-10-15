# Isoq2D Non-Matching Heat Transfer

Run `pdeapp.m` to solve the axisymmetric Navier-Stokes equations in the
quadrilateral fluid mesh and steady heat conduction in a triangular solid
mesh. Gmsh generates the solid triangulation beneath the fluid wall.

The fluid-wall vertices define a spline for the top of the solid, but Gmsh
chooses the solid interface nodes and interior triangulation independently.
Those independently chosen interface nodes are projected onto the fluid's
high-order wall geometry. The two discrete interface polynomials consequently
track the same wall closely without sharing grid nodes, DG nodes, or face
partitions. The coupled solve uses `interfacetype = 1`.

The Gmsh target size is `solidMeshSize = 0.0028` in `pdeapp2.m`. Tight minimum
and maximum size bounds produce roughly 800 triangles with a narrow area
distribution. Decrease this value to refine the solid independently.

`pdeapp1.m` remains a stand-alone flow example. The coupled driver first runs
that complete flow solve, copies the converged conservative and gradient state
into the fluid mesh, and then performs the monolithic fluid-solid solve. The
coupled run uses four MPI ranks for the fluid and two for the solid; both models
use the fluid model's Krylov, preconditioner, and Newton controls. These controls
must agree because the coupled problem is solved as one monolithic system.
