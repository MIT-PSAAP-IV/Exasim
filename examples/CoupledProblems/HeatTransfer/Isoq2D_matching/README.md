# Isoq2D Conformal Heat Transfer

Run `pdeapp.m` to solve the axisymmetric Navier-Stokes equations in the
quadrilateral fluid mesh and steady heat conduction in a triangular solid
mesh. Gmsh generates the solid triangulation beneath the fluid wall.

The solid mesh generator constrains every fluid-wall grid segment to one
Gmsh boundary edge, then copies the fluid grid vertices and high-order DG
face nodes onto the solid interface. The resulting interface is geometrically
conformal. Since the two models use different element families, the coupled
solve selects `interfacetype = 1`.

The Gmsh interior target size is `solidMeshSize = 0.002` in `pdeapp2.m`.
Decrease this value to refine the solid further without changing the shared
fluid-solid interface nodes.

`pdeapp1.m` remains a stand-alone flow example. The coupled driver first runs
that complete flow solve, copies the converged conservative and gradient state
into the fluid mesh, and then performs the monolithic fluid-solid solve. The
coupled run uses four MPI ranks for the fluid and two for the solid; both models
use the fluid model's Krylov, preconditioner, and Newton controls. These controls
must agree because the coupled problem is solved as one monolithic system.
