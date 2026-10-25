# Generalized mixed-mesh interaction

This example is the generalized-interaction counterpart of `../mixedmeshes`.
It intentionally keeps the two physical domains, triangle/quad meshes, PDE
models, polynomial order, and uneven `[2,3]` MPI rank split unchanged.  The
only coupling change is that `interaction.m` describes the edge and both
legacy `mesh.interfacecondition` arrays are empty.

Run `pdeapp.m` from MATLAB.  The manufactured solution is
`sin(pi*x)*sin(pi*y)`.  Set `triangleOnModel1 = false` to reverse the element
families, or `curved = true` to exercise the curved-interface variant.

This one-edge graph is deliberate: it provides a direct legacy-versus-graph
regression before adding more edges.  A generalized run should converge to the
same solution as `../mixedmeshes` with roundoff-level differences when both
use the same rank split.

For the default straight-interface case, the reference comparison uses five
MPI ranks split as `[2,3]`.  Both paths should start with residual `0.182279`,
take 159 GMRES iterations, and finish the Newton update at approximately
`2.36e-11`.  With identical generated meshes, the generalized and legacy
solution files should be byte-identical; their maximum nodal error against the
manufactured solution is approximately `1.64e-5`.
